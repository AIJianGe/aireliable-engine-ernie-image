"""ERNIE-Image 引擎 HTTP 服务（v0.3.0 ComfyUI 子进程模式 + 反代 ComfyUI Web UI）。

8 个标准端点 / CORS / 日志端点 / 静态挂载由 SDK 骨架（``build_engine_app``）
提供；本文件保留引擎特有路由（``/load`` 的 GPU/模型错误映射 + ComfyUI 反代）。

启动：
    uv run --no-sync --project envs/image-ernie-comfyui \\
        python -m engines.ernie_image_909469ed.server --port 7001

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用 GGUF variant 列表 + LoRA 列表 + 支持的分辨率）
    POST /load           启动 ComfyUI 子进程（不预加载模型，首次推理时由 ComfyUI 自动加载）
    POST /unload         释放 ComfyUI 显存（调 /free）或停止子进程
    POST /infer          推理（SSE 流式，提交工作流到 ComfyUI + 订阅 WS 事件 + 下载图片）
    GET  /status         状态查询（含 ComfyUI 子进程状态 + 显存占用）
    POST /lora/load      记录已加载的 LoRA（运行时通过工作流模板注入 LoraLoader 节点）
    POST /lora/unload    从 loaded_loras 移除
    GET  /output/*       静态文件（生成的图片）
    ANY  /{path}         反向代理 → ComfyUI Web UI（最后挂载）

为什么 ``/load`` 由本文件自实现（``skip_endpoints=["/load"]``）：
    本引擎需要把「模型文件缺失」映射为 503 ``model_not_ready``、
    把「CUDA OOM / 子进程启动失败」映射为 503 ``subprocess_failed``，
    与骨架通用的 404/400/500 口径不同，故跳过骨架端点、在本文件实现。

参考实现：``engines/llama_cpp_9cdb76f8/server.py``
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from aireliable_engine import (
    ErrorResponse,
    LoadRequest,
    RuntimeState,
    build_engine_app,
    run_engine_server,
)

from .config import EngineConfig, parse_args
from .engine import ErnieImageEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[ErnieImageEngine] = None
_config: Optional[EngineConfig] = None
# v0.3.9+ 全局复用 AsyncClient（避免每次反代请求新建连接池，5.8MB 文件从 14s 降到 <0.5s）
_proxy_client: Optional[httpx.AsyncClient] = None


def get_engine() -> ErnieImageEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== v0.3.1+ asyncio 噪声日志过滤 =====

def _setup_asyncio_noise_filter() -> None:
    """设置自定义 asyncio 异常处理器，过滤 Windows ProactorEventLoop 噪声日志。

    过滤目标：
        - ConnectionResetError [WinError 10054]：健康检查轮询时 socket 关闭触发
        - _ProactorBasePipeTransport._call_connection_lost 回调异常

    这些异常是 Windows asyncio 已知缺陷，不影响功能（健康检查最终成功）。
    仅过滤特定组合，其他异常仍走默认 handler。
    """
    import asyncio
    import sys

    if sys.platform != "win32":
        return  # 仅 Windows 需要

    loop = asyncio.get_event_loop()

    def _noise_filter(loop, context):
        # 获取异常和回调信息
        exception = context.get("exception")
        callback = context.get("handle")

        # 过滤 ConnectionResetError [WinError 10054]
        if isinstance(exception, ConnectionResetError):
            cb_repr = repr(callback) if callback else ""
            if "_call_connection_lost" in cb_repr or "WinError 10054" in str(exception):
                logger.debug(
                    "已过滤 asyncio 噪声: ConnectionResetError [WinError 10054] "
                    "（Windows ProactorEventLoop 已知缺陷，不影响功能）"
                )
                return

        # 过滤 _call_connection_lost 相关的异常
        if callback is not None:
            cb_repr = repr(callback)
            if "_call_connection_lost" in cb_repr:
                logger.debug(
                    "已过滤 asyncio 噪声: _call_connection_lost 异常 (%s): %s",
                    type(exception).__name__ if exception else "unknown",
                    exception,
                )
                return

        # 其他异常走默认 handler
        loop.default_exception_handler(context)

    loop.set_exception_handler(_noise_filter)
    logger.debug("已设置 asyncio 噪声过滤器（过滤 WinError 10054）")


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: ErnieImageEngine) -> None:
    """启动钩子：绑定全局实例 + 装噪声过滤器 + 初始化反代连接池。"""
    global _engine, _proxy_client

    _engine = engine
    # v0.3.1+ 设置自定义异常处理器
    _setup_asyncio_noise_filter()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, available=%d, loras=%d",
        engine.state_machine.state.value,
        len(engine._manifest_variants),
        len(engine._available_variants),
        len(engine._available_loras),
    )

    # v0.3.9+ 初始化全局反代 AsyncClient（连接池复用，避免每次请求新建）
    _proxy_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=10.0),
        follow_redirects=False,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        trust_env=False,  # 平台内部回环（→ComfyUI），不走系统代理
    )
    logger.info("全局反代 AsyncClient 已初始化（连接池: max=100, keepalive=20）")


async def _on_shutdown(engine: ErnieImageEngine) -> None:
    """关闭钩子：关闭反代 client + 停止 ComfyUI 子进程 + 清空全局引用。"""
    global _engine, _config, _proxy_client

    logger.info("===== ERNIE-Image 引擎关闭 =====")
    try:
        if _proxy_client is not None:
            try:
                await _proxy_client.aclose()
            except Exception as e:
                logger.warning("关闭反代 AsyncClient 失败: %s", e)
            _proxy_client = None
        try:
            await engine.shutdown()
        except Exception as e:
            logger.warning("关闭时清理资源失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


@router.post("/load")
async def load(request: LoadRequest) -> dict:
    """启动 ComfyUI 子进程（若未启动）+ 显存上报 L3。

    ComfyUI 子进程启动后不预加载模型（模型在首次推理时由 ComfyUI 自动加载）。
    错误口径与迁移前一致：
        - 模型文件缺失 → 503 ``model_not_ready``（fail fast）
        - GPU 资源不足（OOM 等）/ 子进程启动失败 → 503 ``subprocess_failed``
        - 档位名非法 → 400；其他 → 500
    """
    try:
        response = await get_engine().load(request)
        return response.model_dump()
    except FileNotFoundError as e:
        # 模型文件缺失，返回 503 + error_type=model_not_ready（fail fast）
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error={
                "type": "model_not_ready",
                "message": str(e),
                "code": 503,
            }).model_dump(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # GPU 资源不足（CUDA OOM）/ 子进程启动失败
        msg = str(e).lower()
        gpu_error_keywords = (
            "cuda out of memory",
            "out of memory",
            "cuda error",
            "cudnn",
            "rocm",
            "hip out of memory",
            "device-side assert",
        )
        if any(kw in msg for kw in gpu_error_keywords):
            logger.error("GPU 资源不足，加载失败: %s", e)
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(error={
                    "type": "subprocess_failed",
                    "message": f"GPU 资源不足: {e}",
                    "code": 503,
                }).model_dump(),
            )
        # 其他 RuntimeError 走通用 500 错误
        logger.exception("加载失败（RuntimeError）")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("加载失败")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 状态机恢复 + auto-load 公共 helper（v0.3.7+）=====

async def auto_load_if_needed(engine: ErnieImageEngine) -> None:
    """确保 ComfyUI 子进程已启动，必要时清理状态机并重新加载。

    v0.3.7+ 关键修复：状态机与子进程状态不一致恢复
    ----------------------------------------------------------------
    历史问题：核心服务重启/异常退出时，Python 引擎父进程崩溃但状态机
    卡在 READY/INFERRING；子进程被 OS 通过 Job Object 自动 kill；之后
    所有 auto-load 都会因为 `ready → loading` 非法转换失败，陷入死循环。

    修复策略（先清理后加载）：
        1. 子进程运行中 → 直接 return
        2. 子进程死了 + 状态机 READY → 调 unload() 清理状态（READY → IDLE）
        3. 子进程死了 + 状态机 INFERRING/ERROR → 强制 reset 到 IDLE
        4. 子进程死了 + 状态机 IDLE/LOADING → 直接 load()

    Args:
        engine: ErnieImageEngine 实例

    Raises:
        RuntimeError: 状态清理后 _load_impl 仍失败（文件缺失、ComfyUI 启动失败等）
    """
    pm_local = engine._process_manager
    sm = engine.state_machine

    # 1. 统一入口：等待任何进行中的启动流程完成（v0.4.2 重构）
    # 覆盖三种窗口期（v0.4.1 修复 HTTP 502 / v0.4.2 修复 WS 1225 + 并发双载）：
    #   a. LOADING 窗口：Popen 已创建 + 端口已分配，但健康检查未通过（实测 ~18-30s，
    #      含 custom_nodes 加载 + DB 迁移）。此时 is_running=True 但端口未监听，
    #      直接放行会连接被拒（HTTP 502 / WS WinError 1225）
    #   b. Popen 创建间隙：另一请求的 load() 已开始（_start_done 已创建）但 Popen
    #      尚未创建，is_running 仍为 False。旧代码会误判"LOADING 但子进程已死"
    #      → 强制 reset → 等锁后二次 load（首个 load 完成 READY 即被 unload 重启）
    #   c. 早已就绪：_start_done 已 set → wait 立即返回，零开销
    # wait_startup_done 语义：_start_done 为 None（无启动流程）时直接返回 is_running，
    # 不阻塞；有进行中的启动流程则等待其结束（成功/失败均解除阻塞）。
    ok = await pm_local.wait_startup_done(timeout=200.0)
    if ok and pm_local.is_running and pm_local.internal_url is not None:
        return
    # 等待失败（启动流程异常结束 / 超时）：落入下方状态清理 + 重新 load 逻辑
    logger.warning(
        "auto_load_if_needed：等待子进程启动完成失败（状态机=%s），尝试清理重载",
        sm.state.value,
    )

    # 2. 子进程死了但状态机卡住：先清理状态再 load
    current_state = sm.state
    if current_state == RuntimeState.READY:
        logger.warning(
            "auto_load_if_needed：状态机 READY 但子进程已死，"
            "先调 unload() 清理状态（避免 ready → loading 非法转换）"
        )
        try:
            await engine.unload()
            logger.info("auto_load_if_needed：unload 完成，状态机已转 IDLE")
        except Exception as e:
            logger.warning("auto_load_if_needed：unload 失败，强制 reset 状态机: %s", e)
            try:
                sm.reset()
            except Exception:
                pass
    elif current_state in (RuntimeState.INFERRING, RuntimeState.ERROR):
        logger.warning(
            "auto_load_if_needed：状态机 %s 但子进程已死，强制 reset 到 IDLE",
            current_state.value,
        )
        try:
            if current_state == RuntimeState.ERROR:
                sm.reset()  # ERROR → IDLE
            else:
                # INFERRING 是非终态：强制转 ERROR 再 reset
                sm.transition(RuntimeState.ERROR)
                sm.reset()
        except Exception as e:
            logger.warning("auto_load_if_needed：状态机 reset 失败: %s", e)
    elif current_state == RuntimeState.LOADING:
        # v0.4.2 修复并发间隙双载：状态机 LOADING 说明有 load() 正在进行
        # （可能处于 load() 开始到 start() 创建 _start_done 的毫秒级间隙，
        # 此时 _start_done=None 且 is_running=False，上方 wait 无法感知）。
        # 旧逻辑直接 reset 会破坏进行中的 load（其完成后 transition(READY)
        # 因状态已被 reset 成 IDLE 而非法转换报错），且本请求获取锁后会发现
        # READY → unload + 重新 load（首个 load 刚完成就被重启，双载）。
        # 现改为轮询等待状态离开 LOADING（上限与启动超时对齐 200s）：
        #   - load 成功（READY + is_running）→ 直接放行
        #   - load 失败（ERROR）→ 落入上方既有 ERROR 清理分支（本轮 elif
        #     已过，直接落到下方 load 重试，ERROR 分支下轮请求处理）
        #   - 超时 → 落入下方 load 重试
        logger.info(
            "auto_load_if_needed：状态机 LOADING（并发 load 进行中），等待其完成"
        )
        deadline = time.monotonic() + 200.0
        while sm.state == RuntimeState.LOADING and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if pm_local.is_running and pm_local.internal_url is not None:
            logger.info("auto_load_if_needed：并发 load 已完成，子进程就绪")
            return

    # 3. 此时状态机应该是 IDLE，调 load 启动 ComfyUI 子进程
    logger.info(
        "auto_load_if_needed：触发 load(variant=None) 启动 ComfyUI 子进程"
    )
    await engine.load(LoadRequest(variant=None))
    logger.info("auto_load_if_needed：成功，ComfyUI 子进程已启动")


# ===== 反向代理 ComfyUI Web UI（最后挂载）=====

# 未 /load 时返回的友好提示页（503）
_UNREADY_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ERNIE-Image 引擎未启动</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         display: flex; align-items: center; justify-content: center;
         min-height: 100vh; margin: 0; background: #f5f5f5; color: #333; }
  .card { background: white; padding: 40px; border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 480px; text-align: center; }
  h1 { font-size: 20px; margin: 0 0 16px; color: #d97706; }
  p { font-size: 14px; line-height: 1.6; color: #666; margin: 8px 0; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
         font-family: Consolas, Monaco, monospace; color: #c41e3a; }
  .hint { margin-top: 20px; padding-top: 16px; border-top: 1px solid #eee;
          font-size: 12px; color: #999; }
</style>
</head>
<body>
  <div class="card">
    <h1>⚠️ ERNIE-Image 引擎未启动</h1>
    <p>ComfyUI 子进程尚未运行，无法提供 Web UI 和推理服务。</p>
    <p>请先在平台详情页点击 <code>启动</code> 按钮（调 <code>POST /load</code>）。</p>
    <p>启动完成后刷新此页面即可访问 ComfyUI 自带 Web UI（用于调试和高级用户）。</p>
    <div class="hint">
      引擎端点：<code>http://127.0.0.1:7001</code><br>
      健康检查：<code>GET /health</code> &nbsp;|&nbsp; 状态：<code>GET /status</code><br>
      推理：<code>POST /infer</code>（SSE 流式）
    </div>
  </div>
</body>
</html>
"""


async def _ensure_subprocess_running() -> Optional[HTMLResponse]:
    """确保 ComfyUI 子进程已启动（反代路由 auto-load 辅助函数）。

    v0.3.0 语义：
        - 用户访问 UI 时自动启动 ComfyUI 子进程（不预加载模型）
        - 模型在首次推理时由 ComfyUI 自动加载
        - 行为对齐 llama_cpp 的 auto-load 逻辑

    v0.3.7+：委托给统一的 auto_load_if_needed helper（含状态机恢复），
    避免子进程死了但状态机卡在 READY 时直接 load 触发非法转换。

    若子进程已运行，返回 None（继续反代）。
    若子进程未运行，自动调 load(variant=None) 启动；
      - 启动成功返回 None（继续反代）
      - 启动失败返回 503 HTMLResponse（显示错误提示页）
    """
    engine = get_engine()
    try:
        await auto_load_if_needed(engine)
        return None
    except Exception as e:
        logger.exception("反代 auto-load 失败：%s", e)
        err_html = _UNREADY_HTML.replace(
            "请先在平台详情页点击 启动 按钮（调 POST /load）。",
            f"自动启动失败：{e}<br><br>请检查 ComfyUI 目录和模型文件是否完整，或手动调 POST /load。",
        )
        return HTMLResponse(content=err_html, status_code=503)


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
)
async def reverse_proxy(path: str, request: Request):
    """反向代理到 ComfyUI 内部端口。

    v0.3.0：子进程未启动时自动调 load（auto-load），
    而非返回 503 提示页。对齐 llama_cpp 的 UI auto-load 规范。

    Args:
        path: 路径（不含前导 /，如 "prompt" / "history" / "view" / "api"）
        request: FastAPI Request 对象
    """
    # 排除 OpenAPI / Swagger UI 路径（FastAPI 自带，但 catch-all 会拦截）
    if path in ("openapi.json", "docs", "redoc"):
        raise HTTPException(status_code=404, detail=f"未找到路径: /{path}")

    # auto-load：子进程未启动时自动启动
    err = await _ensure_subprocess_running()
    if err is not None:
        return err

    engine = get_engine()
    pm_local = engine._process_manager

    # 构造转发请求
    # v0.4.3 修复：使用 ASGI raw_path（原始未解码路径）构造转发 URL。
    # 原因：FastAPI {path:path} 参数已解码（%2F → /），f"{url}/{path}" 再经
    # httpx 发送时 / 不会被重新编码为 %2F。ComfyUI 前端请求 userdata 子目录
    # 文件时用 %2F 编码斜杠（绕过 aiohttp /userdata/{file} 路由中 {file}
    # 不匹配 / 的限制），经反代解码后 → ComfyUI 路由不匹配 → 404
    # （症状：Web UI 侧边栏双击打开 ERNIE-Image 子目录工作流报
    #  "Failed to load file ... 404 Not Found"，目录列表正常但单文件 404）。
    # raw_path 保留浏览器原始编码（%2F / 中文 UTF-8），转发后 ComfyUI
    # 行为与浏览器直连一致（已实测：直连 200，旧反代 404）。
    raw_path_bytes = request.scope.get("raw_path")
    if raw_path_bytes:
        # 原始百分号编码均为 ASCII，安全 decode
        forward_path = raw_path_bytes.decode("ascii", errors="backslashreplace")
    else:
        # fallback：raw_path 缺失时对已解码 path 重新编码（/ 保留）
        from urllib.parse import quote
        forward_path = quote("/" + path, safe="/%")
    target_url = f"{pm_local.internal_url}{forward_path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # 透传请求头（移除 hop-by-hop headers）
    forward_headers = _strip_hop_by_hop_headers(dict(request.headers))

    # ComfyUI origin_only_middleware 兼容（v0.3.8 修复 Web UI 403 问题）：
    # ComfyUI 会拒绝两类请求（返回 403）：
    #   1) Host 与 Origin 域名不匹配（防跨站 POST 劫持）
    #   2) Sec-Fetch-Site: cross-site
    # 反代场景下 Host 被 httpx 重写为内部地址（127.0.0.1:8100），而浏览器
    # 的 Origin（http://127.0.0.1:7001）被原样转发 → 触发不匹配 → 403。
    # module script / modulepreload 请求（<script type="module">、
    # <link rel="modulepreload">）以 CORS 模式发送、必带 Origin 头，
    # 普通 CSS / fetch / 导航请求不带，因此仅 JS 资源全部 403，
    # 页面停在 logo 加载界面。
    # 修复：移除 Origin / Referer / Sec-Fetch-* 头，让 ComfyUI 视为本地请求。
    for _h in ("origin", "referer", "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest"):
        forward_headers.pop(_h, None)

    # v0.3.9 性能修复：禁用上游压缩 + 大 chunk 流式
    # 原因：aiter_bytes() 默认 1KB chunk，5.8MB JS 文件需 5800+ 次 yield/await，
    #       实测反代 14.2s vs 直连 0.15s（慢 90 倍）。
    # 修复：
    #   1. Accept-Encoding: identity → ComfyUI 不压缩，返回原始字节
    #   2. aiter_raw(chunk_size=65536) → 64KB chunk，5.8MB 只需 ~90 次 yield
    #   3. 由于未压缩，无需移除 content-encoding（保持透传一致性）
    forward_headers["accept-encoding"] = "identity"

    # 读取请求体
    body = await request.body()

    logger.debug("反代转发: %s %s -> %s", request.method, path, target_url)

    # v0.3.9+ 使用全局复用 AsyncClient（连接池 + keepalive，避免每次请求新建）
    global _proxy_client
    client = _proxy_client
    if client is None:
        # 兜底：lifespan 未初始化时（不应发生）回退到临时 client
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False,  # 平台内部回环，不走系统代理
            follow_redirects=False,
        )
        logger.warning("全局 _proxy_client 未初始化，回退到临时 client（性能下降）")

    try:
        req = client.build_request(
            request.method,
            target_url,
            headers=forward_headers,
            content=body if body else None,
        )
        response = await client.send(req, stream=True)
    except Exception as e:
        # 仅临时 client 需要关闭；全局 client 由 lifespan 管理
        if _proxy_client is None:
            await client.aclose()
        logger.exception("反代转发失败: %s %s", request.method, path)
        return JSONResponse(
            status_code=502,
            content={"error": {
                "type": "ProxyError",
                "message": f"反向代理 ComfyUI 失败: {e}",
                "code": 502,
            }},
        )

    # 透传响应头（移除 hop-by-hop）
    # v0.3.9：由于请求时设置了 Accept-Encoding: identity，响应不再压缩，
    # 无需移除 content-encoding（保持与上游一致性）
    response_headers = _strip_hop_by_hop_headers(dict(response.headers))
    # 兜底：如果上游仍返回 content-encoding（不应发生），移除避免浏览器二次解压
    response_headers.pop("content-encoding", None)
    response_headers.pop("Content-Encoding", None)

    # 流式响应（SSE / chunked / 普通 JSON 都走这个分支）
    # v0.3.9：用 aiter_raw(chunk_size=65536) 替代 aiter_bytes()，
    # 64KB chunk 大幅减少 yield 次数（5.8MB: 5800 → 90 次）
    is_temp_client = _proxy_client is None

    async def stream_body():
        try:
            async for chunk in response.aiter_raw(chunk_size=65536):
                if chunk:
                    yield chunk
        finally:
            await response.aclose()
            if is_temp_client:
                await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=response.status_code,
        headers=response_headers,
    )


# ===== WebSocket 反向代理（ComfyUI /ws 端点用于实时事件 + 预览图）=====

@router.websocket("/{path:path}")
async def websocket_proxy(websocket: WebSocket, path: str):
    """WebSocket 反向代理到 ComfyUI。

    用于 ComfyUI 实时事件推送（进度 / 执行结果）和预览图流。
    v0.3.0：子进程未启动时自动调 load（auto-load，同 HTTP 反代）。
    v0.3.7+：使用统一的 auto_load_if_needed helper，含状态机恢复逻辑。

    注意：本引擎的 /infer 端点已经通过 ComfyuiClient.subscribe_ws 内部订阅
    ComfyUI WS 事件并转换为 AIReliable SSE 帧推送，前端通常不需要直连
    ComfyUI WS。此 WebSocket 反代主要用于：
        1. ComfyUI 自带 Web UI 的 /ws 连接（用户调试时访问 / 路径）
        2. 高级用户直接订阅 ComfyUI WS 事件（绕过引擎的 SSE 转换）
    """
    engine = get_engine()
    pm_local = engine._process_manager

    # v0.4.2 修复：WS 路径与 HTTP 路径统一走 auto_load_if_needed（无条件调用）。
    # 旧逻辑仅在"未运行"时调 helper，LOADING 窗口期（Popen 已创建 + 端口已分配
    # 但健康检查未通过，实测 ~18-30s）直接跳过 auto-load → accept → connect
    # 未监听端口 → WinError 1225（浏览器每 3s 重连，窗口期内全部失败）。
    # 统一后 helper 覆盖两种情况：
    #   - 未运行 → 状态清理 + load（await 完整启动含健康检查）
    #   - 运行中 → wait_startup_done 等待启动流程完成（就绪或失败）
    #   - 早已就绪（READY）→ 轻量检查直接返回
    try:
        await auto_load_if_needed(engine)
    except Exception as e:
        logger.exception("WS 反代 auto-load 失败：%s", e)
        await websocket.close(code=1011, reason=f"ComfyUI 启动失败: {e}")
        return

    # 接受客户端 WS 连接
    await websocket.accept()

    # 构造到 ComfyUI 的 WS URL
    # ComfyUI WS 端点固定为 /ws，可能带 clientId 查询参数
    target_ws_url = f"{pm_local.internal_url.replace('http', 'ws')}/{path}"
    if websocket.query_params:
        query_str = "&".join(f"{k}={v}" for k, v in websocket.query_params.items())
        target_ws_url = f"{target_ws_url}?{query_str}"

    logger.debug("WS 反代: /%s -> %s", path, target_ws_url)

    # 用 websockets 库连接 ComfyUI（FastAPI 已依赖）
    try:
        import websockets
    except ImportError:
        logger.error("websockets 库未安装，无法反代 WS")
        await websocket.close(code=1011, reason="WS 代理依赖缺失")
        return

    # v0.3.9 修复 WS 断连：过滤要转发的 headers
    # 原因：additional_headers=dict(websocket.headers) 会把客户端的
    #   Sec-WebSocket-* / Host / Origin / Connection / Upgrade 等头
    #   原样转发给 ComfyUI，websockets 库自己也会生成这些头 → 冲突
    #   导致握手失败或连接不稳定（前端表现为"已重新连接 → 又断开"）
    # 修复：只转发必要的非 WS 协议头（User-Agent / Cookie 等），
    #       不转发 Sec-WebSocket-* / Host / Origin / Connection / Upgrade 等
    ws_skip_headers = {
        "host", "origin", "connection", "upgrade", "content-length",
        "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
        "sec-websocket-protocol", "sec-websocket-accept",
        "sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest",
    }
    forward_ws_headers = {
        k: v for k, v in websocket.headers.items()
        if k.lower() not in ws_skip_headers
    }

    # 双向桥接：client <-> ComfyUI
    try:
        async with websockets.connect(
            target_ws_url,
            additional_headers=forward_ws_headers,
            max_size=None,
            ping_interval=20,      # v0.3.9: 启用 ping 保活，避免空闲断连
            ping_timeout=10,
            close_timeout=5,
        ) as upstream_ws:
            import asyncio

            async def client_to_upstream():
                """客户端 → ComfyUI。"""
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in msg and msg["bytes"] is not None:
                            await upstream_ws.send(msg["bytes"])
                        elif "text" in msg and msg["text"] is not None:
                            await upstream_ws.send(msg["text"])
                except WebSocketDisconnect:
                    logger.debug("WS 反代：客户端断开连接")
                except Exception as e:
                    logger.warning("WS 反代 client_to_upstream 异常: %s", e)

            async def upstream_to_client():
                """ComfyUI → 客户端。"""
                try:
                    async for msg in upstream_ws:
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception as e:
                    # v0.3.9: 不再 pass 吞掉异常，记录日志便于排查
                    logger.warning("WS 反代 upstream_to_client 异常: %s", e)

            # 并发执行两个方向的转发
            await asyncio.gather(
                client_to_upstream(),
                upstream_to_client(),
                return_exceptions=True,
            )
    except Exception as e:
        logger.warning("WS 反代异常: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _strip_hop_by_hop_headers(headers: dict) -> dict:
    """移除 hop-by-hop headers（不能被代理转发的头）。

    参考 RFC 7230 §6.1：以下 headers 必须由代理移除：
        Connection / Keep-Alive / Proxy-Authenticate / Proxy-Authorization /
        TE / Trailer / Transfer-Encoding / Upgrade

    另外移除 Host / Content-Length（由下游 client 重新计算）。
    """
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
    return {
        k: v for k, v in headers.items()
        if k.lower() not in hop_by_hop
    }


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        ErnieImageEngine,
        config=config,
        title="ERNIE-Image Engine",
        description=(
            "AIReliable 平台的 ERNIE-Image 引擎（v0.3.0 ComfyUI 子进程模式 + 反代 ComfyUI UI）"
        ),
        version="0.3.0",
        static_dirs={"/output": config.output_root},
        redirect_root=None,
        skip_endpoints=["/load"],
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global app

    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
