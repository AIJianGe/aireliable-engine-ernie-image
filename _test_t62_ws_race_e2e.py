"""T6.2 端到端测试：LOADING 窗口期并发 HTTP + WS 反代（不再 502 / 1225）。

复现用户场景（2026-08-21 11:08:46 日志）：
    引擎 server 运行中，ComfyUI 子进程未启动（IDLE）。浏览器打开
    http://127.0.0.1:7001/ → 页面加载（HTTP）+ WS 连接（/ws?clientId=...）并发到达。
    旧代码：WS 路径仅在"未运行"时调 auto_load_if_needed，LOADING 窗口期
    （is_running=True 但健康检查未过，~18-30s）直接 accept + connect 未监听端口
    → WinError 1225，浏览器每 3s 重连全部失败，直到 ComfyUI 就绪。

    修复后（v0.4.2）：HTTP 与 WS 路径统一走 auto_load_if_needed →
    wait_startup_done 等待启动流程完成（含健康检查）→ 就绪后放行。

验证：
    1. 测试端口起引擎 server（7002，ComfyUI 未启动，IDLE）
    2. 立即并发：1 个 HTTP GET / + 2 个 WS 连接 /ws?clientId=...
    3. 断言：
       a. HTTP 200（等待就绪后返回 ComfyUI Web UI HTML）
       b. 两个 WS 均成功建立（无 1225 / 连接拒绝）
       c. WS 建立后保持 3s 不断开（稳定性）
       d. 日志中"启动 ComfyUI 子进程"仅出现 1 次（无并发双载导致的 unload+重启）
    4. 停止 ComfyUI + server

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_t62_ws_race_e2e.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

import httpx
import uvicorn
import websockets

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.server import create_app, get_engine

TEST_PORT = 7002
LOG_FILE = PROJECT_ROOT / "logs" / "engines" / "ernie_image_909469ed.log"


async def main() -> int:
    print("=" * 60)
    print("T6.2 端到端测试：LOADING 窗口期并发 HTTP + WS 反代")
    print("=" * 60)
    t0 = time.time()

    # 1. 测试配置 + app（引擎 IDLE，ComfyUI 未启动）
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        port=TEST_PORT,
        internal_port=None,
        server_timeout=300.0,
    )
    app = create_app(config)

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=TEST_PORT, log_level="warning",
    ))
    server_task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.2)
    if not server.started:
        print("uvicorn 启动失败")
        return 1
    print(f"[1/5] 引擎 server 已启动（port={TEST_PORT}，ComfyUI 未运行，状态 IDLE）")
    print()

    # 记录测试前的日志大小（用于后续统计"启动 ComfyUI 子进程"出现次数）
    log_size_before = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    # 2. 并发：1 个 HTTP GET / + 2 个 WS 连接（复现浏览器打开页面的行为）
    print("[2/5] 立即并发请求（1 HTTP + 2 WS，模拟浏览器打开 ComfyUI UI）...")
    print("      （LOADING 窗口约 18-30s，期间请求应挂起等待而非失败）")

    async def fetch_http() -> tuple[int, float, int]:
        async with httpx.AsyncClient(timeout=400.0) as c:
            t = time.time()
            r = await c.get(f"http://127.0.0.1:{TEST_PORT}/")
            return r.status_code, time.time() - t, len(r.content)

    async def connect_ws(client_id: str) -> tuple[bool, float, str]:
        """连接引擎 WS 反代，建立后保持 3s 验证稳定性。"""
        t = time.time()
        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{TEST_PORT}/ws?clientId={client_id}",
                open_timeout=400.0,   # LOADING 窗口期等待（握手挂起直到就绪）
                close_timeout=5.0,
            ) as ws:
                dur = time.time() - t
                # 建立 3s 稳定性验证（不发消息，仅保持连接）
                await asyncio.sleep(3.0)
                # 检查连接是否仍开
                closed = ws.state.name == "CLOSED"
                return (not closed), dur, ""
        except Exception as e:
            return False, time.time() - t, str(e)

    http_task = asyncio.create_task(fetch_http())
    ws1_task = asyncio.create_task(connect_ws("test_client_1"))
    ws2_task = asyncio.create_task(connect_ws("test_client_2"))
    # v0.4.2 加强：三个请求同时发起（无间隔），覆盖并发间隙
    # （load() 开始到 start() 创建 _start_done 的毫秒级窗口，
    #   验证 LOADING 轮询等待分支 + 无双载）

    http_code, http_dur, http_size = await http_task
    ws1_ok, ws1_dur, ws1_err = await ws1_task
    ws2_ok, ws2_dur, ws2_err = await ws2_task
    print()

    # 3. 断言 HTTP + WS
    print("[3/5] 验证结果...")
    all_ok = True

    ok = http_code == 200 and http_size > 1000
    tag = "PASS" if ok else "FAIL"
    print(f"      HTTP GET /  : status={http_code}, 耗时={http_dur:.1f}s, 大小={http_size}B  [{tag}]")
    if not ok:
        all_ok = False

    for label, wok, wdur, werr in (("WS-1", ws1_ok, ws1_dur, ws1_err), ("WS-2", ws2_ok, ws2_dur, ws2_err)):
        tag = "PASS" if wok else "FAIL"
        err_info = f", err={werr}" if werr else ""
        print(f"      {label} /ws     : 建立={'成功' if wok else '失败'}, 耗时={wdur:.1f}s{err_info}  [{tag}]")
        if not wok:
            all_ok = False

    print(f"      （验证耗时 {time.time() - t0:.1f}s，含 ComfyUI 启动 + 3s WS 稳定期）")
    print()

    # 4. 断言无并发双载：日志中"启动 ComfyUI 子进程"仅出现 1 次
    print("[4/5] 验证无并发双载（日志检查）...")
    await asyncio.sleep(0.5)  # 等日志落盘
    n_starts = 0
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(log_size_before)
            new_log = f.read()
        n_starts = new_log.count("启动 ComfyUI 子进程:")
    ok = n_starts == 1
    tag = "PASS" if ok else "FAIL"
    print(f"      '启动 ComfyUI 子进程' 出现 {n_starts} 次（期望 1）  [{tag}]")
    if not ok:
        all_ok = False
    print()

    # 5. 清理
    print("[5/5] 清理...")
    try:
        eng = get_engine()
        if eng._process_manager.is_running:
            await eng._process_manager.stop()
            print("      ComfyUI 子进程已停止")
    except Exception as e:
        print(f"      停止 ComfyUI 失败（忽略）: {e}")
    server.should_exit = True
    await asyncio.wait_for(server_task, timeout=10.0)
    print("      server 已退出")

    print()
    print("=" * 60)
    print("T6.2 端到端测试: " + ("PASS ✅（HTTP 200 + WS 稳定建立，无 1225，无双载）" if all_ok else "FAIL ❌"))
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
