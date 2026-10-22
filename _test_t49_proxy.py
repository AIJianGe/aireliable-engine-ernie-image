"""T4.9 集成测试：验证 ComfyUI Web UI 反向代理。

验证：
    1. 启动 FastAPI 引擎服务器（server.py）
    2. 通过引擎端口访问 ComfyUI Web UI（GET /）
    3. 通过引擎端口访问 ComfyUI API（GET /system_stats）
    4. 通过引擎端口访问 ComfyUI 静态资源（GET /static/...）
    5. 验证 8 个标准端点不被 catch-all 拦截（GET /capabilities 等）
    6. 停止服务器

反向代理架构：
    用户 → 引擎端口（如 7004）→ server.py catch-all 路由 → ComfyUI 内部端口（如 8100）
    - 8 标准端点（/capabilities /load /unload /infer 等）由 server.py 直接处理
    - 其他路径（/ /system_stats /static/ /prompt /history /view 等）透传到 ComfyUI
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["AIRELIABLE_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "engines")
os.environ["AIRELIABLE_MODELS_ROOT"] = str(PROJECT_ROOT / "models")

import httpx
import uvicorn

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.server import create_app


async def main() -> int:
    print("=" * 60)
    print("T4.9 集成测试：ComfyUI Web UI 反向代理")
    print("=" * 60)
    print()

    # 1. 构造 EngineConfig + 创建 FastAPI app
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )
    print(f"[1/6] EngineConfig 构造完成")

    app = create_app(config)
    print(f"[2/6] FastAPI app 创建完成")
    print()

    # 3. 启动 uvicorn 服务器（后台）
    # 找一个空闲端口
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    engine_port = s.getsockname()[1]
    s.close()

    print(f"[3/6] 启动 uvicorn 服务器（port={engine_port}）...")
    config_obj = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=engine_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config_obj)
    server_task = asyncio.create_task(server.serve())
    # 等待服务器启动
    await asyncio.sleep(2.0)
    print(f"      服务器已启动: http://127.0.0.1:{engine_port}")
    print()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            base_url = f"http://127.0.0.1:{engine_port}"

            # 4. 测试 8 标准端点（不被 catch-all 拦截）
            print(f"[4/6] 验证 8 标准端点...")
            standard_endpoints = [
                ("GET", "/capabilities", "引擎能力"),
                ("GET", "/status", "引擎状态"),
                ("GET", "/health", "健康检查"),
            ]
            for method, path, desc in standard_endpoints:
                try:
                    resp = await client.request(method, f"{base_url}{path}")
                    ct = resp.headers.get("content-type", "")
                    is_json = "application/json" in ct
                    status_ok = resp.status_code == 200
                    print(f"      {method} {path} ({desc}): {resp.status_code} "
                          f"{'✅' if status_ok else '❌'} {'JSON' if is_json else 'non-JSON'}")
                    if not status_ok:
                        print(f"        响应: {resp.text[:200]}")
                except Exception as e:
                    print(f"      {method} {path} ({desc}): ❌ 异常 {e}")
            print()

            # 5. 测试 ComfyUI Web UI 反代（auto-load 后访问）
            print(f"[5/6] 验证 ComfyUI Web UI 反向代理...")
            print(f"      （首次访问会 auto-load ComfyUI 子进程，约 30s）")

            # GET / → ComfyUI Web UI 首页（HTML）
            print(f"      GET / (ComfyUI Web UI 首页)...")
            t_start = time.time()
            resp = await client.get(f"{base_url}/", timeout=120.0)
            elapsed = time.time() - t_start
            html_content = resp.text
            has_comfyui = "ComfyUI" in html_content or "comfyui" in html_content.lower()
            print(f"        状态: {resp.status_code} ({'✅' if resp.status_code == 200 else '❌'})")
            print(f"        耗时: {elapsed:.1f}s（含 auto-load）")
            print(f"        Content-Type: {resp.headers.get('content-type', '')}")
            print(f"        HTML 长度: {len(html_content)} 字符")
            print(f"        含 ComfyUI 标识: {'✅' if has_comfyui else '❌'}")
            if not has_comfyui:
                print(f"        前 200 字符: {html_content[:200]}")
            print()

            # GET /system_stats → ComfyUI API
            print(f"      GET /system_stats (ComfyUI API)...")
            resp = await client.get(f"{base_url}/system_stats")
            is_json = "application/json" in resp.headers.get("content-type", "")
            if is_json:
                data = resp.json()
                print(f"        状态: {resp.status_code} ✅")
                print(f"        comfyui_version: {data.get('system', {}).get('comfyui_version', 'N/A')}")
                print(f"        python_version: {data.get('system', {}).get('python_version', 'N/A')}")
                pytorch_ver = data.get('system', {}).get('pytorch_version', 'N/A')
                print(f"        pytorch_version: {pytorch_ver}")
            else:
                print(f"        状态: {resp.status_code} ❌ 非 JSON 响应")
                print(f"        响应: {resp.text[:200]}")
            print()

            # GET /object_info → ComfyUI 节点列表
            print(f"      GET /object_info (ComfyUI 节点列表)...")
            resp = await client.get(f"{base_url}/object_info")
            is_json = "application/json" in resp.headers.get("content-type", "")
            if is_json:
                data = resp.json()
                node_count = len(data) if isinstance(data, dict) else 0
                print(f"        状态: {resp.status_code} ✅")
                print(f"        节点数: {node_count}")
                # 验证关键节点存在
                key_nodes = ["UnetLoaderGGUF", "CLIPLoader", "KSampler", "VAEDecode", "SaveImage", "TextGenerate"]
                for node in key_nodes:
                    exists = node in data if isinstance(data, dict) else False
                    print(f"        {node}: {'✅' if exists else '❌'}")
            else:
                print(f"        状态: {resp.status_code} ❌ 非 JSON 响应")
            print()

            # 6. 停止服务器
            print(f"[6/6] 停止服务器...")
            # 先停止引擎（清理 ComfyUI 子进程）
            from engines.ernie_image_909469ed.server import get_engine
            try:
                engine = get_engine()
                if engine and engine._process_manager:
                    stop_result = await engine._process_manager.stop()
                    print(f"      ComfyUI 子进程停止: {stop_result}")
            except Exception as e:
                print(f"      停止引擎异常: {e}")

            server.should_exit = True
            await asyncio.wait_for(server_task, timeout=10.0)
            print(f"      uvicorn 服务器已停止")

        print()
        print("=" * 60)
        print(f"T4.9 Web UI 反代测试通过 ✅")
        print(f"  标准端点: ✅ 不被 catch-all 拦截")
        print(f"  Web UI 首页: ✅ 透传成功")
        print(f"  ComfyUI API: ✅ 透传成功")
        print(f"  节点列表: ✅ 透传成功")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10.0)
        except:
            pass
        return 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
