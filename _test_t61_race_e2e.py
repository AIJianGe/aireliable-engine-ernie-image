"""T6.1 端到端测试：LOADING 窗口期并发反代请求不再 502。

复现用户场景（2026-08-21 10:58:35 GET / → 502）：
    引擎 server 启动（ComfyUI 子进程未运行）→ 浏览器并发请求 /、favicon.ico、
    /api/jobs 等 → 旧代码中 auto_load_if_needed 第 1 分支直接放行（is_running=True
    但健康检查未通过）→ 连接未监听端口 → 502。

    修复后：并发请求在 wait_startup_done() 等待首个 auto-load 完成（含健康检查），
    等待期间请求挂起（非 502），就绪后返回 200。

验证：
    1. 测试端口起引擎 server（7002，ComfyUI 未启动）
    2. 立即并发 3 个 GET /（模拟浏览器同时拉页面 + 资源）
    3. 断言：全部 200（等待 ComfyUI 就绪后返回），无 502
    4. 期间记录每个请求的耗时（应包含 ComfyUI 启动时间 ~18-30s）
    5. 停止 ComfyUI + server

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_t61_race_e2e.py
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

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.server import create_app, get_engine

TEST_PORT = 7002
N_CONCURRENT = 3


async def main() -> int:
    print("=" * 60)
    print("T6.1 端到端测试：LOADING 窗口期并发反代（不再 502）")
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
    # 等 uvicorn 端口就绪
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.2)
    if not server.started:
        print("uvicorn 启动失败")
        return 1
    print(f"[1/4] 引擎 server 已启动（port={TEST_PORT}，ComfyUI 未运行，状态 IDLE）")
    print()

    # 2. 并发 3 个 GET /（复现用户浏览器行为：页面 + 资源并发拉取）
    print(f"[2/4] 立即并发 {N_CONCURRENT} 个 GET /（模拟 LOADING 窗口期访问）...")
    print(f"      （首个请求触发 auto-load；并发请求应等待而非 502，ComfyUI 启动约 18-30s）")

    async def fetch_one(idx: int) -> tuple[int, float, int]:
        async with httpx.AsyncClient(timeout=400.0) as c:
            t = time.time()
            r = await c.get(f"http://127.0.0.1:{TEST_PORT}/")
            return r.status_code, time.time() - t, len(r.content)

    results = await asyncio.gather(*[fetch_one(i) for i in range(N_CONCURRENT)])
    print()

    # 3. 断言
    print("[3/4] 验证结果...")
    all_ok = True
    for i, (code, dur, size) in enumerate(results):
        ok = code == 200 and size > 1000  # ComfyUI Web UI HTML 页面
        tag = "PASS" if ok else "FAIL"
        print(f"      请求{i + 1}: status={code}, 耗时={dur:.1f}s, 大小={size}B  [{tag}]")
        if not ok:
            all_ok = False
    print(f"      （验证耗时 {time.time() - t0:.1f}s，含 ComfyUI 启动）")
    print()

    # 4. 清理：停 ComfyUI + server
    print("[4/4] 清理...")
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
    print("T6.1 端到端测试: " + ("PASS ✅（并发请求等待就绪后 200，无 502）" if all_ok else "FAIL ❌"))
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
