"""T8.1 端到端测试：反代 raw_path 修复（中文工作流 %2F URL 不再 404）。

复现用户场景（2026-08-21 浏览器控制台）：
    经 http://127.0.0.1:7001/ 打开 ComfyUI Web UI → 侧边栏 Workflows 双击
    ERNIE-Image 子目录工作流 → 前端 GET /api/userdata/workflows%2FERNIE-Image%2F
    ERNIE-Image%20Turbo%20%E5%9B%BE%E7%89%87%E7%BC%96%E8%BE%91.json → 404。

根因：反代用 FastAPI {path:path}（已解码，%2F → /）构造转发 URL，
ComfyUI aiohttp /userdata/{file} 路由的 {file} 不匹配含原始 / 的路径 → 404。
修复：转发 URL 改用 ASGI raw_path（原始未解码路径）。

验证：
    1. 经反代请求 %2F 编码的中文工作流 URL → 200 + JSON 内容正确
    2. 三个工作流全部验证
    3. 回归：普通路径（/ 、 /api/object_info、目录列表 API）→ 200
    4. 对照：%2F 编码的英文名文件 → 200

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_t81_rawpath_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

import httpx
import uvicorn

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.server import create_app, get_engine

TEST_PORT = 7002

PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


async def main() -> int:
    print("=" * 64)
    print("T8.1 端到端测试：反代 raw_path 修复（%2F 中文 URL 不再 404）")
    print("=" * 64)
    t0 = time.time()

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
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning"))
    server_task = asyncio.create_task(server.serve())
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.2)
    if not server.started:
        print("uvicorn 启动失败")
        return 1
    print(f"[1/3] 引擎 server 已启动（port={TEST_PORT}，ComfyUI 未运行）")
    print("      （首个请求将触发 auto-load，等待 ComfyUI 启动 18-30s）")

    base = f"http://127.0.0.1:{TEST_PORT}"
    async with httpx.AsyncClient(timeout=400.0) as c:
        # 2. 复现用户 404 场景：%2F 编码的中文工作流 URL（触发 auto-load + 反代）
        print("\n[2/3] 经反代请求三个中文工作流（%2F 编码，复现前端请求）...")
        workflows = [
            "workflows/ERNIE-Image/ERNIE-Image Turbo 文生图.json",
            "workflows/ERNIE-Image/ERNIE-Image Turbo 图生图.json",
            "workflows/ERNIE-Image/ERNIE-Image Turbo 图片编辑.json",
        ]
        for wf in workflows:
            url = f"{base}/api/userdata/" + quote(wf, safe="")
            r = await c.get(url)
            ok = r.status_code == 200
            name = wf.split("/")[-1]
            check(f"[{name}] GET %2F URL → 200（实际 {r.status_code}）", ok)
            if ok:
                try:
                    data = json.loads(r.text)
                    n = len(data.get("nodes", []))
                    check(f"[{name}] JSON 可解析，{n} 节点", n > 0 and "links" in data)
                except Exception as e:
                    check(f"[{name}] JSON 可解析", False, str(e))

        # 3. 回归：普通路径 + 目录列表
        print("\n[3/3] 回归：普通路径...")
        r1 = await c.get(f"{base}/")
        check(f"[/] Web UI 首页 → 200（实际 {r1.status_code}）", r1.status_code == 200)
        r2 = await c.get(f"{base}/api/object_info")
        check(f"[/api/object_info] → 200（实际 {r2.status_code}）", r2.status_code == 200)
        r3 = await c.get(f"{base}/api/userdata", params={"dir": "workflows", "recurse": "true", "split": "false", "full_info": "true"})
        ok3 = r3.status_code == 200 and any(
            "ERNIE-Image" in (f.get("path", "") if isinstance(f, dict) else str(f))
            for f in r3.json()
        )
        check(f"[目录列表] 含 ERNIE-Image 工作流（实际 {r3.status_code}）", ok3)

    # 清理
    try:
        eng = get_engine()
        if eng._process_manager.is_running:
            await eng._process_manager.stop()
            print("\nComfyUI 子进程已停止")
    except Exception as e:
        print(f"停止 ComfyUI 失败（忽略）: {e}")
    server.should_exit = True
    await asyncio.wait_for(server_task, timeout=10.0)
    print("server 已退出")

    print("\n" + "=" * 64)
    print(f"结果：PASS={PASS}  FAIL={FAIL}（总耗时 {time.time()-t0:.1f}s）")
    print("=" * 64)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
