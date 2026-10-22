"""T6.1 单元测试：wait_startup_done 竞态修复（不启动真实 ComfyUI）。

验证 LOADING 窗口期竞态修复的核心语义：
    1. ``_start_done=None``（从未启动）→ 返回 is_running
    2. start() 进行中（event 未 set）→ wait 阻塞 → event set 后返回 True
    3. 启动完成（event set + 进程活着）→ 立即返回 True
    4. 启动失败（event set + 进程已清理）→ 返回 False
    5. 等待超时 → 返回 False

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_startup_race.py
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

from engines.ernie_image_909469ed.comfyui_process import ComfyuiProcessManager
from engines.ernie_image_909469ed.config import EngineConfig

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


def build_pm() -> ComfyuiProcessManager:
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
    )
    return ComfyuiProcessManager(config)


async def main() -> int:
    print("=" * 60)
    print("T6.1 单元测试：wait_startup_done 竞态修复")
    print("=" * 60)

    # [1] 未启动状态：_start_done=None → 返回 is_running=False
    print("\n[1] 未启动状态")
    pm = build_pm()
    r = await pm.wait_startup_done(timeout=1.0)
    check("未启动 → False", r is False, f"got {r}")

    # [2] LOADING 窗口期：event 未 set → 阻塞等待 → set 后返回 True
    print("\n[2] LOADING 窗口期（并发请求等待启动完成）")
    pm2 = build_pm()
    ev = asyncio.Event()
    pm2._start_done = ev

    async def delayed_start_ok():
        """模拟 start() 成功：2s 后 set event + 进程存活。"""
        await asyncio.sleep(2.0)
        ev.set()
        # is_running 由 Popen 对象决定，这里用 mock 验证语义：
        # 直接检查 wait 返回 True 的前提是 is_running=True
        return True

    class FakeProc:
        """Popen 桩：poll() 永远返回 None（进程活着）。"""
        def poll(self):
            return None

    pm2._process = FakeProc()

    t0 = time.time()
    start_task = asyncio.create_task(delayed_start_ok())
    r2 = await pm2.wait_startup_done(timeout=10.0)
    elapsed = time.time() - t0
    await start_task
    check("等待 2s 后返回 True", r2 is True, f"got {r2}")
    check("确实阻塞等待（elapsed>=1.9s）", elapsed >= 1.9, f"elapsed={elapsed:.2f}s")

    # [3] 启动完成态：event 已 set → 立即返回（不阻塞）
    print("\n[3] 启动完成态（立即返回）")
    t1 = time.time()
    r3 = await pm2.wait_startup_done(timeout=10.0)
    e3 = time.time() - t1
    check("立即返回 True", r3 is True and e3 < 0.5, f"got {r3}, elapsed={e3:.3f}s")

    # [4] 启动失败：event set + 进程已清理（_process=None）
    print("\n[4] 启动失败态（event set + 进程已清理）")
    ev2 = asyncio.Event()
    ev2.set()
    pm2._start_done = ev2
    pm2._process = None
    r4 = await pm2.wait_startup_done(timeout=1.0)
    check("返回 False（触发清理重载）", r4 is False, f"got {r4}")

    # [5] 超时：event 永不 set → 200ms 超时返回 False
    print("\n[5] 等待超时")
    pm3 = build_pm()
    ev3 = asyncio.Event()
    pm3._start_done = ev3
    pm3._process = FakeProc()
    t2 = time.time()
    r5 = await pm3.wait_startup_done(timeout=0.2)
    e5 = time.time() - t2
    check("超时返回 False", r5 is False and 0.15 <= e5 < 1.0, f"got {r5}, elapsed={e5:.2f}s")

    # [6] _cleanup_resources 清空 _start_done
    print("\n[6] _cleanup_resources 清理")
    pm4 = build_pm()
    pm4._start_done = asyncio.Event()
    pm4._cleanup_resources()
    check("_start_done 被清空", pm4._start_done is None)

    print("\n" + "=" * 60)
    print(f"结果：PASS={PASS}  FAIL={FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
