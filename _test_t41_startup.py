"""T4.1 测试脚本：ComfyuiProcessManager 启动/停止（临时文件，测试后删除）。

验证：
    1. 实例化 EngineConfig + ComfyuiProcessManager
    2. 启动 ComfyUI 子进程（start(timeout=120)）
    3. 健康检查通过（is_running = True）
    4. 查询 /system_stats 返回 vram_total
    5. 停止子进程（stop()）
    6. 验证已停止（is_running = False）
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 把项目根加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 设置环境变量（模拟平台注入）
import os
os.environ["AIRELIABLE_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "engines")
os.environ["AIRELIABLE_MODELS_ROOT"] = str(PROJECT_ROOT / "models")

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.comfyui_process import ComfyuiProcessManager


async def main() -> int:
    print("=" * 60)
    print("T4.1 测试：ComfyuiProcessManager 启动/停止")
    print("=" * 60)

    # 1. 构造 EngineConfig
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        # 不固定端口，让自动扫描（避免端口冲突）
        internal_port=None,
        # 增加超时到 300s（ComfyUI v0.32.0 首次启动较慢，需要加载 custom_nodes + 数据库迁移）
        server_timeout=300.0,
    )
    print(f"[1/6] EngineConfig 构造完成: {config.to_dict()}")
    print()

    # 2. 实例化 ComfyuiProcessManager
    pm = ComfyuiProcessManager(config)
    print(f"[2/6] ComfyuiProcessManager 实例化完成")
    print(f"      is_running (启动前): {pm.is_running}")
    print()

    # 3. 启动子进程
    print(f"[3/6] 启动 ComfyUI 子进程（timeout=300s）...")
    try:
        start_result = await pm.start(timeout=300.0)
    except Exception as e:
        print(f"      启动失败: {e}")
        # 读取最近日志辅助排查
        logs = pm.read_recent_logs(80)
        if logs:
            print("      最近日志（最后 80 行）:")
            for line in logs.splitlines()[-80:]:
                print(f"        {line}")
        return 1

    print(f"      启动成功: {start_result}")
    print(f"      is_running: {pm.is_running}")
    print(f"      pid: {pm.pid}")
    print(f"      internal_port: {pm.internal_port}")
    print(f"      internal_url: {pm.internal_url}")
    print()

    # 4. 查询 /system_stats
    print(f"[4/6] 查询 /system_stats ...")
    try:
        stats = await pm.get_system_stats()
        print(f"      system_stats: {json.dumps(stats, indent=2, ensure_ascii=False)[:1500]}")
    except Exception as e:
        print(f"      查询失败: {e}")
    print()

    # 5. 查询显存占用
    print(f"[5/6] 查询显存占用 ...")
    try:
        vram_bytes = await pm.get_vram_usage_bytes()
        if vram_bytes is not None:
            vram_gb = vram_bytes / (1024 ** 3)
            print(f"      vram_used: {vram_bytes} bytes ({vram_gb:.2f} GB)")
        else:
            print(f"      vram_used: None（查询失败）")
    except Exception as e:
        print(f"      查询失败: {e}")
    print()

    # 6. 停止子进程
    print(f"[6/6] 停止 ComfyUI 子进程 ...")
    try:
        stop_result = await pm.stop()
        print(f"      stop 结果: {stop_result}")
    except Exception as e:
        print(f"      停止失败: {e}")
        return 1

    print(f"      is_running (停止后): {pm.is_running}")
    print()

    print("=" * 60)
    print("T4.1 测试通过 ✅")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
