"""起 7002 测试实例供浏览器验证（T9 调试用）。

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_run_test_server.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

import uvicorn

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.server import create_app


def main() -> None:
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        port=7002,
        internal_port=None,
        server_timeout=300.0,
    )
    app = create_app(config)
    print("测试实例: http://127.0.0.1:7002/（首次访问触发 auto-load，等 18-30s）")
    uvicorn.run(app, host="127.0.0.1", port=7002, log_level="warning")


if __name__ == "__main__":
    main()
