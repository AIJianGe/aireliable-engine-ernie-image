"""ERNIE-Image 引擎模块入口。

支持 `python -m engines.ernie_image_909469ed` 直接启动。

启动命令：
    uv run --project envs/image-ernie-image-gguf \\
        python -m engines.ernie_image_909469ed.server --port 7001
"""

from .server import main

if __name__ == "__main__":
    main()
