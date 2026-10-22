"""T3.4 对照实验：VAE 编解码往返误差（LoadImage → VAEEncode → VAEDecode）。

目的：
    T3.4 Edit 端到端测试中 mask 外区域平均差=11.9（阈值 2.0 触发 FAIL）。
    该差异的来源有两种可能：
        A. mask 保护失效（KSampler 修改了 mask 外 latent）—— 需要修模板
        B. VAE 编解码固有损失（16ch 高压缩比 VAE + fp16 精度）—— 符合预期
    本脚本跑一条 **不经 KSampler** 的 VAEEncode → VAEDecode 链路，
    测量 VAE 本身的重构误差作为基线：
        - 若基线 ≈ 11 → mask 保护正常，Edit 结果符合预期，仅需放宽测试阈值
        - 若基线 ≈ 0-2 → mask 保护确实失效，需要排查 VAEEncodeForInpaint

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_vae_roundtrip.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

from PIL import Image

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.comfyui_client import (
    ComfyuiClient,
    EVENT_EXECUTION_SUCCESS,
    extract_image_outputs,
)
from engines.ernie_image_909469ed.engine import ErnieImageEngine


class FakeImage:
    def __init__(self, role: str, b64: str):
        self.role = role
        self.b64_json = b64
        self.url = None
        self.path = None


class FakeRequest:
    def __init__(self, images: list):
        self.images = images
        self.workflow = None
        self.loras = []


SIZE = 512


def make_source_b64() -> str:
    """与 _test_edit.py 完全一致的 512x512 渐变源图。"""
    im = Image.new("RGB", (SIZE, SIZE))
    px = im.load()
    for y in range(SIZE):
        for x in range(SIZE):
            px[x, y] = (x * 255 // SIZE, y * 255 // SIZE, 128)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def full_image_diff(a: Image.Image, b: Image.Image) -> tuple[float, int]:
    """全图平均像素差 / 最大差（1/16 采样）。"""
    pa, pb = a.load(), b.load()
    total, count, max_d = 0, 0, 0
    for y in range(0, SIZE, 4):
        for x in range(0, SIZE, 4):
            diff = max(abs(pa[x, y][c] - pb[x, y][c]) for c in range(3))
            total += diff
            max_d = max(max_d, diff)
            count += 1
    return (total / count if count else 0.0), max_d


async def main() -> int:
    print("=" * 60)
    print("对照实验：VAE 编解码往返误差（不经 KSampler）")
    print("=" * 60)
    t0 = time.time()

    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )
    eng = ErnieImageEngine(config)

    task_id = f"vae_rt_{uuid.uuid4().hex[:8]}"
    staged = {}
    saved: list[Path] = []

    try:
        # 1. 落盘源图（与 _test_edit.py 相同的 _stage_input_images 路径）
        print("[1/5] 落盘源图（复用 _stage_input_images，保证处理路径一致）...")
        req = FakeRequest(images=[FakeImage(role="source", b64=make_source_b64())])
        staged = await eng._stage_input_images(req, task_id, "i2i")
        src_filename = staged["source_filename"]
        print(f"      source={src_filename}")
        print()

        # 2. 启动 ComfyUI
        pm = eng._process_manager
        print("[2/5] 启动 ComfyUI 子进程...")
        start_result = await pm.start(timeout=300.0)
        print(f"      启动成功: pid={start_result['pid']}, port={start_result['internal_port']}, "
              f"耗时 {start_result['load_time']:.1f}s")
        print()

        client = ComfyuiClient(pm.internal_url)

        # 3. 构建最小工作流：VAEEncode → VAEDecode（无 KSampler、无 mask）
        vae_filename = eng._find_model_filename("ernie-vae-safetensors")
        print(f"[3/5] 构建往返工作流（VAE={vae_filename}）...")
        workflow = {
            "1": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_filename},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": src_filename},
            },
            "3": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["2", 0], "vae": ["1", 0]},
            },
            "4": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["1", 0]},
            },
            "5": {
                "class_type": "SaveImage",
                "inputs": {"images": ["4", 0], "filename_prefix": "VAE_RT"},
            },
        }
        print()

        # 4. 提交 + 等待
        print("[4/5] 提交工作流...")
        client_id = uuid.uuid4().hex
        submit_result = await client.submit_workflow(workflow, client_id=client_id)
        if "error" in submit_result:
            print(f"      提交失败: {submit_result}")
            return 1
        prompt_id = submit_result.get("prompt_id", "")
        print(f"      提交成功: prompt_id={prompt_id}")

        async for event in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
            et = event.get("type", "")
            if et == EVENT_EXECUTION_SUCCESS:
                print("      [execution_success] 往返完成")
                break
            data = event.get("data", {}) or {}
            if et == "execution_error":
                print(f"      [execution_error] {data}")
                return 1
        print()

        # 5. 下载 + 全图像素比对
        print("[5/5] 下载图片 + 全图像素比对（VAE 重构误差基线）...")
        history = await client.get_history(prompt_id)
        image_outputs = extract_image_outputs(history, prompt_id) if history else []
        if not image_outputs:
            print("      无图片输出")
            return 1
        img_info = image_outputs[0]
        image_bytes = await client.get_image(
            img_info.get("filename", ""), img_info.get("subfolder", ""),
            img_info.get("type", "output"),
        )
        if image_bytes is None:
            print("      下载失败")
            return 1

        output_dir = PROJECT_ROOT / "output" / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        local = output_dir / f"vae_rt_{int(time.time())}_{img_info.get('filename', '')}"
        local.write_bytes(image_bytes)
        saved.append(local)

        with Image.open(local) as out_im:
            out_rgb = out_im.convert("RGB")
            with Image.open(staged["files"][0]) as src_im:
                src_rgb = src_im.convert("RGB")
            avg, max_d = full_image_diff(src_rgb, out_rgb)
            print(f"      全图平均差={avg:.1f}, 最大差={max_d}")
            print(f"      输出尺寸: {out_rgb.size}")
            print()
            if avg < 2.0:
                print(f"      结论: VAE 基线误差很小（{avg:.1f}）→ T3.4 的 mask 外差异 11.9 不是 VAE 损失，需排查 mask 保护")
            else:
                print(f"      结论: VAE 基线误差 ≈ {avg:.1f}（与 T3.4 mask 外差异 11.9 同量级）"
                      f" → mask 保护正常，T3.4 差异主要来自 VAE 编解码固有损失")

        await pm.stop()

        print()
        print("=" * 60)
        print(f"对照实验完成（总耗时 {time.time()-t0:.1f}s）")
        for f in saved:
            print(f"  - {f}")
        print("=" * 60)
        return 0

    finally:
        ErnieImageEngine._cleanup_staged_images(staged.get("files") or [])
        print(f"已清理 input 目录暂存图片（{len(staged.get('files') or [])} 个）")


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
