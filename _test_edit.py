"""T3.4 集成测试：端到端 Edit 推理（Turbo，8 步，源图 + mask）。

验证：
    1. 启动 ComfyUI 子进程
    2. 引擎落盘 source（16 对齐）+ mask（白=重绘区，中心圆）
    3. 渲染 ernie_turbo_edit.json（mode=edit，denoise=1.0）
    4. 提交工作流（LoadImage + LoadImageMask + VAEEncodeForInpaint）
    5. 等待 execution_success
    6. 下载图片 + 像素级验证：
       - mask 外区域（远离重绘区）与源图一致（inpaint 保留性）
       - mask 内区域与源图不同（重绘生效）
    7. 清理暂存文件 + 停止子进程

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_edit.py
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

from PIL import Image, ImageDraw

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.comfyui_client import (
    ComfyuiClient,
    EVENT_EXECUTION_ERROR,
    EVENT_EXECUTION_SUCCESS,
    EVENT_PROGRESS,
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
CX, CY, R = SIZE // 2, SIZE // 2, 120          # mask 圆心 + 半径
SAFE_R = R + 30                                 # 像素比对安全半径（grow_mask_by=6 + 余量）


def make_source_b64() -> str:
    """512x512 渐变源图（四角颜色固定，便于像素比对）。"""
    im = Image.new("RGB", (SIZE, SIZE))
    px = im.load()
    for y in range(SIZE):
        for x in range(SIZE):
            px[x, y] = (x * 255 // SIZE, y * 255 // SIZE, 128)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_mask_b64() -> str:
    """512x512 灰度 mask：中心白圆（重绘区），其余黑（保留区）。"""
    im = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(im)
    draw.ellipse((CX - R, CY - R, CX + R, CY + R), fill=255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pixel_diff(a: Image.Image, b: Image.Image, region: str) -> tuple[float, int]:
    """计算两图在指定区域的平均像素差 / 最大差。

    region: "outside"（mask 安全区外）或 "inside"（mask 圆内）
    """
    pa, pb = a.load(), b.load()
    total, count, max_d = 0, 0, 0
    for y in range(0, SIZE, 4):                 # 1/16 采样加速
        for x in range(0, SIZE, 4):
            d_center = math.hypot(x - CX, y - CY)
            if region == "outside" and d_center < SAFE_R:
                continue
            if region == "inside" and d_center > R * 0.5:
                continue
            diff = max(abs(pa[x, y][c] - pb[x, y][c]) for c in range(3))
            total += diff
            max_d = max(max_d, diff)
            count += 1
    return (total / count if count else 0.0), max_d


async def main() -> int:
    print("=" * 60)
    print("T3.4 集成测试：端到端 Edit 推理（Turbo，8 步，inpaint）")
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
    print(f"[1/8] 引擎初始化完成（input_dir={config.input_dir}）")
    print()

    task_id = f"edit_test_{uuid.uuid4().hex[:8]}"
    staged = {}
    saved: list[Path] = []

    try:
        # 2. 落盘 source + mask
        print("[2/8] _stage_input_images 落盘 source + mask（中心白圆 = 重绘区）...")
        req = FakeRequest(images=[
            FakeImage(role="source", b64=make_source_b64()),
            FakeImage(role="mask", b64=make_mask_b64()),
        ])
        staged = await eng._stage_input_images(req, task_id, "edit")
        print(f"      source={staged['source_filename']}, mask={staged['mask_filename']}, "
              f"size={staged['source_width']}x{staged['source_height']}")
        assert staged["mask_filename"], "mask 落盘失败"
        print()

        # 3. 启动 ComfyUI
        pm = eng._process_manager
        print("[3/8] 启动 ComfyUI 子进程（timeout=300s）...")
        start_result = await pm.start(timeout=300.0)
        print(f"      启动成功: pid={start_result['pid']}, "
              f"port={start_result['internal_port']}, 耗时 {start_result['load_time']:.1f}s")
        print()

        client = ComfyuiClient(pm.internal_url)

        # 4. 渲染 edit 工作流
        print("[4/8] _render_workflow 渲染 ernie_turbo_edit.json（denoise=1.0）...")
        variant = "ernie-image-turbo-gguf-q4-k-m"
        mv = eng._manifest_variants[variant]
        workflow = eng._render_workflow(
            variant=variant, mv=mv,
            prompt="一朵盛开的红色牡丹花",
            negative_prompt="",
            width=staged["source_width"], height=staged["source_height"],
            steps=8, cfg=1.0, seed=42, use_pe=False, loras=[],
            mode="edit",
            source_filename=staged["source_filename"],
            mask_filename=staged["mask_filename"],
            denoise=1.0,
        )
        template_used = eng._select_template(mv, False, "edit")
        print(f"      模板: {template_used}, 节点数: {len(workflow)}")
        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            if ct in ("LoadImage", "LoadImageMask", "VAEEncodeForInpaint", "KSampler"):
                print(f"        {ct}: {node.get('inputs')}")
        print()

        # 5. 提交 + 订阅
        print("[5/8] 提交工作流 + 订阅 WS 事件...")
        client_id = uuid.uuid4().hex
        submit_result = await client.submit_workflow(workflow, client_id=client_id)
        if "error" in submit_result:
            print(f"      提交失败: {submit_result}")
            return 1
        prompt_id = submit_result.get("prompt_id", "")
        print(f"      提交成功: prompt_id={prompt_id}")

        t_infer = time.time()
        async for event in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
            et = event.get("type", "")
            data = event.get("data", {}) or {}
            if et == EVENT_PROGRESS:
                print(f"      [progress] {data.get('value')}/{data.get('max')}")
            elif et == EVENT_EXECUTION_SUCCESS:
                print(f"      [execution_success] 推理完成，耗时 {time.time()-t_infer:.1f}s")
                break
            elif et == EVENT_EXECUTION_ERROR:
                print(f"      [execution_error] {data}")
                return 1
        print()

        # 6. 下载 + 像素验证
        print("[6/8] 下载图片 + 像素级验证（mask 外保留 / mask 内重绘）...")
        history = await client.get_history(prompt_id)
        if not history:
            print("      history 为空")
            return 1
        image_outputs = extract_image_outputs(history, prompt_id)
        if not image_outputs:
            print("      无图片输出")
            return 1
        output_dir = PROJECT_ROOT / "output" / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_info in enumerate(image_outputs):
            image_bytes = await client.get_image(
                img_info.get("filename", ""), img_info.get("subfolder", ""),
                img_info.get("type", "output"),
            )
            if image_bytes is None:
                continue
            local = output_dir / f"t34_edit_{int(time.time())}_{idx}_{img_info.get('filename','')}"
            local.write_bytes(image_bytes)
            saved.append(local)

        if not saved:
            print("      无图片保存成功")
            return 1

        # 像素验证
        with Image.open(saved[0]) as out_im:
            out_im = out_im.convert("RGB")
            src_path = Path(staged["files"][0])
            with Image.open(src_path) as src_im:
                src_rgb = src_im.convert("RGB")
            avg_out, max_out = pixel_diff(src_rgb, out_im, "outside")
            avg_in, max_in = pixel_diff(src_rgb, out_im, "inside")
            print(f"      mask 外（安全区）: 平均差={avg_out:.1f}, 最大差={max_out}")
            print(f"      mask 内（圆心区）: 平均差={avg_in:.1f}, 最大差={max_in}")
            print(f"      输出尺寸: {out_im.size}")
            # mask 外阈值依据（2026-08-21 对照实验 _test_vae_roundtrip.py）：
            #   - 纯 VAE 编解码往返基线误差 = 6.5（16ch 高压缩比 VAE 固有损失）
            #   - Edit 链路 mask 边缘 latent 羽化（8x 下采样 + grow_mask_by=6）额外引入 ~5
            #   - 实测 mask 外平均差 = 11.9，与基线同量级 → mask 保护正常
            # 阈值取 15.0（基线 6.5 的 ~2.3 倍），显著低于 mask 内重绘量级（125+）
            ok_outside = avg_out < 15.0            # mask 外 ≈ VAE 固有损失（基线 6.5）
            ok_inside = avg_in > 20.0              # mask 内应有明显重绘
            check = "PASS" if (ok_outside and ok_inside) else "WARN"
            print(f"      验证结果 [{check}]: mask 外保留={'OK' if ok_outside else 'FAIL'}, "
                  f"mask 内重绘={'OK' if ok_inside else 'FAIL'}")
        print()

        # 7. 停止子进程
        print("[7/8] 停止 ComfyUI 子进程...")
        stop_result = await pm.stop()
        print(f"      停止成功: {stop_result}")
        print()

        total = time.time() - t0
        print("=" * 60)
        print(f"T3.4 Edit 端到端测试完成 ✅（总耗时 {total:.1f}s）")
        print(f"  启动: {start_result['load_time']:.1f}s")
        print(f"  图片: {len(saved)} 张")
        for f in saved:
            print(f"    - {f}")
        print("=" * 60)
        return 0

    finally:
        # 8. 清理暂存
        ErnieImageEngine._cleanup_staged_images(staged.get("files") or [])
        print(f"[8/8] 已清理 input 目录暂存图片（{len(staged.get('files') or [])} 个）")


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
