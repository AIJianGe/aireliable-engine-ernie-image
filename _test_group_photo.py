"""合影助手（B 方案）机制实测：3 轮链式 inpaint + 人物参考图，一次 Queue 完成。

待验证机制（此前未实测过的组合）：
    VAEEncodeForInpaint（noise_mask 区域隔离）+ ReferenceLatent（人物 in-context
    参考）+ denoise 1.0 + 3 轮链式（上一轮 VAEDecode 输出 → 下一轮 pixels 输入）。

拓扑（每轮，base=10/20/30）：
    LoadImageMask(区域蒙版) ──────────────────────→ VAEEncodeForInpaint
    LoadImage(人物N) → VAEEncode → ReferenceLatent → KSampler.positive
    上一轮输出(R1 为背景图) ──────────────────────→ VAEEncodeForInpaint.pixels
    KSampler(denoise 1.0) → VAEDecode → SaveImage

验证点：
    1. 一次提交 3 轮全部执行成功，产出 3 张图
    2. 区域隔离：每轮只有蒙版区域变化，其余区域基本保持（VAE 往返漂移 < 阈值）

运行（需 7001 引擎已启动）：
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_group_photo.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMFY = "http://127.0.0.1:8100"
INPUT_DIR = PROJECT_ROOT / "output" / "input"

GGUF = "ernie-image-turbo-Q4_K_M.gguf"
VAE = "diffusion_pytorch_model.safetensors"
TE = "model_text_only.safetensors"
BG = "ernie_sample.png"
PERSONS = ["test_ref1.jpg", "test_ref2.jpg", "test_ref3.jpg"]
MASKS = ["group_mask_center.png", "group_mask_left.png", "group_mask_right.png"]

# 三个区域（512×512）：中间先画（C 位），再左，再右
REGIONS = {
    "group_mask_center.png": (186, 100, 326, 490),
    "group_mask_left.png": (30, 120, 160, 490),
    "group_mask_right.png": (352, 120, 482, 490),
}


def make_masks() -> None:
    """生成 3 张区域蒙版（RGB，白=重绘区，channel=red 提取）。"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, rect in REGIONS.items():
        img = Image.new("RGB", (512, 512), (0, 0, 0))
        ImageDraw.Draw(img).rectangle(rect, fill=(255, 255, 255))
        img.save(INPUT_DIR / fname)
    print(f"[OK] 3 张区域蒙版已生成到 {INPUT_DIR}")


def build_prompt() -> dict:
    p: dict = {}
    # 共享：unet / clip / 负向 / vae
    p["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF}}
    p["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "stable_diffusion"}}
    p["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}}
    p["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}

    prompts = [
        "参考图中的第一位人物站在画面中间，全身，自然放松的姿势",
        "参考图中的第二位人物站在左侧，与中间的人自然合影",
        "参考图中的第三位人物站在右侧，三人开心合影，光线自然",
    ]
    for i, (mask, person, pos_text) in enumerate(zip(MASKS, PERSONS, prompts)):
        base = 10 * (i + 1)  # 10 / 20 / 30，节点 id = base+0..base+9
        bg, mask_n, person_n, penc, pos, refl, ienc, ks, dec, save = (
            str(base + j) for j in range(10))
        p[bg] = {"class_type": "LoadImage", "inputs": {"image": BG}}  # 仅 R1 用 IMAGE
        p[mask_n] = {"class_type": "LoadImageMask",
                     "inputs": {"image": mask, "channel": "red"}}
        p[person_n] = {"class_type": "LoadImage", "inputs": {"image": person}}
        p[penc] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": [person_n, 0], "vae": ["4", 0]}}
        p[pos] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": pos_text, "clip": ["2", 0]}}
        p[refl] = {"class_type": "ReferenceLatent",
                   "inputs": {"conditioning": [pos, 0], "latent": [penc, 0]}}
        # R1 pixels 来自背景图，R2/R3 来自上一轮 VAEDecode（18 / 28）
        pixels = [bg, 0] if i == 0 else [str(10 * i + 8), 0]
        p[ienc] = {"class_type": "VAEEncodeForInpaint",
                   "inputs": {"pixels": pixels, "vae": ["4", 0],
                              "mask": [mask_n, 0], "grow_mask_by": 6}}
        p[ks] = {"class_type": "KSampler",
                 "inputs": {"model": ["1", 0], "positive": [refl, 0],
                            "negative": ["3", 0], "latent_image": [ienc, 0],
                            "seed": 42, "steps": 8, "cfg": 1.0,
                            "sampler_name": "euler", "scheduler": "simple",
                            "denoise": 1.0}}
        p[dec] = {"class_type": "VAEDecode",
                  "inputs": {"samples": [ks, 0], "vae": ["4", 0]}}
        p[save] = {"class_type": "SaveImage",
                   "inputs": {"images": [dec, 0],
                              "filename_prefix": f"ERNIE_Group_R{i + 1}"}}
    return p


def region_diff(a: Image.Image, b: Image.Image, rect: tuple, pad: int = 40) -> tuple:
    """返回 (蒙版区平均差, 非蒙版区平均差)。rect 为本轮重绘区。"""
    import numpy as np
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(512, x1 + pad), min(512, y1 + pad)
    A = np.asarray(a.convert("RGB"), dtype=np.int16)
    B = np.asarray(b.convert("RGB"), dtype=np.int16)
    d = np.abs(A - B).mean(axis=2)
    m = np.zeros((512, 512), dtype=bool)
    m[y0:y1, x0:x1] = True
    return float(d[m].mean()), float(d[~m].mean())


async def main() -> int:
    make_masks()

    async with httpx.AsyncClient(timeout=600) as hc:
        t0 = time.time()
        r = await hc.post(f"{COMFY}/prompt", json={"prompt": build_prompt()})
        if r.status_code != 200:
            print(f"[FAIL] 提交失败: {r.text[:1500]}")
            return 1
        pid = r.json()["prompt_id"]
        print(f"[OK] 已提交（3 轮链式）prompt_id={pid}，等待执行...")

        while True:
            await asyncio.sleep(4)
            hist = (await hc.get(f"{COMFY}/history/{pid}")).json().get(pid)
            if not hist:
                continue
            st = hist.get("status", {})
            if st.get("completed"):
                break
            if st.get("status_str") == "error":
                errs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                print(f"[FAIL] 执行出错: {json.dumps(errs[:1], ensure_ascii=False)[:1500]}")
                return 1

        dt = time.time() - t0
        # 收集 3 轮输出
        outs = {}
        for o in hist.get("outputs", {}).values():
            for img in o.get("images", []):
                r2 = await hc.get(
                    f"{COMFY}/view?filename={img['filename']}"
                    f"&subfolder={img.get('subfolder', '')}&type={img.get('type', 'output')}")
                outs[img["filename"]] = Image.open(io.BytesIO(r2.content))
        rounds = {k: v for k, v in outs.items() if k.startswith("ERNIE_Group")}
        print(f"[OK] 3 轮执行完成，耗时 {dt:.1f}s，输出 {sorted(rounds)}")
        if len(rounds) < 3:
            print("[FAIL] 输出图不足 3 张")
            return 1

        r1 = rounds[sorted(rounds)[0]]
        r2 = rounds[sorted(rounds)[1]]
        r3 = rounds[sorted(rounds)[2]]
        bg_img = Image.open(INPUT_DIR / BG)

        print("\n区域隔离验证（每轮：蒙版区应大改，其余区应基本不变，<≈10 为 VAE 往返漂移）：")
        ok = True
        for label, prev, cur, rect in (
            ("R1(中间)", bg_img, r1, REGIONS["group_mask_center.png"]),
            ("R2(左侧)", r1, r2, REGIONS["group_mask_left.png"]),
            ("R3(右侧)", r2, r3, REGIONS["group_mask_right.png"]),
        ):
            m, u = region_diff(prev, cur, rect)
            verdict = "PASS" if (m > 15 and u < 12 and u < m) else "FAIL"
            if verdict == "FAIL":
                ok = False
            print(f"  {label}: 蒙版区平均差={m:6.1f}  非蒙版区平均差={u:5.1f}  [{verdict}]")

        print("\n[PASS] 合影助手 3 轮链式机制验证通过" if ok else "\n[FAIL] 区域隔离不符合预期")
        # 保存最终图供人工查看
        out_path = PROJECT_ROOT / "output" / "test_assets" / "group_photo_final.png"
        r3.save(out_path)
        print(f"最终图已保存: {out_path}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
