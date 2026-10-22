"""多图参考（multi-reference）实测：验证 ERNIE-Image 走 Flux2 管线时
ReferenceLatent 链式串联是否可用。

机制（已读源码确认）：
    - comfy/model_base.py:1043  Flux.extra_conds 把 reference_latents 打包为
      CONDList（天然多张），Flux2 继承 Flux（model_base.py:1082）
    - comfy/ldm/flux/model.py:360-398  transformer _forward 循环遍历
      ref_latents 列表，每张参考图编码为 in-context token 拼进 img 序列，
      ref_latents_method 支持 offset/index/uxo-uno/index_timestep_zero
    - comfy_extras/nodes_edit_model.py:6  ReferenceLatent 节点官方描述：
      "you can chain multiple to set multiple reference images"

工作流（2 张参考图）：
    LoadImage×2 → VAEEncode×2 → ReferenceLatent → ReferenceLatent（链式）
    EmptyFlux2LatentImage(512×512) → KSampler(denoise=1.0) → SaveImage

运行（需 7001 引擎已启动，内部 ComfyUI 在 8101）：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_multi_ref.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx

COMFY_URL = "http://127.0.0.1:8100"

GGUF = "ernie-image-turbo-Q4_K_M.gguf"
VAE = "diffusion_pytorch_model.safetensors"
TE = "model_text_only.safetensors"
REF_IMAGES = ["test_ref1.jpg", "test_ref2.jpg"]
# 混合模式：源图作为初始 latent（重绘基础），参考图走 in-context
SRC_IMAGE = "test_ref3.jpg"


def build_prompt() -> dict:
    """混合模式 prompt：源图重绘 + 双参考 in-context。

    节点编号字符串作 key。源图 VAEEncode → KSampler latent_image（denoise 0.6），
    两张参考图 VAEEncode → ReferenceLatent 链式拼进 positive。
    """
    p: dict = {}
    p["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF}}
    p["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE, "type": "stable_diffusion"}}
    p["3"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": "参考第二张图的环境氛围，重构第一张源图的画面",
                         "clip": ["2", 0]}}
    p["4"] = {"class_type": "CLIPTextEncode",
              "inputs": {"text": "", "clip": ["2", 0]}}
    p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": VAE}}
    # 源图：重绘基础（初始 latent）
    p["6"] = {"class_type": "LoadImage", "inputs": {"image": SRC_IMAGE}}
    p["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["5", 0]}}
    # 双参考图：in-context 参考
    p["8"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMAGES[0]}}
    p["9"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMAGES[1]}}
    p["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["5", 0]}}
    p["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["5", 0]}}
    # ReferenceLatent 链式串联（多参考入口）
    p["12"] = {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["3", 0], "latent": ["10", 0]}}
    p["13"] = {"class_type": "ReferenceLatent",
               "inputs": {"conditioning": ["12", 0], "latent": ["11", 0]}}
    # KSampler：源图 latent 做底，denoise 0.6（混合模式核心验证点）
    p["14"] = {"class_type": "KSampler",
               "inputs": {
                   "model": ["1", 0], "positive": ["13", 0], "negative": ["4", 0],
                   "latent_image": ["7", 0], "seed": 42, "steps": 8, "cfg": 1.0,
                   "sampler_name": "euler", "scheduler": "simple", "denoise": 0.6}}
    p["15"] = {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}}
    p["16"] = {"class_type": "SaveImage",
               "inputs": {"images": ["15", 0], "filename_prefix": "ERNIE_MultiRef_Test"}}
    return p


async def main() -> int:
    async with httpx.AsyncClient(timeout=600) as hc:
        # 0. 前置检查
        r = await hc.get(f"{COMFY_URL}/object_info")
        obj = r.json()
        for node in ("ReferenceLatent", "UnetLoaderGGUF", "EmptyFlux2LatentImage"):
            assert node in obj, f"节点不存在: {node}"
        for f in (GGUF,):
            assert f in obj["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0], f"GGUF 不在列表: {f}"
        print("[OK] ReferenceLatent 节点存在，GGUF 在模型列表")

        # 1. 提交
        t0 = time.time()
        r = await hc.post(f"{COMFY_URL}/prompt", json={"prompt": build_prompt()})
        if r.status_code != 200:
            print(f"[FAIL] 提交失败 {r.status_code}: {r.text[:2000]}")
            return 1
        pid = r.json()["prompt_id"]
        print(f"[OK] 已提交 prompt_id={pid}，等待执行（Q4 GGUF，8 步 512×512）...")

        # 2. 轮询历史
        while True:
            await asyncio.sleep(3)
            r = await hc.get(f"{COMFY_URL}/history/{pid}")
            hist = r.json().get(pid)
            if hist is None:
                continue
            status = hist.get("status", {})
            if status.get("completed"):
                outs = []
                for o in hist.get("outputs", {}).values():
                    for img in o.get("images", []):
                        outs.append(img)
                if not outs:
                    print(f"[FAIL] 执行完成但无输出: {json.dumps(status)[:500]}")
                    return 1
                img = outs[0]
                r2 = await hc.get(
                    f"{COMFY_URL}/view?filename={img['filename']}&subfolder={img.get('subfolder','')}&type={img.get('type','output')}")
                out_path = PROJECT_ROOT / "output" / "test_assets" / img["filename"]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(r2.content)
                print(f"[PASS] 多图参考执行成功！耗时 {time.time()-t0:.1f}s")
                print(f"       输出: {out_path} ({len(r2.content)/1024:.1f} KB)")
                return 0
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                print(f"[FAIL] 执行出错: {json.dumps(msgs[:1], ensure_ascii=False)[:1500]}")
                return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
