"""UI 等价复验：与新版多参考图生图 UI 工作流完全相同的拓扑/默认值，
直接提交 ComfyUI API 执行，验证 UI 工作流 Queue 后能跑通。"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

COMFY = "http://127.0.0.1:8100"


def build() -> dict:
    """与 _gen_ui_workflows.build_i2i() 一一对应的 API 格式。

    源图 + 2 参考图均用 UI 默认示例图 ernie_sample.png，denoise 0.6。
    """
    p: dict = {}
    p["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "ernie-image-turbo-Q4_K_M.gguf"}}
    p["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "model_text_only.safetensors", "type": "stable_diffusion"}}
    p["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "一只橘猫趴在窗台上晒太阳，温暖的午后光线，高清细节", "clip": ["2", 0]}}
    p["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}}
    p["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": "diffusion_pytorch_model.safetensors"}}
    p["6"] = {"class_type": "LoadImage", "inputs": {"image": "ernie_sample.png"}}
    p["7"] = {"class_type": "LoadImage", "inputs": {"image": "ernie_sample.png"}}
    p["8"] = {"class_type": "LoadImage", "inputs": {"image": "ernie_sample.png"}}
    p["9"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["5", 0]}}
    p["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["7", 0], "vae": ["5", 0]}}
    p["11"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["5", 0]}}
    p["12"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["3", 0], "latent": ["10", 0]}}
    p["13"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["12", 0], "latent": ["11", 0]}}
    p["14"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["13", 0], "negative": ["4", 0], "latent_image": ["9", 0], "seed": 7, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.6}}
    p["15"] = {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}}
    p["16"] = {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "ERNIE_Image_UI_I2I"}}
    return p


async def main() -> int:
    async with httpx.AsyncClient(timeout=600) as hc:
        t0 = time.time()
        r = await hc.post(f"{COMFY}/prompt", json={"prompt": build()})
        assert r.status_code == 200, r.text[:800]
        pid = r.json()["prompt_id"]
        print(f"submitted {pid}")
        while True:
            await asyncio.sleep(3)
            hist = (await hc.get(f"{COMFY}/history/{pid}")).json().get(pid)
            if not hist:
                continue
            st = hist.get("status", {})
            if st.get("completed"):
                imgs = [i for o in hist.get("outputs", {}).values() for i in o.get("images", [])]
                name = imgs[0]["filename"] if imgs else "none"
                print(f"[PASS] UI 等价执行成功 {time.time()-t0:.1f}s, 输出: {name}")
                return 0
            if st.get("status_str") == "error":
                errs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                print("[FAIL]", json.dumps(errs[:1], ensure_ascii=False)[:800])
                return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
