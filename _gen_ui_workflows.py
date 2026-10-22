"""生成 ERNIE-Image 三个 ComfyUI UI 工作流（litegraph 格式）+ 示例图。

背景：
    引擎的 workflow_templates/ 目录存放 API 格式模板（带 __PLACEHOLDER__ 占位符，
    供引擎 /infer 渲染后提交 /prompt）。用户通过 http://127.0.0.1:7001/ 打开
    ComfyUI Web UI 时，侧边栏 Workflows 里看不到任何工作流（用户目录为空）。

    本脚本把三个功能模式（t2i / i2i / edit）生成为 ComfyUI UI 格式工作流，
    保存到 ``ComfyUI/user/default/workflows/ERNIE-Image/``，用户在 UI 中
    可直接加载、手动调试。参数填实际模型文件名 + 合理默认值。

产物：
    1. input 目录示例图 ×2（供 I2I / Edit 的 LoadImage/LoadImageMask 默认值）：
       - ernie_sample.png       512x512 渐变图
       - ernie_sample_mask.png  512x512 黑底 + 中心白圆（白=重绘区）
    2. UI 工作流 ×4：
       - ERNIE-Image Turbo 文生图.json
       - ERNIE-Image Turbo 图生图.json（v0.4.6 多图参考：源图重绘 +
         2 张参考图 ReferenceLatent 链式 in-context，可复制节点扩展 N 张）
       - ERNIE-Image Turbo 图片编辑.json
       - ERNIE-Image Turbo 合影助手.json（v0.4.7：3 轮链式 inpaint，
         背景+蒙版×3 + 人物参考×3，一次 Queue 把 3 人依次画进背景）

widget 顺序依据（ComfyUI v0.32 源码确认）：
    - UnetLoaderGGUF:            [unet_name]
    - CLIPLoader:                [clip_name, type, device(optional advanced)]
    - CLIPTextEncode:            [text]
    - EmptyFlux2LatentImage:     [width, height, batch_size]
    - KSampler:                  [seed, control("random"), steps, cfg,
                                  sampler_name, scheduler, denoise]
    - VAELoader:                 [vae_name]
    - VAEDecode:                 （无 widget）
    - SaveImage:                 [filename_prefix]
    - LoadImage:                 [image, upload("image" 前端按钮占位)]
    - LoadImageMask:             [image, upload, channel]
    - VAEEncode:                 （无 widget）
    - VAEEncodeForInpaint:       [grow_mask_by]
    - ReferenceLatent:           （无 widget；inputs=[conditioning, latent]）

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_gen_ui_workflows.py
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = (
    PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI"
    / "user" / "default" / "workflows" / "ERNIE-Image"
)
INPUT_DIR = PROJECT_ROOT / "output" / "input"

# 实际模型文件名（与 manifest.yaml 对齐，turbo 变体）
GGUF = "ernie-image-turbo-Q4_K_M.gguf"
VAE = "diffusion_pytorch_model.safetensors"
TE = "model_text_only.safetensors"

SAMPLE = "ernie_sample.png"
SAMPLE_MASK = "ernie_sample_mask.png"


# ===== 示例图生成 =====

def make_sample_images() -> None:
    """生成 I2I/Edit 用的示例图（渐变源图 + 中心白圆 mask）。"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    size = 512

    # 源图：渐变（与 _test_edit.py 一致的测试图）
    src = Image.new("RGB", (size, size))
    px = src.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x * 255 // size, y * 255 // size, 128)
    src.save(INPUT_DIR / SAMPLE, format="PNG")

    # mask：黑底 + 中心白圆（白=重绘区，与引擎 _normalize_mask_image 语义一致）
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [size // 2 - 128, size // 2 - 128, size // 2 + 128, size // 2 + 128],
        fill=255,
    )
    mask.save(INPUT_DIR / SAMPLE_MASK, format="PNG")

    print(f"示例图已生成: {INPUT_DIR / SAMPLE}, {INPUT_DIR / SAMPLE_MASK}")


# ===== litegraph 节点构造 helper =====

def node(nid: int, ntype: str, pos: list, size: list, order: int,
         inputs: list | None = None, outputs: list | None = None,
         widgets: list | None = None, title: str | None = None) -> dict:
    """构造 litegraph 节点。"""
    n = {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets or [],
    }
    if title:
        n["title"] = title
    return n


def ip(name: str, ntype: str, link: int | None) -> dict:
    """input 端口。"""
    return {"name": name, "type": ntype, "link": link}


def op(name: str, ntype: str, links: list, slot_index: int = 0) -> dict:
    """output 端口。slot_index 为输出槽位（LoadImage 的 MASK 在槽 1）。"""
    return {"name": name, "type": ntype, "links": links, "slot_index": slot_index}


def workflow(nodes: list, links: list, groups: list | None = None,
             ds: dict | None = None) -> dict:
    """组装完整 UI 工作流（含 link 回填）。

    回填：遍历 links 数组，把 link_id 写回目标节点 inputs[slot].link 和
    源节点 outputs[slot].links——与 ComfyUI 前端序列化格式一致
    （litegraph configure 时也会从 links 数组重建，回填保证双保险）。

    groups: litegraph 分组框（可选，用于大工作流的视觉分区）。
    ds: extra.ds 视口（可选，默认 offset [300,300] / scale 0.75）。
    """
    by_id = {n["id"]: n for n in nodes}
    for l in links:
        link_id, from_node, from_slot, to_node, to_slot, _ = l
        src = by_id[from_node]
        while len(src["outputs"]) <= from_slot:
            src["outputs"].append({"name": "?", "type": "*", "links": [], "slot_index": len(src["outputs"])})
        outs = src["outputs"][from_slot].setdefault("links", [])
        if link_id not in outs:
            outs.append(link_id)
        dst = by_id[to_node]
        while len(dst["inputs"]) <= to_slot:
            dst["inputs"].append({"name": "?", "type": "*", "link": None})
        dst["inputs"][to_slot]["link"] = link_id

    return {
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": max(l[0] for l in links),
        "nodes": nodes,
        "links": links,
        "groups": groups or [],
        "config": {},
        # v0.4.4 修复：extra.ds 必须为完整格式 {offset: [x, y], scale: n}。
        # 此前只有 scale 缺 offset → 前端加载时 canvas.ds.offset = undefined →
        # computeVisibleArea 读 offset[0] 抛
        # "TypeError: Cannot read properties of undefined (reading '0')" →
        # 弹窗"由于重新加载工作流数据出错，加载被中止"（加载中止）。
        # offset 为画布视口左上角坐标（配合 scale 0.75 让节点居中可见）。
        "extra": {"ds": ds or {"offset": [300, 300], "scale": 0.75}},
        "version": 0.4,
    }


# ===== 公共节点组（三个工作流共享的加载器部分）=====

def loaders_block(ids: dict, pos_x: int) -> tuple[list, list, dict]:
    """返回 (nodes, links, link_id_counter)：UnetLoaderGGUF / CLIPLoader / VAELoader。

    ids: {"unet": 1, "clip": 2, "vae": 7} 节点 id 映射
    """
    nodes = [
        node(ids["unet"], "UnetLoaderGGUF", [pos_x, 210], [315, 58], 0,
             outputs=[op("MODEL", "MODEL", [])],
             widgets=[GGUF], title="Unet Loader (GGUF)"),
        node(ids["clip"], "CLIPLoader", [pos_x, 380], [454, 122], 1,
             outputs=[op("CLIP", "CLIP", [])],
             widgets=[TE, "stable_diffusion", "default"],
             title="CLIP Loader (Text Encoder)"),
        node(ids["vae"], "VAELoader", [pos_x, 620], [315, 58], 2,
             outputs=[op("VAE", "VAE", [])],
             widgets=[VAE], title="VAE Loader"),
    ]
    links = []
    return nodes, links


def text_encode_block(ids: dict, link_start: int) -> tuple[list, list, int]:
    """正向 + 负向 CLIPTextEncode（clip 来源由调用方补 link）。

    ids: {"pos": 3, "neg": 4, "clip": 2}
    返回 (nodes, links, next_link_id)
    """
    nodes = [
        node(ids["pos"], "CLIPTextEncode", [430, 210], [400, 200], 3,
             inputs=[ip("clip", "CLIP", None)],
             outputs=[op("CONDITIONING", "CONDITIONING", [])],
             widgets=["一只橘猫趴在窗台上晒太阳，温暖的午后光线，高清细节"],
             title="正向提示词"),
        node(ids["neg"], "CLIPTextEncode", [430, 460], [400, 130], 4,
             inputs=[ip("clip", "CLIP", None)],
             outputs=[op("CONDITIONING", "CONDITIONING", [])],
             widgets=[""],
             title="负向提示词"),
    ]
    links = [
        [link_start, ids["clip"], 0, ids["pos"], 0, "CLIP"],
        [link_start + 1, ids["clip"], 0, ids["neg"], 0, "CLIP"],
    ]
    return nodes, links, link_start + 2


def ksampler_node(nid: int, pos: list, denoise: float, order: int) -> dict:
    """KSampler 节点（seed 随机 + turbo 默认 8 步 + cfg 1.0）。"""
    return node(nid, "KSampler", pos, [315, 262], order,
                inputs=[
                    ip("model", "MODEL", None),
                    ip("positive", "CONDITIONING", None),
                    ip("negative", "CONDITIONING", None),
                    ip("latent_image", "LATENT", None),
                ],
                outputs=[op("LATENT", "LATENT", [])],
                widgets=[0, "random", 8, 1.0, "euler", "simple", denoise],
                title="KSampler (turbo 8 步)")


# ===== 工作流 1：文生图（T2I）=====

def build_t2i() -> dict:
    ids = {"unet": 1, "clip": 2, "pos": 3, "neg": 4, "latent": 5,
           "ks": 6, "vae": 7, "dec": 8, "save": 9}
    nodes, links = [], []
    lk = 1

    lb, ll = loaders_block(ids, -130)
    nodes += lb
    links += ll

    tb, tl, lk = text_encode_block(ids, lk)
    nodes += tb
    links += tl

    nodes.append(node(ids["latent"], "EmptyFlux2LatentImage", [430, 680], [315, 106], 5,
                      outputs=[op("LATENT", "LATENT", [])],
                      widgets=[1024, 1024, 1],
                      title="空 Latent（Flux2 16ch）"))
    nodes.append(ksampler_node(ids["ks"], [900, 300], 1.0, 6))
    nodes.append(node(ids["dec"], "VAEDecode", [1300, 300], [210, 46], 7,
                      inputs=[ip("samples", "LATENT", None), ip("vae", "VAE", None)],
                      outputs=[op("IMAGE", "IMAGE", [])]))
    nodes.append(node(ids["save"], "SaveImage", [1570, 300], [460, 500], 8,
                      inputs=[ip("images", "IMAGE", None)],
                      widgets=["ERNIE_Image_UI"],
                      title="保存图片"))

    links += [
        [lk, ids["unet"], 0, ids["ks"], 0, "MODEL"],
        [lk + 1, ids["pos"], 0, ids["ks"], 1, "CONDITIONING"],
        [lk + 2, ids["neg"], 0, ids["ks"], 2, "CONDITIONING"],
        [lk + 3, ids["latent"], 0, ids["ks"], 3, "LATENT"],
        [lk + 4, ids["ks"], 0, ids["dec"], 0, "LATENT"],
        [lk + 5, ids["vae"], 0, ids["dec"], 1, "VAE"],
        [lk + 6, ids["dec"], 0, ids["save"], 0, "IMAGE"],
    ]
    return workflow(nodes, links)


# ===== 工作流 2：图生图（I2I，多图参考）=====

def build_i2i() -> dict:
    """v0.4.6 多图参考改造。

    旧版：源图 VAEEncode → KSampler latent_image（denoise 0.6 单图重绘）。
    新版：源图重绘保留 + 2 张参考图走 ReferenceLatent 链式 in-context：

        源图 LoadImage → VAEEncode ──────────────→ KSampler latent_image
        参考图1 LoadImage → VAEEncode ─┐
                                       ├ ReferenceLatent → ReferenceLatent
        参考图2 LoadImage → VAEEncode ─┘        （链式）→ KSampler positive

    机制依据（已实测验证，_test_multi_ref.py）：
        - ReferenceLatent 官方支持链式串联多张参考图
          （comfy_extras/nodes_edit_model.py 节点描述原文
           "you can chain multiple to set multiple reference images"）
        - Flux/Flux2 extra_conds 把 reference_latents 打包为 CONDList，
          transformer 循环编码每张参考为 in-context token
          （comfy/model_base.py:1043、comfy/ldm/flux/model.py:360）
        - 源图重绘（latent_image + denoise）与参考 in-context 可同时工作

    扩展方法：复制"参考图 LoadImage + VAEEncode"一对节点，再链一个
    ReferenceLatent（conditioning 接上一个 ReferenceLatent 输出）即可加第 3+ 张。
    """
    ids = {"unet": 1, "clip": 2, "pos": 3, "neg": 4,
           "src_load": 5, "src_enc": 6,
           "ref1_load": 7, "ref1_enc": 8, "ref2_load": 9, "ref2_enc": 10,
           "refl1": 11, "refl2": 12,
           "ks": 13, "vae": 14, "dec": 15, "save": 16}
    nodes, links = [], []
    lk = 1

    lb, _ = loaders_block(ids, -130)
    nodes += lb

    tb, tl, lk = text_encode_block(ids, lk)
    nodes += tb
    links += tl

    # 源图（重绘基础，denoise 控制保留度）
    nodes.append(node(ids["src_load"], "LoadImage", [430, 660], [340, 314], 5,
                      outputs=[op("IMAGE", "IMAGE", []), op("MASK", "MASK", None)],
                      widgets=[SAMPLE, "image"],
                      title="源图（重绘基础，denoise 越小越像）"))
    nodes.append(node(ids["src_enc"], "VAEEncode", [860, 660], [210, 46], 6,
                      inputs=[ip("pixels", "IMAGE", None), ip("vae", "VAE", None)],
                      outputs=[op("LATENT", "LATENT", [])]))

    # 参考图 ×2（in-context 参考）
    nodes.append(node(ids["ref1_load"], "LoadImage", [430, 1030], [340, 314], 7,
                      outputs=[op("IMAGE", "IMAGE", []), op("MASK", "MASK", None)],
                      widgets=[SAMPLE, "image"],
                      title="参考图 1（人物/风格）"))
    nodes.append(node(ids["ref1_enc"], "VAEEncode", [860, 1030], [210, 46], 8,
                      inputs=[ip("pixels", "IMAGE", None), ip("vae", "VAE", None)],
                      outputs=[op("LATENT", "LATENT", [])]))
    nodes.append(node(ids["ref2_load"], "LoadImage", [430, 1400], [340, 314], 9,
                      outputs=[op("IMAGE", "IMAGE", []), op("MASK", "MASK", None)],
                      widgets=[SAMPLE, "image"],
                      title="参考图 2（环境/物品）"))
    nodes.append(node(ids["ref2_enc"], "VAEEncode", [860, 1400], [210, 46], 10,
                      inputs=[ip("pixels", "IMAGE", None), ip("vae", "VAE", None)],
                      outputs=[op("LATENT", "LATENT", [])]))

    # ReferenceLatent 链式串联（多参考核心）
    nodes.append(node(ids["refl1"], "ReferenceLatent", [1140, 1030], [270, 78], 11,
                      inputs=[ip("conditioning", "CONDITIONING", None),
                              ip("latent", "LATENT", None)],
                      outputs=[op("CONDITIONING", "CONDITIONING", [])],
                      title="参考 1 注入"))
    nodes.append(node(ids["refl2"], "ReferenceLatent", [1140, 1170], [270, 78], 12,
                      inputs=[ip("conditioning", "CONDITIONING", None),
                              ip("latent", "LATENT", None)],
                      outputs=[op("CONDITIONING", "CONDITIONING", [])],
                      title="参考 2 注入"))

    nodes.append(ksampler_node(ids["ks"], [1450, 300], 0.6, 13))
    nodes.append(node(ids["dec"], "VAEDecode", [1760, 300], [210, 46], 14,
                      inputs=[ip("samples", "LATENT", None), ip("vae", "VAE", None)],
                      outputs=[op("IMAGE", "IMAGE", [])]))
    nodes.append(node(ids["save"], "SaveImage", [2030, 300], [460, 500], 15,
                      inputs=[ip("images", "IMAGE", None)],
                      widgets=["ERNIE_Image_UI_I2I"],
                      title="保存图片"))

    links += [
        # 源图编码
        [lk, ids["src_load"], 0, ids["src_enc"], 0, "IMAGE"],
        [lk + 1, ids["vae"], 0, ids["src_enc"], 1, "VAE"],
        # 参考图 1 编码
        [lk + 2, ids["ref1_load"], 0, ids["ref1_enc"], 0, "IMAGE"],
        [lk + 3, ids["vae"], 0, ids["ref1_enc"], 1, "VAE"],
        # 参考图 2 编码
        [lk + 4, ids["ref2_load"], 0, ids["ref2_enc"], 0, "IMAGE"],
        [lk + 5, ids["vae"], 0, ids["ref2_enc"], 1, "VAE"],
        # ReferenceLatent 链：pos → refl1 → refl2
        [lk + 6, ids["pos"], 0, ids["refl1"], 0, "CONDITIONING"],
        [lk + 7, ids["ref1_enc"], 0, ids["refl1"], 1, "LATENT"],
        [lk + 8, ids["refl1"], 0, ids["refl2"], 0, "CONDITIONING"],
        [lk + 9, ids["ref2_enc"], 0, ids["refl2"], 1, "LATENT"],
        # KSampler
        [lk + 10, ids["unet"], 0, ids["ks"], 0, "MODEL"],
        [lk + 11, ids["refl2"], 0, ids["ks"], 1, "CONDITIONING"],
        [lk + 12, ids["neg"], 0, ids["ks"], 2, "CONDITIONING"],
        [lk + 13, ids["src_enc"], 0, ids["ks"], 3, "LATENT"],
        # 输出
        [lk + 14, ids["ks"], 0, ids["dec"], 0, "LATENT"],
        [lk + 15, ids["vae"], 0, ids["dec"], 1, "VAE"],
        [lk + 16, ids["dec"], 0, ids["save"], 0, "IMAGE"],
    ]
    return workflow(nodes, links)


# ===== 工作流 3：图片编辑（Edit / Inpaint）=====

def build_edit() -> dict:
    ids = {"unet": 1, "clip": 2, "pos": 3, "neg": 4, "load": 5, "mask": 6,
           "enc": 7, "ks": 8, "vae": 9, "dec": 10, "save": 11}
    nodes, links = [], []
    lk = 1

    lb, _ = loaders_block(ids, -130)
    nodes += lb

    tb, tl, lk = text_encode_block(ids, lk)
    nodes += tb
    links += tl

    nodes.append(node(ids["load"], "LoadImage", [430, 680], [340, 314], 5,
                      outputs=[op("IMAGE", "IMAGE", []), op("MASK", "MASK", None)],
                      widgets=[SAMPLE, "image"],
                      title="源图（上传或选择）"))
    nodes.append(node(ids["mask"], "LoadImageMask", [430, 1040], [340, 130], 6,
                      outputs=[op("MASK", "MASK", [])],
                      widgets=[SAMPLE_MASK, "image", "red"],
                      title="Mask（白=重绘区）"))
    nodes.append(node(ids["enc"], "VAEEncodeForInpaint", [860, 620], [315, 88], 7,
                      inputs=[ip("pixels", "IMAGE", None),
                              ip("vae", "VAE", None),
                              ip("mask", "MASK", None)],
                      outputs=[op("LATENT", "LATENT", [])],
                      widgets=[6],
                      title="VAE Inpaint 编码"))
    nodes.append(ksampler_node(ids["ks"], [1240, 300], 1.0, 8))
    nodes.append(node(ids["dec"], "VAEDecode", [1630, 300], [210, 46], 9,
                      inputs=[ip("samples", "LATENT", None), ip("vae", "VAE", None)],
                      outputs=[op("IMAGE", "IMAGE", [])]))
    nodes.append(node(ids["save"], "SaveImage", [1900, 300], [460, 500], 10,
                      inputs=[ip("images", "IMAGE", None)],
                      widgets=["ERNIE_Image_UI_EDIT"],
                      title="保存图片"))

    links += [
        [lk, ids["load"], 0, ids["enc"], 0, "IMAGE"],
        [lk + 1, ids["vae"], 0, ids["enc"], 1, "VAE"],
        [lk + 2, ids["mask"], 0, ids["enc"], 2, "MASK"],
        [lk + 3, ids["unet"], 0, ids["ks"], 0, "MODEL"],
        [lk + 4, ids["pos"], 0, ids["ks"], 1, "CONDITIONING"],
        [lk + 5, ids["neg"], 0, ids["ks"], 2, "CONDITIONING"],
        [lk + 6, ids["enc"], 0, ids["ks"], 3, "LATENT"],
        [lk + 7, ids["ks"], 0, ids["dec"], 0, "LATENT"],
        [lk + 8, ids["vae"], 0, ids["dec"], 1, "VAE"],
        [lk + 9, ids["dec"], 0, ids["save"], 0, "IMAGE"],
    ]
    return workflow(nodes, links)


# ===== 工作流 4：合影助手（3 轮链式 inpaint，B 方案）=====

def build_group_photo() -> dict:
    """v0.4.7 合影助手：3 轮链式 inpaint，一次 Queue 把 3 个人依次画进背景。

    拓扑（每轮 base=10/20/30，画布 y=0/760/1520 三行）：

        背景 LoadImage(含 MaskEditor 蒙版) ──MASK──→ VAEEncodeForInpaint
        人物 LoadImage → VAEEncode → ReferenceLatent → KSampler.positive
        pixels：R1=背景图 IMAGE；R2/R3=上一轮 VAEDecode 输出（链式核心）
        KSampler(denoise 1.0) → VAEDecode → SaveImage(ERNIE_Group_R1/2/3)

    机制依据（_test_group_photo.py 实测通过）：
        - VAEEncodeForInpaint 的 noise_mask 与 ReferenceLatent 的
          reference_latents 可同时作用：蒙版区按人物参考生成，
          非蒙版区保持上一轮画面（区域隔离 PASS：蒙版区差~35，
          非蒙版区仅 VAE 往返漂移~5）
        - 3 轮链式在一个 prompt 内顺序执行（52.4s / 512×512）

    使用方法：
        1. 三个"背景+蒙版"节点选同一张背景图，各自右键 →
           Open in MaskEditor，分别涂 中间/左侧/右侧 人物区域
          （白=重绘区；mask 比人体大一圈，给交互手臂留空间）
        2. 三个"人物N参考图"节点各选一张人物照片
        3. 三段提示词按需改写（交互动作写在第 2/3 轮）→ Queue 一次跑完
        4. 加第 4 人：整行复制第 3 轮节点，pixels 接第 3 轮 VAEDecode

    默认（未涂蒙版）时 LoadImage 的 MASK 为全零 → 输出≈输入，安全无害。
    """
    ids = {"unet": 1, "clip": 2, "neg": 3, "vae": 4}
    nodes, links = [], []
    lk = 1
    order = 0

    # 共享加载器（unet/clip/vae）+ 三轮共用负向提示词
    lb, _ = loaders_block(ids, -700)
    nodes += lb
    order += 3
    nodes.append(node(ids["neg"], "CLIPTextEncode", [-700, 860], [400, 130], order,
                      inputs=[ip("clip", "CLIP", None)],
                      outputs=[op("CONDITIONING", "CONDITIONING", [])],
                      widgets=[""], title="负向提示词（三轮共用）"))
    links.append([lk, ids["clip"], 0, ids["neg"], 0, "CLIP"])
    lk += 1

    round_prompts = [
        "参考图中的人物站在画面中间，全身，自然放松的姿势",
        "第二张参考图中的人物站在左侧，与中间的人自然合影",
        "第三张参考图中的人物站在右侧，三人开心合影，光线自然",
    ]
    round_titles = ["中间", "左侧", "右侧"]
    groups = []

    for i in range(3):
        base = 10 * (i + 1)
        y = i * 760
        bg, person, penc, pos, refl, ienc, ks, dec, save = (base + j for j in range(9))

        nodes.append(node(bg, "LoadImage", [60, y], [340, 314], order,
                          outputs=[op("IMAGE", "IMAGE", []),
                                   op("MASK", "MASK", [], 1)],
                          widgets=[SAMPLE, "image"],
                          title=f"背景+蒙版{i + 1}（右键 MaskEditor 涂{round_titles[i]}人物区）"))
        order += 1
        nodes.append(node(pos, "CLIPTextEncode", [480, y], [400, 150], order,
                          inputs=[ip("clip", "CLIP", None)],
                          outputs=[op("CONDITIONING", "CONDITIONING", [])],
                          widgets=[round_prompts[i]],
                          title=f"第 {i + 1} 轮提示词"))
        order += 1
        nodes.append(node(person, "LoadImage", [480, y + 270], [340, 314], order,
                          outputs=[op("IMAGE", "IMAGE", []),
                                   op("MASK", "MASK", None, 1)],
                          widgets=[SAMPLE, "image"],
                          title=f"人物{i + 1}参考图"))
        order += 1
        nodes.append(node(penc, "VAEEncode", [900, y + 270], [210, 46], order,
                          inputs=[ip("pixels", "IMAGE", None), ip("vae", "VAE", None)],
                          outputs=[op("LATENT", "LATENT", [])]))
        order += 1
        nodes.append(node(refl, "ReferenceLatent", [1180, y + 270], [270, 78], order,
                          inputs=[ip("conditioning", "CONDITIONING", None),
                                  ip("latent", "LATENT", None)],
                          outputs=[op("CONDITIONING", "CONDITIONING", [])],
                          title=f"人物{i + 1}参考注入"))
        order += 1
        nodes.append(node(ienc, "VAEEncodeForInpaint", [1180, y], [315, 88], order,
                          inputs=[ip("pixels", "IMAGE", None),
                                  ip("vae", "VAE", None),
                                  ip("mask", "MASK", None)],
                          outputs=[op("LATENT", "LATENT", [])],
                          widgets=[6],
                          title=f"第 {i + 1} 轮 Inpaint 编码"))
        order += 1
        nodes.append(ksampler_node(ks, [1560, y], 1.0, order))
        order += 1
        nodes.append(node(dec, "VAEDecode", [1950, y], [210, 46], order,
                          inputs=[ip("samples", "LATENT", None), ip("vae", "VAE", None)],
                          outputs=[op("IMAGE", "IMAGE", [])]))
        order += 1
        nodes.append(node(save, "SaveImage", [2220, y], [460, 500], order,
                          inputs=[ip("images", "IMAGE", None)],
                          widgets=[f"ERNIE_Group_R{i + 1}"],
                          title=f"第 {i + 1} 轮输出"))
        order += 1

        # pixels 来源：R1=背景图 IMAGE；R2/R3=上一轮 VAEDecode（18 / 28）
        pixels_src = bg if i == 0 else 10 * i + 8
        links += [
            [lk, ids["clip"], 0, pos, 0, "CLIP"],
            [lk + 1, person, 0, penc, 0, "IMAGE"],
            [lk + 2, ids["vae"], 0, penc, 1, "VAE"],
            [lk + 3, pos, 0, refl, 0, "CONDITIONING"],
            [lk + 4, penc, 0, refl, 1, "LATENT"],
            [lk + 5, bg, 1, ienc, 2, "MASK"],
            [lk + 6, ids["vae"], 0, ienc, 1, "VAE"],
            [lk + 7, pixels_src, 0, ienc, 0, "IMAGE"],
            [lk + 8, ids["unet"], 0, ks, 0, "MODEL"],
            [lk + 9, refl, 0, ks, 1, "CONDITIONING"],
            [lk + 10, ids["neg"], 0, ks, 2, "CONDITIONING"],
            [lk + 11, ienc, 0, ks, 3, "LATENT"],
            [lk + 12, ks, 0, dec, 0, "LATENT"],
            [lk + 13, ids["vae"], 0, dec, 1, "VAE"],
            [lk + 14, dec, 0, save, 0, "IMAGE"],
        ]
        lk += 15

        groups.append({
            "id": i + 1,
            "title": f"第 {i + 1} 轮：把人物 {i + 1} 画进{round_titles[i]}（蒙版涂{round_titles[i]}区域）",
            "bounding": [40, y - 70, 2680, 660],
            "color": ["#3f789e", "#a1309b", "#b58b2a"][i],
            "font_size": 24,
            "flags": {},
        })

    return workflow(nodes, links, groups=groups,
                    ds={"offset": [-740, -120], "scale": 0.5})


def main() -> int:
    make_sample_images()

    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("ERNIE-Image Turbo 文生图.json", build_t2i()),
        ("ERNIE-Image Turbo 图生图.json", build_i2i()),
        ("ERNIE-Image Turbo 图片编辑.json", build_edit()),
        ("ERNIE-Image Turbo 合影助手.json", build_group_photo()),
    ]
    for fname, wf in outputs:
        path = WORKFLOWS_DIR / fname
        path.write_text(
            json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_nodes = len(wf["nodes"])
        n_links = len(wf["links"])
        print(f"已生成: {path.relative_to(PROJECT_ROOT)}（{n_nodes} 节点 / {n_links} 连线）")

    print("\n完成。打开 http://127.0.0.1:7001/ → 侧边栏 Workflows → ERNIE-Image 文件夹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
