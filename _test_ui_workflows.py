"""端到端测试：三个 ComfyUI UI 工作流（t2i / i2i / edit）验证与实际执行。

验证层次：
    1. 结构校验：每个节点 type 存在于 /object_info；widgets_values 数量与
       object_info 的 widget 参数数（含 control_after_generate / image_upload
       前端占位）一致；links 与节点 inputs/outputs 双向一致
    2. 语义校验（最强验证）：UI 工作流 → API prompt 转换（模拟 ComfyUI 前端
       graphToPrompt），提交 /prompt 执行，等待完成，下载输出图片验证非空
    3. 像素校验（Edit）：mask 外区域保留源图（低差异）、mask 内重绘（高差异）

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_ui_workflows.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

import httpx
from PIL import Image

from engines.ernie_image_909469ed.comfyui_client import (
    ComfyuiClient,
    EVENT_EXECUTION_SUCCESS,
    extract_image_outputs,
)
from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.comfyui_process import ComfyuiProcessManager

WORKFLOWS_DIR = (
    PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI"
    / "user" / "default" / "workflows" / "ERNIE-Image"
)
INPUT_DIR = PROJECT_ROOT / "output" / "input"

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


# ===== UI → API 转换（模拟 ComfyUI 前端 graphToPrompt）=====

LINK_TYPES = {"MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK"}


def _is_link_type(pdef: list) -> bool:
    """参数是否为 link 类型（连接线输入）。"""
    ptype = pdef[0] if pdef and isinstance(pdef, list) else None
    return isinstance(ptype, str) and ptype in LINK_TYPES


def _widget_param_count(spec: dict) -> int:
    """计算节点应有的 widget 值数量（含前端自动附加的占位 widget）。

    规则：
        - required + optional 中非 link 参数各占 1 个 widget 值
        - 带 control_after_generate 的参数（如 KSampler.seed）额外 +1（前端附加
          "random/fixed/..." 控件）
        - 带 image_upload 标记的参数（如 LoadImage.image）额外 +1（前端附加
          upload 按钮控件，序列化值为 "image"）
    """
    count = 0
    for section in ("required", "optional"):
        for pname, pdef in (spec.get(section) or {}).items():
            if _is_link_type(pdef):
                continue
            count += 1
            opts = pdef[1] if len(pdef) > 1 and isinstance(pdef[1], dict) else {}
            if opts.get("control_after_generate"):
                count += 1
            if opts.get("image_upload"):
                count += 1
    return count


def ui_to_prompt(ui: dict, object_info: dict) -> tuple[dict, list[str]]:
    """UI 工作流 → API prompt 格式（与 ComfyUI 前端 graphToPrompt 语义一致）。

    link 输入以 ``links`` 数组为准（``[link_id, from_node, from_slot, to_node,
    to_slot, type]``），to_slot 索引目标节点 ``inputs`` 数组的参数名——
    不依赖 inputs[].link 回填值（回填由生成脚本保证，前端亦会重建）。
    """
    prompt: dict = {}
    errors: list[str] = []
    node_inputs = {n["id"]: n.get("inputs", []) for n in ui["nodes"]}

    for node in ui["nodes"]:
        ntype = node["type"]
        if ntype not in object_info:
            errors.append(f"节点 {ntype} 不存在于 object_info")
            continue
        prompt[str(node["id"])] = {"class_type": ntype, "inputs": {}}

    # link 输入：遍历 links 数组（to_slot → 目标节点 inputs 数组索引 → 参数名）
    for l in ui["links"]:
        link_id, from_node, from_slot, to_node, to_slot, _ = l
        if str(to_node) not in prompt:
            errors.append(f"link {link_id} 目标节点 {to_node} 不存在")
            continue
        ports = node_inputs.get(to_node, [])
        if to_slot >= len(ports):
            errors.append(f"link {link_id} 的 to_slot={to_slot} 超出节点 {to_node} inputs 数量")
            continue
        pname = ports[to_slot]["name"]
        prompt[str(to_node)]["inputs"][pname] = [str(from_node), from_slot]

    # widget 输入：按 required + optional 顺序消费 widgets_values
    for node in ui["nodes"]:
        ntype = node["type"]
        if ntype not in object_info:
            continue
        spec = object_info[ntype].get("input", {})
        inputs = prompt[str(node["id"])]["inputs"]
        wv = list(node.get("widgets_values") or [])
        wi = 0
        for section in ("required", "optional"):
            for pname, pdef in (spec.get(section) or {}).items():
                if _is_link_type(pdef):
                    continue
                opts = pdef[1] if len(pdef) > 1 and isinstance(pdef[1], dict) else {}
                if wi < len(wv):
                    inputs[pname] = wv[wi]
                wi += 1
                if opts.get("control_after_generate") or opts.get("image_upload"):
                    wi += 1  # 跳过前端附加控件占位值

    return prompt, errors


# ===== 结构校验 =====

def validate_workflow(ui: dict, object_info: dict, name: str) -> bool:
    """校验 UI 工作流结构（节点存在 + widget 数量 + 连线一致 + extra.ds 完整）。"""
    print(f"\n  --- {name} 结构校验 ---")
    all_ok = True
    link_map = {l[0]: l for l in ui["links"]}

    # extra.ds 完整性（v0.4.4 修复项：缺 offset 会导致前端加载中止）
    ds = ui.get("extra", {}).get("ds")
    check(f"[{name}] extra.ds 含 offset+scale",
          isinstance(ds, dict) and isinstance(ds.get("offset"), list) and len(ds["offset"]) == 2
          and isinstance(ds.get("scale"), (int, float)))

    for node in ui["nodes"]:
        ntype = node["type"]
        if ntype not in object_info:
            check(f"[{name}] 节点 {ntype} 已注册", False)
            all_ok = False
            continue
        expected = _widget_param_count(object_info[ntype].get("input", {}))
        actual = len(node.get("widgets_values") or [])
        # VAEDecode / VAEEncode 等无 widget 节点存空列表
        check(f"[{name}] {ntype}#{node['id']} widget 数量（期望 {expected}，实际 {actual}）",
              expected == actual)

    # 连线一致性：每个 input link 都在 links 数组且指向存在的节点
    node_ids = {n["id"] for n in ui["nodes"]}
    for node in ui["nodes"]:
        for port in node.get("inputs", []):
            lid = port.get("link")
            if lid is None:
                continue
            link = link_map.get(lid)
            ok = link is not None and link[1] in node_ids and link[3] in node_ids
            if not ok:
                check(f"[{name}] 连线 {lid} 一致性", False, f"link={link}")
                all_ok = False
    if all_ok:
        check(f"[{name}] 全部连线一致（{len(ui['links'])} 条）", True)
    return all_ok


# ===== 像素比对（Edit 用）=====

def region_diff(a: Image.Image, b: Image.Image, region: str) -> float:
    """mask 外 / mask 内平均像素差（1/16 采样）。"""
    size = 512
    cx = cy = size // 2
    safe_r, r = 196, 128
    pa, pb = a.load(), b.load()
    total, count = 0, 0
    for y in range(0, size, 4):
        for x in range(0, size, 4):
            d = math.hypot(x - cx, y - cy)
            if region == "outside" and d < safe_r:
                continue
            if region == "inside" and d > r * 0.5:
                continue
            total += max(abs(pa[x, y][c] - pb[x, y][c]) for c in range(3))
            count += 1
    return total / count if count else 0.0


async def main() -> int:
    print("=" * 64)
    print("端到端测试：三个 ComfyUI UI 工作流（结构 + 转换 + 实际执行）")
    print("=" * 64)
    t0 = time.time()

    # 1. 启动 ComfyUI
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )
    pm = ComfyuiProcessManager(config)
    print("[1/5] 启动 ComfyUI 子进程...")
    start = await pm.start(timeout=300.0)
    print(f"      就绪: pid={start['pid']}, port={start['internal_port']}，耗时 {start['load_time']:.1f}s")
    client = ComfyuiClient(pm.internal_url)

    try:
        # 2. object_info
        print("\n[2/5] 获取 /object_info...")
        async with httpx.AsyncClient(timeout=60.0) as hc:
            r = await hc.get(f"{pm.internal_url}/object_info")
            object_info = r.json()
        print(f"      已注册 {len(object_info)} 个节点定义")

        # 3. 加载 + 校验三个 UI 工作流
        print("\n[3/5] 结构校验...")
        workflows = {}
        for fname in ("ERNIE-Image Turbo 文生图.json",
                      "ERNIE-Image Turbo 图生图.json",
                      "ERNIE-Image Turbo 图片编辑.json",
                      "ERNIE-Image Turbo 合影助手.json"):
            path = WORKFLOWS_DIR / fname
            ui = json.loads(path.read_text(encoding="utf-8"))
            workflows[fname] = ui
            validate_workflow(ui, object_info, fname[:-5])

        # 4. UI→API 转换 + 逐个执行
        print("\n[4/5] 转换 + 实际执行三个工作流...")
        output_dir = PROJECT_ROOT / "output" / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, list[Path]] = {}

        for fname, ui in workflows.items():
            mode = fname[:-5]
            print(f"\n  --- 执行 {mode} ---")
            prompt, errors = ui_to_prompt(ui, object_info)
            check(f"[{mode}] UI→API 转换无错误", not errors, str(errors[:3]))
            if errors:
                continue

            # 执行（T2I 用 1024 太慢？保持 1024 —— turbo 8 步实测 ~70s 可接受）
            client_id = uuid.uuid4().hex
            submit = await client.submit_workflow(prompt, client_id=client_id)
            if "error" in submit:
                check(f"[{mode}] 提交成功", False, str(submit)[:200])
                continue
            prompt_id = submit.get("prompt_id", "")
            check(f"[{mode}] 提交成功", bool(prompt_id))

            t_exec = time.time()
            done = False
            async for ev in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
                et = ev.get("type", "")
                if et == EVENT_EXECUTION_SUCCESS:
                    done = True
                    break
                if et == "execution_error":
                    check(f"[{mode}] 执行成功", False, str(ev.get("data", {}))[:300])
                    break
            check(f"[{mode}] 执行成功", done, f"耗时 {time.time()-t_exec:.1f}s")
            if not done:
                continue

            history = await client.get_history(prompt_id)
            imgs = extract_image_outputs(history, prompt_id) if history else []
            check(f"[{mode}] 产出图片", len(imgs) > 0)
            if not imgs:
                continue

            saved = []
            for info in imgs:
                data = await client.get_image(
                    info.get("filename", ""), info.get("subfolder", ""), info.get("type", "output"))
                if data is None:
                    continue
                local = output_dir / f"ui_wf_{mode[:4]}_{int(time.time())}_{info.get('filename','')}"
                local.write_bytes(data)
                saved.append(local)
            check(f"[{mode}] 下载保存", len(saved) > 0)
            results[mode] = saved
            print(f"      耗时 {time.time()-t_exec:.1f}s，输出 {len(saved)} 张")

        # 5. Edit 像素校验（mask 外保留 / mask 内重绘）
        print("\n[5/5] Edit 像素级校验...")
        edit_saved = results.get("ERNIE-Image Turbo 图片编辑") or []
        if edit_saved:
            with Image.open(edit_saved[0]) as out_im:
                out_rgb = out_im.convert("RGB")
                with Image.open(INPUT_DIR / "ernie_sample.png") as src_im:
                    src_rgb = src_im.convert("RGB")
                avg_out = region_diff(src_rgb, out_rgb, "outside")
                avg_in = region_diff(src_rgb, out_rgb, "inside")
                print(f"      mask 外平均差={avg_out:.1f}（VAE 固有损失基线 ~6.5，阈值 15）")
                print(f"      mask 内平均差={avg_in:.1f}（重绘生效阈值 20）")
                check("[Edit] mask 外保留源图（< 15）", avg_out < 15.0)
                check("[Edit] mask 内重绘生效（> 20）", avg_in > 20.0)
        else:
            check("[Edit] 像素校验（跳过：无输出）", False)

        print("\n" + "=" * 64)
        print(f"结果：PASS={PASS}  FAIL={FAIL}（总耗时 {time.time()-t0:.1f}s）")
        for mode, saved in results.items():
            for p in saved:
                print(f"  - {p}")
        print("=" * 64)
        return 0 if FAIL == 0 else 1

    finally:
        await pm.stop()
        print("\nComfyUI 子进程已停止")


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
