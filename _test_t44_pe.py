"""T4.4 集成测试：端到端 PE（Prompt Enhancer）推理。

验证：
    1. 启动 ComfyUI 子进程
    2. 加载 ernie_turbo_pe.json 工作流模板（含 CLIPLoader + TextGenerate 节点）
    3. 构建 PE 系统提示词（``<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]``）
    4. 渲染占位符（含 ``__PE_FILENAME__`` / ``__PE_PROMPT__``）
    5. 提交工作流到 ComfyUI（POST /prompt）
    6. 订阅 WS 事件，等待 execution_success
       - TextGenerate 节点先执行 LLM 推理生成增强 prompt
       - KSampler 再用增强 prompt 生成图片
    7. 从 history 获取图片输出，下载验证
    8. 停止子进程

与 T4.2（Turbo 无 PE）的差异：
    - 多加载 PE 模型（Ministral3，~7.15GB），显存占用更高
    - 工作流多 2 个节点：CLIPLoader(节点10) + TextGenerate(节点11)
    - CLIPTextEncode(节点3) 的 text 输入来自 TextGenerate 输出（增强后的 prompt）
    - 推理耗时更长（先 LLM 生成 ~2048 token，再 8 步扩散）

前置条件：
    - PE 模型已转换：``models/image/pe/ernie-image/model_with_tokenizer.safetensors``
      （含 tekken_model 键，否则 Mistral3Tokenizer 初始化报 TypeError）
    - extra_model_paths.yaml 已配置 PE 目录到 clip/text_encoders 路径
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ["AIRELIABLE_LOG_DIR"] = str(PROJECT_ROOT / "logs" / "engines")
os.environ["AIRELIABLE_MODELS_ROOT"] = str(PROJECT_ROOT / "models")

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.comfyui_process import ComfyuiProcessManager
from engines.ernie_image_909469ed.comfyui_client import (
    ComfyuiClient,
    EVENT_EXECUTION_SUCCESS,
    EVENT_EXECUTION_ERROR,
    EVENT_PROGRESS,
    EVENT_EXECUTED,
    extract_image_outputs,
)

import yaml


def build_pe_prompt(prompt: str, width: int, height: int) -> str:
    """构建 PE 系统提示词模板（与 engine.py._build_pe_prompt 一致）。

    PE 模型使用 ErnieTokenizer，不应用内置模板，需传入完整模板字符串。
    模板来源：ComfyUI 官方 blueprint ``Text to Image (Ernie Image Turbo).json``。
    """
    system_prompt = (
        "你是一个专业的文生图 Prompt 增强助手。"
        "你将收到用户的简短图片描述及目标生成分辨率，"
        "请据此扩写为一段内容丰富、细节充分的视觉描述，"
        "以帮助文生图模型生成高质量的图片。"
        "仅输出增强后的描述，不要包含任何解释或前缀。"
    )
    inst_payload = '{{"prompt": "{prompt}", "width": {width}, "height": {height}}}'.format(
        prompt=prompt, width=width, height=height,
    )
    return (
        "<s>"
        f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
        f"[INST]{inst_payload}[/INST]"
    )


def render_workflow(
    template_path: Path,
    manifest_path: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    variant_path: str,
    use_pe: bool,
) -> dict:
    """渲染工作流模板（替换占位符，含 PE 相关占位符）。"""
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    model_files = {mf["id"]: mf["filename"] for mf in manifest["models"]["model_files"]}

    pe_prompt = build_pe_prompt(prompt, width, height) if use_pe else ""

    placeholders = {
        "__PROMPT__": prompt,
        "__NEGATIVE_PROMPT__": negative_prompt,
        "__WIDTH__": width,
        "__HEIGHT__": height,
        "__STEPS__": steps,
        "__CFG__": cfg,
        "__SEED__": seed,
        "__GGUF_FILENAME__": Path(variant_path).name,
        "__VAE_FILENAME__": model_files.get("ernie-vae-safetensors", ""),
        "__TEXT_ENCODER_FILENAME__": model_files.get("ernie-text-encoder-safetensors", ""),
        "__PE_FILENAME__": model_files.get("ernie-pe-safetensors", ""),
        "__PE_PROMPT__": pe_prompt,
    }

    def replace(obj):
        if isinstance(obj, str):
            for ph, val in placeholders.items():
                if obj == ph:
                    return val
                if ph in obj:
                    obj = obj.replace(ph, str(val))
            return obj
        elif isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace(i) for i in obj]
        return obj
    workflow = replace(workflow)
    workflow.pop("_meta", None)
    return workflow


async def main() -> int:
    print("=" * 60)
    print("T4.4 集成测试：端到端 PE（Prompt Enhancer）推理")
    print("=" * 60)

    engine_dir = PROJECT_ROOT / "engines" / "ernie_image_909469ed"
    manifest_path = engine_dir / "manifest.yaml"
    template_path = engine_dir / "workflow_templates" / "ernie_turbo_pe.json"

    # 检查 PE 模型文件存在
    pe_model_path = PROJECT_ROOT / "models" / "image" / "pe" / "ernie-image" / "model_with_tokenizer.safetensors"
    if not pe_model_path.exists():
        print(f"[FAIL] PE 模型文件不存在: {pe_model_path}")
        print("       请先运行 _convert_pe_model.py 转换 PE 模型")
        return 1
    print(f"[0/8] PE 模型文件存在: {pe_model_path.name} ({pe_model_path.stat().st_size/1024/1024/1024:.2f} GB)")
    print()

    # 1. 构造 EngineConfig
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )
    print(f"[1/8] EngineConfig 构造完成")
    print()

    # 2. 启动 ComfyUI 子进程
    pm = ComfyuiProcessManager(config)
    print(f"[2/8] 启动 ComfyUI 子进程（timeout=300s）...")
    t0 = time.time()
    start_result = await pm.start(timeout=300.0)
    print(f"      启动成功: pid={start_result['pid']}, port={start_result['internal_port']}, 耗时 {start_result['load_time']:.1f}s")
    print()

    try:
        client = ComfyuiClient(pm.internal_url)

        # 3. 渲染工作流
        print(f"[3/8] 渲染 PE 工作流模板...")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        turbo_variant = next(
            v for v in manifest["models"]["variants"]
            if v["name"] == "ernie-image-turbo-gguf-q4-k-m"
        )
        prompt = "一只可爱的橘猫坐在窗台上晒太阳"
        workflow = render_workflow(
            template_path=template_path,
            manifest_path=manifest_path,
            prompt=prompt,
            negative_prompt="",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            seed=42,
            variant_path=turbo_variant["path"],
            use_pe=True,
        )
        print(f"      工作流渲染完成: {len(workflow)} 个节点")
        # 打印关键节点（含 PE 节点）
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type", "")
            inputs = node.get("inputs", {})
            if ct == "UnetLoaderGGUF":
                print(f"        [{node_id}] {ct}: unet_name={inputs.get('unet_name')}")
            elif ct == "CLIPLoader":
                clip_name = inputs.get("clip_name", "")
                clip_type = inputs.get("type", "")
                # 区分 Text Encoder 的 CLIPLoader 和 PE 的 CLIPLoader
                role = "PE" if "model_with_tokenizer" in clip_name else "TextEncoder"
                print(f"        [{node_id}] {ct} ({role}): clip_name={clip_name}, type={clip_type}")
            elif ct == "TextGenerate":
                pe_prompt_str = inputs.get("prompt", "")
                print(f"        [{node_id}] {ct}: max_length={inputs.get('max_length')}, "
                      f"temperature={inputs.get('sampling_mode.temperature')}, "
                      f"prompt_len={len(pe_prompt_str)}")
                print(f"              PE prompt 前 80 字符: {pe_prompt_str[:80]}...")
            elif ct == "VAELoader":
                print(f"        [{node_id}] {ct}: vae_name={inputs.get('vae_name')}")
            elif ct == "KSampler":
                print(f"        [{node_id}] {ct}: steps={inputs.get('steps')}, cfg={inputs.get('cfg')}, seed={inputs.get('seed')}")
            elif ct == "CLIPTextEncode":
                text_input = inputs.get("text", "")
                if isinstance(text_input, list):
                    print(f"        [{node_id}] {ct}: text=<from node {text_input[0]}>")
                else:
                    print(f"        [{node_id}] {ct}: text={str(text_input)[:50]}")
        print()

        # 4. 提交工作流
        print(f"[4/8] 提交工作流到 ComfyUI...")
        import uuid
        client_id = uuid.uuid4().hex
        submit_result = await client.submit_workflow(workflow, client_id=client_id)
        if "error" in submit_result:
            print(f"      提交失败: {submit_result}")
            return 1
        prompt_id = submit_result.get("prompt_id", "")
        print(f"      提交成功: prompt_id={prompt_id}, queue_number={submit_result.get('number')}")
        print()

        # 5. 订阅 WS 事件
        print(f"[5/8] 订阅 WebSocket 事件，等待 PE 推理完成...")
        print(f"      （PE 先执行 LLM 推理生成增强 prompt，再执行 8 步扩散）")
        t_infer_start = time.time()
        last_progress = 0.0
        pe_node_executed = False
        sampler_node_executed = False
        async for event in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
            et = event.get("type", "")
            data = event.get("data", {}) or {}
            if et == EVENT_PROGRESS:
                value = data.get("value", 0)
                max_value = data.get("max", 8)
                last_progress = value / max_value if max_value else 0
                print(f"      [progress] {value}/{max_value} ({last_progress*100:.1f}%)")
            elif et == EVENT_EXECUTED:
                node = data.get("node", "")
                # 检查是否是 PE 节点（TextGenerate，节点 11）完成
                if node == "11":
                    pe_node_executed = True
                    elapsed = time.time() - t_infer_start
                    print(f"      [executed] node={node} (TextGenerate/PE) - PE 推理完成，耗时 {elapsed:.1f}s")
                elif node == "6":
                    sampler_node_executed = True
                    print(f"      [executed] node={node} (KSampler) - 扩散采样完成")
                else:
                    print(f"      [executed] node={node}")
            elif et == EVENT_EXECUTION_SUCCESS:
                elapsed = time.time() - t_infer_start
                print(f"      [execution_success] PE 端到端推理完成，耗时 {elapsed:.1f}s")
                print(f"        PE 节点执行: {'✓' if pe_node_executed else '✗'}")
                print(f"        KSampler 执行: {'✓' if sampler_node_executed else '✗'}")
                break
            elif et == EVENT_EXECUTION_ERROR:
                print(f"      [execution_error] {data}")
                import traceback
                traceback.print_exc()
                return 1
        print()

        # 6. 从 history 获取图片输出
        print(f"[6/8] 查询 history 获取图片输出...")
        history = await client.get_history(prompt_id)
        if not history:
            print(f"      history 为空")
            return 1
        image_outputs = extract_image_outputs(history, prompt_id)
        if not image_outputs:
            print(f"      无图片输出")
            return 1
        print(f"      找到 {len(image_outputs)} 张图片: {image_outputs}")
        print()

        # 7. 下载图片，验证文件有效
        print(f"[7/8] 下载图片并验证...")
        output_dir = PROJECT_ROOT / "output" / "image"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for idx, img_info in enumerate(image_outputs):
            filename = img_info.get("filename", "")
            subfolder = img_info.get("subfolder", "")
            image_type = img_info.get("type", "output")
            image_bytes = await client.get_image(filename, subfolder, image_type)
            if image_bytes is None:
                print(f"      下载失败: {img_info}")
                continue
            local_path = output_dir / f"t44_pe_{int(time.time())}_{idx}_{filename}"
            local_path.write_bytes(image_bytes)
            saved_files.append(local_path)
            print(f"      已保存: {local_path.name} ({len(image_bytes)} bytes)")
        if not saved_files:
            print(f"      无图片保存成功")
            return 1
        print()

        # 8. 停止子进程
        print(f"[8/8] 停止 ComfyUI 子进程...")
        stop_result = await pm.stop()
        print(f"      停止成功: {stop_result}")
        print()

        total_time = time.time() - t0
        print("=" * 60)
        print(f"T4.4 PE 测试通过 ✅ （总耗时 {total_time:.1f}s）")
        print(f"  启动: {start_result['load_time']:.1f}s")
        print(f"  PE + 推理: {time.time() - t_infer_start:.1f}s")
        print(f"  图片: {len(saved_files)} 张")
        for f in saved_files:
            print(f"    - {f}")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        await pm.stop()
        return 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
