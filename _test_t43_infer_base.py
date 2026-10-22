"""T4.3 集成测试：端到端 T2V 推理（Base，50 步 1024x1024）。

验证：
    1. 启动 ComfyUI 子进程
    2. 加载 ernie_base_t2i.json 工作流模板（50 步高质量文生图）
    3. 渲染占位符（使用 Base variant：ernie-image-Q4_K_M.gguf，cfg=5.0，steps=50）
    4. 提交工作流到 ComfyUI
    5. 订阅 WS 事件，等待 execution_success
    6. 从 history 获取图片输出，下载验证
    7. 停止子进程

与 T4.2（Turbo 8 步）的差异：
    - 使用 Base variant（ernie-image-Q4_K_M.gguf，4.67GB，与 Turbo 同尺寸但权重不同）
    - 50 步（vs Turbo 8 步），推理耗时更长
    - cfg=5.0（vs Turbo 1.0），更强的引导
    - 验证 ernie_base_t2i.json 模板正确性
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
    extract_image_outputs,
)

import yaml


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
) -> dict:
    """渲染工作流模板（替换占位符）。"""
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    model_files = {mf["id"]: mf["filename"] for mf in manifest["models"]["model_files"]}
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
        "__PE_PROMPT__": "",
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
    print("T4.3 集成测试：端到端 T2V 推理（Base，50 步 1024x1024）")
    print("=" * 60)

    engine_dir = PROJECT_ROOT / "engines" / "ernie_image_909469ed"
    manifest_path = engine_dir / "manifest.yaml"
    template_path = engine_dir / "workflow_templates" / "ernie_base_t2i.json"

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
        print(f"[3/8] 渲染 Base 工作流模板（50 步）...")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        base_variant = next(
            v for v in manifest["models"]["variants"]
            if v["name"] == "ernie-image-gguf-q4-k-m"
        )
        workflow = render_workflow(
            template_path=template_path,
            manifest_path=manifest_path,
            prompt="一只可爱的橘猫坐在窗台上晒太阳，毛茸茸的，阳光洒在身上，温馨治愈",
            negative_prompt="",
            width=1024,
            height=1024,
            steps=50,
            cfg=5.0,
            seed=42,
            variant_path=base_variant["path"],
        )
        print(f"      工作流渲染完成: {len(workflow)} 个节点")
        for node_id, node in workflow.items():
            if isinstance(node, dict):
                ct = node.get("class_type", "")
                inputs = node.get("inputs", {})
                if ct == "UnetLoaderGGUF":
                    print(f"        {ct}: unet_name={inputs.get('unet_name')}")
                elif ct == "CLIPLoader":
                    print(f"        {ct}: clip_name={inputs.get('clip_name')}")
                elif ct == "VAELoader":
                    print(f"        {ct}: vae_name={inputs.get('vae_name')}")
                elif ct == "KSampler":
                    print(f"        {ct}: steps={inputs.get('steps')}, cfg={inputs.get('cfg')}, seed={inputs.get('seed')}")
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
        print(f"[5/8] 订阅 WebSocket 事件，等待 50 步推理完成...")
        t_infer_start = time.time()
        last_progress = 0.0
        progress_count = 0
        async for event in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
            et = event.get("type", "")
            data = event.get("data", {}) or {}
            if et == EVENT_PROGRESS:
                value = data.get("value", 0)
                max_value = data.get("max", 50)
                last_progress = value / max_value if max_value else 0
                progress_count += 1
                # 每 10 步打印一次，避免输出过多
                if value % 10 == 0 or value == max_value:
                    print(f"      [progress] {value}/{max_value} ({last_progress*100:.1f}%)")
            elif et == EVENT_EXECUTION_SUCCESS:
                elapsed = time.time() - t_infer_start
                print(f"      [execution_success] 推理完成，耗时 {elapsed:.1f}s（共 {progress_count} 个 progress 事件）")
                break
            elif et == EVENT_EXECUTION_ERROR:
                print(f"      [execution_error] {data}")
                return 1
            elif et in ("executing", "executed"):
                node = data.get("node", "")
                print(f"      [{et}] node={node}")
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
            local_path = output_dir / f"t43_base_{int(time.time())}_{idx}_{filename}"
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
        print(f"T4.3 Base 测试通过 ✅ （总耗时 {total_time:.1f}s）")
        print(f"  启动: {start_result['load_time']:.1f}s")
        print(f"  推理: {time.time() - t_infer_start:.1f}s（50 步）")
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
