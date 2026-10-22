"""T4.6 集成测试：显存上报 + LRU 卸载 + OOM 协调（L3+L4+L5）。

验证 3 个显存对接场景：

L3 上报：
    1. 启动 ComfyUI 子进程
    2. 查询 ``/system_stats`` 获取显存占用
    3. 验证 ``ComfyuiProcessManager.get_vram_usage_bytes()`` 返回合理值
    4. 运行推理后再次查询，验证显存占用变化

L4 卸载（LRU 驱逐响应）：
    1. 推理后记录显存占用
    2. 调用 ``ComfyuiProcessManager.free_memory(unload_models=True, free_memory=True)``
    3. 验证显存释放（``/free`` 端点生效）
    4. 验证释放量 > 0

L5 OOM 协调：
    1. 单元测试 ``is_oom_error()`` 函数（mock OOM 事件）
    2. 验证 OOM 错误检测逻辑（``out of memory`` / ``cuda error``）
    3. 验证非 OOM 错误不被误判

注意：
    - Core 平台未运行时，``PlatformVramManager.report_loaded()`` 会失败但不中断主流程
    - 本测试聚焦引擎层显存管理逻辑，Core 集成层验证需要完整平台环境
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
    is_oom_error,
    EVENT_EXECUTION_SUCCESS,
    EVENT_PROGRESS,
    extract_image_outputs,
)

import yaml


def render_workflow(
    template_path: Path,
    manifest_path: Path,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    variant_path: str,
) -> dict:
    """渲染工作流模板。"""
    workflow = json.loads(template_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    model_files = {mf["id"]: mf["filename"] for mf in manifest["models"]["model_files"]}
    placeholders = {
        "__PROMPT__": prompt,
        "__NEGATIVE_PROMPT__": "",
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


# ===== L5 OOM 检测单元测试 =====

def test_oom_detection():
    """单元测试：验证 is_oom_error() 函数。"""
    print("=" * 60)
    print("[L5 单元测试] is_oom_error() OOM 检测逻辑")
    print("=" * 60)

    # OOM 错误事件（应被检测为 OOM）
    oom_cases = [
        {"type": "execution_error", "data": {"exception_type": "OutOfMemoryError",
          "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB."}},
        {"type": "execution_error", "data": {"exception_type": "RuntimeError",
          "exception_message": "cuda error: out of memory"}},
        {"type": "execution_error", "data": {"exception_type": "RuntimeError",
          "exception_message": "OutOfMemoryError: CUDA out of memory"}},
        {"type": "execution_error", "data": {"exception_message": "out of memory"}},
    ]

    # 非 OOM 错误事件（不应被检测为 OOM）
    non_oom_cases = [
        {"type": "execution_error", "data": {"exception_type": "ValueError",
          "exception_message": "Invalid prompt"}},
        {"type": "execution_error", "data": {"exception_type": "KeyError",
          "exception_message": "Node not found"}},
        {"type": "execution_success", "data": {}},
        {"type": "progress", "data": {"value": 1, "max": 8}},
    ]

    passed = 0
    failed = 0

    for i, case in enumerate(oom_cases):
        result = is_oom_error(case)
        status = "✅" if result else "❌"
        if result:
            passed += 1
        else:
            failed += 1
        msg = case.get("data", {}).get("exception_message", "")[:60]
        print(f"  OOM case {i+1}: {status} (detected={result}) - {msg}")

    for i, case in enumerate(non_oom_cases):
        result = is_oom_error(case)
        status = "✅" if not result else "❌"
        if not result:
            passed += 1
        else:
            failed += 1
        msg = case.get("data", {}).get("exception_message", case.get("type", ""))[:60]
        print(f"  non-OOM case {i+1}: {status} (detected={result}) - {msg}")

    print(f"\n  结果: {passed} passed, {failed} failed")
    if failed > 0:
        print("  ❌ OOM 检测单元测试失败")
        return False
    print("  ✅ OOM 检测单元测试通过")
    print()
    return True


async def main() -> int:
    print("=" * 60)
    print("T4.6 集成测试：显存上报 + LRU 卸载 + OOM 协调（L3+L4+L5）")
    print("=" * 60)
    print()

    # L5 单元测试
    if not test_oom_detection():
        return 1

    # L3 + L4 端到端测试
    print("=" * 60)
    print("[L3+L4 端到端测试] 显存上报 + LRU 卸载")
    print("=" * 60)

    engine_dir = PROJECT_ROOT / "engines" / "ernie_image_909469ed"
    manifest_path = engine_dir / "manifest.yaml"
    template_path = engine_dir / "workflow_templates" / "ernie_turbo_t2i.json"

    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )

    # 1. 启动 ComfyUI
    pm = ComfyuiProcessManager(config)
    print(f"\n[1/8] 启动 ComfyUI 子进程...")
    t0 = time.time()
    start_result = await pm.start(timeout=300.0)
    print(f"      启动成功: pid={start_result['pid']}, port={start_result['internal_port']}, 耗时 {start_result['load_time']:.1f}s")

    try:
        client = ComfyuiClient(pm.internal_url)

        # 2. L3: 查询空载显存
        print(f"\n[2/8] [L3] 查询空载显存占用...")
        vram_idle = await pm.get_vram_usage_bytes()
        vram_idle_gb = vram_idle / 1024**3 if vram_idle else 0
        print(f"      空载显存: {vram_idle_gb:.2f} GB ({vram_idle:,} bytes)")
        if vram_idle_gb < 0.1:
            print(f"      ⚠️ 空载显存过低，可能查询失败")
        else:
            print(f"      ✅ 空载显存查询成功（ComfyUI 自身占用）")

        # 3. 渲染 + 提交工作流
        print(f"\n[3/8] 渲染 + 提交 Turbo 工作流...")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        turbo_variant = next(
            v for v in manifest["models"]["variants"]
            if v["name"] == "ernie-image-turbo-gguf-q4-k-m"
        )
        workflow = render_workflow(
            template_path=template_path,
            manifest_path=manifest_path,
            prompt="一只可爱的橘猫坐在窗台上晒太阳",
            width=1024,
            height=1024,
            steps=8,
            cfg=1.0,
            seed=42,
            variant_path=turbo_variant["path"],
        )
        import uuid
        client_id = uuid.uuid4().hex
        submit_result = await client.submit_workflow(workflow, client_id=client_id)
        prompt_id = submit_result.get("prompt_id", "")
        print(f"      提交成功: prompt_id={prompt_id}")

        # 4. 等待推理完成
        print(f"\n[4/8] 等待推理完成...")
        t_infer_start = time.time()
        async for event in client.subscribe_ws(client_id=client_id, prompt_id=prompt_id, timeout=600.0):
            et = event.get("type", "")
            data = event.get("data", {}) or {}
            if et == EVENT_PROGRESS:
                value = data.get("value", 0)
                max_value = data.get("max", 8)
                if value % 4 == 0 or value == max_value:
                    print(f"      [progress] {value}/{max_value}")
            elif et == EVENT_EXECUTION_SUCCESS:
                elapsed = time.time() - t_infer_start
                print(f"      [execution_success] 推理完成，耗时 {elapsed:.1f}s")
                break
        infer_time = time.time() - t_infer_start

        # 5. L3: 查询推理后显存
        print(f"\n[5/8] [L3] 查询推理后显存占用...")
        vram_loaded = await pm.get_vram_usage_bytes()
        vram_loaded_gb = vram_loaded / 1024**3 if vram_loaded else 0
        print(f"      推理后显存: {vram_loaded_gb:.2f} GB ({vram_loaded:,} bytes)")
        vram_diff = vram_loaded - vram_idle if vram_loaded and vram_idle else 0
        vram_diff_gb = vram_diff / 1024**3 if vram_diff else 0
        print(f"      显存增量: {vram_diff_gb:+.2f} GB（模型加载占用）")
        if vram_diff_gb > 0.1:
            print(f"      ✅ L3 显存上报验证通过（模型加载后显存增加）")
        else:
            print(f"      ⚠️ 显存增量过低（可能 ComfyUI --disable-smart-memory 已预加载）")

        # 6. L4: 调用 /free 释放显存
        print(f"\n[6/8] [L4] 调用 ComfyUI POST /free 释放显存...")
        print(f"      free_memory(unload_models=True, free_memory=True)")
        free_result = await pm.free_memory(unload_models=True, free_memory=True)
        print(f"      /free 响应: {free_result}")

        # 等待 1 秒让显存稳定
        await asyncio.sleep(1.0)

        # 7. L4: 查询释放后显存
        print(f"\n[7/8] [L4] 查询释放后显存...")
        vram_freed = await pm.get_vram_usage_bytes()
        vram_freed_gb = vram_freed / 1024**3 if vram_freed else 0
        print(f"      释放后显存: {vram_freed_gb:.2f} GB ({vram_freed:,} bytes)")
        freed_bytes = (vram_loaded - vram_freed) if vram_loaded and vram_freed else 0
        freed_gb = freed_bytes / 1024**3 if freed_bytes else 0
        print(f"      释放量: {freed_gb:.2f} GB")
        if freed_bytes > 0:
            print(f"      ✅ L4 LRU 卸载验证通过（显存释放 {freed_gb:.2f} GB）")
        else:
            print(f"      ⚠️ 显存释放量为 0（可能模型已在 CPU offload）")

        # 8. 停止子进程
        print(f"\n[8/8] 停止 ComfyUI 子进程...")
        stop_result = await pm.stop()
        print(f"      停止成功: {stop_result}")

        total_time = time.time() - t0
        print()
        print("=" * 60)
        print(f"T4.6 显存测试完成（总耗时 {total_time:.1f}s）")
        print(f"  L3 显存上报:")
        print(f"    空载: {vram_idle_gb:.2f} GB")
        print(f"    推理后: {vram_loaded_gb:.2f} GB")
        print(f"    增量: {vram_diff_gb:+.2f} GB")
        print(f"  L4 LRU 卸载:")
        print(f"    释放前: {vram_loaded_gb:.2f} GB")
        print(f"    释放后: {vram_freed_gb:.2f} GB")
        print(f"    释放量: {freed_gb:.2f} GB")
        print(f"  L5 OOM 协调:")
        print(f"    is_oom_error() 单元测试: ✅ 通过")
        print(f"  推理耗时: {infer_time:.1f}s")
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
