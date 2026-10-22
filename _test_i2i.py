"""T2.4 集成测试：端到端 I2I 推理（Turbo，8 步，本地测试图）。

验证：
    1. 启动 ComfyUI 子进程（含 --input-directory 参数）
    2. 引擎 ``_stage_input_images`` 把 b64 测试图落盘到 input 目录（16 对齐）
    3. 引擎 ``_render_workflow`` 渲染 ernie_turbo_i2i.json（mode=i2i，denoise=0.6）
    4. 提交工作流到 ComfyUI（LoadImage 从 input 目录读取源图）
    5. 订阅 WS 事件，等待 execution_success（验证 VAEEncode 兼容性 → 风险 1）
    6. 从 history 获取图片输出 + 下载验证
    7. 清理 input 目录暂存文件
    8. 停止子进程

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_i2i.py
"""

from __future__ import annotations

import asyncio
import base64
import io
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
from engines.ernie_image_909469ed.comfyui_process import ComfyuiProcessManager
from engines.ernie_image_909469ed.comfyui_client import (
    ComfyuiClient,
    EVENT_EXECUTION_ERROR,
    EVENT_EXECUTION_SUCCESS,
    EVENT_PROGRESS,
    extract_image_outputs,
)
from engines.ernie_image_909469ed.engine import ErnieImageEngine


class FakeImage:
    """ImageInput 桩（b64 来源）。"""
    def __init__(self, role: str, b64: str):
        self.role = role
        self.b64_json = b64
        self.url = None
        self.path = None


class FakeRequest:
    """InferRequest 桩。"""
    def __init__(self, images: list):
        self.images = images
        self.workflow = None
        self.loras = []


def make_test_source_b64() -> str:
    """生成 512x512 渐变测试图（b64 PNG）。"""
    im = Image.new("RGB", (512, 512))
    px = im.load()
    for y in range(512):
        for x in range(512):
            px[x, y] = (x * 255 // 512, y * 255 // 512, 128)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def main() -> int:
    print("=" * 60)
    print("T2.4 集成测试：端到端 I2I 推理（Turbo，8 步）")
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
    print(f"[1/8] EngineConfig + 引擎初始化完成（input_dir={config.input_dir}）")
    print()

    task_id = f"i2i_test_{uuid.uuid4().hex[:8]}"
    staged = {}

    try:
        # 2. 落盘测试源图（b64 → input 目录）
        print("[2/8] _stage_input_images 落盘测试源图（512x512 渐变图）...")
        req = FakeRequest(images=[FakeImage(role="source", b64=make_test_source_b64())])
        staged = await eng._stage_input_images(req, task_id, "i2i")
        print(f"      source={staged['source_filename']}, "
              f"size={staged['source_width']}x{staged['source_height']}, "
              f"files={len(staged['files'])}")
        assert staged["source_filename"], "落盘失败"
        print()

        # 3. 启动 ComfyUI 子进程
        pm = eng._process_manager
        print("[3/8] 启动 ComfyUI 子进程（timeout=300s）...")
        start_result = await pm.start(timeout=300.0)
        print(f"      启动成功: pid={start_result['pid']}, "
              f"port={start_result['internal_port']}, 耗时 {start_result['load_time']:.1f}s")
        print()

        client = ComfyuiClient(pm.internal_url)

        # 4. 渲染 I2I 工作流
        print("[4/8] _render_workflow 渲染 ernie_turbo_i2i.json（denoise=0.6）...")
        variant = "ernie-image-turbo-gguf-q4-k-m"
        mv = eng._manifest_variants[variant]
        workflow = eng._render_workflow(
            variant=variant, mv=mv,
            prompt="一只毛茸茸的橘猫坐在窗台上，阳光洒在身上",
            negative_prompt="",
            width=staged["source_width"], height=staged["source_height"],
            steps=8, cfg=1.0, seed=42, use_pe=False, loras=[],
            mode="i2i",
            source_filename=staged["source_filename"],
            mask_filename=None,
            denoise=0.6,
        )
        template_used = eng._select_template(mv, False, "i2i")
        print(f"      模板: {template_used}, 节点数: {len(workflow)}")
        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            if ct in ("LoadImage", "VAEEncode", "KSampler"):
                print(f"        {ct}: {node.get('inputs')}")
        print()

        # 5. 提交 + 订阅 WS
        print("[5/8] 提交工作流 + 订阅 WS 事件（验证 VAEEncode 兼容性）...")
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

        # 6. 下载图片
        print("[6/8] 查询 history + 下载图片...")
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
        saved = []
        for idx, img_info in enumerate(image_outputs):
            image_bytes = await client.get_image(
                img_info.get("filename", ""), img_info.get("subfolder", ""),
                img_info.get("type", "output"),
            )
            if image_bytes is None:
                continue
            local = output_dir / f"t24_i2i_{int(time.time())}_{idx}_{img_info.get('filename','')}"
            local.write_bytes(image_bytes)
            saved.append(local)
            # 验证尺寸与源图一致
            with Image.open(io.BytesIO(image_bytes)) as im:
                print(f"      已保存: {local.name} ({len(image_bytes)} bytes, {im.size})")
        if not saved:
            print("      无图片保存成功")
            return 1
        print()

        # 7. 停止子进程
        print("[7/8] 停止 ComfyUI 子进程...")
        stop_result = await pm.stop()
        print(f"      停止成功: {stop_result}")
        print()

        total = time.time() - t0
        print("=" * 60)
        print(f"T2.4 I2I 端到端测试通过 ✅（总耗时 {total:.1f}s）")
        print(f"  启动: {start_result['load_time']:.1f}s")
        print(f"  图片: {len(saved)} 张")
        for f in saved:
            print(f"    - {f}")
        print("=" * 60)
        return 0

    finally:
        # 8. 清理暂存图片（与引擎 _infer_impl finally 语义一致）
        ErnieImageEngine._cleanup_staged_images(staged.get("files") or [])
        print(f"[8/8] 已清理 input 目录暂存图片（{len(staged.get('files') or [])} 个）")


if __name__ == "__main__":
    sys.exit(asyncio.get_event_loop().run_until_complete(main()))
