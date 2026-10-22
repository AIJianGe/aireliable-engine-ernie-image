"""ERNIE-Image 引擎推理核心（v0.3.0 ComfyUI 子进程模式，推倒重来）。

封装 ComfyUI 子进程管理 + 工作流模板 + 推理调度，提供：
    - ComfyUI 子进程启动/停止（通过 ``ComfyuiProcessManager``）
    - ComfyUI 工作流提交/事件订阅（通过 ``ComfyuiClient``）
    - Com�UI Web UI 反代（由 ``server.py`` 实现）
    - 显存对接 L1-L5（L1 CLI 参数 / L3 上报 / L4 LRU / L5 OOM 协调）
    - variant 扫描（从 manifest.yaml 读取 + 扫描 unet 目录判断可用性）
    - LoRA 扫描（扫描 lora_dir，运行时通过工作流模板注入 LoraLoader 节点）

继承 ``CustomEngine``（integrated 范式），实现 7 个 ``_xxx_impl`` 方法。

v0.3.0 关键变更（与 v0.2 范式 B 的差异）：
    1. 继承 ``CustomEngine``（不再是 ``WorkflowEngine``）
    2. 删除节点注册表 / 工作流 yaml 加载 / WorkflowExecutor 执行
    3. 推理通过 ComfyUI HTTP + WebSocket API 驱动（不再走 SDK 工作流执行器）
    4. 模型加载由 ComfyUI 内部完成（不再在 _load_impl 中加载权重）
    5. SSE 帧由 ComfyUI WS 事件转换而来（不再由 SDK ProgressReporter 生成）

8 端点契约不变：
    - GET  /capabilities   ← _capabilities_impl
    - POST /load           ← _load_impl（启动 ComfyUI 子进程 + 显存上报 L3）
    - POST /unload         ← _unload_impl（ComfyUI /free 释放显存 L4 + 停止子进程）
    - POST /infer          ← _infer_impl（submit workflow + 订阅 WS + 下载图片 + SSE 推送）
    - GET  /status         ← _status_impl（返回状态 + ComfyUI 子进程状态）
    - POST /lora/load      ← _load_lora_impl（记录到 loaded_loras，下次推理生效）
    - POST /lora/unload    ← _unload_lora_impl（从 loaded_loras 移除）
    - GET  /health         ← BaseEngine.health

显存对接架构（详见分析文档第六部分）：
    - L1 启动前预检查：平台 Core 主导（已有），引擎层不参与
    - L1 启动参数：``ComfyuiProcessManager._build_command`` 配置 ``--normalvram --disable-smart-memory``
    - L2 ComfyUI 内部：ComfyUI 自管（normalvram 档位，不上 offload 到 CPU）
    - L3 上报：推理后查询 ``/system_stats`` + 调 ``PlatformVramManager.report_loaded``
    - L4 LRU 响应：``_unload_impl`` 调 ComfyUI ``POST /free`` 释放显存
    - L5 OOM 协调：检测 WS ``execution_error`` + ``is_oom_error`` → ``/free`` + ``request_vram`` → 重试

参考文档：
    - ANALYSIS-TASKS/ERNIE_IMAGE_COMFYUI_MIGRATION_ANALYSIS.md
    - ANALYSIS-TASKS/ERNIE_IMAGE_COMFYUI_MIGRATION_TASKS.md T2.4
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
    make_error_frame,
    make_final_frame,
    make_frame,
)
from aireliable_engine.vram_manager import PlatformVramManager

from .comfyui_client import (
    EVENT_EXECUTED,
    EVENT_EXECUTION_ERROR,
    EVENT_EXECUTION_INTERRUPTED,
    EVENT_EXECUTION_START,
    EVENT_EXECUTION_SUCCESS,
    EVENT_PROGRESS,
    ComfyuiClient,
    extract_image_outputs,
    is_oom_error,
)
from .comfyui_process import ComfyuiProcessManager
from .config import EngineConfig
from .schemas import (
    SUPPORTED_SIZE_STRINGS,
    ErnieCapabilitiesResponse,
    ErnieVariantInfo,
    size_to_wh,
)
from .task_reporting import ErnieTaskReporting

logger = logging.getLogger(__name__)

# v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）。
# ERNIE-Image 的 inputs 是自由 dict（无 pydantic Inputs 模型），
# 此处按 _infer_impl 实际读取的字段手写 schema（engine.py L610-L672）。
_INPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "正向提示词"},
        "negative_prompt": {"type": "string", "default": "", "description": "反向提示词"},
        "size": {"type": "string", "default": "1024x1024",
                 "description": "输出尺寸 WxH（8 的倍数；i2i/edit 模式忽略，用源图尺寸）"},
        "num_inference_steps": {"type": "integer", "description": "采样步数（缺省用 variant default_steps）"},
        "guidance_scale": {"type": "number", "description": "CFG 引导系数（缺省用 variant default_guidance）"},
        "seed": {"type": "integer", "description": "随机种子（缺省随机）"},
        "use_pe": {"type": "boolean", "default": False,
                   "description": "是否启用 Prompt Enhancer（variant 需 supports_pe）"},
        "denoise": {"type": "number", "default": 0.6, "minimum": 0.0, "maximum": 1.0,
                    "description": "去噪强度（仅 i2i 模式；edit 由蒙版控制，t2i 固定 1.0）"},
    },
    "required": ["prompt"],
}


# ===== manifest 中 variant 元数据 =====

class _ManifestVariant(BaseModel):
    """manifest.yaml 中单个 variant 的元数据（内部用）。"""

    name: str
    alias: Optional[str] = None
    base_model: str = ""
    precision: str = ""
    loader: str = ""
    vram_required: str = "8GB"
    path: str = ""
    default_steps: int = 50
    default_guidance: float = 5.0
    supports_pe: bool = True
    model_file_ids: list[str] = []


# ===== ErnieImageEngine =====

class ErnieImageEngine(CustomEngine):
    """ERNIE-Image 引擎（v0.3.0 ComfyUI 子进程模式）。

    继承 ``CustomEngine``（integrated 范式），通过子进程方式启动 ComfyUI，
    把推理请求转换为 ComfyUI 工作流 JSON 提交执行，订阅 WebSocket 事件
    转换为 AIReliable SSE 帧推送。

    生命周期：
        1. ``__init__(config)``          — 加载 manifest + 扫描 variants / loras
        2. ``_load_impl(variant)``       — 启动 ComfyUI 子进程（首次加载时）
        3. ``_infer_impl(request, stream)`` — 提交工作流 + 订阅 WS + 下载图片
        4. ``_unload_impl()``            — 释放 ComfyUI 显存（不停止子进程）或停止子进程
        5. 引擎关闭时                    — 停止 ComfyUI 子进程

    工作流模板：
        位于 ``workflow_templates/`` 目录，JSON 格式（ComfyUI API 格式）。
        支持的模板：
            - ``ernie_turbo_t2i.json``  — Turbo 文生图（8 步快速）
            - ``ernie_base_t2i.json``  — Base 文生图（50 步高质量）
            - ``ernie_turbo_pe.json``  — Turbo + Prompt Enhancer（8 步 + PE 增强）
        模板含占位符（如 ``__PROMPT__``），运行时由 ``_render_workflow`` 替换为实际值。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )

        self.config: EngineConfig = config
        self.output_root = config.output_root

        # ComfyUI 子进程管理器（lazy 启动，在 _load_impl 中调 start()）
        self._process_manager: ComfyuiProcessManager = ComfyuiProcessManager(config)
        # ComfyUI HTTP + WS 客户端（lazy 创建，每次推理时根据当前端口创建）
        self._client: Optional[ComfyuiClient] = None

        # 平台显存管理器（用于 L3 上报 / L4 LRU 响应 / L5 OOM 协调）
        self._vram_manager: PlatformVramManager = PlatformVramManager(
            core_url=config.core_url,
            engine_id=config.engine_id,
        )

        # 任务中心上报（v0.2.16 统一任务管理；gateway tasks.db 为唯一真相源）
        self.task_reporting = ErnieTaskReporting(config.engine_id)

        # manifest 数据
        self._manifest_variants: dict[str, _ManifestVariant] = self._load_manifest_variants()
        self.default_variant = self._read_default_variant_from_manifest(
            Path(__file__).resolve().parent / "manifest.yaml"
        )

        # variant 可用性缓存（扫描 unet_dir 判断 GGUF 是否存在）
        self._available_variants: set[str] = self._scan_available_variants()

        # LoRA 扫描结果（list of filename）
        self._available_loras: list[str] = self._scan_loras()
        # 已加载的 LoRA（name -> weight，运行时通过工作流模板注入 LoraLoader 节点）
        self._loaded_loras: dict[str, float] = {}

        # ComfyUI 加载状态
        self._loaded_variant: Optional[str] = None
        self._comfyui_started_at: Optional[float] = None
        self._last_infer_at: Optional[float] = None
        # 最近一次推理的显存占用（bytes，来自 /system_stats）
        self._last_vram_used_bytes: Optional[int] = None
        # 最近一次使用时间（load/infer 时更新，status 时上报）
        self.last_used_at: Optional[float] = None

        logger.info(
            "ERNIE-Image 引擎初始化完成（v0.3.0 ComfyUI 子进程模式）: "
            "comfyui_dir=%s, variants=%d, available=%d, loras=%d, default_variant=%s",
            config.comfyui_dir,
            len(self._manifest_variants),
            len(self._available_variants),
            len(self._available_loras),
            self.default_variant,
        )

    # ============ manifest 加载 + variant 扫描 ============

    def _load_manifest_variants(self) -> dict[str, _ManifestVariant]:
        """从 manifest.yaml 读取 variants 元数据。

        Returns:
            ``dict[variant_name -> _ManifestVariant]``
        """
        manifest_path = Path(__file__).resolve().parent / "manifest.yaml"
        if not manifest_path.exists():
            logger.warning("manifest.yaml 不存在: %s", manifest_path)
            return {}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
        except Exception as e:
            logger.warning("读取 manifest.yaml 失败: %s", e)
            return {}

        variants: dict[str, _ManifestVariant] = {}
        for v in (manifest.get("models") or {}).get("variants") or []:
            try:
                mv = _ManifestVariant(**v)
                variants[mv.name] = mv
            except Exception as e:
                logger.warning("解析 variant 失败（跳过）: %s, error=%s", v.get("name"), e)

        logger.info(
            "从 manifest 加载 %d 个 variant: %s",
            len(variants), list(variants.keys()),
        )
        return variants

    def _scan_available_variants(self) -> set[str]:
        """扫描 unet_dir 判断哪些 variant 的 GGUF 主权重已下载。

        判断逻辑：variant.path 字段（如 ``image/unet/ernie-image-Q4_K_M.gguf``）
        相对 models_root 拼接后文件存在 → available。

        Returns:
            ``set[variant_name]``
        """
        available: set[str] = set()
        for name, mv in self._manifest_variants.items():
            if not mv.path:
                continue
            # path 是相对 models_root 的路径（如 image/unet/ernie-image-Q4_K_M.gguf）
            gguf_path = self.config.models_root / mv.path
            if gguf_path.exists() and gguf_path.is_file():
                available.add(name)
                logger.debug("variant 可用: %s (path=%s)", name, gguf_path)
            else:
                logger.debug("variant 不可用（文件不存在）: %s (path=%s)", name, gguf_path)

        logger.info(
            "扫描 unet_dir 完毕: %d/%d 个 variant 可用",
            len(available), len(self._manifest_variants),
        )
        return available

    def _scan_loras(self) -> list[str]:
        """扫描 lora_dir 下的 LoRA 文件名列表。

        Returns:
            ``list[filename]``（如 ``["style.safetensors", "anime.safetensors"]``）
        """
        loras: list[str] = []
        lora_dir = self.config.lora_dir
        if not lora_dir.exists():
            logger.info("LoRA 目录不存在: %s", lora_dir)
            return loras

        supported_exts = {".safetensors", ".pt", ".bin", ".ckpt"}
        try:
            for f in sorted(lora_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in supported_exts:
                    loras.append(f.name)
        except Exception as e:
            logger.warning("扫描 LoRA 目录失败: %s", e)

        logger.info("扫描 LoRA 目录完毕: %d 个 LoRA 可用", len(loras))
        return loras

    def _refresh_scan(self) -> None:
        """刷新 variants + loras 扫描结果（供 /capabilities 调用）。"""
        self._available_variants = self._scan_available_variants()
        self._available_loras = self._scan_loras()

    # ============ 8 端点实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities。

        每次调用刷新扫描结果，确保新下载的 GGUF / LoRA 能被识别。
        """
        self._refresh_scan()

        variants: list[VariantInfo] = []
        for name, mv in self._manifest_variants.items():
            gguf_filename = Path(mv.path).name if mv.path else ""
            gguf_path = self.config.models_root / mv.path if mv.path else None
            file_size = gguf_path.stat().st_size if gguf_path and gguf_path.exists() else None

            variants.append(ErnieVariantInfo(
                name=name,
                base_model=mv.base_model,
                precision=mv.precision,
                loader=mv.loader,
                vram_required=mv.vram_required,
                default_steps=mv.default_steps,
                default_guidance=mv.default_guidance,
                supports_pe=mv.supports_pe,
                available=name in self._available_variants,
                path=gguf_filename,
                file_size=file_size,
            ))

        return ErnieCapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=list(self._available_loras),
            supported_sizes=SUPPORTED_SIZE_STRINGS,
            default_variant=self.default_variant,
            # v0.4.0 多模式：支持的功能模式 + 模板注册表（含存在性校验）
            supported_modes=list(self.SUPPORTED_MODES),
            workflow_templates=self._build_template_registry(),
            # v0.4.1+ MCP 化：推理输入 JSON Schema（手写，见模块级 _INPUT_SCHEMA）
            input_schema=_INPUT_SCHEMA,
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load — 启动 ComfyUI 子进程（若未启动）+ 显存上报 L3。

        ComfyUI 子进程启动后不会预加载模型（模型在首次推理时由 ComfyUI 自动加载），
        此处仅确保子进程已启动 + 健康检查通过。

        Args:
            variant: 要加载的 variant 名（用于记录 + 上报，不实际加载权重）。
                     None 时用 default_variant。

        Returns:
            LoadResponse: 含 variant / state / vram_used / load_time
        """
        # 确定 variant
        if variant is None:
            variant = self.default_variant
        if variant is None and self._manifest_variants:
            variant = next(iter(self._manifest_variants))
        if variant is None:
            raise RuntimeError("无法确定 variant（manifest 未声明 variants？）")

        if variant not in self._manifest_variants:
            raise RuntimeError(
                f"未知 variant: {variant}（manifest 声明的 variants: "
                f"{list(self._manifest_variants.keys())}）"
            )

        if variant not in self._available_variants:
            mv = self._manifest_variants[variant]
            raise RuntimeError(
                f"variant 不可用（GGUF 主权重未下载）: {variant}\n"
                f"  期望路径: {self.config.models_root / mv.path}\n"
                f"  请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮下载"
            )

        # 启动 ComfyUI 子进程（若未启动）
        load_time: Optional[float] = None
        if not self._process_manager.is_running:
            logger.info("启动 ComfyUI 子进程（variant=%s）", variant)
            start_result = await self._process_manager.start(
                timeout=self.config.server_timeout
            )
            load_time = start_result.get("load_time", 0.0)
            self._comfyui_started_at = time.time()
            logger.info(
                "ComfyUI 子进程已启动: pid=%d, port=%d, 耗时 %.2fs",
                start_result.get("pid", 0),
                start_result.get("internal_port", 0),
                load_time,
            )
        else:
            logger.info(
                "ComfyUI 子进程已在运行（pid=%d, port=%d），跳过启动",
                self._process_manager.pid,
                self._process_manager.internal_port,
            )

        # 更新 client（端口可能变化）
        if self._process_manager.internal_url:
            self._client = ComfyuiClient(self._process_manager.internal_url)

        # 显存上报 L3（查询 ComfyUI /system_stats）
        vram_used_str = "0GB"
        try:
            vram_bytes = await self._process_manager.get_vram_usage_bytes()
            if vram_bytes is not None:
                self._last_vram_used_bytes = vram_bytes
                vram_used_str = f"{vram_bytes / (1024 ** 3):.2f}GB"
                # 上报到平台（L3）
                await self._vram_manager.report_loaded(
                    variant=variant,
                    vram_used_bytes=vram_bytes,
                )
                logger.info(
                    "[VRAM L3] 上报显存占用: variant=%s, vram=%s",
                    variant, vram_used_str,
                )
        except Exception as e:
            logger.warning("[VRAM L3] 上报显存占用失败: %s", e)

        self._loaded_variant = variant
        self.last_used_at = time.time()

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used=vram_used_str,
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload — 释放 ComfyUI 显存（L4 LRU 响应）。

        优先调 ComfyUI ``POST /free`` 释放显存（保留子进程，下次推理更快），
        释放失败时降级为停止子进程。
        """
        vram_freed = "0GB"

        if self._process_manager.is_running:
            # L4: 优先调 /free 释放显存（不停止子进程）
            try:
                vram_before = await self._process_manager.get_vram_usage_bytes()
                freed_ok = await self._process_manager.free_memory(
                    unload_models=True, free_memory=True
                )
                if freed_ok:
                    # 等待显存释放稳定
                    await asyncio.sleep(1.0)
                    vram_after = await self._process_manager.get_vram_usage_bytes()
                    if vram_before is not None and vram_after is not None:
                        freed_bytes = max(0, vram_before - vram_after)
                        vram_freed = f"{freed_bytes / (1024 ** 3):.2f}GB"
                    logger.info(
                        "[VRAM L4] ComfyUI /free 成功: vram_freed=%s", vram_freed
                    )
                else:
                    logger.warning("[VRAM L4] ComfyUI /free 失败，降级为停止子进程")
                    stop_result = await self._process_manager.stop()
                    vram_freed = stop_result.get("vram_freed", "unknown")
            except Exception as e:
                logger.warning("[VRAM L4] 释放显存异常，降级为停止子进程: %s", e)
                try:
                    stop_result = await self._process_manager.stop()
                    vram_freed = stop_result.get("vram_freed", "unknown")
                except Exception as stop_err:
                    logger.error("停止子进程失败: %s", stop_err)
        else:
            logger.info("ComfyUI 子进程未运行，无需释放显存")

        # 上报显存已释放（L3）
        try:
            await self._vram_manager.report_unloaded()
            logger.info("[VRAM L3] 上报显存已释放")
        except Exception as e:
            logger.warning("[VRAM L3] 上报显存释放失败: %s", e)

        self._loaded_variant = None
        self._last_vram_used_bytes = None
        self._client = None

        return UnloadResponse(
            state="idle",
            vram_freed=vram_freed,
        )

    async def _status_impl(self) -> StatusResponse:
        """GET /status — 返回引擎状态 + ComfyUI 子进程状态。"""
        # 查询当前显存占用（若子进程在运行）
        vram_used = "0GB"
        if self._process_manager.is_running:
            try:
                vram_bytes = await self._process_manager.get_vram_usage_bytes()
                if vram_bytes is not None:
                    self._last_vram_used_bytes = vram_bytes
                    vram_used = f"{vram_bytes / (1024 ** 3):.2f}GB"
            except Exception:
                pass

        # ComfyUI 子进程状态
        comfyui_status = {
            "running": self._process_manager.is_running,
            "pid": self._process_manager.pid,
            "internal_port": self._process_manager.internal_port,
            "internal_url": self._process_manager.internal_url,
            "started_at": self._comfyui_started_at,
            "vram_used_bytes": self._last_vram_used_bytes,
        }

        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self._loaded_variant,
            vram_used=vram_used,
            loaded_loras=list(self._loaded_loras.keys()),
            last_used_at=self.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "paradigm": "integrated",  # v0.3.0 范式声明
                "comfyui": comfyui_status,
                "available_variants": sorted(self._available_variants),
                "available_loras": list(self._available_loras),
            },
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load — 记录已加载的 LoRA（运行时通过工作流模板注入）。

        ComfyUI 模式下 LoRA 不会持久加载到显存，而是在每次推理时通过
        workflow_template 的 LoraLoader 节点动态注入。此处仅维护一个列表，
        ``_infer_impl`` 在渲染工作流时会把这些 LoRA 注入到工作流中。
        """
        if name not in self._available_loras:
            raise FileNotFoundError(
                f"LoRA 不存在: {name}（可用: {self._available_loras}）"
            )

        self._loaded_loras[name] = weight
        logger.info("LoRA 已记录: name=%s, weight=%.2f", name, weight)

        return LoraLoadResponse(
            name=name,
            weight=weight,
            loaded_loras=list(self._loaded_loras.keys()),
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload — 从 loaded_loras 移除指定 LoRA。"""
        if name in self._loaded_loras:
            del self._loaded_loras[name]
            logger.info("LoRA 已移除: name=%s", name)
        else:
            logger.warning("LoRA 未在 loaded_loras 中: name=%s", name)

        return LoraUnloadResponse(
            name=name,
            loaded_loras=list(self._loaded_loras.keys()),
        )

    # ============ 推理实现（核心）============

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer — 提交工作流到 ComfyUI + 订阅 WS + 下载图片 + SSE 推送。

        流程：
            1. 确定 variant（request.variant > self._loaded_variant > default_variant）
            2. 确保 ComfyUI 子进程已启动（若未启动则调 _load_impl）
            3. 选择工作流模板（根据 variant 类型 + 是否启用 PE）
            4. 渲染工作流（替换占位符为实际参数）
            5. 提交工作流到 ComfyUI（POST /prompt）
            6. 订阅 WebSocket 事件，转换为 AIReliable SSE 帧推送
            7. 检测 OOM（L5）：execution_error + is_oom_error → 释放显存 + 重试
            8. 推理完成后下载图片并保存到 output_root
            9. 上报显存占用（L3）

        Args:
            request: 推理请求（含 variant / inputs / loras / task_id）
            stream: SSE 流（推送进度帧 + 最终结果帧）
        """
        task_id = request.task_id or f"ernie_{uuid.uuid4().hex[:16]}"
        start_time = time.time()

        # 1. 确定 variant
        variant = request.variant or self._loaded_variant or self.default_variant
        if variant is None:
            stream.put_threadsafe(make_error_frame(
                "未指定 variant 且无 default_variant（请检查 manifest.yaml）",
                error_type="variant_not_specified",
            ))
            return

        if variant not in self._manifest_variants:
            stream.put_threadsafe(make_error_frame(
                f"未知 variant: {variant}（可用: {list(self._manifest_variants.keys())}）",
                error_type="unknown_variant",
            ))
            return

        mv = self._manifest_variants[variant]

        # 2. 确保 ComfyUI 子进程已启动（若未启动则启动）
        if not self._process_manager.is_running:
            stream.put_threadsafe(make_frame(0.0, "starting_comfyui"))
            try:
                load_resp = await self._load_impl(variant)
                logger.info(
                    "推理触发 auto-load: variant=%s, vram=%s, load_time=%.2fs",
                    variant, load_resp.vram_used, load_resp.load_time or 0.0,
                )
            except Exception as e:
                stream.put_threadsafe(make_error_frame(
                    f"启动 ComfyUI 失败: {e}",
                    error_type="comfyui_start_failed",
                ))
                return

        if self._client is None:
            self._client = ComfyuiClient(self._process_manager.internal_url)

        # 3. 提取推理参数
        inputs = request.inputs if isinstance(request.inputs, dict) else {}
        prompt = str(inputs.get("prompt", ""))
        negative_prompt = str(inputs.get("negative_prompt", "") or "")
        size = str(inputs.get("size", "") or "1024x1024")
        try:
            width, height = size_to_wh(size)
        except ValueError as e:
            stream.put_threadsafe(make_error_frame(
                f"无效的 size 参数: {e}",
                error_type="invalid_size",
            ))
            return

        # 确保 width / height 是 8 的倍数（ComfyUI 要求）
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)

        steps = int(inputs.get("num_inference_steps", 0) or mv.default_steps)
        cfg = float(inputs.get("guidance_scale", 0.0) or mv.default_guidance)
        seed_input = inputs.get("seed")
        seed = int(seed_input) if seed_input not in (None, 0, "") else random.randint(0, 2**32 - 1)
        use_pe = bool(inputs.get("use_pe", False)) and mv.supports_pe

        # v0.4.0 多模式：判定 mode（t2i / i2i / edit）
        mode, mode_error = self._determine_mode(request)
        if mode_error:
            stream.put_threadsafe(make_error_frame(
                mode_error, error_type="invalid_workflow_mode",
            ))
            return

        # v0.4.0 多模式：暂存输入图片（i2i / edit 模式，落盘到 ComfyUI input 目录）
        staged: dict[str, Any] = {
            "source_filename": None, "mask_filename": None,
            "source_width": 0, "source_height": 0, "files": [],
        }
        if mode in ("i2i", "edit"):
            stream.put_threadsafe(make_frame(0.0, "staging_images"))
            try:
                staged = await self._stage_input_images(request, task_id, mode)
            except Exception as e:
                stream.put_threadsafe(make_error_frame(
                    f"输入图片处理失败: {e}",
                    error_type="stage_image_failed",
                ))
                return

        # v0.4.0 多模式：i2i / edit 用源图实际尺寸（忽略 inputs 的 width/height）。
        # 原因：Gateway 总是填默认 1024x1024，引擎无法区分"用户显式"与"默认值"，
        # 用源图尺寸是最确定的行为（见分析文档 3.5）。
        if mode in ("i2i", "edit") and staged.get("source_width"):
            width = int(staged["source_width"])
            height = int(staged["source_height"])

        # v0.4.0 多模式：denoise（i2i 可调默认 0.6；edit 固定 1.0 由 noise_mask 控制区域；
        # t2i 固定 1.0）
        denoise = 1.0
        if mode == "i2i":
            try:
                denoise = float(inputs.get("denoise", 0.6) or 0.6)
            except (TypeError, ValueError):
                denoise = 0.6
            denoise = min(1.0, max(0.0, denoise))

        # 任务中心上报：任务创建（v0.2.16 统一任务管理）
        # 上报分工（见 task_reporting.py 模块注释）：
        #   转发链路（request.task_id 非空）只报 created（gateway 维护进度/终态）；
        #   直连链路报全生命周期（created→running→progress→终态）。
        # 位置在 mode 判定 + staging 成功之后：task_type 精确、且任务一旦创建
        # 必有终态（succeeded/failed），不产生滞留 queued 的孤儿记录。
        gateway_forwarded = bool(request.task_id)
        await self.task_reporting.on_infer_created(
            task_id,
            gateway_forwarded=gateway_forwarded,
            mode=mode,
            request_summary={
                "prompt": prompt[:200],
                "mode": mode,
                "size": f"{width}x{height}",
                "seed": seed,
                "steps": steps,
                "loras": [s.name for s in (request.loras or [])]
                or list(self._loaded_loras.keys()),
            },
            variant=variant,
        )
        self.task_reporting.on_infer_running(
            task_id, gateway_forwarded=gateway_forwarded,
        )

        # 推理成败标记 + 错误详情（finally 中据此上报任务中心终态）
        infer_succeeded = False
        error_detail: Optional[dict[str, Any]] = None

        # v0.4.0：try/finally 保证暂存图片在任何路径（含错误提前 return）下都被清理
        try:
            # 推送开始帧
            stream.put_threadsafe(make_frame(
                0.0, "workflow_start",
                task_id=task_id,
                variant=variant,
                engine_id=self.engine_id,
                params={
                    "prompt": prompt[:200],  # 截断防止过长
                    "negative_prompt": negative_prompt[:200],
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "use_pe": use_pe,
                    "mode": mode,          # v0.4.0 多模式
                    "loras": list(self._loaded_loras.keys()),
                },
            ))

            # 4. 选择 + 渲染工作流模板
            try:
                workflow_data = self._render_workflow(
                    variant=variant,
                    mv=mv,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    use_pe=use_pe,
                    loras=request.loras or [],
                    mode=mode,                        # v0.4.0 多模式
                    source_filename=staged.get("source_filename"),
                    mask_filename=staged.get("mask_filename"),
                    denoise=denoise,
                )
            except Exception as e:
                logger.exception("渲染工作流模板失败")
                error_detail = {"type": "workflow_render_failed", "message": str(e)}
                stream.put_threadsafe(make_error_frame(
                    f"渲染工作流模板失败: {e}",
                    error_type="workflow_render_failed",
                ))
                return

            # 5. 提交工作流到 ComfyUI
            client_id = uuid.uuid4().hex
            try:
                submit_result = await self._client.submit_workflow(
                    prompt_data=workflow_data,
                    client_id=client_id,
                )
            except Exception as e:
                error_detail = {"type": "workflow_submit_failed", "message": str(e)}
                stream.put_threadsafe(make_error_frame(
                    f"提交工作流到 ComfyUI 失败: {e}",
                    error_type="workflow_submit_failed",
                ))
                return

            if "error" in submit_result:
                error_detail = {
                    "type": "workflow_rejected",
                    "message": str(submit_result.get("error", "unknown")),
                }
                stream.put_threadsafe(make_error_frame(
                    f"ComfyUI 拒绝工作流: {submit_result.get('error', 'unknown')}",
                    error_type="workflow_rejected",
                    node_errors=submit_result.get("node_errors", {}),
                ))
                return

            prompt_id = submit_result.get("prompt_id", "")
            logger.info(
                "工作流已提交: prompt_id=%s, variant=%s, mode=%s, steps=%d, size=%dx%d, seed=%d",
                prompt_id, variant, mode, steps, width, height, seed,
            )

            # 6. 订阅 WebSocket 事件 + 转换为 SSE 帧
            # 任务中心进度 hook（仅直连链路生效；WS 事件循环内 fire-and-forget）
            def _task_progress_hook(progress: float, stage: str) -> None:
                self.task_reporting.on_infer_progress(
                    task_id,
                    gateway_forwarded=gateway_forwarded,
                    progress=progress, stage=stage,
                )

            try:
                await self._subscribe_and_stream(
                    stream=stream,
                    client_id=client_id,
                    prompt_id=prompt_id,
                    task_id=task_id,
                    variant=variant,
                    total_steps=steps,
                    progress_hook=_task_progress_hook,
                )
            except Exception as e:
                logger.exception("订阅 ComfyUI WebSocket 异常")
                error_detail = {"type": "ws_subscribe_error", "message": str(e)}
                stream.put_threadsafe(make_error_frame(
                    f"订阅 ComfyUI WebSocket 异常: {e}",
                    error_type="ws_subscribe_error",
                ))
                return

            # 7. 推理完成：下载图片 + 推送最终帧
            elapsed = time.time() - start_time
            self._last_infer_at = time.time()
            self.last_used_at = time.time()

            # 查询 ComfyUI history 获取输出文件信息
            image_paths: list[str] = []
            try:
                history = await self._client.get_history(prompt_id)
                if history:
                    image_outputs = extract_image_outputs(history, prompt_id)
                    for img_info in image_outputs:
                        # 下载图片到本地
                        local_path = await self._download_and_save_image(
                            img_info, task_id=task_id,
                        )
                        if local_path:
                            image_paths.append(local_path)
            except Exception as e:
                logger.warning("下载图片失败: %s", e)

            # 上报显存占用（L3）
            try:
                vram_bytes = await self._process_manager.get_vram_usage_bytes()
                if vram_bytes is not None:
                    self._last_vram_used_bytes = vram_bytes
                    await self._vram_manager.report_loaded(
                        variant=variant,
                        vram_used_bytes=vram_bytes,
                    )
                    logger.info(
                        "[VRAM L3] 推理后上报显存: variant=%s, vram=%.2fGB",
                        variant, vram_bytes / (1024 ** 3),
                    )
            except Exception as e:
                logger.warning("[VRAM L3] 推理后上报显存失败: %s", e)

            # 推送最终帧（result 先构建：任务中心成功上报复用同一结构）
            result = {
                "task_id": task_id,
                "variant": variant,
                "mode": mode,        # v0.4.0 多模式
                "images": image_paths,
                "image_urls": [
                    f"/output/image/{Path(p).name}" for p in image_paths
                ],
            }
            # 任务中心成功上报（直连链路 await 先入库再返回最终帧；
            # 转发链路跳过——gateway 调度器消费 SSE 帧后自行 mark_succeeded）
            await self.task_reporting.on_infer_succeeded(
                task_id, gateway_forwarded=gateway_forwarded, result=result,
            )
            infer_succeeded = True
            stream.put_threadsafe(make_final_frame(
                result=result,
                timing={
                    "total_seconds": elapsed,
                    "variant": variant,
                    "steps": steps,
                    "size": f"{width}x{height}",
                    "seed": seed,
                },
            ))
            logger.info(
                "推理完成: task_id=%s, variant=%s, mode=%s, 耗时 %.2fs, 图片数=%d",
                task_id, variant, mode, elapsed, len(image_paths),
            )
        except Exception as e:  # noqa: BLE001
            # 推理执行体异常在引擎内消化（推 engine_error 帧后返回）：
            # 单次推理失败 ≠ 引擎故障，SDK 基类 finally 会把状态回 READY。
            # （若异常冒泡到 SDK infer()，状态机进 ERROR 并卡死，需重新 load 才能恢复。）
            logger.exception("推理执行失败：task_id=%s", task_id)
            error_detail = {"type": "engine_error", "message": str(e)}
            stream.put_threadsafe(make_error_frame(
                f"推理失败: {e}", "engine_error",
            ))
        finally:
            # 任务中心失败上报（仅直连链路；未成功即失败，覆盖所有错误分支
            # 与意外异常。fire-and-forget，不阻断清理）
            if not infer_succeeded:
                self.task_reporting.on_infer_failed(
                    task_id,
                    gateway_forwarded=gateway_forwarded,
                    error=error_detail or {
                        "type": "engine_error", "message": "推理未完成（未知原因）",
                    },
                )
            # v0.4.0：清理 input 目录暂存图片（无论成功 / 失败 / 提前 return）
            self._cleanup_staged_images(staged.get("files") or [])

    # ============ 工作流模板渲染 ============

    def _render_workflow(
        self,
        variant: str,
        mv: _ManifestVariant,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        use_pe: bool,
        loras: list,
        mode: str = "t2i",
        source_filename: Optional[str] = None,
        mask_filename: Optional[str] = None,
        denoise: float = 1.0,
    ) -> dict:
        """渲染工作流模板（替换占位符为实际参数）。

        v0.4.0 模板选择三维化（variant × mode × PE，见分析文档 5.2）：

            ============  ==============  =================  =====================
            mode          base            turbo              turbo + PE
            ============  ==============  =================  =====================
            t2i           ernie_base_t2i  ernie_turbo_t2i    ernie_turbo_pe
            i2i           ernie_base_i2i  ernie_turbo_i2i    ernie_turbo_i2i_pe
            edit          ernie_base_edit ernie_turbo_edit   ernie_turbo_edit_pe
            ============  ==============  =================  =====================

            （``ernie_turbo_pe.json`` 为 v0.3.x 历史名 = t2i + PE，保留不重命名）

        模板中的占位符（``__PROMPT__`` / ``__WIDTH__`` 等）会被替换为实际值。
        替换是递归的：遍历 dict / list，遇到字符串占位符就替换。

        v0.4.0 新增占位符（i2i / edit 模板用，见分析文档 4.7）：
            - ``__SOURCE_IMAGE__``: 源图文件名（LoadImage 节点 image 输入）
            - ``__MASK_IMAGE__``: mask 文件名（LoadImageMask 节点 image 输入）
            - ``__DENOISE__``: KSampler denoise（i2i 默认 0.6 可调；edit 固定 1.0）

        Args:
            variant: variant 名
            mv: manifest variant 元数据
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图片宽度
            height: 图片高度
            steps: 推理步数
            cfg: 引导系数
            seed: 随机种子
            use_pe: 是否启用 Prompt Enhancer
            loras: LoRA 列表（来自 InferRequest.loras + self._loaded_loras）
            mode: 功能模式（t2i / i2i / edit）
            source_filename: 暂存源图文件名（i2i / edit，``_stage_input_images`` 产出）
            mask_filename: 暂存 mask 文件名（edit）
            denoise: KSampler denoise 值（i2i 可调；edit/t2i 固定 1.0）

        Returns:
            dict: ComfyUI API 格式的工作流 JSON
        """
        # 选择模板（三维：variant × mode × PE）
        template_name = self._select_template(mv, use_pe, mode)
        template_path = self.config.workflow_templates_dir / template_name

        if not template_path.exists():
            raise FileNotFoundError(
                f"工作流模板不存在: {template_path} "
                f"（请检查 workflow_templates/ 目录）"
            )

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                workflow_data = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"加载工作流模板失败: {template_path}, error={e}"
            ) from e

        # 查找各组件文件名（从 manifest variant 推导）
        gguf_filename = Path(mv.path).name if mv.path else ""
        vae_filename = self._find_model_filename("ernie-vae-safetensors")
        text_encoder_filename = self._find_model_filename("ernie-text-encoder-safetensors")
        pe_filename = self._find_model_filename("ernie-pe-safetensors")

        # 合并 LoRA（request.loras + self._loaded_loras）
        merged_loras: dict[str, float] = dict(self._loaded_loras)
        for lora in loras:
            if hasattr(lora, "name"):
                merged_loras[lora.name] = lora.weight
            elif isinstance(lora, dict):
                name = lora.get("name")
                if name:
                    merged_loras[name] = float(lora.get("weight", 1.0))

        # 构建 PE 系统提示词（仅 use_pe=True 时使用）
        pe_prompt = self._build_pe_prompt(prompt, width, height) if use_pe else ""

        # 占位符替换映射
        placeholders: dict[str, Any] = {
            "__PROMPT__": prompt,
            "__NEGATIVE_PROMPT__": negative_prompt,
            "__WIDTH__": width,
            "__HEIGHT__": height,
            "__STEPS__": steps,
            "__CFG__": cfg,
            "__SEED__": seed,
            "__GGUF_FILENAME__": gguf_filename,
            "__VAE_FILENAME__": vae_filename,
            "__TEXT_ENCODER_FILENAME__": text_encoder_filename,
            "__PE_FILENAME__": pe_filename,
            "__PE_PROMPT__": pe_prompt,
            # v0.4.0 多模式新增（i2i / edit 模板，命名与分析文档 4.7 一致）
            "__SOURCE_IMAGE__": source_filename or "",
            "__MASK_IMAGE__": mask_filename or "",
            "__DENOISE__": denoise,
        }

        # 递归替换占位符
        workflow_data = self._replace_placeholders(workflow_data, placeholders)

        # 注入 LoRA 节点（若有 LoRA 加载）
        if merged_loras:
            workflow_data = self._inject_lora_nodes(workflow_data, merged_loras)

        # 删除 _meta 字段（ComfyUI API 不需要）
        workflow_data.pop("_meta", None)

        logger.info(
            "工作流模板渲染完成: template=%s, variant=%s, nodes=%d, loras=%d",
            template_name, variant, len(workflow_data), len(merged_loras),
        )
        return workflow_data

    def _select_template(self, mv: _ManifestVariant, use_pe: bool, mode: str = "t2i") -> str:
        """根据 variant 类型 × 功能模式 × 是否启用 PE 选择工作流模板。

        v0.4.0 三维模板矩阵（见分析文档 5.2）：

            ============  ==============  =================  =====================
            mode          base            turbo              turbo + PE
            ============  ==============  =================  =====================
            t2i           ernie_base_t2i  ernie_turbo_t2i    ernie_turbo_pe
            i2i           ernie_base_i2i  ernie_turbo_i2i    ernie_turbo_i2i_pe
            edit          ernie_base_edit ernie_turbo_edit   ernie_turbo_edit_pe
            ============  ==============  =================  =====================

        特殊约定：
            - t2i + PE 用历史名 ``ernie_turbo_pe.json``（v0.3.x 已存在，不重命名）
            - PE 仅 Turbo variant 支持（``mv.supports_pe`` 且 base_model 含 turbo）
            - mode 非法值兜底为 t2i（``_determine_mode`` 已做前置校验，此处防御）

        Args:
            mv: manifest variant 元数据（取 ``base_model`` / ``supports_pe``）
            use_pe: 是否启用 Prompt Enhancer
            mode: 功能模式（t2i / i2i / edit）

        Returns:
            str: 模板文件名（位于 ``workflow_templates/`` 目录）
        """
        if mode not in self.SUPPORTED_MODES:
            mode = "t2i"
        is_turbo = "turbo" in mv.base_model.lower()
        pe_effective = use_pe and mv.supports_pe and is_turbo

        if mode == "t2i":
            # t2i + PE 保留 v0.3.x 历史命名
            if pe_effective:
                return "ernie_turbo_pe.json"
            return "ernie_turbo_t2i.json" if is_turbo else "ernie_base_t2i.json"
        # i2i / edit：PE 版命名为 ernie_turbo_{mode}_pe.json
        if pe_effective:
            return f"ernie_turbo_{mode}_pe.json"
        return f"ernie_{'turbo' if is_turbo else 'base'}_{mode}.json"

    # v0.4.0 模板注册表（capabilities 声明用）
    # mode → 该 mode 的全部模板文件（与 _select_template 三维矩阵一致）
    _TEMPLATE_REGISTRY: dict[str, list[str]] = {
        "t2i": [
            "ernie_base_t2i.json",
            "ernie_turbo_t2i.json",
            "ernie_turbo_pe.json",      # 历史名 = t2i + PE
        ],
        "i2i": [
            "ernie_base_i2i.json",
            "ernie_turbo_i2i.json",
            "ernie_turbo_i2i_pe.json",
        ],
        "edit": [
            "ernie_base_edit.json",
            "ernie_turbo_edit.json",
            "ernie_turbo_edit_pe.json",
        ],
    }

    def _build_template_registry(self) -> dict[str, list[str]]:
        """构建模板注册表（mode → 实际存在的模板文件清单）。

        扫描 ``workflow_templates/`` 目录，只返回存在的文件；
        缺失的模板会被过滤掉并在日志中告警（便于排查模板未部署问题）。

        Returns:
            ``dict[mode -> list[模板文件名]]``
        """
        registry: dict[str, list[str]] = {}
        for mode, templates in self._TEMPLATE_REGISTRY.items():
            existing = [
                t for t in templates
                if (self.config.workflow_templates_dir / t).exists()
            ]
            missing = [t for t in templates if t not in existing]
            if missing:
                logger.warning(
                    "模板注册表：mode=%s 缺失 %d 个模板文件: %s（目录=%s）",
                    mode, len(missing), missing, self.config.workflow_templates_dir,
                )
            registry[mode] = existing
        return registry

    def _build_pe_prompt(self, prompt: str, width: int, height: int) -> str:
        """构建 PE（Prompt Enhancer）系统提示词模板。

        PE 模型（Ministral3）使用 ``ErnieTokenizer``（``comfy/text_encoders/ernie.py``），
        该 tokenizer 不应用任何内置模板，直接 tokenize 传入字符串。因此必须传入完整的
        ``<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]`` 模板。

        模板来源：ComfyUI 官方 blueprint ``Text to Image (Ernie Image Turbo).json``
        节点 95（TextGenerate）的 prompt 输入。

        Args:
            prompt: 用户原始正向提示词
            width: 目标图片宽度
            height: 目标图片高度

        Returns:
            str: 完整的 PE 系统提示词（含 SYSTEM_PROMPT + INST + JSON 参数）
        """
        # 系统 prompt（中文）：指导 PE 模型扩写简短描述为丰富的视觉描述
        system_prompt = (
            "你是一个专业的文生图 Prompt 增强助手。"
            "你将收到用户的简短图片描述及目标生成分辨率，"
            "请据此扩写为一段内容丰富、细节充分的视觉描述，"
            "以帮助文生图模型生成高质量的图片。"
            "仅输出增强后的描述，不要包含任何解释或前缀。"
        )
        # INST 部分：JSON 格式的用户输入（prompt / width / height）
        # 注意：JSON 中的引号需转义为 \"，避免破坏 ComfyUI 工作流 JSON 结构
        inst_payload = '{{"prompt": "{prompt}", "width": {width}, "height": {height}}}'.format(
            prompt=prompt,
            width=width,
            height=height,
        )
        # 组装完整模板（与官方 blueprint 一致）
        pe_prompt = (
            "<s>"
            f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
            f"[INST]{inst_payload}[/INST]"
        )
        logger.debug(
            "PE 系统提示词构建完成: prompt_len=%d, width=%d, height=%d, "
            "template_len=%d",
            len(prompt), width, height, len(pe_prompt),
        )
        return pe_prompt

    # ===== v0.4.0 多模式工作流：mode 判定 + 输入图片落盘 =====

    # 支持的功能模式（mode）
    SUPPORTED_MODES = ("t2i", "i2i", "edit")

    def _determine_mode(self, request: InferRequest) -> tuple[str, Optional[str]]:
        """判定功能模式（t2i / i2i / edit）。

        判定优先级（见分析文档 4.1）：
            1. ``request.workflow`` 显式指定（"t2i"/"i2i"/"edit"，不区分大小写）
            2. 自动推断（根据 ``request.images`` 的 role）：
               - images 为空 → t2i
               - 有 role=source 且无 role=mask → i2i
               - 有 role=mask → edit
            3. 兜底 → t2i

        Args:
            request: 推理请求

        Returns:
            tuple[str, Optional[str]]: (mode, 错误信息)。
            合法时错误信息为 None；workflow 传非法值时 mode 为空串 + 错误信息。
        """
        workflow = (request.workflow or "").strip().lower() if request.workflow else ""
        if workflow:
            if workflow in self.SUPPORTED_MODES:
                return workflow, None
            return "", (
                f"无效的 workflow 参数: {request.workflow!r}"
                f"（支持: {' / '.join(self.SUPPORTED_MODES)}）"
            )

        # 自动推断（根据 images 的 role）
        images = request.images or []
        if not images:
            return "t2i", None
        roles = set()
        for img in images:
            role = getattr(img, "role", None) or "source"
            roles.add(role.lower())
        if "mask" in roles:
            return "edit", None
        return "i2i", None

    async def _stage_input_images(
        self,
        request: InferRequest,
        task_id: str,
        mode: str,
    ) -> dict[str, Any]:
        """把 ``request.images`` 落盘到 ComfyUI input 目录（i2i / edit 模式）。

        处理流程（见分析文档 4.4）：
            1. 按 role 提取 source 图（role=source 或无 role）和 mask 图（role=mask）
            2. 来源解析：url → httpx 异步下载（60s 超时）；
               b64_json → base64 解码；path → 直接读文件
            3. source 图：PIL 打开 → RGB → 16 对齐 resize → 存 PNG
               （16 对齐与 Flux2 latent 压缩比一致，保证 VAEEncode 尺寸确定）
            4. mask 图：归一化为灰度 PNG（白=重绘区）→ resize 到源图尺寸
               （RGBA 输入取 alpha 反转，灰度/RGB 输入直接取亮度）
            5. 命名：``{task_id}_source.png`` / ``{task_id}_mask.png``（task_id 唯一）

        Args:
            request: 推理请求（含 images）
            task_id: 任务 ID（用于生成唯一文件名）
            mode: 功能模式（i2i / edit；t2i 不落盘直接返回空）

        Returns:
            dict: ``{
                "source_filename": str | None,   # LoadImage 用文件名
                "mask_filename": str | None,     # LoadImageMask 用文件名
                "source_width": int,             # 源图宽（16 对齐后）
                "source_height": int,            # 源图高（16 对齐后）
                "files": list[Path],             # 已落盘文件（推理结束后清理）
            }``

        Raises:
            RuntimeError: edit 模式缺 mask / 缺 source / 图片下载解码失败
        """
        result: dict[str, Any] = {
            "source_filename": None,
            "mask_filename": None,
            "source_width": 0,
            "source_height": 0,
            "files": [],
        }
        if mode == "t2i":
            return result

        images = request.images or []
        source_img = None
        mask_img = None
        for img in images:
            role = (getattr(img, "role", None) or "source").lower()
            if role == "mask" and mask_img is None:
                mask_img = img
            elif role in ("source", "reference", "control") and source_img is None:
                source_img = img
        # 无 role 的图片兜底为 source
        if source_img is None and images:
            source_img = images[0]

        if source_img is None:
            raise RuntimeError(
                f"{mode} 模式需要至少一张源图（images 中 role=source）"
            )
        if mode == "edit" and mask_img is None:
            raise RuntimeError(
                "edit 模式需要 mask 图（images 中 role=mask），"
                "或改用 workflow='i2i' 做整图重绘"
            )

        input_dir = self.config.input_dir
        input_dir.mkdir(parents=True, exist_ok=True)

        # --- source 图落盘（16 对齐 resize）---
        source_bytes = await self._read_image_bytes(source_img, "source")
        source_filename = f"{task_id}_source.png"
        source_path = input_dir / source_filename
        try:
            from PIL import Image
            with Image.open(__import__("io").BytesIO(source_bytes)) as im:
                im = im.convert("RGB")
                w, h = im.size
                aligned_w = max(16, (w // 16) * 16)
                aligned_h = max(16, (h // 16) * 16)
                if (aligned_w, aligned_h) != (w, h):
                    im = im.resize((aligned_w, aligned_h))
                    logger.info(
                        "源图尺寸 16 对齐 resize: %dx%d → %dx%d", w, h, aligned_w, aligned_h,
                    )
                im.save(source_path, format="PNG")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"源图处理失败: {e}") from e
        result["source_filename"] = source_filename
        result["source_width"] = aligned_w
        result["source_height"] = aligned_h
        result["files"].append(source_path)

        # --- mask 图落盘（归一化：白=重绘区）---
        if mode == "edit" and mask_img is not None:
            mask_bytes = await self._read_image_bytes(mask_img, "mask")
            mask_filename = f"{task_id}_mask.png"
            mask_path = input_dir / mask_filename
            try:
                from PIL import Image
                with Image.open(__import__("io").BytesIO(mask_bytes)) as im:
                    mask = self._normalize_mask_image(im)
                    # resize 到源图尺寸（16 对齐后），消除尺寸不一致
                    mask = mask.resize(
                        (result["source_width"], result["source_height"]),
                    )
                    mask.save(mask_path, format="PNG")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"mask 图处理失败: {e}") from e
            result["mask_filename"] = mask_filename
            result["files"].append(mask_path)

        logger.info(
            "输入图片落盘完成: mode=%s, source=%s (%dx%d), mask=%s, dir=%s",
            mode, source_filename, result["source_width"], result["source_height"],
            result["mask_filename"], input_dir,
        )
        return result

    async def _read_image_bytes(self, img: Any, label: str) -> bytes:
        """解析 ImageInput 的三种来源（url / b64_json / path）为原始字节。

        Args:
            img: ImageInput 对象（或含 url/b64_json/path 字段的 dict）
            label: 图片用途标签（source / mask，用于报错信息）

        Returns:
            bytes: 图片原始字节

        Raises:
            RuntimeError: 三种来源都不可用 / 下载超时 / 文件不存在
        """
        url = getattr(img, "url", None)
        b64 = getattr(img, "b64_json", None)
        path = getattr(img, "path", None)
        # dict 兼容（直接调引擎 API 的场景）
        if isinstance(img, dict):
            url = img.get("url")
            b64 = img.get("b64_json")
            path = img.get("path")

        if b64:
            import base64
            try:
                return base64.b64decode(b64)
            except Exception as e:
                raise RuntimeError(f"{label} 图 b64_json 解码失败: {e}") from e
        if url:
            import httpx
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp.content
            except Exception as e:
                raise RuntimeError(f"{label} 图下载失败（url={url[:200]}）: {e}") from e
        if path:
            p = Path(path)
            if not p.exists():
                raise RuntimeError(f"{label} 图文件不存在: {p}")
            return p.read_bytes()
        raise RuntimeError(f"{label} 图缺少 url / b64_json / path 来源字段")

    @staticmethod
    def _normalize_mask_image(mask_im) -> "Any":
        """把任意形态的 mask 图归一化为灰度图（白=重绘区，即 mask=1）。

        支持两种入参形态（见分析文档 3.4）：
            - RGBA（alpha 表示编辑区，OpenAI 风格：透明=编辑区）→ 取 alpha 反转
            - 灰度 / RGB（白色=编辑区）→ 直接取亮度

        Args:
            mask_im: PIL Image 对象

        Returns:
            PIL Image（mode="L"）：白色（255）区域 = 重绘区
        """
        # RGBA / LA / P(带透明) → alpha 反转（透明处变白=重绘区）
        if mask_im.mode in ("RGBA", "LA") or (
            mask_im.mode == "P" and "transparency" in mask_im.info
        ):
            alpha = mask_im.getchannel("A")
            # alpha=0（透明）→ 白（255）；alpha=255（不透明）→ 黑（0）
            inverted = alpha.point(lambda v: 255 - v)
            return inverted.convert("L")
        # 灰度 / RGB：直接取亮度（白=重绘区）
        return mask_im.convert("L")

    @staticmethod
    def _cleanup_staged_images(files: list) -> None:
        """推理结束后清理 input 目录的暂存图片。

        Args:
            files: ``_stage_input_images`` 返回的 ``files`` 列表
        """
        for f in files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception as e:
                logger.debug("清理暂存图片失败: %s, %s", f, e)

    def _find_model_filename(self, model_file_id: str) -> str:
        """从 manifest.model_files 中查找指定 id 的 filename。

        Args:
            model_file_id: model_files 中的 id 字段（如 ``ernie-vae-safetensors``）

        Returns:
            str: 文件名（如 ``diffusion_pytorch_model.safetensors``）
        """
        manifest_path = Path(__file__).resolve().parent / "manifest.yaml"
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            for mf in (manifest.get("models") or {}).get("model_files") or []:
                if mf.get("id") == model_file_id:
                    return mf.get("filename", "")
        except Exception as e:
            logger.warning("查找 model_files 失败: %s", e)
        return ""

    def _replace_placeholders(self, obj: Any, placeholders: dict[str, Any]) -> Any:
        """递归替换占位符。

        遍历 dict / list，遇到字符串中的占位符就替换为实际值。
        非字符串类型（int / float / bool）原样返回。
        """
        if isinstance(obj, str):
            for placeholder, value in placeholders.items():
                if placeholder in obj:
                    # 若整串就是占位符，返回原类型值（保留 int/float）
                    if obj == placeholder:
                        return value
                    # 否则做字符串替换
                    obj = obj.replace(placeholder, str(value))
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_placeholders(v, placeholders) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_placeholders(item, placeholders) for item in obj]
        return obj

    def _inject_lora_nodes(self, workflow: dict, loras: dict[str, float]) -> dict:
        """注入 LoraLoader 节点到工作流中。

        在 UnetLoaderGGUF 和 KSampler 之间插入 LoraLoader 节点，
        修改 KSampler 的 model 输入为最后一个 LoraLoader 的输出。

        Args:
            workflow: 工作流 dict（已被占位符替换）
            loras: ``{filename: weight}``

        Returns:
            dict: 注入 LoraLoader 后的工作流
        """
        # 找到 UnetLoaderGGUF 节点（输出 model）和 KSampler 节点（输入 model）
        unet_node_id: Optional[str] = None
        sampler_node_id: Optional[str] = None
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            if class_type == "UnetLoaderGGUF":
                unet_node_id = node_id
            elif class_type == "KSampler":
                sampler_node_id = node_id

        if not unet_node_id or not sampler_node_id:
            logger.warning("未找到 UnetLoaderGGUF 或 KSampler 节点，跳过 LoRA 注入")
            return workflow

        # 保存原 KSampler 的 model 输入（应该指向 UnetLoaderGGUF）
        sampler = workflow[sampler_node_id]
        original_model_input = sampler["inputs"].get("model", [unet_node_id, 0])

        # 链式注入 LoraLoader 节点
        prev_node_id = unet_node_id
        # 用大数字作为 LoRA 节点 ID（避免与现有 ID 冲突）
        lora_node_base = 100
        for idx, (lora_name, weight) in enumerate(loras.items()):
            lora_node_id = str(lora_node_base + idx)
            workflow[lora_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": [prev_node_id, 0],
                    "clip": ["2", 0],  # CLIP 来自 CLIPLoaderGGUF（节点 2）
                    "lora_name": lora_name,
                    "strength_model": weight,
                    "strength_clip": weight,
                },
            }
            prev_node_id = lora_node_id

        # 修改 KSampler 的 model 输入为最后一个 LoraLoader 的输出
        sampler["inputs"]["model"] = [prev_node_id, 0]

        return workflow

    # ============ WebSocket 事件订阅 + SSE 转换 ============

    async def _subscribe_and_stream(
        self,
        stream: SSEStream,
        client_id: str,
        prompt_id: str,
        task_id: str,
        variant: str,
        total_steps: int,
        progress_hook: Optional[Any] = None,
    ) -> None:
        """订阅 ComfyUI WebSocket 事件，转换为 AIReliable SSE 帧推送。

        事件处理：
            - ``progress`` → 推送进度帧（progress = value / max）
            - ``execution_start`` → 推送 stage=execution_start 帧
            - ``executing`` → 推送 stage=node_executing 帧（含 node_id）
            - ``executed`` → 推送 stage=node_executed 帧（含 node_id + output）
            - ``execution_success`` → 推送 stage=execution_success 帧（结束）
            - ``execution_error`` → 检测 OOM，若 OOM 触发 L5 协调；否则推送错误帧
            - ``execution_interrupted`` → 推送中断帧
            - ``timeout`` / ``ws_error`` → 推送错误帧

        L5 OOM 协调：
            检测到 OOM 错误时：
                1. 调 ComfyUI ``POST /free`` 释放显存
                2. 调 ``PlatformVramManager.request_vram`` 向平台请求更多显存
                3. 不自动重试（避免无限循环），推送 OOM 错误帧让调用方决策

        Args:
            stream: SSE 流
            client_id: WebSocket 客户端 ID（必须与 submit_workflow 一致）
            prompt_id: ComfyUI 工作流 ID
            task_id: 任务 ID（用于日志）
            variant: variant 名（用于日志）
            total_steps: 总推理步数（用于计算进度百分比）
        """
        if self._client is None:
            stream.put_threadsafe(make_error_frame(
                "ComfyUI 客户端未初始化",
                error_type="client_not_initialized",
            ))
            return

        async for event in self._client.subscribe_ws(
            client_id=client_id,
            prompt_id=prompt_id,
            timeout=self.config.server_timeout * 5,  # WS 单次事件超时
        ):
            event_type = event.get("type", "unknown")
            data = event.get("data", {}) or {}

            if event_type == EVENT_PROGRESS:
                # 推理进度（value / max）
                value = int(data.get("value", 0))
                max_value = int(data.get("max", total_steps or 1))
                progress = value / max_value if max_value > 0 else 0.0
                stream.put_threadsafe(make_frame(
                    progress, "sampling",
                    step=value,
                    total_steps=max_value,
                    task_id=task_id,
                ))
                # 任务中心进度（v0.2.16；仅直连链路，hook 内部判断）
                if progress_hook is not None:
                    try:
                        progress_hook(progress, "sampling")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("任务中心进度上报异常（不阻断推理）: %s", e)

            elif event_type == EVENT_EXECUTION_START:
                stream.put_threadsafe(make_frame(
                    0.0, "execution_start",
                    prompt_id=prompt_id,
                    task_id=task_id,
                ))

            elif event_type == "executing":
                node_id = data.get("node", "")
                stream.put_threadsafe(make_frame(
                    0.5, "node_executing",
                    node_id=node_id,
                    task_id=task_id,
                ))

            elif event_type == EVENT_EXECUTED:
                node_id = data.get("node", "")
                stream.put_threadsafe(make_frame(
                    0.95, "node_executed",
                    node_id=node_id,
                    task_id=task_id,
                ))

            elif event_type == EVENT_EXECUTION_SUCCESS:
                stream.put_threadsafe(make_frame(
                    1.0, "execution_success",
                    prompt_id=prompt_id,
                    task_id=task_id,
                ))
                logger.info(
                    "ComfyUI 工作流执行成功: prompt_id=%s, task_id=%s",
                    prompt_id, task_id,
                )
                return

            elif event_type == EVENT_EXECUTION_ERROR:
                # L5 OOM 检测
                if is_oom_error(event):
                    logger.error(
                        "[VRAM L5] ComfyUI OOM: prompt_id=%s, task_id=%s, event=%s",
                        prompt_id, task_id, data,
                    )
                    # 1. 释放 ComfyUI 显存
                    try:
                        await self._process_manager.free_memory(
                            unload_models=True, free_memory=True
                        )
                    except Exception as e:
                        logger.warning("[VRAM L5] 释放显存失败: %s", e)

                    # 2. 向平台请求更多显存（触发 LRU）
                    try:
                        # 估算需要的显存（保守用 variant 的 vram_required）
                        required_gb = self._parse_vram_required_gb(
                            self._manifest_variants.get(variant)
                        )
                        required_bytes = int(required_gb * 1024 ** 3)
                        await self._vram_manager.request_vram(required_bytes)
                    except Exception as e:
                        logger.warning("[VRAM L5] 请求显存失败: %s", e)

                    stream.put_threadsafe(make_error_frame(
                        "显存不足（OOM），已释放当前模型并请求平台 LRU 协调，"
                        "请重试推理或降低分辨率 / 步数",
                        error_type="oom",
                        prompt_id=prompt_id,
                        variant=variant,
                    ))
                else:
                    # 非 OOM 错误
                    exception_type = data.get("exception_type", "unknown")
                    exception_message = data.get("exception_message", "")
                    stream.put_threadsafe(make_error_frame(
                        f"ComfyUI 工作流执行失败: {exception_type}: {exception_message}",
                        error_type="execution_error",
                        exception_type=exception_type,
                        prompt_id=prompt_id,
                    ))
                return

            elif event_type == EVENT_EXECUTION_INTERRUPTED:
                stream.put_threadsafe(make_error_frame(
                    "推理被中断",
                    error_type="execution_interrupted",
                    prompt_id=prompt_id,
                ))
                return

            elif event_type in ("timeout", "ws_error"):
                error_msg = data.get("error", "未知错误")
                stream.put_threadsafe(make_error_frame(
                    f"ComfyUI WebSocket {event_type}: {error_msg}",
                    error_type=event_type,
                    prompt_id=prompt_id,
                ))
                return

            # 其他事件忽略（如 status 队列状态变化）

    def _parse_vram_required_gb(self, mv: Optional[_ManifestVariant]) -> float:
        """解析 variant 的 vram_required 字段为 GB 数。"""
        if mv is None or not mv.vram_required:
            return 8.0  # 默认 8GB
        # 解析 "8GB" / "16GB" 等格式
        s = mv.vram_required.upper().replace("GB", "").strip()
        try:
            return float(s)
        except ValueError:
            return 8.0

    # ============ 图片下载 + 保存 ============

    async def _download_and_save_image(
        self,
        img_info: dict,
        task_id: str,
    ) -> Optional[str]:
        """将 ComfyUI 输出图片落到 output/image/（v0.2.16 硬链接，物理一份）。

        v0.2.16 统一任务管理改造（T3.2，消除 G4 双份）：

        - ComfyUI 启动参数 ``--output-directory {output_root}/image``
          （comfyui_process._build_command）→ ComfyUI 生成的原文件已在
          ``output/image/{subfolder}/{filename}`` 下
        - 本方法改为对原文件创建**硬链接** ``output/image/{task_id}_{filename}``：
          同一 inode、两个文件名，磁盘物理一份。ComfyUI history 面板（原文件名）
          与任务中心 ``/output``（task_id 前缀名）双视图共存
        - ComfyUI 清理其文件名时，硬链接引用计数保证平台侧文件仍可访问
        - fallback：硬链接失败（跨盘符/权限/文件缺失）或 type != "output"
          （temp 预览图原文件在 ComfyUI cwd/temp，不在 output 目录树）→
          HTTP 下载复制（旧路径）+ warning，不阻断推理

        Args:
            img_info: ``{"filename": ..., "subfolder": ..., "type": ...}``
            task_id: 任务 ID（用于命名文件）

        Returns:
            str | None: 保存的本地路径；失败返回 None
        """
        if self._client is None:
            return None

        filename = img_info.get("filename", "")
        subfolder = img_info.get("subfolder", "")
        image_type = img_info.get("type", "output")

        if not filename:
            return None

        # 保存到 output_root/image/{task_id}_{filename}
        image_dir = self.output_root / "image"
        image_dir.mkdir(parents=True, exist_ok=True)

        # 用 task_id 前缀 + 原文件名
        local_filename = f"{task_id}_{filename}" if task_id else filename
        local_path = image_dir / local_filename

        # 1. 优先硬链接 ComfyUI 原文件（type=output 且文件已在本目录树内）
        if image_type == "output" and subfolder is not None:
            src_path = image_dir / subfolder / filename if subfolder else image_dir / filename
            if src_path.exists() and src_path != local_path:
                try:
                    os.link(src_path, local_path)
                    logger.info(
                        "图片已硬链接（物理一份，双视图共存）: %s -> %s",
                        src_path, local_path,
                    )
                    return str(local_path)
                except FileExistsError:
                    # 目标已存在（同 task_id 重试场景）：视为成功
                    return str(local_path)
                except OSError as e:
                    logger.warning(
                        "硬链接失败，回退 HTTP 下载复制（%s -> %s，原因：%s）",
                        src_path, local_path, e,
                    )

        # 2. fallback：HTTP 下载 /view 复制（temp 类型 / 硬链接不可用）
        image_bytes = await self._client.get_image(filename, subfolder, image_type)
        if image_bytes is None:
            logger.warning("下载图片失败: filename=%s, subfolder=%s", filename, subfolder)
            return None

        try:
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            logger.info(
                "图片已保存（复制回退）: %s (%d bytes)",
                local_path, len(image_bytes),
            )
            return str(local_path)
        except Exception as e:
            logger.warning("保存图片失败: %s", e)
            return None

    # ============ 引擎关闭清理 ============

    async def shutdown(self) -> None:
        """引擎关闭时清理资源（停止 ComfyUI 子进程 + 关闭 vram_manager）。"""
        logger.info("ERNIE-Image 引擎关闭中...")
        try:
            if self._process_manager.is_running:
                await self._process_manager.stop()
        except Exception as e:
            logger.warning("停止 ComfyUI 子进程失败: %s", e)

        try:
            await self._vram_manager.close()
        except Exception as e:
            logger.warning("关闭 PlatformVramManager 失败: %s", e)

        logger.info("ERNIE-Image 引擎已关闭")
