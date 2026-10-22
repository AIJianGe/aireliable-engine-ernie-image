"""任务中心对接（ERNIE-Image 引擎，v0.2.16+）。

与 ``qwen_image_5e8f3a21/task_reporting.py`` 同模式，设计依据
TASK_CENTER_ENGINE_HISTORY_UNIFY_ANALYSIS.md（方案 A）——gateway ``data/tasks.db``
为任务元数据**唯一真相源**。

上报分工规则（避免双写终态）：
    - **gateway 转发链路**（``request.task_id`` 非空）：gateway 已建任务并维护
      进度/终态（调度器消费 SSE 帧后 mark_succeeded/mark_failed），
      引擎只报 created（upsert 更新状态，不带 request/task_type——
      gateway UPDATE 分支不覆盖二者，此为双保险）
    - **直连链路**（``request.task_id`` 为空）：引擎报全生命周期
      created（INSERT，带 task_type + request 摘要）→ running → progress →
      succeeded/failed。succeeded 用 await（先入库再返回结果帧）

task_type 映射（与 gateway 任务中心筛选语义对齐）：
    t2i → image_generation；i2i / edit → image_editing

与 FLUX 引擎的差异（workflow 范式）：
    - 本引擎无自建 WebUI 历史视图（ComfyUI 前端自带 history 面板，
      走 ComfyUI 原生 /history API，不经本模块）→ 无 fetch_history_page
    - 图片物理唯一：ComfyUI output 目录原文件 + ``output/image/`` 硬链接
      （见 engine._download_and_save_image，T3.2），任务中心与 ComfyUI
      history 面板双视图共存，数据只有一份

容错原则（与 SDK PlatformTaskReporter 一致）：
    gateway 不可达时所有上报降级（warning 日志），**不阻断推理主流程**。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aireliable_engine import PlatformTaskReporter

logger = logging.getLogger(__name__)

# 模式 → 任务中心 task_type（与 gateway /v1/images/* 端点取值一致）
TASK_TYPE_BY_MODE: dict[str, str] = {
    "t2i": "image_generation",
    "i2i": "image_editing",
    "edit": "image_editing",
}


class ErnieTaskReporting:
    """ERNIE 引擎任务上报薄封装（惰性初始化，容错优先）。"""

    def __init__(self, engine_id: str):
        self._engine_id = engine_id
        # 自建实例（显式 engine_id，独立调试脱离平台时 env 未注入也能正确归属）
        self._reporter: Optional[PlatformTaskReporter] = None

    async def _ensure_reporter(self) -> Optional[PlatformTaskReporter]:
        """惰性初始化上报器（首次 awaited 上报时创建，捕获主 loop）。"""
        if self._reporter is None:
            try:
                self._reporter = PlatformTaskReporter(engine_id=self._engine_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("任务上报器初始化失败（不阻断推理）: %s", e)
        return self._reporter

    # ===== 推理生命周期埋点（engine.py _infer_impl 调用）=====

    async def on_infer_created(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        mode: str,
        request_summary: dict[str, Any],
        variant: str,
    ) -> None:
        """推理入口上报任务创建（mode 判定成功后调用，task_type 精确）。

        - 转发链路：仅 upsert 状态（gateway 侧 request/task_type 为权威，
          且 UPDATE 分支本就不覆盖，双保险）
        - 直连链路：INSERT 新任务，携带 task_type + request 摘要
          （摘要含 prompt/mode/size/seed/steps/loras）
        """
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            if gateway_forwarded:
                await reporter.report_created(task_id=task_id, variant=variant)
            else:
                await reporter.report_created(
                    task_id=task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=TASK_TYPE_BY_MODE.get(mode, "image_generation"),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("report_created 失败（不阻断推理）: %s", e)

    def on_infer_running(self, task_id: str, *, gateway_forwarded: bool) -> None:
        """推理开始上报 running（仅直连链路；转发链路由 gateway 调度器维护）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_running(task_id)

    def on_infer_progress(
        self, task_id: str, *, gateway_forwarded: bool, progress: float, stage: str,
    ) -> None:
        """推理进度上报（仅直连链路，fire-and-forget，WS 事件循环内调用安全）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_progress(task_id, progress=progress, stage=stage)

    async def on_infer_succeeded(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        result: dict[str, Any],
    ) -> None:
        """推理成功上报（仅直连链路；转发链路 gateway 消费 SSE 帧后自行 mark_succeeded）。

        使用 await（而非 fire）：先入库再返回结果帧（SDK 3s 超时容错）。
        result 含 images（本地路径）+ image_urls（/output/image/ 视图 URL），
        与 make_final_frame 的 result 结构一致。
        """
        if gateway_forwarded:
            return
        reporter = await self._ensure_reporter()
        if reporter is None:
            return
        try:
            await reporter.report_succeeded(task_id=task_id, result=result)
        except Exception as e:  # noqa: BLE001
            logger.warning("report_succeeded 失败（不阻断推理）: %s", e)

    def on_infer_failed(
        self,
        task_id: str,
        *,
        gateway_forwarded: bool,
        error: dict[str, Any],
    ) -> None:
        """推理失败上报（仅直连链路，fire-and-forget）。"""
        if gateway_forwarded or self._reporter is None:
            return
        self._reporter.fire_report_failed(task_id, error=error)
