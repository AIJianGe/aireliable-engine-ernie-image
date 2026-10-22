"""ComfyUI HTTP + WebSocket 客户端（v0.3.0）。

封装 ComfyUI 的 HTTP API 和 WebSocket 事件订阅，供 ``ErnieImageEngine._infer_impl``
调用。

HTTP API：
    - POST /prompt              提交工作流 JSON
    - GET  /history/{prompt_id}  查询工作流执行历史（含输出文件信息）
    - GET  /view                 下载图片（filename + subfolder + type）
    - GET  /system_stats         查询系统状态（含显存，L3 上报）
    - POST /free                 释放显存（L4 LRU 驱逐响应）

WebSocket 事件（``/ws?clientId=<id>``）：
    - progress           推理进度（``{"value": N, "max": M}``）
    - executing          节点开始执行（``{"node": "<id>", "prompt_id": "..."}``）
    - executed           节点执行完成（含输出文件信息）
    - execution_start    工作流开始执行
    - execution_success  工作流执行完成（成功）
    - execution_error    工作流执行失败（含异常信息，用于 L5 OOM 检测）
    - execution_interrupted  工作流被中断
    - status             队列状态变化

设计要点：
    - HTTP 用 ``httpx.AsyncClient``，超时 600s（支持长推理）
    - WebSocket 用 ``websockets`` 库（FastAPI 已依赖）
    - ``subscribe_ws`` 是异步迭代器，逐个产出解析后的事件 dict
    - 失败不抛异常，返回错误 dict 让调用方决策（便于 L5 OOM 协调）
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


# ===== ComfyUI 事件类型常量 =====

# WebSocket 事件类型（来自 ComfyUI server.py 的 send() 方法）
EVENT_PROGRESS = "progress"                    # 推理进度（step/value + max）
EVENT_EXECUTING = "executing"                 # 节点开始执行
EVENT_EXECUTED = "executed"                   # 节点执行完成（含输出文件）
EVENT_EXECUTION_START = "execution_start"     # 工作流开始执行
EVENT_EXECUTION_SUCCESS = "execution_success" # 工作流执行完成（成功）
EVENT_EXECUTION_ERROR = "execution_error"     # 工作流执行失败（L5 OOM 检测）
EVENT_EXECUTION_INTERRUPTED = "execution_interrupted"  # 工作流被中断
EVENT_STATUS = "status"                       # 队列状态变化

# 终态事件（执行结束，迭代器应退出）
TERMINAL_EVENTS = frozenset({
    EVENT_EXECUTION_SUCCESS,
    EVENT_EXECUTION_ERROR,
    EVENT_EXECUTION_INTERRUPTED,
})


# ===== ComfyuiClient =====

class ComfyuiClient:
    """ComfyUI HTTP + WebSocket 客户端。

    用法：
        ```python
        client = ComfyuiClient("http://127.0.0.1:8100")
        result = await client.submit_workflow(prompt_data, client_id)
        async for event in client.subscribe_ws(client_id, prompt_id=result["prompt_id"]):
            if event["type"] == EVENT_PROGRESS:
                print(f"进度: {event['data']['value']}/{event['data']['max']}")
            elif event["type"] == EVENT_EXECUTION_SUCCESS:
                break
        history = await client.get_history(result["prompt_id"])
        image_bytes = await client.get_image(filename, subfolder, "output")
        ```

    设计：
        - HTTP 方法无状态，每次创建新的 ``httpx.AsyncClient``
        - WebSocket 方法用 ``websockets`` 库，长连接
        - 失败返回错误 dict 而非抛异常，便于调用方决策
    """

    def __init__(self, base_url: str, timeout: float = 600.0):
        """
        Args:
            base_url: ComfyUI 基础 URL（如 ``http://127.0.0.1:8100``）
            timeout: HTTP 超时秒数（默认 600s，支持长推理）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ============ HTTP API ============

    async def submit_workflow(
        self,
        prompt_data: dict,
        client_id: Optional[str] = None,
    ) -> dict:
        """提交工作流 JSON 到 ComfyUI。

        Args:
            prompt_data: ComfyUI API 格式的工作流 JSON
                ````json
                {
                  "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "..."}},
                  "2": {"class_type": "KSampler", "inputs": {...}}
                }
                ````
            client_id: WebSocket 客户端 ID（用于接收事件）；None 则自动生成

        Returns:
            dict: ComfyUI 返回的响应
                - 成功：``{"prompt_id": "<uuid>", "number": <queue_number>, "node_errors": {}}``
                - 失败：``{"error": ..., "node_errors": ...}``

        Raises:
            RuntimeError: 提交失败（网络错误 / 4xx / 5xx）
        """
        if client_id is None:
            client_id = uuid.uuid4().hex

        url = f"{self.base_url}/prompt"
        payload = {
            "prompt": prompt_data,
            "client_id": client_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.post(url, json=payload)
                if r.status_code == 200:
                    return r.json()
                # 4xx/5xx 错误
                error_text = r.text[:500]
                logger.error(
                    "提交工作流失败: status=%d, url=%s, response=%s",
                    r.status_code, url, error_text,
                )
                raise RuntimeError(
                    f"ComfyUI 提交工作流失败（status={r.status_code}）: {error_text}"
                )
        except httpx.HTTPError as e:
            logger.error("提交工作流网络错误: %s", e)
            raise RuntimeError(f"ComfyUI 网络错误: {e}") from e

    async def get_history(self, prompt_id: str) -> Optional[dict]:
        """查询工作流执行历史。

        Args:
            prompt_id: ComfyUI 返回的 prompt_id

        Returns:
            dict | None: 历史记录 JSON；查询失败返回 None

        响应结构：
            ````json
            {
              "<prompt_id>": {
                "prompt": [...],
                "outputs": {
                  "<node_id>": {
                    "images": [
                      {"filename": "ERNIE_Image_00001_.png",
                       "subfolder": "",
                       "type": "output"}
                    ]
                  }
                },
                "status": {
                  "status_str": "success",
                  "completed": true,
                  "messages": [...]
                }
              }
            }
            ````
        """
        url = f"{self.base_url}/history/{prompt_id}"
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                logger.warning("GET /history/%s 返回 %d", prompt_id, r.status_code)
                return None
        except Exception as e:
            logger.warning("GET /history/%s 失败: %s", prompt_id, e)
            return None

    async def get_image(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output",
    ) -> Optional[bytes]:
        """下载 ComfyUI 生成的图片。

        Args:
            filename: 文件名（如 ``ERNIE_Image_00001_.png``）
            subfolder: 子目录（通常为空）
            image_type: 类型（``output`` / ``temp``）

        Returns:
            bytes | None: 图片字节数据；失败返回 None
        """
        url = f"{self.base_url}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    return r.content
                logger.warning(
                    "GET /view 返回 %d: filename=%s, subfolder=%s, type=%s",
                    r.status_code, filename, subfolder, image_type,
                )
                return None
        except Exception as e:
            logger.warning("GET /view 失败: %s", e)
            return None

    async def get_system_stats(self) -> Optional[dict]:
        """查询 ComfyUI 系统状态（含显存）。

        供 ``ComfyuiProcessManager.get_system_stats`` 调用（也可独立使用）。

        Returns:
            dict | None: 系统状态 JSON；失败返回 None
        """
        url = f"{self.base_url}/system_stats"
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                return None
        except Exception as e:
            logger.warning("GET /system_stats 失败: %s", e)
            return None

    async def free_memory(
        self,
        unload_models: bool = True,
        free_memory: bool = True,
    ) -> bool:
        """调 ``POST /free`` 释放 ComfyUI 显存（L4 LRU 驱逐响应）。

        Args:
            unload_models: 是否卸载已加载的模型
            free_memory: 是否释放显存

        Returns:
            bool: True 成功 / False 失败
        """
        url = f"{self.base_url}/free"
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.post(url, json=payload)
                if r.status_code == 200:
                    logger.info(
                        "[VRAM] ComfyUI /free 成功: unload_models=%s, free_memory=%s",
                        unload_models, free_memory,
                    )
                    return True
                logger.warning("[VRAM] ComfyUI /free 返回 %d", r.status_code)
                return False
        except Exception as e:
            logger.warning("[VRAM] ComfyUI /free 失败: %s", e)
            return False

    async def interrupt(self) -> bool:
        """调 ``POST /interrupt`` 中断当前正在执行的工作流。

        用于 L5 OOM 协调：检测到 OOM 时先中断当前推理，再释放显存重试。

        Returns:
            bool: True 成功 / False 失败
        """
        url = f"{self.base_url}/interrupt"
        try:
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                r = await client.post(url)
                return r.status_code == 200
        except Exception as e:
            logger.warning("POST /interrupt 失败: %s", e)
            return False

    # ============ WebSocket 事件订阅 ============

    async def subscribe_ws(
        self,
        client_id: str,
        prompt_id: Optional[str] = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[dict]:
        """订阅 ComfyUI WebSocket 事件流。

        连接 ``ws://<host>:<port>/ws?clientId=<client_id>``，逐个产出解析后的事件 dict。

        事件 dict 结构：
            ````python
            {
              "type": "progress",       # 事件类型
              "data": {"value": 5, "max": 50},  # 事件数据
              "prompt_id": "...",        # 关联的 prompt_id（从 data 中提取）
            }
            ````

        终态事件（``execution_success`` / ``execution_error`` /
        ``execution_interrupted``）产出后自动退出迭代器。

        Args:
            client_id: WebSocket 客户端 ID（必须与 ``submit_workflow`` 的一致）
            prompt_id: 关联的 prompt_id（仅用于日志，不发送给 ComfyUI）
            timeout: 单次事件等待超时秒数（默认 600s）

        Yields:
            dict: 事件 dict（含 ``type`` / ``data`` / ``prompt_id``）
        """
        import websockets
        from websockets.exceptions import WebSocketException

        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={client_id}"
        logger.info(
            "订阅 ComfyUI WebSocket: %s (prompt_id=%s)",
            ws_url, prompt_id or "<unknown>",
        )

        try:
            async with websockets.connect(ws_url, max_size=None) as ws:
                # 用 wait_for 实现单次事件超时
                import asyncio
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "ComfyUI WebSocket 事件等待超时（%ss），退出订阅",
                            timeout,
                        )
                        yield {
                            "type": "timeout",
                            "data": {"timeout": timeout},
                            "prompt_id": prompt_id,
                        }
                        return

                    # 解析事件 JSON
                    # ComfyUI WebSocket 二进制消息是 PreviewImage（JPEG/PNG 字节流），
                    # 格式：前 8 字节 [event_type(4B)][format(4B)] + 图片字节。
                    # AIReliable SSE 流不需要预览图（最终图片通过 /view 下载），
                    # 直接跳过二进制消息避免 utf-8 解码失败噪声日志。
                    if isinstance(raw, (bytes, bytearray)):
                        # 二进制预览图消息（BinaryEventTypes.PREVIEW_IMAGE=1 / PREVIEW_IMAGE = 1）
                        # 前 4 字节是事件类型，0x01 表示预览图
                        if len(raw) >= 4 and raw[3] == 0x01:
                            logger.debug(
                                "ComfyUI WS 二进制预览图消息已跳过（%d 字节）",
                                len(raw),
                            )
                        else:
                            logger.debug(
                                "ComfyUI WS 未知二进制消息已跳过（%d 字节, header=%r）",
                                len(raw), raw[:8] if len(raw) >= 8 else raw,
                            )
                        continue

                    try:
                        event = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning("ComfyUI WS 事件解析失败: %s, raw=%r", e, raw[:200])
                        continue

                    # 规范化事件结构
                    event_type = event.get("type", "unknown")
                    event_data = event.get("data", {})
                    # 从 data 中提取 prompt_id（ComfyUI 的事件 data 含 prompt_id）
                    event_prompt_id = event_data.get("prompt_id") if isinstance(event_data, dict) else None
                    normalized = {
                        "type": event_type,
                        "data": event_data,
                        "prompt_id": event_prompt_id or prompt_id,
                    }

                    # 调试日志（仅非 progress 事件，progress 太频繁）
                    if event_type != EVENT_PROGRESS:
                        logger.debug(
                            "ComfyUI WS 事件: type=%s, prompt_id=%s",
                            event_type, event_prompt_id or "<none>",
                        )

                    yield normalized

                    # 终态事件：退出迭代器
                    if event_type in TERMINAL_EVENTS:
                        logger.info(
                            "ComfyUI WS 终态事件: %s, prompt_id=%s",
                            event_type, event_prompt_id or "<none>",
                        )
                        return

        except WebSocketException as e:
            logger.error("ComfyUI WebSocket 连接异常: %s", e)
            yield {
                "type": "ws_error",
                "data": {"error": str(e)},
                "prompt_id": prompt_id,
            }
        except Exception as e:
            logger.exception("ComfyUI WebSocket 订阅异常")
            yield {
                "type": "ws_error",
                "data": {"error": str(e)},
                "prompt_id": prompt_id,
            }


# ===== 事件解析辅助函数 =====

def is_oom_error(event: dict) -> bool:
    """判断事件是否为 OOM（显存不足）错误。

    用于 L5 OOM 协调：检测到 OOM 时，先释放显存再重试。

    判断依据（满足任一）：
        - ``exception_type`` 含 ``OutOfMemory`` / ``CUDA`` / ``MemoryError``
        - ``exception_message`` 含 ``out of memory`` / ``cuda error`` / ``CUDA error``

    Args:
        event: WebSocket 事件 dict

    Returns:
        bool: True 表示是 OOM 错误
    """
    if event.get("type") != EVENT_EXECUTION_ERROR:
        return False

    data = event.get("data", {})
    if not isinstance(data, dict):
        return False

    exception_type = str(data.get("exception_type", "")).lower()
    exception_message = str(data.get("exception_message", "")).lower()

    oom_keywords = [
        "outofmemory",
        "out of memory",
        "cuda error",
        "memoryerror",
        "runtimeerror: cuda",
        "device-side assert",
    ]

    text = f"{exception_type} {exception_message}"
    return any(kw in text for kw in oom_keywords)


def extract_image_outputs(history: dict, prompt_id: str) -> list[dict]:
    """从 ComfyUI history 响应中提取图片输出列表。

    Args:
        history: ``get_history(prompt_id)`` 返回的 dict
        prompt_id: 要提取的 prompt_id

    Returns:
        list[dict]: 图片输出列表，每项含 ``filename`` / ``subfolder`` / ``type``
    """
    if not history or prompt_id not in history:
        return []

    prompt_history = history[prompt_id]
    outputs = prompt_history.get("outputs", {})
    if not isinstance(outputs, dict):
        return []

    images: list[dict] = []
    for _node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        node_images = node_output.get("images", [])
        if isinstance(node_images, list):
            for img in node_images:
                if isinstance(img, dict) and "filename" in img:
                    images.append({
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
    return images
