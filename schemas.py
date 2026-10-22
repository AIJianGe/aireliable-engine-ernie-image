"""ERNIE-Image 引擎专用数据模型（v0.3.0 ComfyUI 子进程模式）。

继承 SDK 通用 schemas，扩展 ERNIE-Image 特有字段。

v0.3.0 关键变更：
    - 推倒重来：删除 v0.2 范式 B 相关的 ComponentLoadStatus / ErnieStatusResponse 扩展
      （状态信息直接通过 StatusResponse.extra 字段返回）
    - 保留 ErnieVariantInfo（扩展 path / file_size 字段）
    - 保留 SUPPORTED_SIZE_STRINGS / size_to_wh / wh_to_size 工具函数
    - 修正导入路径：从 ``engines.base.schemas`` 改为 ``aireliable_engine``
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from aireliable_engine import (
    CapabilitiesResponse,
    InferRequest as BaseInferRequest,
    LoadResponse,
    LoraSpec,
    StatusResponse,
    VariantInfo,
)


# ===== Variant Info（ERNIE-Image 专用）=====

class ErnieVariantInfo(VariantInfo):
    """ERNIE-Image variant 信息（继承通用 + 扩展）。

    扩展字段：
        - path：GGUF 文件名（ComfyUI UnetLoaderGGUF 节点的 unet_name 参数）
        - file_size：文件字节数（用于前端展示）
    """

    path: str = ""                                       # GGUF 文件名
    file_size: Optional[int] = None                      # 文件字节数


# ===== Capabilities（v0.4.0 多模式扩展）=====

class ErnieCapabilitiesResponse(CapabilitiesResponse):
    """ERNIE-Image 能力声明（继承通用 + 扩展，v0.4.0 多模式工作流）。

    扩展字段：
        - supported_modes：支持的功能模式（t2i / i2i / edit）
        - workflow_templates：模板注册表（mode → 该 mode 可用的模板文件清单，
          含存在性校验结果，供前端与 Gateway 展示 / 排查模板缺失）
    """

    supported_modes: list[str] = Field(default_factory=lambda: ["t2i"])
    workflow_templates: dict[str, list[str]] = Field(default_factory=dict)


# ===== OpenAI 兼容出参（保留 v0.1 兼容）=====

class ImageData(BaseModel):
    """OpenAI 兼容的图片数据。"""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """OpenAI 兼容的图片生成响应。"""

    created: int
    data: list[ImageData] = Field(default_factory=list)


# ===== 工具函数 =====

def size_to_wh(size: str) -> tuple[int, int]:
    """把 "1024x1024" 转成 (1024, 1024)。"""
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception as e:
        raise ValueError(
            f"无效的 size 格式: {size}（应为 WxH，如 1024x1024）"
        ) from e


def wh_to_size(width: int, height: int) -> str:
    """把 (1024, 1024) 转成 "1024x1024"。"""
    return f"{width}x{height}"


# 支持的分辨率列表（前端 UI 下拉框用）
# ERNIE-Image 官方推荐 1K 分辨率，2K 需要更大显存
SUPPORTED_SIZE_STRINGS: list[str] = [wh_to_size(w, h) for w, h in [
    # 1K 标准（官方推荐）
    (1024, 1024), (848, 1264), (1264, 848),
    (768, 1376), (1376, 768), (896, 1200), (1200, 896),
    # 1K 常用比例
    (720, 1280), (1280, 720), (1024, 768), (768, 1024),
    # 2K 高清（约 4× 显存，请确认 8GB+ VRAM）
    (1280, 1280), (1920, 1080), (1080, 1920), (2048, 1024), (2048, 2048),
    # 小尺寸测试用（v0.3.0 新增，便于快速验证）
    (512, 512), (256, 256),
]]
