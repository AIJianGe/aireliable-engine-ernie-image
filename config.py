"""ERNIE-Image 引擎配置（v0.3.0 ComfyUI 子进程模式重构版）。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.3.0 关键变更：
    - 推倒重来：删除 WorkflowEngine 相关字段（workflows_dir / gguf_eager_dequant / dtype 等）
    - 新增 ComfyUI 子进程配置（comfyui_dir / python_exe / extra_model_paths_yaml / server_log_dir）
    - 新增 workflow_templates_dir（ComfyUI JSON 工作流模板目录）
    - 新增 core_url（读 AIRELIABLE_CORE_URL，平台显存 API 调用）
    - 新增 ComfyUI 启动参数（vram_mode / disable_smart_memory 等）
    - 日志路径读 AIRELIABLE_LOG_DIR（v0.2.15+ 规范）

环境变量优先级（高 → 低）：
    命令行参数 > 环境变量 > 默认值

关键环境变量（由平台 engine_manager._build_subprocess_env 注入）：
    AIRELIABLE_CORE_URL          平台 API 地址（http://127.0.0.1:8001）
    AIRELIABLE_MODELS_ROOT       models 根目录
    AIRELIABLE_MODEL_LAYOUT_BASE 当前引擎的 model_layout.base（image/unet/）
    AIRELIABLE_LORA_DIR          LoRA 目录
    AIRELIABLE_UI_DIR            UI 目录（未使用，保留兼容）
    AIRELIABLE_LOG_DIR           引擎日志目录（{project_root}/logs/engines）
    AIRELIABLE_ENGINE_ID         当前引擎 ID
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aireliable_engine import EngineConfig as BaseEngineConfig


# ===== 常量 =====

ENGINE_ID = "ernie_image_909469ed"
ENGINE_NAME = "ERNIE-Image (ComfyUI)"
CAPABILITY = "text-to-image"
ENGINE_TYPE = "image"

# 默认端口（与 config/registry.yaml 一致）
DEFAULT_PORT = 7001

# 模型目录布局（相对 MODELS_ROOT）
# v0.3.0：保持平台现有目录结构（v1.2 平台规范优先原则）
_DEFAULT_UNET_SUBDIR = "image/unet/"                       # GGUF 主权重
_DEFAULT_VAE_SUBDIR = "image/vae/ernie-image/"             # VAE
_DEFAULT_TEXT_ENCODER_SUBDIR = "shared/text_encoders/ernie-image/"  # Text Encoder
_DEFAULT_PE_SUBDIR = "image/pe/ernie-image/"               # Prompt Enhancer
_DEFAULT_TOKENIZER_SUBDIR = "shared/tokenizer/ernie-image/"  # Tokenizer
_DEFAULT_LORA_SUBDIR = "image/loras/ernie-image/"          # LoRA


def _resolve_subdir(env_key: str, default: str) -> str:
    """解析子目录（优先读环境变量，fallback 到默认值）。"""
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val.rstrip("/") + "/"
    return default


# 运行时常量（在模块加载时根据环境变量解析）
UNET_SUBDIR = _resolve_subdir("AIRELIABLE_MODEL_LAYOUT_BASE", _DEFAULT_UNET_SUBDIR)
VAE_SUBDIR = _resolve_subdir("AIRELIABLE_VAE_DIR", _DEFAULT_VAE_SUBDIR)
TEXT_ENCODER_SUBDIR = _resolve_subdir("AIRELIABLE_TEXT_ENCODER_DIR", _DEFAULT_TEXT_ENCODER_SUBDIR)
PE_SUBDIR = _resolve_subdir("AIRELIABLE_PE_DIR", _DEFAULT_PE_SUBDIR)
TOKENIZER_SUBDIR = _resolve_subdir("AIRELIABLE_TOKENIZER_DIR", _DEFAULT_TOKENIZER_SUBDIR)
LORA_SUBDIR = _resolve_subdir("AIRELIABLE_LORA_DIR", _DEFAULT_LORA_SUBDIR)

# ComfyUI 默认路径（相对项目根）
# v0.3.0：ComfyUI 仓库克隆到 envs/image-ernie-comfyui/ComfyUI/
_DEFAULT_COMFYUI_DIR = "envs/image-ernie-comfyui/ComfyUI"
_DEFAULT_COMFYUI_PYTHON = "envs/image-ernie-comfyui/.venv/Scripts/python.exe"

# 隐藏端口范围（ComfyUI 内部监听，仅 127.0.0.1）
DEFAULT_INTERNAL_PORT_START = 8100
DEFAULT_INTERNAL_PORT_END = 8200

# ComfyUI 子进程启动超时（秒）
# 实测 ComfyUI v0.32.0 首次启动约 65s（含 custom_nodes 加载 + 数据库迁移），
# 预留缓冲设为 180s（非首次启动约 20-30s）
DEFAULT_SERVER_TIMEOUT = 180.0

# 健康检查间隔（秒）
DEFAULT_HEALTH_CHECK_INTERVAL = 5.0

# ComfyUI VRAM 档位（v1.2 显存对接 L1）
# 取值：normalvram（默认）/ lowvram / novram / highvram / gpu_only / cpu
# 平台规范：用 normalvram，禁用 lowvram（与平台 LRU 冲突）
DEFAULT_VRAM_MODE = "normalvram"

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为：normalvram，ComfyUI 自动探测显存）
#   gpu     ：强制全 GPU（vram_mode=highvram，显存不足直接报错）
#   gpu+cpu ：GPU 优先，显存不足时模型自动溢出 CPU（vram_mode=lowvram，
#             ComfyUI 按需换入换出）
#   cpu     ：强制 CPU（vram_mode=cpu，--cpu 参数）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


@dataclass
class EngineConfig(BaseEngineConfig):
    """引擎运行配置（v0.3.0 ComfyUI 子进程模式）。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径（模型）=====
    unet_dir: Path = field(init=False)               # GGUF 主权重目录
    vae_dir: Path = field(init=False)
    text_encoder_dir: Path = field(init=False)
    pe_dir: Path = field(init=False)
    tokenizer_dir: Path = field(init=False)
    lora_dir: Path = field(init=False)

    # ===== 推导路径（日志 + ComfyUI 子进程）=====
    log_dir: Path = field(init=False)                # 引擎 Python 进程日志目录
    server_log_dir: Path = field(init=False)         # ComfyUI 子进程日志目录（覆写模式）
    # v0.4.0 多模式工作流：ComfyUI 输入图片暂存目录（--input-directory 指向）
    # i2i/edit 模式下，引擎把 InferRequest.images 落盘到这里，LoadImage 节点读取
    input_dir: Path = field(init=False)

    # ===== ComfyUI 子进程配置 =====
    comfyui_dir: Path = field(default_factory=lambda: Path(_DEFAULT_COMFYUI_DIR))
    python_exe: Path = field(default_factory=lambda: Path(_DEFAULT_COMFYUI_PYTHON))
    # extra_model_paths.yaml 路径（映射平台 models/ 到 ComfyUI 虚拟路径）
    extra_model_paths_yaml: Path = field(init=False)
    # ComfyUI JSON 工作流模板目录
    workflow_templates_dir: Path = field(init=False)

    # ===== 隐藏端口范围（ComfyUI 内部端口）=====
    internal_port_start: int = DEFAULT_INTERNAL_PORT_START
    internal_port_end: int = DEFAULT_INTERNAL_PORT_END
    # v0.3.0 固定内部端口（若设置则忽略端口范围；None 时自动扫描）
    internal_port: Optional[int] = None

    # ===== 子进程管理 =====
    server_timeout: float = DEFAULT_SERVER_TIMEOUT
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL

    # ===== ComfyUI 启动参数（v1.2 显存对接 L1）=====
    # VRAM 档位：normalvram（默认）/ lowvram / novram / highvram
    # 平台规范：用 normalvram，禁用 lowvram（与平台 LRU 冲突）
    vram_mode: str = DEFAULT_VRAM_MODE
    # 禁用 smart memory（避免与平台 LRU 重复）
    disable_smart_memory: bool = True
    # 禁用自动打开浏览器
    disable_auto_launch: bool = True
    # 预览方式：auto（WebSocket 推送预览图）
    preview_method: str = "auto"

    # ===== 设备 =====
    device: str = "cuda"

    # ===== 设备策略（v0.3.5+ 新增）=====
    # auto / gpu / gpu+cpu / cpu，见模块顶部 DEVICE_POLICIES 注释。
    # 平台通过 AIRELIABLE_DEVICE_POLICY 环境变量注入（UI 设置 > 默认 auto），
    # 映射到 vram_mode（gpu→highvram / gpu+cpu→lowvram / cpu→cpu），
    # 仅当用户未显式配置 vram_mode（--vram-mode / COMFYUI_VRAM_MODE）时生效。
    # 默认 "auto" 由基类 EngineConfig 提供。

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.unet_dir = self.models_root / UNET_SUBDIR
        self.vae_dir = self.models_root / VAE_SUBDIR
        self.text_encoder_dir = self.models_root / TEXT_ENCODER_SUBDIR
        self.pe_dir = self.models_root / PE_SUBDIR
        self.tokenizer_dir = self.models_root / TOKENIZER_SUBDIR
        self.lora_dir = self.models_root / LORA_SUBDIR

        # 引擎 Python 进程日志目录（v0.2.15+ 规范：读 AIRELIABLE_LOG_DIR）
        log_dir_env = os.environ.get("AIRELIABLE_LOG_DIR")
        if log_dir_env:
            self.log_dir = Path(log_dir_env)
        else:
            # fallback：项目根 / logs / engines
            self.log_dir = self.project_root / "logs" / "engines"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ComfyUI 子进程日志目录（logs/engines/comfyui_server/，与 llama_cpp_server 一致）
        self.server_log_dir = self.log_dir / "comfyui_server"
        self.server_log_dir.mkdir(parents=True, exist_ok=True)

        # extra_model_paths.yaml 路径（引擎目录下）
        engine_dir = Path(__file__).resolve().parent
        self.extra_model_paths_yaml = engine_dir / "extra_model_paths.yaml"

        # ComfyUI 工作流模板目录（引擎目录下 workflow_templates/）
        self.workflow_templates_dir = engine_dir / "workflow_templates"
        self.workflow_templates_dir.mkdir(parents=True, exist_ok=True)

        # v0.4.0 修复（T2.4 测试发现的根因）：output_root 相对路径 → 绝对路径（锚定 project_root）。
        # 原因：ComfyUI 子进程以 cwd=comfyui_dir 启动，"--output-directory output/image" 与
        # "--input-directory output/input" 中的相对路径会被 ComfyUI 按 comfyui_dir 解析
        # （os.path.abspath），导致引擎落盘位置与 ComfyUI 读取位置错位
        # （症状：LoadImage 校验 "Invalid image file"）。
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()
        if not self.output_root.is_absolute():
            self.output_root = (self.project_root / self.output_root).resolve()

        # 输出根目录
        self.output_root.mkdir(parents=True, exist_ok=True)

        # v0.4.0 多模式工作流：ComfyUI 输入图片暂存目录（output/input/，
        # 与 --output-directory 指向的 output/image/ 对称）
        self.input_dir = self.output_root / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # 设备探测
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        # 读取平台 Core URL（AIRELIABLE_CORE_URL，用于显存上报）
        if self.core_url is None:
            self.core_url = os.environ.get("AIRELIABLE_CORE_URL")

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "unet_dir": str(self.unet_dir),
            "vae_dir": str(self.vae_dir),
            "text_encoder_dir": str(self.text_encoder_dir),
            "pe_dir": str(self.pe_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "lora_dir": str(self.lora_dir),
            "log_dir": str(self.log_dir),
            "server_log_dir": str(self.server_log_dir),
            "comfyui_dir": str(self.comfyui_dir),
            "python_exe": str(self.python_exe),
            "extra_model_paths_yaml": str(self.extra_model_paths_yaml),
            "workflow_templates_dir": str(self.workflow_templates_dir),
            "internal_port": self.internal_port,
            "internal_port_start": self.internal_port_start,
            "internal_port_end": self.internal_port_end,
            "vram_mode": self.vram_mode,
            "disable_smart_memory": self.disable_smart_memory,
            "device": self.device,
            "device_policy": self.device_policy,
            "core_url": self.core_url,
        }


def parse_args() -> EngineConfig:
    """从命令行参数 + 环境变量构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="ERNIE-Image 引擎服务（ComfyUI 子进程模式）")

    # 网络参数
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--host", type=str, default=None, help="监听地址")

    # 路径参数
    parser.add_argument(
        "--models-root", type=str, default=None,
        help="模型根目录（默认: ./models 或 AIRELIABLE_MODELS_ROOT 环境变量）",
    )
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="项目根目录（默认: 当前目录）",
    )
    parser.add_argument(
        "--comfyui-dir", type=str, default=None,
        help="ComfyUI 仓库目录（默认: envs/image-ernie-comfyui/ComfyUI/）",
    )
    parser.add_argument(
        "--python-exe", type=str, default=None,
        help="ComfyUI 子进程 Python 解释器路径（默认: envs/image-ernie-comfyui/.venv/Scripts/python.exe）",
    )

    # 隐藏端口范围
    parser.add_argument(
        "--internal-port-start", type=int, default=None,
        help=f"ComfyUI 内部端口范围起始（默认 {DEFAULT_INTERNAL_PORT_START}）",
    )
    parser.add_argument(
        "--internal-port-end", type=int, default=None,
        help=f"ComfyUI 内部端口范围结束（默认 {DEFAULT_INTERNAL_PORT_END}）",
    )
    parser.add_argument(
        "--internal-port", type=int, default=None,
        help="固定 ComfyUI 内部端口（设置后忽略端口范围扫描）",
    )

    # ComfyUI 启动参数
    parser.add_argument(
        "--vram-mode", type=str, default=None,
        choices=["normalvram", "lowvram", "novram", "highvram", "gpu_only", "cpu"],
        help="ComfyUI VRAM 档位（默认 normalvram，禁用 lowvram 与平台 LRU 冲突）",
    )
    parser.add_argument(
        "--enable-smart-memory", action="store_true", default=None,
        help="启用 ComfyUI smart memory（默认禁用，避免与平台 LRU 重复）",
    )

    # 设备策略（v0.3.5+ 新增）
    parser.add_argument(
        "--device-policy", type=str, default=None,
        choices=list(DEVICE_POLICIES),
        help=(
            "设备策略（默认: 环境变量 AIRELIABLE_DEVICE_POLICY 或 auto）。"
            "auto=引擎自行决定；gpu=强制全 GPU；"
            "gpu+cpu=GPU 优先显存不足溢出 CPU；cpu=强制 CPU"
        ),
    )

    # 设备
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu/mps）")

    # 日志
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR）",
    )

    args = parser.parse_args()

    # 合并环境变量
    def _env_or(arg_val: str | None, env_key: str, default: str | None = None) -> str | None:
        if arg_val is not None:
            return arg_val
        return os.environ.get(env_key, default)

    port = int(args.port) if args.port else int(os.environ.get("ENGINE_PORT", DEFAULT_PORT))
    host = args.host or "127.0.0.1"
    models_root = Path(_env_or(args.models_root, "AIRELIABLE_MODELS_ROOT", "models"))
    project_root = Path(args.project_root or ".")

    comfyui_dir = Path(_env_or(args.comfyui_dir, "COMFYUI_DIR", _DEFAULT_COMFYUI_DIR))
    python_exe = Path(_env_or(args.python_exe, "COMFYUI_PYTHON_EXE", _DEFAULT_COMFYUI_PYTHON))

    # 内部端口
    internal_port_start = (
        args.internal_port_start if args.internal_port_start is not None
        else int(os.environ.get("COMFYUI_INTERNAL_PORT_START", DEFAULT_INTERNAL_PORT_START))
    )
    internal_port_end = (
        args.internal_port_end if args.internal_port_end is not None
        else int(os.environ.get("COMFYUI_INTERNAL_PORT_END", DEFAULT_INTERNAL_PORT_END))
    )
    internal_port = args.internal_port or (
        int(p) if (p := os.environ.get("COMFYUI_INTERNAL_PORT")) else None
    )

    # VRAM 模式
    vram_mode = args.vram_mode or os.environ.get("COMFYUI_VRAM_MODE", DEFAULT_VRAM_MODE)

    # v0.3.5+ 设备策略：命令行 > 环境变量 AIRELIABLE_DEVICE_POLICY（平台注入）> 默认 auto
    # 映射到 vram_mode（仅当用户未显式配置 vram_mode 时）：
    #   gpu     → highvram（全部模型常驻 GPU，显存不足直接报错）
    #   gpu+cpu → lowvram（GPU 优先，ComfyUI 按需把模型换入换出，显存不足溢出 CPU）
    #   cpu     → cpu（--cpu 纯 CPU 推理）
    #   auto    → 不干预（normalvram，ComfyUI 自动探测）
    device_policy = (
        args.device_policy
        or os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip()
        or "auto"
    )
    if device_policy not in DEVICE_POLICIES:
        logging.getLogger(__name__).warning(
            "非法设备策略 %r（合法值：%s），回退 auto。"
            "来源：命令行 --device-policy 或环境变量 AIRELIABLE_DEVICE_POLICY。",
            device_policy, DEVICE_POLICIES,
        )
        device_policy = "auto"
    vram_mode_user_set = (
        args.vram_mode is not None
        or bool(os.environ.get("COMFYUI_VRAM_MODE"))
    )
    if device_policy != "auto" and not vram_mode_user_set:
        policy_to_vram = {"gpu": "highvram", "gpu+cpu": "lowvram", "cpu": "cpu"}
        vram_mode = policy_to_vram[device_policy]
        logging.getLogger(__name__).info(
            "设备策略 %s → vram_mode=%s（由 AIRELIABLE_DEVICE_POLICY 生效，"
            "用户未显式配置 vram_mode）", device_policy, vram_mode,
        )
    elif device_policy != "auto" and vram_mode_user_set:
        logging.getLogger(__name__).info(
            "设备策略 %s 与显式 vram_mode=%s 并存，vram_mode 优先", device_policy, vram_mode,
        )

    # smart memory（默认禁用 True=disable）
    disable_smart_memory = True
    if args.enable_smart_memory:
        disable_smart_memory = False
    elif os.environ.get("COMFYUI_ENABLE_SMART_MEMORY"):
        disable_smart_memory = False

    # 设备：cpu 策略强制 cpu（引擎 Python 进程侧同样降级）
    device = args.device or "cuda"
    if device_policy == "cpu" and device == "cuda":
        device = "cpu"

    return EngineConfig(
        port=port,
        host=host,
        models_root=models_root,
        project_root=project_root,
        comfyui_dir=comfyui_dir,
        python_exe=python_exe,
        internal_port_start=internal_port_start,
        internal_port_end=internal_port_end,
        internal_port=internal_port,
        vram_mode=vram_mode,
        disable_smart_memory=disable_smart_memory,
        device=device,
        device_policy=device_policy,
        log_level=args.log_level or os.environ.get("LOG_LEVEL", "INFO"),
    )
