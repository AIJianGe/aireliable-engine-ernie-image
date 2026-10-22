"""ComfyUI 子进程生命周期管理（v0.3.0 ComfyUI 子进程模式）。

职责：
    1. 启动 ComfyUI 子进程（``python main.py`` + 平台规范的 CLI 参数）
    2. 健康检查（轮询 ``GET /system_stats``）
    3. 停止子进程（``taskkill /T /F`` 杀进程树 + Job Object 兜底）
    4. 显存释放（调 ComfyUI ``POST /free`` 端点，对接平台 L4 LRU 驱逐响应）
    5. 查询显存状态（``GET /system_stats``，对接平台 L3 显存上报）

设计参考：``engines/llama_cpp_9cdb76f8/llama_process.py`` 的 LlamaProcessManager

关键设计：
    - subprocess.Popen 启动 ComfyUI 主进程（Python 解释器 + main.py）
    - Windows: ``creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`` + Job Object 绑定
    - Linux/Mac: ``start_new_session=True`` + ``os.killpg`` 杀进程组
    - stdout/stderr 合并重定向到 ``logs/engines/comfyui_server/comfyui_server.log``（覆写）
    - 隐藏端口范围 8100-8200，自动扫描空闲端口

平台规范优先（v1.2）：
    - ComfyUI 启动参数用 ``--normalvram --disable-smart-memory``（避免与平台 LRU 冲突）
    - 通过 ``--extra-model-paths-config extra_model_paths.yaml`` 映射平台模型目录
    - 日志路径读 ``AIRELIABLE_LOG_DIR`` 环境变量
    - 显存状态通过 HTTP 协议对接 Core（``PlatformVramManager``），不侵入 ComfyUI 代码
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import EngineConfig

logger = logging.getLogger(__name__)


# ===== 端口分配工具 =====

def allocate_internal_port(start: int = 8100, end: int = 8200) -> int:
    """从指定范围内找空闲端口。

    与 llama_process.allocate_internal_port 行为一致：
    顺序尝试 ``start..end``，找到能 bind 的端口即返回。

    Args:
        start: 起始端口（默认 8100）
        end: 结束端口（默认 8200）

    Returns:
        int: 可用的端口号

    Raises:
        RuntimeError: 范围内无空闲端口
    """
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无空闲端口可用（范围 {start}-{end}）")


# ===== ComfyuiProcessManager =====

class ComfyuiProcessManager:
    """ComfyUI 子进程管理器（参考 LlamaProcessManager）。

    封装 ComfyUI 子进程的启动 / 停止 / 健康检查 / 显存释放 / 显存查询，
    供 ``ErnieImageEngine._load_impl`` / ``_unload_impl`` / ``_infer_impl`` 调用。

    生命周期：
        1. ``__init__(config)``  — 仅持有 config，不启动子进程
        2. ``start()``           — 启动 ComfyUI 子进程 + 健康检查
        3. ``is_running``         — 运行中状态查询
        4. ``free_memory()``     — 调 ``POST /free`` 释放显存（L4 LRU 驱逐响应）
        5. ``get_system_stats()`` — 调 ``GET /system_stats`` 查询显存（L3 显存上报）
        6. ``stop()``             — 杀进程树 + 清理资源
    """

    def __init__(self, config: EngineConfig):
        """
        Args:
            config: 引擎配置（含 comfyui_dir / python_exe / vram_mode 等）
        """
        self.config = config

        # 子进程状态
        self._process: Optional[subprocess.Popen] = None
        self._internal_port: Optional[int] = None
        self._log_file_handle: Optional[Any] = None
        self._start_time: Optional[float] = None
        # Windows Job Object 句柄（绑定子进程生命周期，父进程死时 OS 自动 kill 子进程）
        self._job_handle: Optional[Any] = None
        # v0.4.1：启动流程完成事件（解决 LOADING 窗口期反代 502 竞态）
        # start() 开头创建（未 set），启动流程结束（成功/失败）后 set；
        # _cleanup_resources() 清空。供 wait_startup_done() 等待。
        self._start_done: Optional[asyncio.Event] = None

    # ============ 状态属性 ============

    @property
    def is_running(self) -> bool:
        """子进程是否在运行。"""
        return self._process is not None and self._process.poll() is None

    @property
    def internal_port(self) -> Optional[int]:
        """当前隐藏端口。"""
        return self._internal_port

    @property
    def internal_url(self) -> Optional[str]:
        """ComfyUI 内部 URL（如 ``http://127.0.0.1:8100``）。"""
        if self._internal_port is None:
            return None
        return f"http://{self.config.host}:{self._internal_port}"

    @property
    def pid(self) -> Optional[int]:
        """子进程 PID。"""
        if self._process is None:
            return None
        return self._process.pid

    # ============ 启动 ============

    async def start(self, timeout: float = 120.0) -> dict:
        """启动 ComfyUI 子进程。

        流程：
            1. 校验 ComfyUI 目录和 Python 解释器存在
            2. 分配隐藏端口
            3. 构造 CLI 命令（含 ``--normalvram --disable-smart-memory``）
            4. 打开日志文件（覆写模式）
            5. ``subprocess.Popen`` 启动子进程
            6. Windows Job Object 绑定
            7. 健康检查（轮询 ``GET /system_stats``）

        Args:
            timeout: 健康检查超时秒数（默认 120s，ComfyUI 首次启动较慢）

        Returns:
            dict: ``{"internal_port": int, "pid": int, "load_time": float}``

        Raises:
            FileNotFoundError: ComfyUI 目录或 Python 解释器不存在
            RuntimeError: 启动失败 / 健康检查超时 / 子进程意外退出
        """
        if self.is_running:
            raise RuntimeError(
                f"已有 ComfyUI 子进程在运行（pid={self.pid}），请先 stop()"
            )

        # v0.4.1：创建启动完成事件（局部引用，finally 中 set，保证失败路径也能解除等待者阻塞）
        start_done = asyncio.Event()
        self._start_done = start_done

        # 1. 校验 ComfyUI 目录和 Python 解释器
        comfyui_main = self.config.comfyui_dir / "main.py"
        if not comfyui_main.exists():
            raise FileNotFoundError(
                f"ComfyUI main.py 不存在: {comfyui_main} "
                f"（请检查 comfyui_dir={self.config.comfyui_dir}）"
            )
        if not self.config.python_exe.exists():
            raise FileNotFoundError(
                f"Python 解释器不存在: {self.config.python_exe} "
                f"（请检查 envs/image-ernie-comfyui/.venv/）"
            )

        # 2. 分配隐藏端口
        if self.config.internal_port is not None:
            # 固定端口模式
            self._internal_port = self.config.internal_port
            logger.info("使用固定内部端口: %d", self._internal_port)
        else:
            self._internal_port = allocate_internal_port(
                self.config.internal_port_start,
                self.config.internal_port_end,
            )
            logger.info("分配隐藏端口: %d", self._internal_port)

        # 2.5 动态生成 extra_model_paths.yaml（用 AIRELIABLE_MODELS_ROOT 绝对路径）
        # 平台规范优先：若 AIRELIABLE_MODELS_ROOT 注入了绝对路径，覆盖静态 yaml 文件
        self._generate_extra_model_paths()

        # 3. 构造 CLI 命令
        cmd = self._build_command()

        # 4. 打开日志文件（覆写模式，与 llama_cpp_server 一致）
        log_file = self.config.server_log_dir / "comfyui_server.log"
        self._log_file_handle = open(log_file, "w", encoding="utf-8")
        logger.info("ComfyUI 子进程日志: %s（覆写模式）", log_file)

        # 5. 启动子进程
        self._start_time = time.time()
        logger.info("启动 ComfyUI 子进程: cmd=%s", " ".join(str(c) for c in cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(self.config.comfyui_dir),
                creationflags=_get_creation_flags(),
                start_new_session=(sys.platform != "win32"),
                close_fds=True,
            )
        except Exception as e:
            logger.exception("启动 ComfyUI 子进程失败")
            self._cleanup_resources()
            raise RuntimeError(f"启动 ComfyUI 子进程失败: {e}") from e

        logger.info("ComfyUI 子进程已启动: pid=%d, port=%d", self._process.pid, self._internal_port)

        # 6. Windows Job Object 绑定
        self._assign_to_job_object(self._process)

        # 7. 健康检查
        try:
            await self.wait_for_ready(timeout=timeout)
        except Exception as e:
            # 健康检查失败，停止子进程
            await self.stop()
            raise RuntimeError(f"ComfyUI 健康检查失败: {e}") from e
        finally:
            # v0.4.1：启动流程结束（成功/失败）均解除 wait_startup_done 等待者阻塞。
            # 用局部引用 set（stop() 可能已把 self._start_done 清成 None）
            start_done.set()

        load_time = time.time() - self._start_time
        logger.info("ComfyUI 启动完成: 耗时 %.2fs", load_time)

        return {
            "internal_port": self._internal_port,
            "pid": self._process.pid,
            "load_time": load_time,
        }

    def _build_command(self) -> list[str]:
        """构造 ComfyUI 启动命令。

        命令格式：
            ``python main.py --listen <host> --port <port>
            --disable-auto-launch --preview-method <method>
            --output-directory <dir>
            --extra-model-paths-config <yaml>
            [--<vram_mode>] [--disable-smart-memory]``

        平台规范优先（v1.2 显存对接 L1）：
            - 默认 ``normalvram``（**不传任何 vram 参数**，让 ComfyUI 自动探测显存决定档位）
            - ComfyUI v0.32.0 实际支持的 vram 参数：``--highvram`` / ``--lowvram`` /
              ``--novram`` / ``--gpu-only`` / ``--cpu``
            - **ComfyUI v0.32.0 不再支持** ``--normalvram``（默认行为即 normalvram）
            - 禁用 ``--lowvram``（与平台 LRU 冲突）
            - 启用 ``--disable-smart-memory``（避免与平台 LRU 重复）

        Returns:
            list[str]: 完整命令行
        """
        cmd: list[str] = [
            str(self.config.python_exe),
            "main.py",
            "--listen", self.config.host,
            "--port", str(self._internal_port),
            "--disable-auto-launch",
            "--preview-method", self.config.preview_method,
            "--output-directory", str(self.config.output_root / "image"),
            # v0.4.0 多模式工作流：输入图片暂存目录（i2i/edit 模式 LoadImage 节点读取）
            "--input-directory", str(self.config.input_dir),
            "--extra-model-paths-config", str(self.config.extra_model_paths_yaml),
        ]

        # VRAM 档位（平台规范：默认 normalvram 不传任何参数）
        # ComfyUI v0.32.0 支持的 vram 参数：--highvram / --lowvram / --novram / --gpu-only / --cpu
        # normalvram 不是显式参数，而是默认行为（不传任何 vram 参数让 ComfyUI 自动探测）
        vram_mode = (self.config.vram_mode or "normalvram").lower()
        if vram_mode != "normalvram":
            # 非默认档位才传参数（highvram / lowvram / novram / gpu-only / cpu）
            cmd.append(f"--{vram_mode}")

        # 禁用 smart memory（避免与平台 LRU 重复）
        if self.config.disable_smart_memory:
            cmd.append("--disable-smart-memory")

        return cmd

    def _generate_extra_model_paths(self) -> None:
        """动态生成 extra_model_paths.yaml（平台规范优先）。

        平台通过 ``AIRELIABLE_MODELS_ROOT`` 注入绝对路径时，覆盖静态 yaml 文件，
        确保 ComfyUI 能找到平台模型目录。

        生成逻辑：
            1. 从 ``config.models_root`` 读取绝对路径
            2. 拼接各子目录（unet/vae/text_encoder/pe/tokenizer/lora）
            3. 写入 ``config.extra_model_paths_yaml``（覆盖静态文件）

        平台规范优先原则：
            - 模型文件位置遵循 manifest.yaml 声明
            - 文件名保持平台现状（不强制改名为 ComfyUI 标准命名）
        """
        import yaml

        models_root = self.config.models_root
        # v0.4.5 修复：models_root 可能是相对路径（手动启动未注入
        # AIRELIABLE_MODELS_ROOT 时默认 "models"）。ComfyUI 对 yaml 中的
        # 相对路径按「yaml 文件所在目录」解析（utils/extra_config.py
        # load_extra_path_config → engines/ernie_image_909469ed/models/...，
        # 不存在）→ Web UI 报缺失模型。此处统一 resolve 为绝对路径
        # （基于 CWD = 项目根），平台启动与手动启动行为一致。
        if not models_root.is_absolute():
            models_root = models_root.resolve()
            logger.info(
                "models_root 为相对路径，已 resolve 为绝对路径: %s", models_root
            )

        # 构造 extra_model_paths.yaml 数据结构
        # 用绝对路径，避免相对路径解析问题
        # v1.3（T4.4）：PE 目录也映射到 clip/text_encoders（CLIPLoader 查找路径），
        # 用换行符分隔多路径（ComfyUI utils/extra_config.py:24 用 split("\n") 解析）
        te_path = str(models_root / "shared" / "text_encoders" / "ernie-image") + "/"
        pe_path = str(models_root / "image" / "pe" / "ernie-image") + "/"
        # ComfyUI extra_model_paths 用 \n 分隔同类型的多个路径
        clip_paths = te_path + "\n" + pe_path

        data = {
            "ernie_image_platform": {
                "unet": str(models_root / "image" / "unet") + "/",
                "diffusion_models": str(models_root / "image" / "unet") + "/",
                "vae": str(models_root / "image" / "vae" / "ernie-image") + "/",
                "clip": clip_paths,
                "text_encoders": clip_paths,
                "llm": pe_path,
                "tokenizers": str(models_root / "shared" / "tokenizer" / "ernie-image" / "tokenizer") + "/",
                "loras": str(models_root / "image" / "loras" / "ernie-image") + "/",
            }
        }

        yaml_path = self.config.extra_model_paths_yaml
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(
                "动态生成 extra_model_paths.yaml: %s（models_root=%s）",
                yaml_path, models_root,
            )
        except Exception as e:
            logger.warning("动态生成 extra_model_paths.yaml 失败（用静态文件兜底）: %s", e)

    async def wait_for_ready(self, timeout: float = 120.0) -> None:
        """轮询 ``GET /system_stats`` 直到 ComfyUI 就绪。

        Args:
            timeout: 超时秒数

        Raises:
            RuntimeError: 子进程意外退出
            TimeoutError: 健康检查超时
        """
        if self._internal_port is None or self._process is None:
            raise RuntimeError("子进程未启动")

        deadline = time.time() + timeout
        url = f"{self.internal_url}/system_stats"
        logger.info("ComfyUI 健康检查: %s（超时 %.0fs）", url, timeout)

        last_error: Optional[str] = None
        while time.time() < deadline:
            # 检查子进程是否意外退出
            if self._process.poll() is not None:
                recent_logs = self.read_recent_logs(50)
                raise RuntimeError(
                    f"ComfyUI 子进程意外退出（exit_code={self._process.returncode}）"
                    f"\n最近日志:\n{recent_logs}"
                )

            # 轮询 /system_stats
            try:
                async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                    r = await client.get(url)
                    if r.status_code == 200:
                        data = r.json()
                        logger.info(
                            "ComfyUI 就绪: version=%s, python=%s, pytorch=%s, device=%s",
                            data.get("system", {}).get("comfyui_version", "?"),
                            data.get("system", {}).get("python_version", "?"),
                            data.get("system", {}).get("pytorch_version", "?"),
                            data.get("devices", [{}])[0].get("name", "?") if data.get("devices") else "?",
                        )
                        return
            except Exception as e:
                last_error = str(e)

            await asyncio.sleep(2.0)

        raise TimeoutError(
            f"ComfyUI 健康检查超时（{timeout}s）: last_error={last_error}"
        )

    async def wait_startup_done(self, timeout: float = 200.0) -> bool:
        """等待正在进行的 ``start()`` 流程结束（v0.4.1 新增）。

        解决 LOADING 窗口期竞态：``Popen`` 已创建 + 端口已分配，但健康检查
        未通过（ComfyUI 初始化需 ~18-30s，含 custom_nodes 加载 + DB 迁移）。
        此窗口内若反代请求直接连接会 502（All connection attempts failed），
        应先在此等待启动流程结束后再放行。

        Args:
            timeout: 等待超时秒数（默认 200s，覆盖 ComfyUI 首次启动 ~65s + 缓冲）

        Returns:
            bool: 启动流程结束后子进程是否仍在运行
                - True: 启动成功（或早已就绪），可继续反代
                - False: 等待超时 / 启动失败后进程已清理，调用方走清理重载逻辑
        """
        if self._start_done is None:
            # 无进行中的启动流程：直接反馈当前状态
            return self.is_running
        try:
            await asyncio.wait_for(self._start_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("wait_startup_done 等待超时（%.0fs），子进程可能仍在初始化", timeout)
            return False
        return self.is_running

    # ============ 停止 ============

    async def stop(self) -> dict:
        """停止 ComfyUI 子进程（释放所有显存）。

        v0.3.1+ 增强进程树清理（与 LlamaProcessManager 一致）：
            - Windows: ``taskkill /T /F /PID <pid>``（杀整个进程树）
            - Linux/Mac: ``os.killpg``（杀整个进程组）

        Returns:
            dict: ``{"state": "idle", "vram_freed": "...", "pid": int, "stop_time": float}``
        """
        if self._process is None:
            return {"state": "idle", "vram_freed": "0GB"}

        pid = self._process.pid
        start_time = time.time()
        logger.info("停止 ComfyUI 子进程: pid=%d", pid)

        # 估算释放的显存（查询 /system_stats 对比启动前/后）
        vram_freed = await self._estimate_vram_freed()

        # 优先用进程树清理
        try:
            await self._kill_process_tree(pid)
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                    timeout=3.0,
                )
                logger.info("ComfyUI 进程树已清理: pid=%d", pid)
            except asyncio.TimeoutError:
                logger.warning("进程树清理后 3s 内未退出，回退到 kill: pid=%d", pid)
                try:
                    self._process.kill()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("进程树清理失败，回退到 terminate/kill: %s", e)
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                        timeout=5.0,
                    )
                    logger.info("ComfyUI 已优雅退出: pid=%d", pid)
                except asyncio.TimeoutError:
                    logger.warning("ComfyUI 5s 内未退出，强制 kill: pid=%d", pid)
                    self._process.kill()
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._process.wait, 3.0,
                        )
                    except Exception:
                        pass
            except Exception as e2:
                logger.error("停止子进程异常: %s", e2)
                try:
                    self._process.kill()
                except Exception:
                    pass

        self._cleanup_resources()
        stop_time = time.time() - start_time
        return {
            "state": "idle",
            "vram_freed": vram_freed,
            "pid": pid,
            "stop_time": stop_time,
        }

    async def _kill_process_tree(self, pid: int) -> None:
        """杀整个进程树（与 LlamaProcessManager 一致）。

        Windows: ``taskkill /T /F /PID <pid>``
        Linux/Mac: ``os.killpg(os.getpgid(pid), SIGKILL)``

        Args:
            pid: 根进程 PID
        """
        if sys.platform == "win32":
            cmd = ["taskkill", "/T", "/F", "/PID", str(pid)]
            logger.info("Windows 进程树清理: %s", " ".join(cmd))
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                if proc.returncode == 0:
                    logger.info("进程树清理成功: pid=%d", pid)
                else:
                    logger.debug(
                        "taskkill 返回 %d: stdout=%s, stderr=%s",
                        proc.returncode,
                        stdout.decode(errors="replace"),
                        stderr.decode(errors="replace"),
                    )
            except asyncio.TimeoutError:
                logger.warning("taskkill 超时: pid=%d", pid)
            except Exception as e:
                logger.warning("taskkill 异常: %s", e)
        else:
            try:
                import os
                import signal
                pgid = os.getpgid(pid)
                logger.info("Linux 进程组清理: pgid=%d", pgid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning("killpg 异常: %s", e)

    def _assign_to_job_object(self, process: subprocess.Popen) -> None:
        """Windows: 把子进程加入 Job Object。

        Job Object 配置：
            - ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``：父进程死时 OS 自动 kill Job 内所有进程
        作用：
            - 防止平台异常退出时 ComfyUI 变孤儿继续占用显存
            - 仅 Windows 需要，Linux/Mac 已用 ``start_new_session`` + 进程组

        Args:
            process: ``subprocess.Popen`` 对象
        """
        if sys.platform != "win32":
            return

        try:
            import ctypes

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9
            PROCESS_ALL_ACCESS = 0x1F0FFF

            kernel32 = ctypes.windll.kernel32
            kernel32.CreateJobObjectW = ctypes.WINFUNCTYPE(
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p,
                use_last_error=True,
            )(("CreateJobObjectW", kernel32))
            kernel32.SetInformationJobObject = ctypes.WINFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_uint32,
                use_last_error=True,
            )(("SetInformationJobObject", kernel32))
            kernel32.AssignProcessToJobObject = ctypes.WINFUNCTYPE(
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                use_last_error=True,
            )(("AssignProcessToJobObject", kernel32))
            kernel32.OpenProcess = ctypes.WINFUNCTYPE(
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32,
                use_last_error=True,
            )(("OpenProcess", kernel32))

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            ret = kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ret:
                raise ctypes.WinError(ctypes.get_last_error())

            process_handle = kernel32.OpenProcess(
                PROCESS_ALL_ACCESS, False, process.pid,
            )
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())

            ret = kernel32.AssignProcessToJobObject(job_handle, process_handle)
            if not ret:
                raise ctypes.WinError(ctypes.get_last_error())

            self._job_handle = job_handle
            logger.info(
                "ComfyUI 子进程已绑定 Job Object: pid=%d, job=%d",
                process.pid, job_handle,
            )

        except Exception as e:
            logger.warning("Job Object 绑定失败（不阻断启动）: %s", e)

    def _cleanup_resources(self) -> None:
        """清理子进程资源（关闭日志文件句柄 + Job Object 句柄）。"""
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

        # Windows Job Object 句柄不需要显式关闭（进程退出后 OS 自动回收）
        self._job_handle = None

        self._process = None
        self._internal_port = None
        self._start_time = None
        # v0.4.1：清空启动完成事件（下次 start() 重新创建；
        # start() 失败路径的 finally 用局部引用 set，不依赖此处的清空）
        self._start_done = None

    # ============ 日志读取 ============

    def read_recent_logs(self, n_lines: int = 100) -> str:
        """读取 ComfyUI 子进程日志文件的最近 N 行。

        用于启动失败时返回 ``recent_logs`` 给前端（与 LlamaProcessManager 一致）。

        Args:
            n_lines: 行数（默认 100）

        Returns:
            str: 最近 N 行日志（拼接为字符串）；文件不存在返回空串
        """
        log_file = self.config.server_log_dir / "comfyui_server.log"
        if not log_file.exists():
            return ""

        try:
            # 用 GBK 解码兜底（与 decode_subprocess_output 一致）
            with open(log_file, "rb") as f:
                lines = f.readlines()
            recent = lines[-n_lines:] if len(lines) > n_lines else lines
            return b"".join(recent).decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("读取 ComfyUI 日志失败: %s", e)
            return ""

    # ============ ComfyUI HTTP API（显存对接 L3/L4/L5）============

    async def get_system_stats(self) -> Optional[dict]:
        """调 ``GET /system_stats`` 查询 ComfyUI 系统状态（含显存）。

        用于 L3 显存上报：推理后查询显存占用，调 ``PlatformVramManager.report_loaded``。

        Returns:
            dict | None: ComfyUI 返回的系统状态 JSON；子进程未启动或查询失败返回 None

        响应结构示例：
            ````json
            {
              "system": {
                "comfyui_version": "v0.32.0",
                "python_version": "3.11.10",
                "pytorch_version": "2.11.0+cu128"
              },
              "devices": [
                {
                  "name": "Quadro RTX 8000",
                  "type": "cuda",
                  "vram_total": 51539148800,
                  "vram_free": 50094669824,
                  "torch_vram_total": 0,
                  "torch_vram_free": 0
                }
              ]
            }
            ````
        """
        if not self.is_running or self.internal_url is None:
            return None

        url = f"{self.internal_url}/system_stats"
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                logger.warning("GET /system_stats 返回 %d: %s", r.status_code, r.text[:200])
                return None
        except Exception as e:
            logger.warning("GET /system_stats 失败: %s", e)
            return None

    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> bool:
        """调 ``POST /free`` 释放 ComfyUI 显存（L4 LRU 驱逐响应）。

        平台 L4 设计：Core 触发 LRU 驱逐时调引擎 ``/unload``，
        引擎 ``_unload_impl`` 调本方法释放 ComfyUI 显存。

        Args:
            unload_models: 是否卸载已加载的模型（默认 True）
            free_memory: 是否释放显存（默认 True）

        Returns:
            bool: True 成功 / False 失败或子进程未启动
        """
        if not self.is_running or self.internal_url is None:
            return False

        url = f"{self.internal_url}/free"
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                r = await client.post(url, json=payload)
                if r.status_code == 200:
                    logger.info("[VRAM] ComfyUI /free 成功: 卸载模型=%s, 释放显存=%s",
                                unload_models, free_memory)
                    return True
                logger.warning("[VRAM] ComfyUI /free 返回 %d: %s",
                               r.status_code, r.text[:200])
                return False
        except Exception as e:
            logger.warning("[VRAM] ComfyUI /free 失败: %s", e)
            return False

    async def get_vram_usage_bytes(self) -> Optional[int]:
        """查询 ComfyUI 当前显存占用（字节）。

        用于 L3 显存上报：推理后查询 ``vram_total - vram_free`` 得到占用值，
        调 ``PlatformVramManager.report_loaded`` 上报到 Core。

        Returns:
            int | None: 显存占用字节数；查询失败返回 None
        """
        stats = await self.get_system_stats()
        if not stats or not stats.get("devices"):
            return None
        try:
            device = stats["devices"][0]
            vram_total = int(device.get("vram_total", 0))
            vram_free = int(device.get("vram_free", 0))
            return max(0, vram_total - vram_free)
        except Exception as e:
            logger.warning("解析 ComfyUI 显存数据失败: %s", e)
            return None

    async def _estimate_vram_freed(self) -> str:
        """停止子进程时估算释放的显存。

        ComfyUI 子进程退出后无法查询显存，简化为查询停止前的显存占用。
        若查询失败返回 "unknown"。

        Returns:
            str: 释放的显存字符串（如 "7.20GB" / "unknown"）
        """
        vram_used = await self.get_vram_usage_bytes()
        if vram_used is None or vram_used == 0:
            return "0GB"
        return f"{vram_used / (1024 ** 3):.2f}GB"


# ===== 辅助函数 =====

def _get_creation_flags() -> int:
    """获取 Windows 子进程创建标志（不弹 cmd 窗口）。

    与 ``llama_process._get_creation_flags`` 一致：
        - ``CREATE_NO_WINDOW = 0x08000000``
        - ``CREATE_NEW_PROCESS_GROUP = 0x00000200``（允许 Ctrl+C 不传递到子进程）

    Returns:
        int: 创建标志（非 Windows 返回 0）
    """
    if sys.platform == "win32":
        return 0x08000000 | 0x00000200
    return 0
