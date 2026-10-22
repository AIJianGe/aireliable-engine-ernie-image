"""diffusers 风格配置文件清理脚本（T2.10，v0.3.0 ComfyUI 迁移）。

职责：
    扫描 manifest.yaml 中声明的 model_file target_dir，删除 ComfyUI 不需要的
    diffusers 风格小配置文件（config.json / generation_config.json /
    scheduler_config.json / model_index.json 等 < 5KB 文件）。

平台规范优先（v1.2）：
    - **不重命名模型文件**：GGUF / safetensors / tokenizer 文件全部保留原名
    - **只删除 < 5KB 的 diffusers 配置 JSON**：这些是 diffusers `from_pretrained()`
      需要的，ComfyUI 走 GGUF loader + 原生节点，完全不需要
    - manifest.yaml 已删除这些文件的声明，本脚本只清理历史遗留文件

幂等性：
    - 执行后写入 `.cleanup_done` marker 文件到 models_root 根目录
    - 下次启动时若 marker 存在则跳过（除非 ``force=True``）
    - marker 文件含版本号 + 执行时间 + 删除文件列表，便于追溯

调用方式：
    1. 引擎 ``_load_impl`` 首次启动时自动调用：
       ````python
       from ._cleanup_diffusers_configs import cleanup_diffusers_configs
       cleanup_diffusers_configs(models_root, manifest_model_files, logger)
       ````
    2. 命令行直接运行（手动清理）：
       ````bash
       python _cleanup_diffusers_configs.py --models-root D:/.../AIReliable/models --force
       ````

关联分析：``ANALYSIS-TASKS/ERNIE_IMAGE_COMFYUI_MIGRATION_ANALYSIS.md`` 第四部分 §4.5.2 方案 C
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

# ===== 清理规则配置 =====

# ComfyUI 不需要的 diffusers 风格配置文件名（白名单，全部 < 5KB）
# 这些是 diffusers `from_pretrained()` 需要的元数据，ComfyUI 走 GGUF loader 不用
DIFFUSERS_CONFIG_FILENAMES: frozenset[str] = frozenset({
    "config.json",
    "generation_config.json",
    "scheduler_config.json",
    "model_index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
})

# 但是：tokenizer.json 是 ComfyUI-GGUF tekken tokenizer 需要的，**绝对不能删**
# （上面 DIFFUSERS_CONFIG_FILENAMES 白名单不含 tokenizer.json，所以安全）
KEEP_FILENAMES: frozenset[str] = frozenset({
    "tokenizer.json",  # ComfyUI-GGUF tekken tokenizer 必需
})

# 文件大小阈值：> 5KB 的文件不删除（保守策略，防止误删大权重）
# config.json 等通常 < 2KB，tokenizer 配置文件 < 5KB
MAX_CONFIG_FILE_SIZE_BYTES = 5 * 1024  # 5KB

# marker 文件名（写入 models_root 根目录）
MARKER_FILENAME = ".ernie_image_cleanup_done"

# marker 文件版本（升级清理逻辑时递增，触发重新清理）
MARKER_VERSION = "1.0"


def _is_diffusers_config_file(path: Path) -> bool:
    """判断文件是否是需要清理的 diffusers 配置文件。

    判断条件（全部满足才删除）：
        1. 文件名在 ``DIFFUSERS_CONFIG_FILENAMES`` 白名单中
        2. 文件名不在 ``KEEP_FILENAMES`` 保留列表中
        3. 文件大小 < ``MAX_CONFIG_FILE_SIZE_BYTES``（5KB）

    Args:
        path: 文件路径

    Returns:
        bool: 是否需要清理
    """
    name = path.name
    if name in KEEP_FILENAMES:
        return False
    if name not in DIFFUSERS_CONFIG_FILENAMES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return size < MAX_CONFIG_FILE_SIZE_BYTES


def _read_marker(models_root: Path) -> Optional[dict[str, Any]]:
    """读取清理 marker 文件（若存在）。

    Args:
        models_root: models 根目录

    Returns:
        dict | None: marker 内容，不存在返回 None
    """
    marker_path = models_root / MARKER_FILENAME
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_marker(models_root: Path, deleted_files: list[str]) -> None:
    """写入清理 marker 文件。

    Args:
        models_root: models 根目录
        deleted_files: 本次删除的文件列表（相对 models_root 的相对路径）
    """
    marker_path = models_root / MARKER_FILENAME
    content = {
        "version": MARKER_VERSION,
        "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": int(time.time()),
        "deleted_files": deleted_files,
        "deleted_count": len(deleted_files),
    }
    try:
        marker_path.write_text(
            json.dumps(content, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # marker 写入失败不影响主流程


def _get_target_dirs_from_manifest(
    manifest_model_files: Iterable[dict[str, Any]],
) -> list[str]:
    """从 manifest.yaml 的 model_files 提取所有 target_dir（去重）。

    Args:
        manifest_model_files: manifest.yaml 的 ``models.model_files`` 列表

    Returns:
        list[str]: 去重后的 target_dir 列表（相对 models_root 的相对路径）
    """
    seen: set[str] = set()
    result: list[str] = []
    for f in manifest_model_files:
        td = f.get("target_dir")
        if td and td not in seen:
            # 规范化路径分隔符（Windows / Linux 兼容）
            normalized = td.replace("\\", "/").rstrip("/")
            seen.add(normalized)
            result.append(normalized)
    return result


def cleanup_diffusers_configs(
    models_root: Path,
    manifest_model_files: Iterable[dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    force: bool = False,
) -> dict[str, Any]:
    """清理 diffusers 风格配置文件（幂等）。

    流程：
        1. 读取 marker 文件，若已执行过则跳过（除非 ``force=True``）
        2. 从 manifest 提取所有 target_dir
        3. 递归扫描每个 target_dir，找出 diffusers 配置文件
        4. 删除符合条件的文件（白名单 + 大小阈值）
        5. 写入 marker 文件记录本次清理结果

    Args:
        models_root: models 根目录（绝对路径）
        manifest_model_files: manifest.yaml 的 ``models.model_files`` 列表
        logger: 日志记录器，None 则用 ``logging.getLogger(__name__)``
        force: True 则强制重新清理（忽略 marker 文件）

    Returns:
        dict: 清理结果
            - ``skipped`` (bool): 是否因 marker 跳过
            - ``deleted_files`` (list[str]): 删除的文件相对路径列表
            - ``deleted_count`` (int): 删除文件数
            - ``scanned_dirs`` (list[str]): 扫描的目录列表
            - ``marker_version`` (str): marker 版本
    """
    log = logger or logging.getLogger(__name__)
    models_root = Path(models_root).resolve()

    # 1. 检查 marker
    if not force:
        marker = _read_marker(models_root)
        if marker is not None and marker.get("version") == MARKER_VERSION:
            log.info(
                "[cleanup] diffusers 配置清理已执行过（version=%s, at=%s），跳过（force=False）",
                marker.get("version"),
                marker.get("executed_at"),
            )
            return {
                "skipped": True,
                "deleted_files": [],
                "deleted_count": 0,
                "scanned_dirs": [],
                "marker_version": marker.get("version"),
            }

    # 2. 从 manifest 提取 target_dir
    target_dirs = _get_target_dirs_from_manifest(manifest_model_files)
    if not target_dirs:
        log.warning("[cleanup] manifest 未声明任何 target_dir，跳过清理")
        return {
            "skipped": True,
            "deleted_files": [],
            "deleted_count": 0,
            "scanned_dirs": [],
            "marker_version": MARKER_VERSION,
        }

    log.info("[cleanup] 扫描 %d 个 target_dir: %s", len(target_dirs), target_dirs)

    # 3. 扫描 + 删除
    deleted_files: list[str] = []
    for rel_dir in target_dirs:
        target = models_root / rel_dir
        if not target.exists():
            log.debug("[cleanup] 目录不存在，跳过: %s", target)
            continue
        if not target.is_dir():
            log.debug("[cleanup] 不是目录，跳过: %s", target)
            continue

        # 递归扫描目录下所有文件
        for fpath in target.rglob("*"):
            if not fpath.is_file():
                continue
            if not _is_diffusers_config_file(fpath):
                continue

            # 计算相对路径（用于 marker 记录 + 日志）
            try:
                rel = fpath.relative_to(models_root).as_posix()
            except ValueError:
                rel = str(fpath)

            try:
                size = fpath.stat().st_size
                fpath.unlink()
                deleted_files.append(rel)
                log.info("[cleanup] 删除 diffusers 配置: %s (%d bytes)", rel, size)
            except OSError as e:
                log.warning("[cleanup] 删除失败: %s (%s)", rel, e)

    # 4. 写 marker
    _write_marker(models_root, deleted_files)

    log.info(
        "[cleanup] 清理完成: 删除 %d 个 diffusers 配置文件，marker 已写入 %s",
        len(deleted_files),
        models_root / MARKER_FILENAME,
    )

    return {
        "skipped": False,
        "deleted_files": deleted_files,
        "deleted_count": len(deleted_files),
        "scanned_dirs": target_dirs,
        "marker_version": MARKER_VERSION,
    }


# ===== 命令行入口 =====

def _load_manifest_model_files(manifest_path: Path) -> list[dict[str, Any]]:
    """从 manifest.yaml 读取 model_files 列表。

    Args:
        manifest_path: manifest.yaml 路径

    Returns:
        list[dict]: model_files 列表
    """
    import yaml  # type: ignore[import-untyped]

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return (data.get("models") or {}).get("model_files") or []


def main() -> int:
    """命令行入口。

    用法：
        ````bash
        # 默认清理（幂等，已清理过则跳过）
        python _cleanup_diffusers_configs.py \\
            --models-root D:/myProInstallDir/AIReliable/models \\
            --manifest ../../engines/ernie_image_909469ed/manifest.yaml

        # 强制重新清理
        python _cleanup_diffusers_configs.py \\
            --models-root D:/myProInstallDir/AIReliable/models \\
            --manifest ../../engines/ernie_image_909469ed/manifest.yaml \\
            --force
        ````
    """
    parser = argparse.ArgumentParser(
        description="清理 ERNIE-Image diffusers 风格配置文件（T2.10）",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        required=True,
        help="models 根目录绝对路径",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="manifest.yaml 路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新清理（忽略 marker 文件）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用 DEBUG 日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    models_root = Path(args.models_root).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not models_root.exists():
        print(f"ERROR: models_root 不存在: {models_root}")
        return 1
    if not manifest_path.exists():
        print(f"ERROR: manifest 不存在: {manifest_path}")
        return 1

    model_files = _load_manifest_model_files(manifest_path)
    if not model_files:
        print(f"ERROR: manifest 未声明任何 model_files: {manifest_path}")
        return 1

    result = cleanup_diffusers_configs(
        models_root=models_root,
        manifest_model_files=model_files,
        force=args.force,
    )

    print("\n========== 清理结果 ==========")
    print(f"跳过: {result['skipped']}")
    print(f"删除文件数: {result['deleted_count']}")
    print(f"扫描目录: {result['scanned_dirs']}")
    print(f"marker 版本: {result['marker_version']}")
    if result["deleted_files"]:
        print("\n删除的文件列表:")
        for f in result["deleted_files"]:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
