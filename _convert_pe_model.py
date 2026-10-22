"""为 PE（Prompt Enhancer）模型添加 tekken_model 键。

问题：
    - `models/image/pe/ernie-image/model.safetensors`（7.14GB）
      已是 text-only 格式（无 `language_model.` 前缀，无 `vision_tower.*`），
      含 `lm_head.weight`（支持文本生成）。
    - 但缺少 `tekken_model` 键，ComfyUI Mistral3Tokenizer 初始化时会报
      `TypeError: the JSON object must be str, bytes or bytearray, not NoneType`。

方案（平台规范优先原则）：
    1. **不重新下载**：用现有 PE safetensors 文件。
    2. **不修改原文件**：保留 `model.safetensors` 不变，新增 `model_with_tokenizer.safetensors`。
    3. **嵌入 tekken tokenizer**：从 `shared/tokenizer/ernie-image/pe_tokenizer/tokenizer.json`
       重建 tekken_model 字节（torch.ByteTensor），写入 `tekken_model` 键。
    4. **保留所有原始权重**：PE 模型已含 lm_head，支持 TextGenerate 文本生成。

验证：
    - 转换后 `tekken_model` 键存在且能被 `from_tekken_json()` 解析
    - `lm_head.weight` 保留（TextGenerate 需要）
    - `model.layers.0.post_attention_layernorm.weight` 存在（detect_te_model 识别 Ministral3-3B）
    - hidden_size=3072

幂等性：
    - 检查 marker 文件 `.pe_model_converted`
    - marker 包含源文件 mtime + size + 目标文件 size + tokenizer 源 mtime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch

# 复用 text_encoder 转换脚本的 tekken 构建 logic
sys.path.insert(0, str(Path(__file__).parent))
from _convert_text_encoder import build_tekken_model_from_tokenizer_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "models" / "image" / "pe" / "ernie-image" / "model.safetensors"
DEFAULT_DST = DEFAULT_SRC.parent / "model_with_tokenizer.safetensors"
DEFAULT_MARKER = DEFAULT_SRC.parent / ".pe_model_converted"
DEFAULT_TOKENIZER_SRC = (
    PROJECT_ROOT / "models" / "shared" / "tokenizer" / "ernie-image" / "pe_tokenizer" / "tokenizer.json"
)


def convert(
    src: Path,
    dst: Path,
    tokenizer_src: Path = DEFAULT_TOKENIZER_SRC,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """为 PE 模型添加 tekken_model 键。

    Args:
        src: 源文件路径（PE `model.safetensors`）
        dst: 目标文件路径（`model_with_tokenizer.safetensors`）
        tokenizer_src: HuggingFace pe_tokenizer.json 路径
        force: 跳过 marker 校验，强制重新转换
        verbose: 打印调试信息

    Returns:
        转换统计 dict
    """
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")
    if not tokenizer_src.exists():
        raise FileNotFoundError(f"tokenizer.json 不存在: {tokenizer_src}")

    src_size = src.stat().st_size
    src_mtime = src.stat().st_mtime
    tok_mtime = tokenizer_src.stat().st_mtime

    # 幂等性检查
    if not force and dst.exists() and DEFAULT_MARKER.exists():
        try:
            marker_data = json.loads(DEFAULT_MARKER.read_text(encoding="utf-8"))
            if (
                marker_data.get("src_size") == src_size
                and abs(marker_data.get("src_mtime", 0) - src_mtime) < 1.0
                and abs(marker_data.get("tokenizer_mtime", 0) - tok_mtime) < 1.0
                and marker_data.get("dst_size") == dst.stat().st_size
            ):
                print(f"[SKIP] 已转换过，跳过: {dst}")
                print(f"       如需强制重新转换，请使用 --force 参数")
                return {"skipped": True, "src_size": src_size, "dst_size": dst.stat().st_size}
        except Exception as e:
            print(f"[WARN] marker 文件读取失败，将重新转换: {e}")

    print(f"[1/5] 加载源 safetensors: {src}")
    print(f"      文件大小: {src_size:,} bytes ({src_size/1024/1024/1024:.2f} GB)")

    tensors_to_save = {}
    stats = {
        "total_keys": 0,
        "kept_keys": 0,
        "has_lm_head": False,
        "tekken_model_added": False,
        "skipped": False,
    }

    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        stats["total_keys"] = len(keys)
        print(f"      总 key 数: {len(keys)}")

        for key in keys:
            tensor = f.get_tensor(key)
            tensors_to_save[key] = tensor.contiguous()
            stats["kept_keys"] += 1

            if key == "lm_head.weight":
                stats["has_lm_head"] = True
                if verbose:
                    print(f"  [KEEP] {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

        # 验证关键键
        if "lm_head.weight" not in tensors_to_save:
            raise RuntimeError(
                f"PE 模型缺少 lm_head.weight，无法支持 TextGenerate 文本生成:\n"
                f"  源文件: {src}"
            )
        if "model.layers.0.post_attention_layernorm.weight" not in tensors_to_save:
            raise RuntimeError(
                f"PE 模型缺少关键键，无法被 ComfyUI 识别为 Ministral3-3B:\n"
                f"  源文件: {src}"
            )

    print(f"[2/5] 原始权重加载完成: {stats['kept_keys']} keys")
    print(f"      Has lm_head.weight: {stats['has_lm_head']}")

    # 验证 hidden_size
    hidden_size = tensors_to_save["model.layers.0.post_attention_layernorm.weight"].shape[0]
    if hidden_size != 3072:
        print(f"[WARN] hidden_size={hidden_size}，不是预期的 3072（Ministral3-3B）")
    else:
        print(f"      验证: hidden_size={hidden_size} ✓ (Ministral3-3B)")

    # 重建 tekken_model tokenizer
    print(f"[3/5] 重建 tekken tokenizer:")
    tekken_tensor = build_tekken_model_from_tokenizer_json(tokenizer_src)
    if tekken_tensor.dtype != torch.uint8:
        tekken_tensor = tekken_tensor.to(torch.uint8)
    tensors_to_save["tekken_model"] = tekken_tensor
    stats["tekken_model_added"] = True
    print(f"      tekken_model: shape={tuple(tekken_tensor.shape)}, dtype={tekken_tensor.dtype}")
    print(f"      字节数: {tekken_tensor.numel():,} ({tekken_tensor.numel()/1024:.1f} KB)")

    print(f"[4/5] 保存目标 safetensors: {dst}")
    metadata = {
        "format": "pt",
        "arch": "ministral3_3b",
        "role": "prompt_enhancer",
        "converted_from": "pe_model_add_tokenizer",
        "converted_by": "aireliable_ernie_image_migration",
        "convert_time": str(int(time.time())),
    }
    save_file(tensors_to_save, str(dst), metadata=metadata)
    dst_size = dst.stat().st_size
    print(f"      目标文件大小: {dst_size:,} bytes ({dst_size/1024/1024/1024:.2f} GB)")

    # 写入 marker
    print(f"[5/5] 写入 marker: {DEFAULT_MARKER}")
    marker_data = {
        "src_path": str(src),
        "src_size": src_size,
        "src_mtime": src_mtime,
        "tokenizer_path": str(tokenizer_src),
        "tokenizer_mtime": tok_mtime,
        "dst_path": str(dst),
        "dst_size": dst_size,
        "convert_time": int(time.time()),
        "stats": stats,
        "metadata": metadata,
    }
    DEFAULT_MARKER.write_text(json.dumps(marker_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"转换完成 ✓")
    stats["src_size"] = src_size
    stats["dst_size"] = dst_size
    return stats


def verify(dst: Path = DEFAULT_DST) -> bool:
    """验证转换后的 PE 文件。"""
    print(f"\n=== 验证 ===")
    print(f"文件: {dst}")

    if not dst.exists():
        print(f"[FAIL] 文件不存在")
        return False

    with safe_open(str(dst), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"总 key 数: {len(keys)}")

        if "tekken_model" not in keys:
            print(f"[FAIL] tekken_model 键缺失")
            return False

        if "lm_head.weight" not in keys:
            print(f"[FAIL] lm_head.weight 缺失（TextGenerate 需要）")
            return False

        if "model.layers.0.post_attention_layernorm.weight" not in keys:
            print(f"[FAIL] 关键键缺失")
            return False

        # 验证 tekken_model JSON 可解析
        try:
            tekken_tensor = f.get_tensor("tekken_model")
            tekken_bytes = tekken_tensor.numpy().tobytes()
            tekken_data = json.loads(tekken_bytes.decode("utf-8"))
            print(
                f"  tekken JSON 解析成功: vocab={len(tekken_data.get('vocab', []))}, "
                f"special_tokens={len(tekken_data.get('special_tokens', []))}"
            )
        except Exception as e:
            print(f"[FAIL] tekken_model JSON 解析失败: {e}")
            return False

        # 验证 hidden_size
        w = f.get_tensor("model.layers.0.post_attention_layernorm.weight")
        if w.shape[0] != 3072:
            print(f"[FAIL] hidden_size={w.shape[0]}，不是 3072")
            return False

        # 验证 lm_head
        lm_head = f.get_tensor("lm_head.weight")
        print(f"  lm_head.weight: shape={tuple(lm_head.shape)}, dtype={lm_head.dtype}")

        print(f"[OK] ✓ PE 模型验证通过（含 tekken_model + lm_head）")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="为 PE 模型添加 tekken_model 键（ComfyUI TextGenerate 需要）"
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"源文件路径（默认: {DEFAULT_SRC}）")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help=f"目标文件路径（默认: {DEFAULT_DST}）")
    parser.add_argument(
        "--tokenizer-src",
        type=Path,
        default=DEFAULT_TOKENIZER_SRC,
        help=f"pe_tokenizer.json 路径（默认: {DEFAULT_TOKENIZER_SRC}）",
    )
    parser.add_argument("--force", action="store_true", help="强制重新转换")
    parser.add_argument("--verbose", action="store_true", help="打印调试信息")
    parser.add_argument("--verify-only", action="store_true", help="只验证，不转换")
    args = parser.parse_args()

    if args.verify_only:
        ok = verify(args.dst)
        sys.exit(0 if ok else 1)

    stats = convert(
        args.src,
        args.dst,
        tokenizer_src=args.tokenizer_src,
        force=args.force,
        verbose=args.verbose,
    )
    ok = verify(args.dst)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
