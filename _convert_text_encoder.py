"""将 multimodal Mistral3 safetensors 转换为 ComfyUI 兼容的 text-only 格式。

问题：
    - `models/shared/text_encoders/ernie-image/model.safetensors`（7.17GB）
      是 multimodal Mistral3ForConditionalGeneration 模型，键名带 `language_model.`
      前缀，且包含 `vision_tower.*` 视觉塔权重。
    - ComfyUI 的 `detect_te_model()`（comfy/sd.py:1565）只识别直接以
      `model.layers.0.post_attention_layernorm.weight` 命名的 Ministral3-3B 权重。
    - 结果：te_model 检测失败 → 退化到 SD1ClipModel → 推理时报
      `Cannot copy out of meta tensor; no data!`
    - 即使检测成功，普通 safetensors 缺少 `tekken_model` 字段
      （ComfyUI-GGUF 的 gguf_clip_loader 会从 GGUF metadata 重建），
      导致 Mistral3Tokenizer 初始化时报
      `TypeError: the JSON object must be str, bytes or bytearray, not NoneType`

方案（平台规范优先原则 v1.2）：
    1. **不重新下载**：用现有 multimodal safetensors 文件，不消耗带宽。
    2. **不修改原文件**：保留 `model.safetensors` 不变，新增 `model_text_only.safetensors`。
    3. **键名规范化**：剥离 `language_model.` 前缀，丢弃 `vision_tower.*` 视觉塔权重。
    4. **保留 embedding/lm_head**：ERNIE-Image 文本编码只需要 LLM 部分。
    5. **嵌入 tekken tokenizer**：从 `shared/tokenizer/ernie-image/tokenizer/tokenizer.json`
       重建 tekken_model 字节（torch.ByteTensor），写入 `tekken_model` 键。

验证：
    - 转换后键名应与 `models/image/pe/ernie-image/model.safetensors`（PE 文件，
      已是 text-only）格式一致：`model.layers.X.*` / `model.embed_tokens.weight` / `lm_head.weight`
    - ComfyUI `detect_te_model()` 应返回 `TEModel.MINISTRAL_3_3B`（hidden_size=3072）
    - `tekken_model` 键存在且能被 `from_tekken_json()` 解析

幂等性：
    - 检查 marker 文件 `.text_encoder_converted`
    - marker 包含源文件 mtime + size + 目标文件 size + tokenizer 源 mtime
    - 源/目标文件未变化时跳过
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Windows 下使用 safetensors 加载/保存
from safetensors import safe_open
from safetensors.torch import save_file

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "models" / "shared" / "text_encoders" / "ernie-image" / "model.safetensors"
DEFAULT_DST = DEFAULT_SRC.parent / "model_text_only.safetensors"
DEFAULT_MARKER = DEFAULT_SRC.parent / ".text_encoder_converted"
DEFAULT_TOKENIZER_SRC = (
    PROJECT_ROOT / "models" / "shared" / "tokenizer" / "ernie-image" / "tokenizer" / "tokenizer.json"
)


def _bytes_to_unicode() -> dict:
    """Reproduce transformers.convert_slow_tokenizer.bytes_to_unicode.

    ComfyUI-GGUF 的 gguf_tekken_tokenizer_loader 用这个把 token str 转 bytes。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def build_tekken_model_from_tokenizer_json(tokenizer_json_path: Path) -> torch.ByteTensor:
    """从 HuggingFace tokenizer.json 重建 tekken_model 字节。

    ComfyUI-GGUF 的 gguf_tekken_tokenizer_loader() 从 GGUF metadata 重建 tekken
    tokenizer。但普通 safetensors 没有 GGUF metadata，所以需要从 HuggingFace
    tokenizer.json 重建相同格式的数据。

    tekken_model 是 torch.ByteTensor，内容是 JSON 序列化的 dict：
        {
            "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
            "vocab": [{"rank": int, "token_bytes": base64, "token_str": str}, ...],
            "special_tokens": [{"rank": int, "token_str": str, "is_control": True}, ...]
        }

    Args:
        tokenizer_json_path: HuggingFace tokenizer.json 路径

    Returns:
        torch.ByteTensor（1D，包含 JSON 序列化的 tekken tokenizer 数据）
    """
    print(f"  从 tokenizer.json 重建 tekken tokenizer: {tokenizer_json_path}")
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        hf_data = json.load(f)

    model = hf_data.get("model", {})
    vocab = model.get("vocab", {})
    added_tokens = hf_data.get("added_tokens", [])

    if len(vocab) != 131072:
        raise ValueError(
            f"tokenizer.json vocab size {len(vocab)} 不是 131072（Mistral tekken 标志）"
        )

    # Mistral 原生 tekken 格式约定：
    #   - default_num_special_tokens = 32（id 0-31 是 Mistral 标准特殊 token）
    #   - default_vocab_size = 131072（总词表大小，含特殊 token）
    #   - vocab 中只存 BPE token（rank 从 0 开始），实际 ID = rank + 32
    #   - special_tokens 中存 32 个特殊 token（rank 从 0 开始），实际 ID = rank
    #   - 与 HuggingFace tokenizer.json 的原 ID 完全对齐
    #
    # ComfyUI 0.32.0 的 from_tekken_json (bpe_tokenizer.py:270) 强制读取
    # config["default_num_special_tokens"]，所以这个字段必须存在。
    # 而 ComfyUI-GGUF 的 gguf_tekken_tokenizer_loader 生成的 config 缺少此字段，
    # 这是上游的兼容性问题，我们这里直接补齐。
    SPECIAL_TOKEN_OFFSET = 32

    tekken_data = {
        "config": {
            "num_vocab_tokens": 150000,
            "default_vocab_size": 131072,
            "default_num_special_tokens": SPECIAL_TOKEN_OFFSET,
        },
        "vocab": [],
        "special_tokens": [],
    }

    # 构建 id -> token 字符串映射
    id_to_token = {}
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str

    # _bytes_to_unicode() 返回 {byte: unicode_char}（与 transformers.bytes_to_unicode 一致）
    # ComfyUI-GGUF 用反向映射 {unicode_char: byte} 将 token 字符串转回 bytes
    byte_encoder = _bytes_to_unicode()  # {byte: unicode_char}
    byte_decoder = {v: k for k, v in byte_encoder.items()}  # {unicode_char: byte}

    # 1) 特殊 token：取 id < SPECIAL_TOKEN_OFFSET 的（Mistral 原生 32 个）
    #    这些在 HuggingFace tokenizer.json 中作为 added_tokens 出现
    special_id_to_content = {
        t["id"]: t["content"] for t in added_tokens if "id" in t and t["id"] < SPECIAL_TOKEN_OFFSET
    }

    for idx in range(SPECIAL_TOKEN_OFFSET):
        token_str = special_id_to_content.get(idx, id_to_token.get(idx, f"<UNKNOWN_{idx}>"))
        tekken_data["special_tokens"].append(
            {"rank": idx, "token_str": token_str, "is_control": True}
        )

    # 2) BPE token：id >= SPECIAL_TOKEN_OFFSET，rank 从 0 开始重新计数
    #    实际 ID（在模型嵌入表中）= rank + SPECIAL_TOKEN_OFFSET
    max_vocab = 131072 - SPECIAL_TOKEN_OFFSET  # 131040
    for idx in range(SPECIAL_TOKEN_OFFSET, len(vocab)):
        if len(tekken_data["vocab"]) >= max_vocab:
            # 超过 max_vocab 的 token 会被 from_tekken_json 过滤掉，不存了
            break
        token_str = id_to_token.get(idx, "")
        if not token_str:
            # 缺失的 token（理论上不会发生），用占位
            token_str = f"<MISSING_{idx}>"

        # 对齐 gguf_tekken_tokenizer_loader：
        #   tok = bytes([decoder[char] for char in token])
        # 其中 decoder = {v: k for k, v in bytes_to_unicode().items()}，是反向映射
        # token_str 是 HuggingFace tokenizer.json 的 vocab key（已是 unicode str）
        try:
            tok_bytes = bytes([byte_decoder[c] for c in token_str if c in byte_decoder])
        except (KeyError, ValueError):
            # 有些 token str 可能包含 byte_decoder 中没有的字符，直接用 utf-8
            tok_bytes = token_str.encode("utf-8", errors="replace")

        tekken_data["vocab"].append(
            {
                "rank": len(tekken_data["vocab"]),
                "token_bytes": base64.b64encode(tok_bytes).decode("ascii"),
                "token_str": tok_bytes.decode("utf-8", errors="replace"),
            }
        )

    print(
        f"    special_tokens: {len(tekken_data['special_tokens'])}, "
        f"vocab: {len(tekken_data['vocab'])} (max_vocab={max_vocab})"
    )

    # 序列化为 JSON 字节
    json_bytes = json.dumps(tekken_data, ensure_ascii=False).encode("utf-8")
    return torch.ByteTensor(list(json_bytes))


def convert(
    src: Path,
    dst: Path,
    tokenizer_src: Path = DEFAULT_TOKENIZER_SRC,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """转换 multimodal Mistral3 safetensors 到 text-only 格式。

    Args:
        src: 源文件路径（multimodal `model.safetensors`）
        dst: 目标文件路径（text-only `model_text_only.safetensors`）
        tokenizer_src: HuggingFace tokenizer.json 路径（用于重建 tekken_model）
        force: 跳过 marker 校验，强制重新转换
        verbose: 打印每个保留/丢弃的 key（调试用）

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
                print(f"       源文件: {src_size:,} bytes, mtime={time.ctime(src_mtime)}")
                print(f"       目标文件: {dst.stat().st_size:,} bytes")
                print(f"       如需强制重新转换，请使用 --force 参数")
                return {
                    "skipped": True,
                    "src_size": src_size,
                    "dst_size": dst.stat().st_size,
                }
        except Exception as e:
            print(f"[WARN] marker 文件读取失败，将重新转换: {e}")

    print(f"[1/6] 加载源 safetensors: {src}")
    print(f"      文件大小: {src_size:,} bytes ({src_size/1024/1024/1024:.2f} GB)")

    # 加载所有 tensor
    tensors_to_save = {}
    stats = {
        "total_keys": 0,
        "kept_keys": 0,
        "dropped_vision_tower": 0,
        "stripped_prefix": 0,
        "lm_head_keys": 0,
        "tekken_model_added": False,
        "skipped": False,
    }

    with safe_open(str(src), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        stats["total_keys"] = len(keys)
        print(f"      总 key 数: {len(keys)}")

        for key in keys:
            # 跳过视觉塔权重（ERNIE-Image 文本编码不需要）
            if key.startswith("vision_tower."):
                stats["dropped_vision_tower"] += 1
                if verbose:
                    print(f"  [DROP] {key}")
                continue

            # 剥离 language_model. 前缀
            new_key = key
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                stats["stripped_prefix"] += 1

            # lm_head 保持原样（无前缀）
            if new_key == "lm_head.weight":
                stats["lm_head_keys"] += 1

            tensor = f.get_tensor(key)
            tensors_to_save[new_key] = tensor.contiguous()  # contiguous 避免 view 问题
            stats["kept_keys"] += 1

            if verbose:
                print(f"  [KEEP] {key} -> {new_key} ({tensor.shape}, {tensor.dtype})")

    print(f"[2/6] 键名转换完成:")
    print(f"      总 key 数: {stats['total_keys']}")
    print(f"      保留 key 数: {stats['kept_keys']}")
    print(f"      丢弃 vision_tower: {stats['dropped_vision_tower']}")
    print(f"      剥离前缀: {stats['stripped_prefix']}")
    print(f"      lm_head 键: {stats['lm_head_keys']}")

    # 验证关键键存在
    required_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ]
    missing = [k for k in required_keys if k not in tensors_to_save]
    if missing:
        raise RuntimeError(
            f"转换后关键键缺失，无法被 ComfyUI 识别为 Ministral3-3B：\n"
            f"  缺失: {missing}\n"
            f"  这说明源文件结构异常，请检查 {src}"
        )

    # 验证 hidden_size=3072（Ministral3-3B 标志）
    hidden_size = tensors_to_save["model.layers.0.post_attention_layernorm.weight"].shape[0]
    if hidden_size != 3072:
        print(f"[WARN] hidden_size={hidden_size}，不是预期的 3072（Ministral3-3B）")
    else:
        print(f"      验证: hidden_size={hidden_size} ✓ (Ministral3-3B)")

    # 重建 tekken_model tokenizer
    print(f"[3/6] 重建 tekken tokenizer:")
    tekken_tensor = build_tekken_model_from_tokenizer_json(tokenizer_src)
    # safetensors 不支持存储 1D ByteTensor，需要转成 int8 然后 reshape
    # 实际上 safetensors 支持 torch.uint8 (torch.ByteTensor 在新版 torch 里就是 uint8)
    # 但 ComfyUI 读取时用 .item() 或 .tolist() 访问，需要保持 1D ByteTensor
    # safetensors.torch.save_file 支持 torch.uint8
    if tekken_tensor.dtype != torch.uint8:
        tekken_tensor = tekken_tensor.to(torch.uint8)
    tensors_to_save["tekken_model"] = tekken_tensor
    stats["tekken_model_added"] = True
    print(f"      tekken_model: shape={tuple(tekken_tensor.shape)}, dtype={tekken_tensor.dtype}")
    print(f"      字节数: {tekken_tensor.numel():,} ({tekken_tensor.numel()/1024:.1f} KB)")

    print(f"[4/6] 保存目标 safetensors: {dst}")
    # 使用 safetensors.torch.save_file 直接保存
    # 注意：需要 metadata 字段供 ComfyUI 检测 arch
    metadata = {
        "format": "pt",
        "arch": "ministral3_3b",
        "converted_from": "multimodal_mistral3",
        "converted_by": "aireliable_ernie_image_migration",
        "convert_time": str(int(time.time())),
    }
    save_file(tensors_to_save, str(dst), metadata=metadata)
    dst_size = dst.stat().st_size
    print(f"      目标文件大小: {dst_size:,} bytes ({dst_size/1024/1024/1024:.2f} GB)")

    # 写入 marker
    print(f"[5/6] 写入 marker: {DEFAULT_MARKER}")
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

    print(f"[6/6] 转换完成 ✓")
    stats["src_size"] = src_size
    stats["dst_size"] = dst_size
    return stats


def verify(dst: Path = DEFAULT_DST) -> bool:
    """验证转换后的文件能否被 ComfyUI 正确识别为 Ministral3-3B。

    通过模拟 ComfyUI 的 detect_te_model 逻辑验证。
    """
    print(f"\n=== 验证 ===")
    print(f"文件: {dst}")

    if not dst.exists():
        print(f"[FAIL] 文件不存在")
        return False

    with safe_open(str(dst), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"总 key 数: {len(keys)}")

        # 检查 tekken_model 键存在
        if "tekken_model" not in keys:
            print(f"[FAIL] tekken_model 键缺失（tokenizer 无法初始化）")
            return False
        tekken_tensor = f.get_tensor("tekken_model")
        print(f"tekken_model: shape={tuple(tekken_tensor.shape)}, dtype={tekken_tensor.dtype}")

        # 验证 tekken_model JSON 可解析
        try:
            tekken_bytes = tekken_tensor.numpy().tobytes()
            tekken_data = json.loads(tekken_bytes.decode("utf-8"))
            print(
                f"  tekken JSON 解析成功: vocab={len(tekken_data.get('vocab', []))}, "
                f"special_tokens={len(tekken_data.get('special_tokens', []))}"
            )
        except Exception as e:
            print(f"[FAIL] tekken_model JSON 解析失败: {e}")
            return False

        # 验证 config 字段完整性（ComfyUI 0.32.0 from_tekken_json 强制要求）
        config = tekken_data.get("config", {})
        required_config_keys = ["num_vocab_tokens", "default_vocab_size", "default_num_special_tokens"]
        missing_config = [k for k in required_config_keys if k not in config]
        if missing_config:
            print(f"[FAIL] tekken config 缺失字段: {missing_config}")
            print(f"       当前 config: {config}")
            print(f"       ComfyUI 0.32.0 from_tekken_json 强制读取 default_num_special_tokens")
            return False

        special_offset = config["default_num_special_tokens"]
        default_vocab = config["default_vocab_size"]
        max_vocab = default_vocab - special_offset
        print(
            f"  config: default_vocab_size={default_vocab}, "
            f"default_num_special_tokens={special_offset}, max_vocab={max_vocab}"
        )

        # 验证 vocab rank 范围 [0, max_vocab)
        vocab_ranks = [w.get("rank", -1) for w in tekken_data.get("vocab", [])]
        if vocab_ranks:
            rank_min, rank_max = min(vocab_ranks), max(vocab_ranks)
            print(f"  vocab rank 范围: [{rank_min}, {rank_max}]（期望 [0, {max_vocab-1}]）")
            if rank_min < 0 or rank_max >= max_vocab:
                print(f"[FAIL] vocab rank 越界（max_vocab={max_vocab}）")
                return False
            # 实际 ID = rank + special_offset，应在 [special_offset, default_vocab) 范围
            actual_id_max = rank_max + special_offset
            if actual_id_max >= default_vocab:
                print(f"[FAIL] vocab 实际 ID 越界（{actual_id_max} >= {default_vocab}）")
                return False

        # 验证 special_tokens rank 范围 [0, special_offset)
        special_ranks = [w.get("rank", -1) for w in tekken_data.get("special_tokens", [])]
        if special_ranks:
            srank_min, srank_max = min(special_ranks), max(special_ranks)
            print(f"  special_tokens rank 范围: [{srank_min}, {srank_max}]（期望 [0, {special_offset-1}]）")
            if srank_min < 0 or srank_max >= special_offset:
                print(f"[FAIL] special_tokens rank 越界（应 < {special_offset}）")
                return False

        # 模拟 from_tekken_json 的关键检查（不实际构造 BPETokenizer，避免依赖）
        # 仅检查数据结构完整性
        for w in tekken_data.get("vocab", [])[:3]:
            if "token_bytes" not in w or "rank" not in w:
                print(f"[FAIL] vocab 项缺少 token_bytes/rank 字段: {w}")
                return False
            try:
                base64.b64decode(w["token_bytes"])
            except Exception as e:
                print(f"[FAIL] vocab 项 token_bytes 不是有效 base64: {e}")
                return False

        for w in tekken_data.get("special_tokens", [])[:3]:
            if "rank" not in w:
                print(f"[FAIL] special_tokens 项缺少 rank 字段: {w}")
                return False
            if "token_str" not in w and "token_bytes" not in w:
                print(f"[FAIL] special_tokens 项缺少 token_str/token_bytes 字段: {w}")
                return False

        print(f"  tekken 结构验证 ✓ (符合 ComfyUI 0.32.0 from_tekken_json 期望)")

        # 检查关键键
        check_key = "model.layers.0.post_attention_layernorm.weight"
        if check_key not in keys:
            print(f"[FAIL] 关键键缺失: {check_key}")
            print(f"  前 10 个键: {keys[:10]}")
            return False

        weight = f.get_tensor(check_key)
        print(f"关键键 {check_key}: shape={tuple(weight.shape)}, dtype={weight.dtype}")

        # 模拟 detect_te_model 逻辑
        if "model.layers.0.post_attention_layernorm.weight" in keys:
            w = weight
            if 'model.layers.0.self_attn.q_norm.weight' in keys:
                # Qwen3 路径（不是我们要的）
                print(f"[INFO] 走 Qwen3 路径（含 q_norm）")
            if w.shape[0] == 5120:
                print(f"[FAIL] 走 MISTRAL3_24B 路径（hidden=5120）")
                return False
            if w.shape[0] == 3072:
                print(f"[OK] ✓ 走 MINISTRAL_3_3B 路径（hidden=3072）")
                print(f"     ComfyUI 将加载 comfy.text_encoders.ernie.te()")
                print(f"     tokenizer 将使用 comfy.text_encoders.ernie.ErnieTokenizer")
                print(f"     tekken_model 已嵌入，Mistral3Tokenizer 可正确初始化")
                return True

        print(f"[FAIL] 未识别为 Ministral3-3B")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="转换 multimodal Mistral3 safetensors 到 ComfyUI 兼容的 text-only 格式"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help=f"源文件路径（默认: {DEFAULT_SRC}）",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"目标文件路径（默认: {DEFAULT_DST}）",
    )
    parser.add_argument(
        "--tokenizer-src",
        type=Path,
        default=DEFAULT_TOKENIZER_SRC,
        help=f"HuggingFace tokenizer.json 路径（默认: {DEFAULT_TOKENIZER_SRC}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新转换，跳过 marker 校验",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个保留/丢弃的 key",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证，不转换",
    )
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
