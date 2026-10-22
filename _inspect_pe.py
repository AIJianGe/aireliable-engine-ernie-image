"""检查 PE 模型文件结构（是否含 tekken_model、是否 text-only 格式）。"""
from pathlib import Path
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PE_PATH = PROJECT_ROOT / "models" / "image" / "pe" / "ernie-image" / "model.safetensors"
TE_PATH = PROJECT_ROOT / "models" / "shared" / "text_encoders" / "ernie-image" / "model_text_only.safetensors"


def inspect(path: Path, label: str) -> None:
    print(f"=== {label} ===")
    print(f"路径: {path}")
    if not path.exists():
        print(f"[FAIL] 文件不存在")
        return
    print(f"大小: {path.stat().st_size:,} bytes ({path.stat().st_size/1024/1024/1024:.2f} GB)")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"总 key 数: {len(keys)}")
        print(f"Has tekken_model: {'tekken_model' in keys}")
        print(f"Has lm_head.weight: {'lm_head.weight' in keys}")
        print(f"Has model.embed_tokens.weight: {'model.embed_tokens.weight' in keys}")
        print(f"Has vision_tower.*: {any(k.startswith('vision_tower.') for k in keys)}")
        print(f"Has language_model.* prefix: {any(k.startswith('language_model.') for k in keys)}")
        print(f"First 5 keys: {keys[:5]}")
        print(f"Last 5 keys: {keys[-5:]}")

        # 检查 hidden_size
        if "model.layers.0.post_attention_layernorm.weight" in keys:
            w = f.get_tensor("model.layers.0.post_attention_layernorm.weight")
            print(f"hidden_size (layer 0): {w.shape[0]} (期望 3072 for Ministral3-3B)")

        # 检查 metadata
        try:
            meta = f.metadata()
            if meta:
                print(f"Metadata: {dict(meta)}")
        except Exception:
            pass
    print()


if __name__ == "__main__":
    inspect(PE_PATH, "PE 模型 (ernie-image/pe/model.safetensors)")
    inspect(TE_PATH, "Text Encoder (已转换的 model_text_only.safetensors)")
