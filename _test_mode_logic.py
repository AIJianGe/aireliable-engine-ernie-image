"""v0.4.0 多模式逻辑单元测试（不需要启动 ComfyUI）。

验证：
    1. ``_determine_mode``：workflow 显式指定 + images role 自动推断 + 非法值报错
    2. ``_select_template``：三维模板矩阵（variant × mode × PE）10 组用例
    3. ``_build_template_registry``：9 个模板文件全部存在
    4. ``_capabilities_impl``：supported_modes / workflow_templates 字段
    5. ``_normalize_mask_image``：RGBA 透明反转 + 灰度直通
    6. ``_stage_input_images``：本地图片落盘（16 对齐 + role 提取 + finally 清理语义）

运行：
    cd d:\\myProInstallDir\\AIReliable
    envs\\image-ernie-comfyui\\.venv\\Scripts\\python.exe engines\\ernie_image_909469ed\\_test_mode_logic.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AIRELIABLE_LOG_DIR", str(PROJECT_ROOT / "logs" / "engines"))
os.environ.setdefault("AIRELIABLE_MODELS_ROOT", str(PROJECT_ROOT / "models"))

import io

from PIL import Image

from engines.ernie_image_909469ed.config import EngineConfig
from engines.ernie_image_909469ed.engine import ErnieImageEngine


PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


# ===== 测试用 request / image 桩 =====

class FakeImage:
    def __init__(self, role: str | None = None, path: str | None = None,
                 b64: str | None = None):
        self.role = role
        self.path = path
        self.b64_json = b64
        self.url = None


class FakeRequest:
    def __init__(self, workflow: str | None = None, images: list | None = None):
        self.workflow = workflow
        self.images = images or []


def build_engine() -> ErnieImageEngine:
    config = EngineConfig(
        models_root=PROJECT_ROOT / "models",
        project_root=PROJECT_ROOT,
        comfyui_dir=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / "ComfyUI",
        python_exe=PROJECT_ROOT / "envs" / "image-ernie-comfyui" / ".venv" / "Scripts" / "python.exe",
        internal_port=None,
        server_timeout=300.0,
    )
    return ErnieImageEngine(config)


def test_determine_mode(eng: ErnieImageEngine) -> None:
    print("\n[1] _determine_mode")
    cases = [
        # (request, 期望 mode, 期望有错)
        (FakeRequest(), "t2i", False),
        (FakeRequest(workflow="t2i"), "t2i", False),
        (FakeRequest(workflow="i2i"), "i2i", False),
        (FakeRequest(workflow="EDIT"), "edit", False),          # 大小写不敏感
        (FakeRequest(workflow="bad"), "", True),                # 非法值
        (FakeRequest(workflow=""), "t2i", False),               # 空串走自动推断
        (FakeRequest(images=[FakeImage()]), "i2i", None),       # 无 role → source → i2i
        (FakeRequest(images=[FakeImage("source")]), "i2i", False),
        (FakeRequest(images=[FakeImage("source"), FakeImage("mask")]), "edit", False),
        (FakeRequest(images=[FakeImage("mask")]), "edit", False),
    ]
    for i, (req, exp_mode, exp_err) in enumerate(cases):
        mode, err = eng._determine_mode(req)
        if exp_err is None:
            # None 表示期望无错（case 6 的 None 与 False 同义）
            exp_err = False
        check(
            f"case{i}: workflow={req.workflow!r} images={len(req.images)}",
            mode == exp_mode and bool(err) == bool(exp_err),
            f"got mode={mode!r} err={err!r}",
        )


def test_select_template(eng: ErnieImageEngine) -> None:
    print("\n[2] _select_template 三维矩阵")
    mv_turbo = eng._manifest_variants.get("ernie-image-turbo-gguf-q4-k-m")
    mv_base = eng._manifest_variants.get("ernie-image-gguf-q4-k-m")
    assert mv_turbo and mv_base, f"manifest variants 缺失: {list(eng._manifest_variants)}"

    matrix = [
        (mv_turbo, False, "t2i", "ernie_turbo_t2i.json"),
        (mv_turbo, True, "t2i", "ernie_turbo_pe.json"),        # 历史名
        (mv_base, False, "t2i", "ernie_base_t2i.json"),
        (mv_base, True, "t2i", "ernie_base_t2i.json"),          # base 不支持 PE
        (mv_turbo, False, "i2i", "ernie_turbo_i2i.json"),
        (mv_turbo, True, "i2i", "ernie_turbo_i2i_pe.json"),
        (mv_base, False, "i2i", "ernie_base_i2i.json"),
        (mv_turbo, False, "edit", "ernie_turbo_edit.json"),
        (mv_turbo, True, "edit", "ernie_turbo_edit_pe.json"),
        (mv_base, False, "edit", "ernie_base_edit.json"),
        (mv_turbo, False, "invalid", "ernie_turbo_t2i.json"),   # 非法 mode 兜底 t2i
    ]
    for mv, pe, mode, exp in matrix:
        got = eng._select_template(mv, pe, mode)
        turbo = "turbo" in mv.base_model.lower()
        check(
            f"turbo={turbo} pe={pe} mode={mode}",
            got == exp,
            f"got {got!r} expect {exp!r}",
        )


def test_template_registry(eng: ErnieImageEngine) -> None:
    print("\n[3] _build_template_registry")
    reg = eng._build_template_registry()
    check("三个 mode 均有键", set(reg.keys()) == {"t2i", "i2i", "edit"}, str(reg.keys()))
    for mode, templates in reg.items():
        check(f"mode={mode} 模板齐全（3 个）", len(templates) == 3, str(templates))


def test_capabilities(eng: ErnieImageEngine) -> None:
    print("\n[4] _capabilities_impl 扩展字段")
    caps = asyncio.get_event_loop().run_until_complete(eng._capabilities_impl())
    d = caps.model_dump()
    check("supported_modes", d.get("supported_modes") == ["t2i", "i2i", "edit"],
          str(d.get("supported_modes")))
    wt = d.get("workflow_templates") or {}
    check("workflow_templates 含 3 mode", set(wt.keys()) == {"t2i", "i2i", "edit"},
          str(wt.keys()))
    check("isinstance ErnieCapabilitiesResponse",
          caps.__class__.__name__ == "ErnieCapabilitiesResponse")


def test_normalize_mask() -> None:
    print("\n[5] _normalize_mask_image")
    # RGBA 全透明 → 全白（全重绘）
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    m = ErnieImageEngine._normalize_mask_image(rgba)
    mean = sum(m.getdata()) / (64 * 64)
    check("RGBA 全透明 → mask 全白", m.mode == "L" and mean == 255.0, f"mean={mean}")
    # RGBA 全不透明 → 全黑（不重绘）
    rgba2 = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    m2 = ErnieImageEngine._normalize_mask_image(rgba2)
    mean2 = sum(m2.getdata()) / (64 * 64)
    check("RGBA 全不透明 → mask 全黑", mean2 == 0.0, f"mean={mean2}")
    # 灰度白 → 白（直通）
    gray = Image.new("L", (64, 64), 255)
    m3 = ErnieImageEngine._normalize_mask_image(gray)
    check("灰度白 → 直通", sum(m3.getdata()) / (64 * 64) == 255.0)
    # RGB 白 → 白
    rgb = Image.new("RGB", (64, 64), (255, 255, 255))
    m4 = ErnieImageEngine._normalize_mask_image(rgb)
    check("RGB 白 → 直通", sum(m4.getdata()) / (64 * 64) == 255.0)


def test_stage_input_images(eng: ErnieImageEngine) -> None:
    print("\n[6] _stage_input_images（本地图片）")
    loop = asyncio.get_event_loop()

    # 构造测试源图（100x60，非 16 对齐 → 期望 96x48）
    src = Image.new("RGB", (100, 60), (200, 100, 50))
    src_buf = io.BytesIO()
    src.save(src_buf, format="PNG")
    import base64
    src_b64 = base64.b64encode(src_buf.getvalue()).decode()

    # 构造测试 mask（RGBA 半透明 → 中间值）
    mask = Image.new("RGBA", (100, 60), (0, 0, 0, 128))
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()

    # i2i：只有 source
    req = FakeRequest(images=[FakeImage(role="source", b64=src_b64)])
    staged = loop.run_until_complete(eng._stage_input_images(req, "test_task_001", "i2i"))
    check("i2i: source_filename", staged["source_filename"] == "test_task_001_source.png",
          str(staged["source_filename"]))
    check("i2i: 无 mask", staged["mask_filename"] is None)
    check("i2i: 16 对齐 100x60 → 96x48",
          (staged["source_width"], staged["source_height"]) == (96, 48),
          f"got {staged['source_width']}x{staged['source_height']}")
    check("i2i: files 数量 1", len(staged["files"]) == 1)
    p = Path(staged["files"][0])
    check("i2i: 文件存在", p.exists())
    with Image.open(p) as im:
        check("i2i: 落盘尺寸 96x48", im.size == (96, 48), str(im.size))
    # 清理
    ErnieImageEngine._cleanup_staged_images(staged["files"])
    check("i2i: 清理后文件不存在", not p.exists())

    # edit：source + mask
    req2 = FakeRequest(images=[
        FakeImage(role="source", b64=src_b64),
        FakeImage(role="mask", b64=mask_b64),
    ])
    staged2 = loop.run_until_complete(eng._stage_input_images(req2, "test_task_002", "edit"))
    check("edit: source + mask 都落盘",
          staged2["source_filename"] == "test_task_002_source.png"
          and staged2["mask_filename"] == "test_task_002_mask.png")
    check("edit: files 数量 2", len(staged2["files"]) == 2)
    mp = Path(staged2["files"][1])
    with Image.open(mp) as mim:
        check("edit: mask 尺寸与源图一致（96x48）", mim.size == (96, 48), str(mim.size))
        check("edit: mask 为灰度 L", mim.mode == "L", mim.mode)
        # alpha=128 → 255-128=127
        vals = set(mim.getdata())
        check("edit: mask 值 = 255-alpha = 127", vals == {127}, str(vals))
    ErnieImageEngine._cleanup_staged_images(staged2["files"])
    check("edit: 清理完成", not mp.exists() and not Path(staged2["files"][0]).exists())

    # 异常：edit 缺 mask → RuntimeError
    req3 = FakeRequest(images=[FakeImage(role="source", b64=src_b64)])
    try:
        loop.run_until_complete(eng._stage_input_images(req3, "test_task_003", "edit"))
        check("edit 缺 mask 应报错", False)
    except RuntimeError as e:
        check("edit 缺 mask 报 RuntimeError", "mask" in str(e))

    # 异常：i2i 无任何图片 → RuntimeError
    try:
        loop.run_until_complete(eng._stage_input_images(FakeRequest(), "t4", "i2i"))
        check("i2i 无图应报错", False)
    except RuntimeError as e:
        check("i2i 无图报 RuntimeError", "源图" in str(e))


def main() -> int:
    print("=" * 60)
    print("v0.4.0 多模式逻辑单元测试（不启动 ComfyUI）")
    print("=" * 60)

    eng = build_engine()
    print(f"引擎初始化 OK：input_dir={eng.config.input_dir}")

    test_determine_mode(eng)
    test_select_template(eng)
    test_template_registry(eng)
    test_capabilities(eng)
    test_normalize_mask()
    test_stage_input_images(eng)

    print("\n" + "=" * 60)
    print(f"结果：PASS={PASS}  FAIL={FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
