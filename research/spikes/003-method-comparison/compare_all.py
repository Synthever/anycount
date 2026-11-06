"""
Spike 003: Side-by-Side Visual & Quantitative Comparison Across All 6 Methods
1. Metode 1: Count-Anything (SAM3 3.4GB)
2. Metode 2: Offline ONNX Model Pack (2.77 MB) + LLM Validator + Distilled v2
3. Metode 3: Gemini 3.8 Flash — Single-Pass
4. Metode 4: Gemini 3.8 Flash — 2x2 Tiled Grid + Point-NMS
5. Metode 5: GPT-6 Luna (gpt-6-luna-xhigh) — Single-Pass
6. Metode 6: GPT-6 Luna (gpt-6-luna-xhigh) — 2x2 Tiled Grid + Point-NMS
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def make_montage(
    panels: List[Tuple[str, Path]],
    out_path: Path,
    target_height: int = 380,
) -> None:
    loaded = []
    for title, path in panels:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = target_height / float(h)
        new_w = int(round(w * scale))
        resized = img.resize((new_w, target_height), Image.Resampling.LANCZOS)
        loaded.append((title, resized))

    if not loaded:
        return

    header_h = 42
    pad = 12
    total_w = sum(im.size[0] for _, im in loaded) + pad * (len(loaded) + 1)
    total_h = target_height + header_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=14)
        except Exception:
            font = ImageFont.load_default()

    x_cursor = pad
    for title, im in loaded:
        draw.text((x_cursor + 4, pad + 10), title, fill=(245, 245, 245), font=font)
        canvas.paste(im, (x_cursor, header_h + pad))
        x_cursor += im.size[0] + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    research_root = repo_root / "research"
    s0_dir = research_root / "spikes" / "000-count-anything-baseline" / "results"
    s1_dir = research_root / "spikes" / "001-gemini-single-pass" / "results"
    s2_dir = research_root / "spikes" / "002-gemini-tiled-grid" / "results"
    s4_dir = research_root / "spikes" / "004-offline-onnx-llm-validator" / "results"
    s5_dir = research_root / "spikes" / "005-gpt6-luna-single-pass" / "results"
    s6_dir = research_root / "spikes" / "006-gpt6-luna-tiled-grid" / "results"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ca_cats = repo_root / "docs" / "image" / "count-anything" / "result" / "hasil-kucing.jpg"
    ca_containers = repo_root / "docs" / "image" / "count-anything" / "result" / "hasil-container.jpg"
    ca_apples = Path("/root/workspace/count-anything/exp/infer/apples__apples__20261104_094222/apples__apples.jpg")

    make_montage(
        [
            ("M1: Count-Anything (2, 23s)", ca_cats),
            ("M3: Gemini Single (2, 5.7s)", s1_dir / "cats__cats.jpg"),
            ("M5: GPT-6 Single (2, 1.7s)", s5_dir / "cats__cats.jpg"),
            ("M3: Gemini 'remote' (2, 5.0s)", s1_dir / "cats__remote_control.jpg"),
            ("M5: GPT-6 'remote' (2, 2.2s)", s5_dir / "cats__remote_control.jpg"),
        ],
        out_dir / "montage_cats_and_remotes.jpg",
    )

    make_montage(
        [
            ("M3: Gemini Single (24/24, 13.9s)", s1_dir / "coins__coins.jpg"),
            ("M4: Gemini 2x2 (24/24, 20.8s)", s2_dir / "coins__coins__2x2.jpg"),
            ("M5: GPT-6 Single (24/24, 3.9s)", s5_dir / "coins__coins.jpg"),
            ("M6: GPT-6 2x2 (25/24, 3.7s)", s6_dir / "coins__coins__2x2.jpg"),
        ],
        out_dir / "montage_coins.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (64, 28.1s)", ca_containers),
            ("M3: Gemini Single (29, 6.0s)", s1_dir / "containers__container.jpg"),
            ("M4: Gemini 2x2 (65, 31.5s)", s2_dir / "containers__container__2x2.jpg"),
            ("M5: GPT-6 Single (24, 3.2s)", s5_dir / "containers__container.jpg"),
            ("M6: GPT-6 2x2 (64, 13.8s)", s6_dir / "containers__container__2x2.jpg"),
        ],
        out_dir / "montage_containers.jpg",
    )

    if ca_apples.exists():
        make_montage(
            [
                ("M1: Count-Anything (339, 23s)", ca_apples),
                ("M3: Gemini Single (289, 39.9s)", s1_dir / "apples__apples.jpg"),
                ("M4: Gemini 2x2 (336, 21.1s)", s2_dir / "apples__apples__2x2.jpg"),
                ("M5: GPT-6 Single (264, 20.7s)", s5_dir / "apples__apples.jpg"),
                ("M6: GPT-6 2x2 (325, 38.6s)", s6_dir / "apples__apples__2x2.jpg"),
            ],
            out_dir / "montage_apples.jpg",
        )

    make_montage(
        [
            ("M1: Count-Anything (41/43, 13.6s)", s0_dir / "tennis-ball__tennis_ball.jpg"),
            ("M3: Gemini Single (42/43, 7.7s)", s1_dir / "tennis-ball__tennis_ball.jpg"),
            ("M4: Gemini 2x2 (42/43, 10.7s)", s2_dir / "tennis-ball__tennis_ball__2x2.jpg"),
            ("M5: GPT-6 Single (83 DRIFT, 8.7s)", s5_dir / "tennis-ball__tennis_ball.jpg"),
            ("M6: GPT-6 2x2 (41/43, 6.8s)", s6_dir / "tennis-ball__tennis_ball__2x2.jpg"),
        ],
        out_dir / "montage_tennis_ball.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (125, 13.6s)", s0_dir / "Heidelbeeren__blueberry.jpg"),
            ("M3: Gemini Single (122, 12.4s)", s1_dir / "Heidelbeeren__blueberry.jpg"),
            ("M4: Gemini 2x2 (122, 11.3s)", s2_dir / "Heidelbeeren__blueberry__2x2.jpg"),
            ("M5: GPT-6 Single (103, 9.0s)", s5_dir / "Heidelbeeren__blueberry.jpg"),
            ("M6: GPT-6 2x2 (117, 10.1s)", s6_dir / "Heidelbeeren__blueberry__2x2.jpg"),
        ],
        out_dir / "montage_heidelbeeren.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (131, 13.6s)", s0_dir / "Heidelbeeren-bnw__blueberry.jpg"),
            ("M3: Gemini Single (112, 12.1s)", s1_dir / "Heidelbeeren-bnw__blueberry.jpg"),
            ("M4: Gemini 2x2 (119, 11.5s)", s2_dir / "Heidelbeeren-bnw__blueberry__2x2.jpg"),
            ("M5: GPT-6 Single (133, 11.2s)", s5_dir / "Heidelbeeren-bnw__blueberry.jpg"),
            ("M6: GPT-6 2x2 (116, 13.8s)", s6_dir / "Heidelbeeren-bnw__blueberry__2x2.jpg"),
        ],
        out_dir / "montage_heidelbeeren_bnw.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (137, 16.6s)", s0_dir / "Heidelbeeren-bnw2__blueberry.jpg"),
            ("M3: Gemini Single (128, 14.0s)", s1_dir / "Heidelbeeren-bnw2__blueberry.jpg"),
            ("M4: Gemini 2x2 (127, 11.0s)", s2_dir / "Heidelbeeren-bnw2__blueberry__2x2.jpg"),
            ("M5: GPT-6 Single (92, 8.8s)", s5_dir / "Heidelbeeren-bnw2__blueberry.jpg"),
            ("M6: GPT-6 2x2 (116, 16.7s)", s6_dir / "Heidelbeeren-bnw2__blueberry__2x2.jpg"),
        ],
        out_dir / "montage_heidelbeeren_bnw2.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (128, 13.6s)", s0_dir / "candy-ball__candy.jpg"),
            ("M3: Gemini Single (128, 19.0s)", s1_dir / "candy-ball__candy.jpg"),
            ("M4: Gemini 2x2 (145, 14.1s)", s2_dir / "candy-ball__candy__2x2.jpg"),
            ("M5: GPT-6 Single (102, 9.0s)", s5_dir / "candy-ball__candy.jpg"),
            ("M6: GPT-6 2x2 (144, 9.8s)", s6_dir / "candy-ball__candy__2x2.jpg"),
        ],
        out_dir / "montage_candy_ball.jpg",
    )

    make_montage(
        [
            ("M1: Count-Anything (123, 16.6s)", s0_dir / "candy-bnw__candy.jpg"),
            ("M3: Gemini Single (116, 12.5s)", s1_dir / "candy-bnw__candy.jpg"),
            ("M4: Gemini 2x2 (144, 12.8s)", s2_dir / "candy-bnw__candy__2x2.jpg"),
            ("M5: GPT-6 Single (101, 20.3s)", s5_dir / "candy-bnw__candy.jpg"),
            ("M6: GPT-6 2x2 (112, 5.6s)", s6_dir / "candy-bnw__candy__2x2.jpg"),
        ],
        out_dir / "montage_candy_bnw.jpg",
    )

    # Metode 2 Flywheel montage copy
    if (s4_dir / "apple_val_1__flywheel_montage.jpg").exists():
        import shutil
        shutil.copy2(
            s4_dir / "apple_val_1__flywheel_montage.jpg",
            out_dir / "montage_m2_offline_onnx_flywheel.jpg",
        )

    print("Generated all 6-method comparison montages in:", out_dir)


if __name__ == "__main__":
    main()
