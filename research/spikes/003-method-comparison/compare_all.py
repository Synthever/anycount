"""
Spike 003: Side-by-Side Visual & Quantitative Comparison
Compares Count-Anything (SAM3 3.4GB) vs Gemini 3.8 Flash Single-Pass vs Gemini 3.8 Flash Tiled (2x2 + Point-NMS).
Generates labeled side-by-side comparison montages in results/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def make_montage(
    panels: List[Tuple[str, Path]],
    out_path: Path,
    target_height: int = 420,
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

    header_h = 44
    pad = 12
    total_w = sum(im.size[0] for _, im in loaded) + pad * (len(loaded) + 1)
    total_h = target_height + header_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=16)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            font = ImageFont.load_default()

    x_cursor = pad
    for title, im in loaded:
        draw.text((x_cursor + 6, pad + 10), title, fill=(245, 245, 245), font=font)
        canvas.paste(im, (x_cursor, header_h + pad))
        x_cursor += im.size[0] + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    research_root = repo_root / "research"
    s1_dir = research_root / "spikes" / "001-gemini-single-pass" / "results"
    s2_dir = research_root / "spikes" / "002-gemini-tiled-grid" / "results"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ca_cats = repo_root / "docs" / "image" / "count-anything" / "result" / "hasil-kucing.jpg"
    ca_containers = repo_root / "docs" / "image" / "count-anything" / "result" / "hasil-container.jpg"
    ca_apples = Path("/root/workspace/count-anything/exp/infer/apples__apples__20261104_094222/apples__apples.jpg")

    make_montage(
        [
            ("Count-Anything SAM3 (count=2, 23s CPU)", ca_cats),
            ("Gemini Single-Pass (count=2, 5.7s)", s1_dir / "cats__cats.jpg"),
            ("Gemini Open-Vocab 'remote' (count=2, 5.0s)", s1_dir / "cats__remote_control.jpg"),
        ],
        out_dir / "montage_cats_and_remotes.jpg",
    )

    make_montage(
        [
            ("Gemini Single-Pass (count=24/24, 13.9s)", s1_dir / "coins__coins.jpg"),
            ("Gemini 2x2 Tiled + NMS (count=24/24, 20.8s)", s2_dir / "coins__coins__2x2.jpg"),
        ],
        out_dir / "montage_coins.jpg",
    )

    make_montage(
        [
            ("Count-Anything SAM3 (count=64, 28.1s A100)", ca_containers),
            ("Gemini Single-Pass (count=29, MISSES DENSE)", s1_dir / "containers__container.jpg"),
            ("Gemini 2x2 Tiled + NMS (count=65, 1.6% err)", s2_dir / "containers__container__2x2.jpg"),
        ],
        out_dir / "montage_containers.jpg",
    )

    if ca_apples.exists():
        make_montage(
            [
                ("Count-Anything SAM3 (count=339, 23s CPU)", ca_apples),
                ("Gemini Single-Pass (count=289, 39.9s)", s1_dir / "apples__apples.jpg"),
                ("Gemini 2x2 Tiled + NMS (count=343, 21.1s)", s2_dir / "apples__apples__2x2.jpg"),
            ],
            out_dir / "montage_apples.jpg",
        )

    print("Generated comparison montages in:", out_dir)


if __name__ == "__main__":
    main()
