"""Train a tennis-ball detector from LLM pseudo-labels + synthetic dense composites."""
from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/workspace/anycount/research/packs/tennis-ball")
RAW = ROOT / "raw"
LABELS = ROOT / "labels"
DS = ROOT / "dataset"
MODELS = ROOT / "models"
YOLO_BASE = Path("/root/workspace/countthis/pilot/yolo11n.pt")
SEED = 7
random.seed(SEED)
np.random.seed(SEED)


def points_of(rec: dict) -> list[tuple[float, float]]:
    return [(float(p["x"]), float(p["y"])) for p in rec.get("points", [])]


def pseudo_boxes(points: list[tuple[float, float]], w: int, h: int) -> list[tuple[float, float, float]]:
    """cx, cy, size using the median distance to the 3 nearest neighbours."""
    if not points:
        return []
    pts = np.array(points, dtype=float)
    n = len(pts)
    min_px, max_px = 10.0, 0.30 * min(w, h)
    out = []
    for i in range(n):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        k = min(3, max(1, n - 1))
        d_nn = float(np.median(np.sort(d)[:k])) if n > 1 else min_px / 0.92
        size = float(np.clip(d_nn * 0.92, min_px, max_px))
        out.append((float(pts[i, 0]), float(pts[i, 1]), size))
    return out


def to_yolo_line(cx: float, cy: float, size: float, w: int, h: int) -> str:
    bw, bh = size / w, size / h
    return f"0 {cx / w:.6f} {cy / h:.6f} {bw:.6f} {bh:.6f}"


def write_pair(split: str, stem: str, img: Image.Image, lines: list[str]) -> None:
    img_dir = DS / "images" / split
    lbl_dir = DS / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img.save(img_dir / f"{stem}.jpg", quality=92)
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def build() -> dict:
    if DS.exists():
        shutil.rmtree(DS)
    recs = []
    for path in sorted(RAW.glob("*.jpg")):
        lab = LABELS / f"{path.stem}.json"
        if not lab.exists():
            continue
        rec = json.loads(lab.read_text())
        if rec["count"] <= 0:
            continue
        recs.append((path, rec))
    print(f"labelled images with points: {len(recs)}")

    random.shuffle(recs)
    val_cut = max(1, round(len(recs) * 0.15))
    val_set, train_set = recs[:val_cut], recs[val_cut:]

    crops: list[Image.Image] = []
    for path, rec in train_set:
        im = Image.open(path).convert("RGB")
        if (im.width, im.height) != (rec["width"], rec["height"]):
            im = im.resize((rec["width"], rec["height"]), Image.LANCZOS)
        pts = points_of(rec)
        w, h = im.size
        boxes = pseudo_boxes(pts, w, h)
        write_pair("train", f"{path.stem}_full", im, [to_yolo_line(*b, w, h) for b in boxes])
        for idx, (cx, cy, size) in enumerate(boxes):
            half = int(max(size * 0.6, 12))
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
            if x1 - x0 >= 24 and y1 - y0 >= 24:
                crops.append(im.crop((x0, y0, x1, y1)))
        ow, oh = round(w * 0.08), round(h * 0.08)
        for r in range(2):
            for c in range(2):
                tx0 = max(0, round(c * w / 2) - ow)
                tx1 = min(w, round((c + 1) * w / 2) + ow)
                ty0 = max(0, round(r * h / 2) - oh)
                ty1 = min(h, round((r + 1) * h / 2) + oh)
                tw, th = tx1 - tx0, ty1 - ty0
                lines = []
                for (cx, cy, size) in boxes:
                    if tx0 <= cx < tx1 and ty0 <= cy < ty1:
                        lines.append(to_yolo_line(cx - tx0, cy - ty0, size, tw, th))
                if lines:
                    write_pair("train", f"{path.stem}_t{r}{c}", im.crop((tx0, ty0, tx1, ty1)), lines)

    for path, rec in val_set:
        im = Image.open(path).convert("RGB")
        if (im.width, im.height) != (rec["width"], rec["height"]):
            im = im.resize((rec["width"], rec["height"]), Image.LANCZOS)
        w, h = im.size
        boxes = pseudo_boxes(points_of(rec), w, h)
        write_pair("val", f"{path.stem}_full", im, [to_yolo_line(*b, w, h) for b in boxes])

    # synthetic dense composites: paste real ball crops onto real backgrounds
    composites = 0
    if crops:
        backgrounds = [Image.open(p).convert("RGB") for p, _ in train_set]
        for i in range(28):
            bg = random.choice(backgrounds).copy()
            bw, bh = bg.size
            lines = []
            for _ in range(random.randint(35, 70)):
                crop = random.choice(crops)
                scale = random.uniform(0.35, 0.9) * (min(bw, bh) * 0.055 / max(1, min(crop.size)))
                size = max(14, int(min(crop.size) * scale))
                patch = crop.resize((size, size), Image.LANCZOS)
                cx, cy = random.uniform(size, bw - size), random.uniform(size, bh - size)
                bg.paste(patch, (int(cx - size / 2), int(cy - size / 2)))
                lines.append(to_yolo_line(cx, cy, size, bw, bh))
            write_pair("train", f"synth_{i:03d}", bg, lines)
            composites += 1
    print(f"composites: {composites}")
    DS.joinpath("data.yaml").write_text(
        f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: tennis_ball\n"
    )
    counts = {
        "train": len(list((DS / 'images' / 'train').glob('*.jpg'))),
        "val": len(list((DS / 'images' / 'val').glob('*.jpg'))),
    }
    print(counts)
    return counts


def train(epochs: int = 35, imgsz: int = 640) -> Path:
    from ultralytics import YOLO
    from onnxruntime.quantization import QuantType, quantize_dynamic

    MODELS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_BASE))
    model.train(
        data=str(DS / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=1.0,
        mosaic=0.5,
        close_mosaic=10,
        workers=4,
        device="cpu",
        project=str(ROOT / "runs"),
        name="tennis_v1",
        exist_ok=True,
        verbose=True,
    )
    best = ROOT / "runs" / "tennis_v1" / "weights" / "best.pt"
    exported = Path(YOLO(str(best)).export(format="onnx", imgsz=imgsz, opset=12))
    int8 = MODELS / "tennis_ball_v1_int8.onnx"
    quantize_dynamic(str(exported), str(int8), weight_type=QuantType.QInt8)
    print(f"onnx int8: {int8} ({int8.stat().st_size/1e6:.2f} MB)")
    return int8


if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        train()
    else:
        build()
