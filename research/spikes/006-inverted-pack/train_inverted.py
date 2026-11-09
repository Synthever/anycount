"""Train a tennis-ball pack on RGB-negative images, same protocol as train_tennis.py.

Reuses the canonical dataset builder (identical seed, split, pseudo-boxes and
synthetic composites) but every image written to disk is inverted first, so the
inverted pack differs from the shipped pack in one variable only: image colour.

Usage:
    python3 train_inverted.py            # build dataset_inverted/
    python3 train_inverted.py --train    # train 35 epochs + export INT8 ONNX
"""
from __future__ import annotations

import sys
from pathlib import Path

PACK = Path("/root/workspace/anycount/research/packs/tennis-ball")
sys.path.insert(0, str(PACK))

import train_tennis as base  # noqa: E402

from PIL import ImageOps  # noqa: E402

INV_DS = PACK / "dataset_inverted"
OUT_ONNX = PACK / "models" / "tennis_ball_v1_inverted_int8.onnx"
RUN_NAME = "tennis_v1_inverted"


def build() -> dict:
    base.DS = INV_DS
    original_write_pair = base.write_pair

    def write_pair(split: str, stem: str, img, lines: list[str]) -> None:
        original_write_pair(split, stem, ImageOps.invert(img.convert("RGB")), lines)

    base.write_pair = write_pair
    return base.build()


def train(epochs: int = 35, imgsz: int = 640) -> Path:
    from ultralytics import YOLO
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = YOLO(str(base.YOLO_BASE))
    model.train(
        data=str(INV_DS / "data.yaml"),
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
        project=str(PACK / "runs"),
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
    )
    best = PACK / "runs" / RUN_NAME / "weights" / "best.pt"
    exported = Path(YOLO(str(best)).export(format="onnx", imgsz=imgsz, opset=12))
    quantize_dynamic(str(exported), str(OUT_ONNX), weight_type=QuantType.QInt8)
    print(f"onnx int8: {OUT_ONNX} ({OUT_ONNX.stat().st_size / 1e6:.2f} MB)")
    return OUT_ONNX


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    else:
        print(build())
