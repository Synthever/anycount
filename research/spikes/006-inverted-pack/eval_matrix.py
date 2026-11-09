"""2x2 evaluation matrix: {normal, inverted} model x {normal, inverted} input.

Runs both tennis-ball packs through the exact worker functions used by M2/M3
(`run_pass` / `run_tiled_pass`), on the held-out validation split plus the app's
tennis-ball sample. Reference counts are the pseudo-labels for the val split and
43 for the sample (benchmark figure, not human-verified).

Usage: python3 eval_matrix.py [--out eval_matrix.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PACK = Path("/root/workspace/anycount/research/packs/tennis-ball")
WORKERS = Path("/root/workspace/anycount/prototype/workers")
SAMPLES = Path("/root/workspace/anycount/research/samples")
OUT = Path(__file__).with_name("eval_matrix.json")

sys.path.insert(0, str(WORKERS))
import onnx_worker as worker  # noqa: E402

MODELS = {
    "normal": PACK / "models" / "tennis_ball_v1_int8.onnx",
    "inverted": PACK / "models" / "tennis_ball_v1_inverted_int8.onnx",
}
IMGSZ = 640
CONF = 0.25


def infer(sess, inp_name: str, img: Image.Image, mode: str) -> int:
    if mode == "tiled":
        points, _ = worker.run_tiled_pass(sess, inp_name, img, IMGSZ, 1.0, 1.0)
    else:
        points = worker.run_pass(sess, inp_name, img, IMGSZ, CONF)
    return len(points)


def ref_count(stem: str) -> int | None:
    lab = PACK / "dataset" / "labels" / "val" / f"{stem}.txt"
    if lab.exists():
        return len([ln for ln in lab.read_text().splitlines() if ln.strip()])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import onnxruntime as ort

    sessions = {}
    for name, path in MODELS.items():
        if not path.exists():
            raise SystemExit(f"missing model: {path}")
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        sessions[name] = (sess, sess.get_inputs()[0].name)

    # Input variants share stems, so each row compares like with like.
    val_normal = sorted((PACK / "dataset" / "images" / "val").glob("*.jpg"))
    val_inverted = sorted((PACK / "dataset_inverted" / "images" / "val").glob("*.jpg"))
    if [p.stem for p in val_normal] != [p.stem for p in val_inverted]:
        raise SystemExit("val stems differ between dataset/ and dataset_inverted/")

    sample = SAMPLES / "tennis-ball.jpg"
    sample_inv = Path(__file__).with_name("_sample_inverted.jpg")
    ImageOps.invert(Image.open(sample).convert("RGB")).save(sample_inv, quality=90)

    rows = []
    for variant, paths in (("normal", val_normal), ("inverted", val_inverted)):
        for path in paths:
            img = Image.open(path).convert("RGB")
            for model_name, (sess, inp) in sessions.items():
                for mode in ("single", "tiled"):
                    rows.append(
                        {
                            "set": "val",
                            "image": path.stem,
                            "input": variant,
                            "model": model_name,
                            "mode": mode,
                            "count": infer(sess, inp, img, mode),
                            "ref": ref_count(path.stem),
                        }
                    )

    for variant, path in (("normal", sample), ("inverted", sample_inv)):
        img = Image.open(path).convert("RGB")
        for model_name, (sess, inp) in sessions.items():
            for mode in ("single", "tiled"):
                rows.append(
                    {
                        "set": "sample",
                        "image": path.stem,
                        "input": variant,
                        "model": model_name,
                        "mode": mode,
                        "count": infer(sess, inp, img, mode),
                        "ref": 43,
                    }
                )

    def mape(xs):
        vals = [abs(r["count"] - r["ref"]) / r["ref"] for r in xs if r["ref"]]
        return round(100.0 * float(np.mean(vals)), 2) if vals else None

    summary = []
    for model_name in MODELS:
        for variant in ("normal", "inverted"):
            for mode in ("single", "tiled"):
                cell = [
                    r
                    for r in rows
                    if r["model"] == model_name
                    and r["input"] == variant
                    and r["mode"] == mode
                ]
                summary.append(
                    {
                        "model": model_name,
                        "input": variant,
                        "mode": mode,
                        "n": len(cell),
                        "meanCount": round(float(np.mean([r["count"] for r in cell])), 2),
                        "mapePct": mape(cell),
                    }
                )

    payload = {"rows": rows, "summary": summary}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'model':9s} {'input':9s} {'mode':7s} {'n':>2s} {'mean':>7s} {'mape%':>7s}")
    for s in summary:
        print(
            f"{s['model']:9s} {s['input']:9s} {s['mode']:7s} {s['n']:2d} "
            f"{s['meanCount']:7.2f} {str(s['mapePct']):>7s}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
