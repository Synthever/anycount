"""Compare preprocessing variants (none / grayscale / CLAHE) with the shipped pack.

Offline: the normal tennis-ball pack through the M2/M3 worker functions.
VLM: DeepSeek single-pass through the existing spike-005 harness (temperature 0.1).

Inputs: the 9 held-out validation images and the app's tennis-ball sample. Reference
counts are the pseudo-labels (val) and 43 (sample, benchmark figure, not verified).

Usage: python3 run_variants.py [--out variants.json] [--skip-vlm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PACK = Path("/root/workspace/anycount/research/packs/tennis-ball")
WORKERS = Path("/root/workspace/anycount/prototype/workers")
SAMPLES = Path("/root/workspace/anycount/research/samples")
VLM_SPIKE = Path("/root/workspace/anycount/research/spikes/005-deepseek-single-pass")
PROTO_ENV = Path("/root/workspace/anycount/prototype/.env")
TMP = Path(__file__).with_name("_tmp")
OUT = Path(__file__).with_name("variants.json")

sys.path.insert(0, str(WORKERS))
import onnx_worker as worker  # noqa: E402

MODEL = PACK / "models" / "tennis_ball_v1_int8.onnx"
IMGSZ = 640
CONF = 0.25
VARIANTS = ("normal", "gray", "clahe")


def apply_variant(img: Image.Image, variant: str) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    if variant == "normal":
        return Image.fromarray(rgb)
    if variant == "gray":
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def vlm_env() -> None:
    """Load gateway URL/key from prototype/.env without printing the secret."""
    if os.environ.get("LLM_API_KEY"):
        return
    for line in PROTO_ENV.read_text().splitlines():
        if line.startswith("VLM_API_KEY="):
            os.environ["LLM_API_KEY"] = line.split("=", 1)[1].strip()
        elif line.startswith("VLM_API_URL="):
            os.environ["LLM_API_URL"] = line.split("=", 1)[1].strip()


def ref_count(stem: str) -> int | None:
    lab = PACK / "dataset" / "labels" / "val" / f"{stem}.txt"
    if lab.exists():
        return len([ln for ln in lab.read_text().splitlines() if ln.strip()])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-vlm", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort

    TMP.mkdir(exist_ok=True)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    images = [(p.stem, p, ref_count(p.stem)) for p in sorted((PACK / "dataset" / "images" / "val").glob("*.jpg"))]
    images.append(("tennis-ball", SAMPLES / "tennis-ball.jpg", 43))

    rows = []
    for stem, src, ref in images:
        base_img = Image.open(src).convert("RGB")
        for variant in VARIANTS:
            img = apply_variant(base_img, variant)
            path = TMP / f"{stem}__{variant}.jpg"
            img.save(path, quality=90)

            single = len(worker.run_pass(sess, inp, img, IMGSZ, CONF))
            tiled, _ = worker.run_tiled_pass(sess, inp, img, IMGSZ, 1.0, 1.0)
            rows.append(
                {
                    "image": stem,
                    "variant": variant,
                    "ref": ref,
                    "m2": single,
                    "m3": len(tiled),
                }
            )
            print(f"{stem:14s} {variant:6s} m2={single:3d} m3={len(tiled):3d} ref={ref}", flush=True)

    if not args.skip_vlm:
        vlm_env()
        sys.path.insert(0, str(VLM_SPIKE))
        import run_single_pass_deepseek as vlm  # noqa: E402

        for stem, src, ref in images:
            for variant in VARIANTS:
                path = TMP / f"{stem}__{variant}.jpg"
                try:
                    res = vlm.call_deepseek_points(path, "tennis ball")
                except Exception as exc:  # keep the offline rows even if the gateway fails
                    res = {"count": None, "error": str(exc)}
                m6 = res.get("count")
                for row in rows:
                    if row["image"] == stem and row["variant"] == variant:
                        row["m6"] = m6
                print(f"{stem:14s} {variant:6s} m6={m6}", flush=True)

    def summarize(key: str):
        out = []
        for variant in VARIANTS:
            cell = [r for r in rows if r["variant"] == variant and r.get(key) is not None]
            if not cell:
                continue
            counts = [r[key] for r in cell]
            mape = 100.0 * float(
                np.mean([abs(r[key] - r["ref"]) / r["ref"] for r in cell if r["ref"]])
            )
            out.append(
                {
                    "variant": variant,
                    "n": len(cell),
                    "mean": round(float(np.mean(counts)), 2),
                    "mapePct": round(mape, 2),
                }
            )
        return out

    payload = {"rows": rows, "summary": {k: summarize(k) for k in ("m2", "m3", "m6")}}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print()
    for key in ("m2", "m3", "m6"):
        print(key, json.dumps(payload["summary"][key]))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
