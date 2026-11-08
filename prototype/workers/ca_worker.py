"""
Thin worker for Method 1: Count-Anything (SAM3 3.4 GB checkpoint).
Outputs a single JSON line prefixed with ANYCOUNT_JSON: containing pixel coordinates
scaled to target width/height.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--orig-width", type=int, required=True)
    parser.add_argument("--orig-height", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.ca_repo)
    from count_anything import CountAnything

    img_path = Path(args.image).resolve()
    with Image.open(img_path) as im:
        w, h = im.size
    sx = args.orig_width / float(w)
    sy = args.orig_height / float(h)

    t0 = time.perf_counter()
    ckpt = Path(args.ca_repo) / "checkpoints" / "count_anything.pt"
    model = CountAnything(str(ckpt), output_dir=args.work_dir)
    result = model(str(img_path), args.query)[0]
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    pts = [
        {"x": round(float(p["x"]) * sx, 2), "y": round(float(p["y"]) * sy, 2)}
        for p in result.pred_points
    ]
    payload = {
        "count": len(pts),
        "points": pts,
        "elapsed_ms": elapsed_ms,
        "model_size_mb": 3410.0,
    }
    print("ANYCOUNT_JSON:" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
