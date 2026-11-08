"""
Worker for Method 2:
1. `infer`: Runs quantized INT8 ONNX model pack (single 640 pass + automatic 2x2 tiled pass on dense scenes).
2. `train`: Distills a new single-class INT8 ONNX model pack (~2.9 MB) from LLM Validator points.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


def letterbox_640(im: Image.Image, imgsz: int = 640) -> Tuple[np.ndarray, float, int, int]:
    w, h = im.size
    scale = min(imgsz / float(w), imgsz / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x = (imgsz - nw) // 2
    pad_y = (imgsz - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.expand_dims(arr, 0), scale, pad_x, pad_y


def box_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thr]
    return keep


def run_pass(sess, inp_name: str, im: Image.Image, imgsz: int, conf_thr: float) -> List[Tuple[float, float, float]]:
    blob, scale, pad_x, pad_y = letterbox_640(im, imgsz)
    out = sess.run(None, {inp_name: blob})[0]
    preds = np.squeeze(out, axis=0)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    scores = preds[:, 4]
    mask = scores >= conf_thr
    preds = preds[mask]
    scores = scores[mask]
    if len(preds) == 0:
        return []
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - bw / 2.0 - pad_x) / scale
    y1 = (cy - bh / 2.0 - pad_y) / scale
    x2 = (cx + bw / 2.0 - pad_x) / scale
    y2 = (cy + bh / 2.0 - pad_y) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = box_nms(boxes, scores, 0.45)
    w, h = im.size
    pts = []
    for idx in keep:
        px = float(np.clip((boxes[idx, 0] + boxes[idx, 2]) / 2.0, 0, w - 1))
        py = float(np.clip((boxes[idx, 1] + boxes[idx, 3]) / 2.0, 0, h - 1))
        pts.append((px, py, float(scores[idx])))
    return pts


def cmd_infer(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(args.onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    imgsz = int(inp_shape[2]) if isinstance(inp_shape[2], int) else 640

    im = Image.open(args.image).convert("RGB")
    w, h = im.size
    sx = args.orig_width / float(w)
    sy = args.orig_height / float(h)

    single_pts = run_pass(sess, inp_name, im, imgsz, conf_thr=0.25)
    mode = f"single_{imgsz}"
    grid_boxes = None

    # Escalate to 2x2 Tiled ONNX pass when scene is dense
    if len(single_pts) >= 48 or (w >= 900 and len(single_pts) == 0):
        tiled_probe = []
        cell_w, cell_h = w / 2.0, h / 2.0
        pad_x, pad_y = cell_w * 0.08, cell_h * 0.08
        seam_tol = min(cell_w, cell_h) * 0.04
        raw_boxes = []
        for r in range(2):
            for c in range(2):
                cx0, cy0 = c * cell_w, r * cell_h
                cx1, cy1 = (c + 1) * cell_w, (r + 1) * cell_h
                raw_boxes.append([int(round(cx0 * sx)), int(round(cy0 * sy)), int(round(cx1 * sx)), int(round(cy1 * sy))])
                x0 = max(0, int(round(cx0 - (pad_x if c > 0 else 0))))
                y0 = max(0, int(round(cy0 - (pad_y if r > 0 else 0))))
                x1 = min(w, int(round(cx1 + (pad_x if c < 1 else 0))))
                y1 = min(h, int(round(cy1 + (pad_y if r < 1 else 0))))
                crop = im.crop((x0, y0, x1, y1))
                for lx, ly, sc in run_pass(sess, inp_name, crop, imgsz, conf_thr=0.35):
                    gx, gy = x0 + lx, y0 + ly
                    if cx0 - seam_tol <= gx <= cx1 + seam_tol and cy0 - seam_tol <= gy <= cy1 + seam_tol:
                        tiled_probe.append((gx, gy, sc, f"r{r}_c{c}"))
        if len(tiled_probe) > len(single_pts) * 1.35:
            kept = []
            for gx, gy, sc, tid in tiled_probe:
                if not any(math.hypot(gx - kx, gy - ky) < 16.0 for kx, ky, _, _ in kept):
                    kept.append((gx, gy, sc, tid))
            single_pts = [(gx, gy, sc) for gx, gy, sc, _ in kept]
            mode = f"tiled_2x2_{imgsz}+point_nms"
            grid_boxes = raw_boxes

    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
    size_mb = round(Path(args.onnx).stat().st_size / (1024 * 1024), 2)
    points = [{"x": round(px * sx, 2), "y": round(py * sy, 2)} for px, py, _ in single_pts]
    print("ANYCOUNT_JSON:" + json.dumps({
        "count": len(points),
        "points": points,
        "grid_boxes": grid_boxes,
        "elapsed_ms": elapsed_ms,
        "model_size_mb": size_mb,
        "mode": mode,
    }), flush=True)


def cmd_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from onnxruntime.quantization import quantize_dynamic, QuantType

    t0 = time.perf_counter()
    pts_data = json.loads(Path(args.points_json).read_text(encoding="utf-8"))
    pts = pts_data["points"]
    im = Image.open(args.image).convert("RGB")
    w, h = im.size

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "images" / "train").mkdir(parents=True)
    (work / "images" / "val").mkdir(parents=True)
    (work / "labels" / "train").mkdir(parents=True)
    (work / "labels" / "val").mkdir(parents=True)

    # Estimate box size from nearest neighbors
    yolo_lines = []
    for i, p1 in enumerate(pts):
        dists = sorted(
            math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])
            for j, p2 in enumerate(pts) if i != j
        )
        d_nn = dists[min(1, len(dists) - 1)] if dists else min(w, h) * 0.12
        bsize = max(14.0, min(d_nn * 0.90, min(w, h) * 0.35))
        cx = max(0.01, min(0.99, p1["x"] / w))
        cy = max(0.01, min(0.99, p1["y"] / h))
        bw = max(0.02, min(0.95, bsize / w))
        bh = max(0.02, min(0.95, bsize / h))
        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    lbl_text = "\n".join(yolo_lines) + "\n"
    for idx in range(10):
        im.save(work / "images" / "train" / f"sample_{idx}.jpg", quality=92)
        (work / "labels" / "train" / f"sample_{idx}.txt").write_text(lbl_text, encoding="utf-8")
    im.save(work / "images" / "val" / "val_0.jpg", quality=92)
    (work / "labels" / "val" / "val_0.txt").write_text(lbl_text, encoding="utf-8")

    data_yaml = work / "data.yaml"
    data_yaml.write_text(
        f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n  0: {args.slug}\n",
        encoding="utf-8",
    )

    model = YOLO(args.base_weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(work / "runs"),
        name="distill",
        exist_ok=True,
        verbose=False,
    )
    best_pt = work / "runs" / "distill" / "weights" / "best.pt"
    exported = YOLO(str(best_pt)).export(format="onnx", imgsz=640, opset=12)
    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(exported), str(out_onnx), weight_type=QuantType.QInt8)
    size_mb = round(out_onnx.stat().st_size / (1024 * 1024), 2)
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
    shutil.rmtree(work, ignore_errors=True)

    print("ANYCOUNT_JSON:" + json.dumps({
        "onnx_path": str(out_onnx),
        "size_mb": size_mb,
        "epochs": args.epochs,
        "elapsed_ms": elapsed_ms,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--onnx", required=True)
    p_inf.add_argument("--image", required=True)
    p_inf.add_argument("--orig-width", type=int, required=True)
    p_inf.add_argument("--orig-height", type=int, required=True)

    p_trn = sub.add_parser("train")
    p_trn.add_argument("--slug", required=True)
    p_trn.add_argument("--image", required=True)
    p_trn.add_argument("--points-json", required=True)
    p_trn.add_argument("--base-weights", required=True)
    p_trn.add_argument("--out-onnx", required=True)
    p_trn.add_argument("--work-dir", required=True)
    p_trn.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()
    if args.cmd == "infer":
        cmd_infer(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
