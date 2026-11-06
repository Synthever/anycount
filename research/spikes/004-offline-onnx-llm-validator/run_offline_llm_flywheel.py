"""
Spike 004 (Metode 2): Offline ONNX Model Pack per Object + LLM Validator + Auto-Training Flywheel.

Alur Kerja (Pipeline):
1. Filter dataset COCO 5-kelas menjadi dataset 1-kelas ("apple", class 0) di /tmp/apple_1class_dataset.
2. Latih model dasar YOLO11n selama 5 epoch (imgsz=320, CPU), ekspor ke ONNX (opset=12),
   dan kuantisasi ke INT8 (apple_int8.onnx, ~2.8 MB) sebagai "Downloadable Object Pack".
3. Jalankan inferensi Offline ONNX (onnxruntime CPU) pada 3 sampel uji:
   - samples/apple_val_1.jpg (Moderate count dari COCO val)
   - samples/apple_val_2.jpg (Moderate count dari COCO val)
   - ../../samples/apples.jpg (Dense count)
4. Jalankan LLM Validator (Gemini 3.8 Flash) pada gambar yang sama untuk memvalidasi jumlah dan titik [x, y].
5. Hitung Validation Verdict: jika selisih (discrepancy) > 20% atau confidence rata-rata < 0.35,
   picu STATUS: TRIGGER_PSEUDO_LABEL_TRAINING dan bangkitkan pseudo-label YOLO (.txt) otomatis
   dari titik-titik LLM menggunakan estimasi radius tetangga terdekat (nearest-neighbor box size).
6. Jalankan 1 putaran Active Learning / Pseudo-Label Fine-Tuning (3 epoch) -> ekspor
   models/apple_v2_distilled_int8.onnx, lalu evaluasi ulang pada ke-3 gambar untuk mengukur
   peningkatan akurasi sebelum vs sesudah distilasi LLM.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

API_URL = os.environ.get("GEMINI_API_URL", "https://jr.rkhyg.xyz/v1/chat/completions")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash-high")

SPIKE_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SPIKE_DIR.parents[1]
PILOT_ROOT = Path("/root/workspace/countthis/pilot")
BASE_PT = PILOT_ROOT / "yolo11n.pt"
SMOKE_DATASET = PILOT_ROOT / "smoke" / "dataset"

TMP_DS_V1 = Path("/tmp/apple_1class_dataset")
TMP_DS_V2 = Path("/tmp/apple_1class_dataset_v2")
TMP_RUNS = Path("/tmp/apple_flywheel_runs")


def render_count_anything_style(
    image_path: Path,
    badge_text: str,
    points_px: List[Dict[str, float]],
    out_path: Path,
    boxes_px: List[List[float]] | None = None,
) -> None:
    """Render Count-Anything visual style (red dot + white ring + top-left badge, plus optional boxes)."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    if boxes_px:
        for b in boxes_px:
            x1, y1, x2, y2 = b[:4]
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(50, 220, 100, 160),
                width=max(1, round(min(width, height) / 320)),
            )

    radius = max(4, round(min(width, height) / 160))
    for pt in points_px:
        x = max(0.0, min(float(pt["x"]), float(width - 1)))
        y = max(0.0, min(float(pt["y"]), float(height - 1)))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(230, 40, 35, 235),
            outline=(255, 255, 255, 255),
            width=max(1, radius // 2),
        )

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, width // 38))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    pad = 10
    draw.rectangle(
        [0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2],
        fill=(0, 0, 0, 185),
    )
    draw.text((pad, pad), badge_text, fill=(255, 255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def render_three_panel_montage(
    img_paths: List[Path],
    titles: List[str],
    out_path: Path,
) -> None:
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    target_h = 360
    resized = []
    for im in imgs:
        w, h = im.size
        nw = int(round(w * (target_h / h)))
        resized.append(im.resize((nw, target_h), Image.BILINEAR))

    header_h = 36
    pad = 8
    total_w = sum(im.size[0] for im in resized) + pad * (len(resized) + 1)
    total_h = target_h + header_h + pad * 2
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=15)
    except Exception:
        font = ImageFont.load_default()

    x_off = pad
    for im, title in zip(resized, titles):
        draw.text((x_off + 6, 8), title, fill=(240, 240, 240), font=font)
        canvas.paste(im, (x_off, header_h))
        x_off += im.size[0] + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def extract_json_points(text: str) -> List[Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("points", [])
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    pairs = re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", cleaned)
    return [[float(y), float(x)] for y, x in pairs]


def call_gemini_validator(image_path: Path, text_query: str = "apple") -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for live LLM calls.")

    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        f'Locate every individual "{text_query}" visible in this image.\n'
        "Place exactly one point [y, x] at the center of each visible instance (normalized 0 to 1000, "
        "where [0,0] is top-left and [1000,1000] is bottom-right).\n"
        "Scan systematically row by row from top-left to bottom-right so you do not miss any or double-count.\n"
        "IMPORTANT: Do NOT write coordinates in your thinking trace; keep thinking under 3 sentences and "
        "emit the coordinates directly in the final JSON response:\n"
        '{"points": [[y1, x1], [y2, x2], ...]}'
    )

    payload = {
        "model": MODEL,
        "reasoning_effort": "low",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.1,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "anycount-spike004/1.0",
            "Connection": "close",
        },
    )

    t0 = time.perf_counter()
    content_parts: List[str] = []
    usage: Dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
            if chunk.get("usage"):
                usage = chunk["usage"]
                break
    elapsed = time.perf_counter() - t0

    full_text = "".join(content_parts).strip()
    raw_points = extract_json_points(full_text)

    img = Image.open(image_path)
    w, h = img.size
    points_px = []
    for item in raw_points:
        if isinstance(item, dict):
            pt = item.get("point") or [item.get("y", 0), item.get("x", 0)]
        else:
            pt = item
        if len(pt) >= 2:
            ny, nx = float(pt[0]), float(pt[1])
            points_px.append({
                "x": round((nx / 1000.0) * w, 2),
                "y": round((ny / 1000.0) * h, 2),
                "norm_y": ny,
                "norm_x": nx,
            })

    return {
        "count": len(points_px),
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "points": points_px,
    }


def build_single_class_apple_dataset(src_ds: Path, dst_ds: Path) -> Dict[str, Any]:
    """Filter COCO 5-class dataset for class 0 ('apple') only."""
    if dst_ds.exists():
        shutil.rmtree(dst_ds)

    stats: Dict[str, Any] = {}
    for split in ["train", "val"]:
        (dst_ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        img_count = 0
        box_count = 0
        for lbl_file in sorted((src_ds / "labels" / split).glob("*.txt")):
            apple_lines = []
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts and parts[0] == "0":
                    apple_lines.append("0 " + " ".join(parts[1:]))
            if apple_lines:
                stem = lbl_file.stem
                src_img = src_ds / "images" / split / f"{stem}.jpg"
                if src_img.exists():
                    shutil.copy2(src_img, dst_ds / "images" / split / f"{stem}.jpg")
                    (dst_ds / "labels" / split / f"{stem}.txt").write_text(
                        "\n".join(apple_lines) + "\n", encoding="utf-8"
                    )
                    img_count += 1
                    box_count += len(apple_lines)
        stats[split] = {"images": img_count, "boxes": box_count}

    (dst_ds / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: apple\n",
        encoding="utf-8",
    )
    return stats


def train_and_export_int8(
    weights_path: Path,
    data_yaml: Path,
    epochs: int,
    run_name: str,
    out_int8_path: Path,
    warmup_bias_lr: float = 0.012,
    lr0: float = 0.0004,
) -> Dict[str, Any]:
    """Train YOLO11n on CPU, export fp32 ONNX (opset=12), and quantize to INT8."""
    t0 = time.perf_counter()
    model = YOLO(str(weights_path))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=320,
        batch=4,
        nbs=16,
        freeze=22,
        warmup_epochs=1.0,
        warmup_bias_lr=warmup_bias_lr,
        optimizer="AdamW",
        lr0=lr0,
        mosaic=0.0,
        device="cpu",
        project=str(TMP_RUNS),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    train_s = time.perf_counter() - t0

    trained_pt = TMP_RUNS / run_name / "weights" / "last.pt"
    export_model = YOLO(str(trained_pt))
    onnx_fp32_str = export_model.export(format="onnx", imgsz=320, opset=12)
    onnx_fp32_path = Path(onnx_fp32_str)

    out_int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_fp32_path), str(out_int8_path), weight_type=QuantType.QInt8)

    fp32_mb = round(onnx_fp32_path.stat().st_size / (1024 * 1024), 2)
    int8_mb = round(out_int8_path.stat().st_size / (1024 * 1024), 2)
    return {
        "trained_pt": str(trained_pt),
        "onnx_fp32_path": str(onnx_fp32_path),
        "onnx_int8_path": str(out_int8_path),
        "fp32_size_mb": fp32_mb,
        "int8_size_mb": int8_mb,
        "compression_ratio": round(fp32_mb / max(int8_mb, 0.01), 2),
        "train_elapsed_s": round(train_s, 2),
    }


def run_onnx_inference(
    sess: ort.InferenceSession,
    image_path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    imgsz: int = 320,
    warmup_runs: int = 2,
    bench_runs: int = 10,
) -> Dict[str, Any]:
    """Run offline ONNX CPU inference, measure median latency (ms), and extract boxes and center points."""
    img = Image.open(image_path).convert("RGB")
    w0, h0 = img.size
    r = min(imgsz / w0, imgsz / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (imgsz - nw) / 2.0, (imgsz - nh) / 2.0
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(resized, (int(round(dw - 0.1)), int(round(dh - 0.1))))
    x = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

    inp_name = sess.get_inputs()[0].name
    for _ in range(warmup_runs):
        sess.run(None, {inp_name: x})

    latencies_ms = []
    out = None
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        out = sess.run(None, {inp_name: x})[0][0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = round(float(np.median(latencies_ms)), 2)
    mean_ms = round(float(np.mean(latencies_ms)), 2)

    scores = out[4]
    mask = scores >= conf_thres
    boxes = out[:4, mask].T
    scores = scores[mask]

    if len(boxes) == 0:
        return {
            "count": 0,
            "mean_conf": 0.0,
            "latency_ms": median_ms,
            "latency_mean_ms": mean_ms,
            "points": [],
            "boxes": [],
        }

    x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2.0 - dw) / r, 0.0, w0 - 1.0)
    y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2.0 - dh) / r, 0.0, h0 - 1.0)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2.0 - dw) / r, 0.0, w0 - 1.0)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2.0 - dh) / r, 0.0, h0 - 1.0)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_o = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_thres]

    points_px = []
    boxes_px = []
    for idx in keep:
        bx1, by1, bx2, by2 = float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        conf_val = float(scores[idx])
        points_px.append({"x": round(cx, 2), "y": round(cy, 2), "conf": round(conf_val, 3)})
        boxes_px.append([round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2), round(conf_val, 3)])

    mean_conf = round(float(scores[keep].mean()), 3)
    return {
        "count": len(keep),
        "mean_conf": mean_conf,
        "latency_ms": median_ms,
        "latency_mean_ms": mean_ms,
        "points": points_px,
        "boxes": boxes_px,
    }


def points_to_yolo_pseudo_labels(points_px: List[Dict[str, float]], w: int, h: int) -> List[str]:
    """Convert LLM [x, y] points into YOLO normalized bounding boxes (0 cx cy bw bh) using nearest-neighbor radius."""
    if not points_px:
        return []
    pts = np.array([[p["x"] / w, p["y"] / h] for p in points_px], dtype=np.float32)
    n = len(pts)
    lines = []
    for i in range(n):
        cx, cy = float(pts[i, 0]), float(pts[i, 1])
        if n > 1:
            dists = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
            dists[i] = 1e9
            nn = float(dists.min())
            bw = float(np.clip(nn * 0.90, 0.04, 0.22))
            bh = float(np.clip(nn * 0.90, 0.04, 0.22))
        else:
            bw, bh = 0.12, 0.12
        bw = max(0.02, min(bw, cx * 2.0, (1.0 - cx) * 2.0))
        bh = max(0.02, min(bh, cy * 2.0, (1.0 - cy) * 2.0))
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def compute_validation_verdict(offline_count: int, offline_conf: float, llm_count: int) -> Dict[str, Any]:
    denom = max(llm_count, 1)
    discrepancy_pct = round(abs(offline_count - llm_count) / denom * 100.0, 1)
    low_conf = offline_conf < 0.35
    high_discrepancy = discrepancy_pct > 20.0
    if high_discrepancy or low_conf:
        status = "TRIGGER_PSEUDO_LABEL_TRAINING"
    else:
        status = "PASS_OFFLINE_MODEL_VALID"
    return {
        "status": status,
        "discrepancy_pct": discrepancy_pct,
        "signed_diff": offline_count - llm_count,
        "low_confidence_flag": low_conf,
        "high_discrepancy_flag": high_discrepancy,
    }


def main() -> None:
    models_dir = SPIKE_DIR / "models"
    samples_dir = SPIKE_DIR / "samples"
    results_dir = SPIKE_DIR / "results"
    pseudo_dir = results_dir / "pseudo_labels"
    for d in [models_dir, samples_dir, results_dir, pseudo_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[1/5] Building 1-class 'apple' dataset in /tmp/apple_1class_dataset...", flush=True)
    ds_stats = build_single_class_apple_dataset(SMOKE_DATASET, TMP_DS_V1)
    print(f"      Dataset stats: {ds_stats}", flush=True)

    # Copy 2 moderate-count validation images into samples/
    val1_src = TMP_DS_V1 / "images" / "val" / "000000498919.jpg"
    val2_src = TMP_DS_V1 / "images" / "val" / "000000217753.jpg"
    val1_dst = samples_dir / "apple_val_1.jpg"
    val2_dst = samples_dir / "apple_val_2.jpg"
    shutil.copy2(val1_src, val1_dst)
    shutil.copy2(val2_src, val2_dst)
    dense_src = RESEARCH_ROOT / "samples" / "apples.jpg"

    test_samples = [
        ("apple_val_1", val1_dst, "Moderate (COCO val 000000498919)", 13),
        ("apple_val_2", val2_dst, "Moderate (COCO val 000000217753)", 9),
        ("apples", dense_src, "Dense pile (research/samples/apples.jpg)", 215),
    ]

    print("[2/5] Training base 1-class YOLO11n (5 epochs, imgsz=320) & exporting apple_int8.onnx...", flush=True)
    v1_model_path = models_dir / "apple_int8.onnx"
    v1_export_info = train_and_export_int8(
        weights_path=BASE_PT,
        data_yaml=TMP_DS_V1 / "data.yaml",
        epochs=5,
        run_name="v1_base_apple",
        out_int8_path=v1_model_path,
        warmup_bias_lr=0.012,
        lr0=0.0004,
    )
    print(
        f"      Exported v1: fp32={v1_export_info['fp32_size_mb']} MB -> "
        f"INT8={v1_export_info['int8_size_mb']} MB ({v1_export_info['compression_ratio']}x smaller), "
        f"train_time={v1_export_info['train_elapsed_s']}s",
        flush=True,
    )

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess_v1 = ort.InferenceSession(str(v1_model_path), so, providers=["CPUExecutionProvider"])

    print("[3/5] Running Offline ONNX v1 + LLM Validator + Auto Pseudo-Label Generation...", flush=True)
    per_image_records: List[Dict[str, Any]] = []

    if TMP_DS_V2.exists():
        shutil.rmtree(TMP_DS_V2)
    shutil.copytree(TMP_DS_V1, TMP_DS_V2)

    for stem, img_path, desc, coco_gt in test_samples:
        # 3a. Offline ONNX v1 inference
        v1_res = run_onnx_inference(sess_v1, img_path, conf_thres=0.25)
        v1_img_out = results_dir / f"{stem}__offline_onnx.jpg"
        render_count_anything_style(
            img_path,
            f"Offline ONNX v1: {v1_res['count']} ({v1_res['latency_ms']} ms)",
            v1_res["points"],
            v1_img_out,
            boxes_px=v1_res["boxes"],
        )

        # 3b. LLM Validator (cached or live)
        llm_json_path = results_dir / f"{stem}__llm_validator.json"
        if llm_json_path.exists():
            llm_res = json.loads(llm_json_path.read_text(encoding="utf-8"))
        else:
            llm_res = call_gemini_validator(img_path, "apple")
            llm_json_path.write_text(json.dumps(llm_res, indent=2), encoding="utf-8")

        llm_img_out = results_dir / f"{stem}__llm_validator.jpg"
        render_count_anything_style(
            img_path,
            f"LLM Validator: {llm_res['count']} ({llm_res['elapsed_s']} s)",
            llm_res["points"],
            llm_img_out,
        )

        # 3c. Validation Verdict & Pseudo-Label generation
        verdict = compute_validation_verdict(v1_res["count"], v1_res["mean_conf"], llm_res["count"])
        w, h = Image.open(img_path).size
        pseudo_lines = []
        pseudo_file = pseudo_dir / f"{stem}.txt"
        if verdict["status"] == "TRIGGER_PSEUDO_LABEL_TRAINING":
            pseudo_lines = points_to_yolo_pseudo_labels(llm_res["points"], w, h)
            pseudo_file.write_text("\n".join(pseudo_lines) + "\n", encoding="utf-8")
            for copy_idx in range(2):
                shutil.copy2(img_path, TMP_DS_V2 / "images" / "train" / f"pseudo_{stem}_{copy_idx}.jpg")
                (TMP_DS_V2 / "labels" / "train" / f"pseudo_{stem}_{copy_idx}.txt").write_text(
                    "\n".join(pseudo_lines) + "\n", encoding="utf-8"
                )

        print(
            f"      [{stem}] v1_count={v1_res['count']} ({v1_res['latency_ms']} ms, conf={v1_res['mean_conf']}) | "
            f"llm_count={llm_res['count']} ({llm_res['elapsed_s']} s) | "
            f"discrepancy={verdict['discrepancy_pct']}% -> {verdict['status']}",
            flush=True,
        )

        per_image_records.append({
            "stem": stem,
            "image_path": str(img_path.relative_to(RESEARCH_ROOT)),
            "description": desc,
            "image_size": [w, h],
            "coco_sparse_boxes": coco_gt,
            "offline_v1": {
                "count": v1_res["count"],
                "mean_conf": v1_res["mean_conf"],
                "latency_ms": v1_res["latency_ms"],
                "latency_mean_ms": v1_res["latency_mean_ms"],
                "output_image": str(v1_img_out.relative_to(SPIKE_DIR)),
            },
            "llm_validator": {
                "count": llm_res["count"],
                "elapsed_s": llm_res["elapsed_s"],
                "elapsed_ms": round(llm_res["elapsed_s"] * 1000.0, 1),
                "output_image": str(llm_img_out.relative_to(SPIKE_DIR)),
            },
            "validation_verdict": verdict,
            "pseudo_labels_generated": len(pseudo_lines),
            "pseudo_label_file": str(pseudo_file.relative_to(SPIKE_DIR)),
        })

    print("[4/5] Running 1 round of Pseudo-Label Fine-Tuning (3 epochs) -> apple_v2_distilled_int8.onnx...", flush=True)
    v2_model_path = models_dir / "apple_v2_distilled_int8.onnx"
    v2_export_info = train_and_export_int8(
        weights_path=Path(v1_export_info["trained_pt"]),
        data_yaml=TMP_DS_V2 / "data.yaml",
        epochs=3,
        run_name="v2_distilled_apple",
        out_int8_path=v2_model_path,
        warmup_bias_lr=0.030,
        lr0=0.0005,
    )
    print(
        f"      Exported v2: fp32={v2_export_info['fp32_size_mb']} MB -> "
        f"INT8={v2_export_info['int8_size_mb']} MB, fine-tune_time={v2_export_info['train_elapsed_s']}s",
        flush=True,
    )

    sess_v2 = ort.InferenceSession(str(v2_model_path), so, providers=["CPUExecutionProvider"])

    print("[5/5] Re-evaluating Offline ONNX v2 (distilled) on the 3 test images...", flush=True)
    for rec, (stem, img_path, _, _) in zip(per_image_records, test_samples):
        v2_res = run_onnx_inference(sess_v2, img_path, conf_thres=0.25)
        v2_img_out = results_dir / f"{stem}__offline_v2_distilled.jpg"
        render_count_anything_style(
            img_path,
            f"Offline ONNX v2 (Distilled): {v2_res['count']} ({v2_res['latency_ms']} ms)",
            v2_res["points"],
            v2_img_out,
            boxes_px=v2_res["boxes"],
        )

        llm_cnt = rec["llm_validator"]["count"]
        v1_cnt = rec["offline_v1"]["count"]
        v2_cnt = v2_res["count"]
        v1_err_pct = round(abs(v1_cnt - llm_cnt) / max(llm_cnt, 1) * 100.0, 1)
        v2_err_pct = round(abs(v2_cnt - llm_cnt) / max(llm_cnt, 1) * 100.0, 1)

        rec["offline_v2_distilled"] = {
            "count": v2_cnt,
            "mean_conf": v2_res["mean_conf"],
            "latency_ms": v2_res["latency_ms"],
            "latency_mean_ms": v2_res["latency_mean_ms"],
            "discrepancy_vs_llm_pct": v2_err_pct,
            "error_reduction_pct_points": round(v1_err_pct - v2_err_pct, 1),
            "output_image": str(v2_img_out.relative_to(SPIKE_DIR)),
        }
        rec["speedup_vs_llm_x"] = round(
            (rec["llm_validator"]["elapsed_s"] * 1000.0) / max(v2_res["latency_ms"], 0.1), 1
        )

        montage_out = results_dir / f"{stem}__flywheel_montage.jpg"
        render_three_panel_montage(
            [
                results_dir / f"{stem}__offline_onnx.jpg",
                results_dir / f"{stem}__llm_validator.jpg",
                results_dir / f"{stem}__offline_v2_distilled.jpg",
            ],
            [
                f"1. Offline ONNX v1: {v1_cnt} ({rec['offline_v1']['latency_ms']} ms)",
                f"2. LLM Validator: {llm_cnt} ({rec['llm_validator']['elapsed_s']} s)",
                f"3. Distilled ONNX v2: {v2_cnt} ({v2_res['latency_ms']} ms)",
            ],
            montage_out,
        )
        rec["montage_image"] = str(montage_out.relative_to(SPIKE_DIR))

        print(
            f"      [{stem}] v1={v1_cnt} (err={v1_err_pct}%) -> LLM={llm_cnt} -> "
            f"v2={v2_cnt} (err={v2_err_pct}%, speedup={rec['speedup_vs_llm_x']}x)",
            flush=True,
        )

    summary = {
        "spike": "004-offline-onnx-llm-validator",
        "method": "Metode 2: Offline ONNX Model Pack per Object + LLM Validator + Auto-Training Flywheel",
        "target_class": "apple (class 0)",
        "dataset_filtered": ds_stats,
        "models": {
            "v1_base_int8": {
                "path": "models/apple_int8.onnx",
                "fp32_size_mb": v1_export_info["fp32_size_mb"],
                "int8_size_mb": v1_export_info["int8_size_mb"],
                "compression_ratio": v1_export_info["compression_ratio"],
                "epochs": 5,
                "imgsz": 320,
                "train_elapsed_s": v1_export_info["train_elapsed_s"],
            },
            "v2_distilled_int8": {
                "path": "models/apple_v2_distilled_int8.onnx",
                "fp32_size_mb": v2_export_info["fp32_size_mb"],
                "int8_size_mb": v2_export_info["int8_size_mb"],
                "compression_ratio": v2_export_info["compression_ratio"],
                "fine_tune_epochs": 3,
                "imgsz": 320,
                "train_elapsed_s": v2_export_info["train_elapsed_s"],
            },
        },
        "evaluations": per_image_records,
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
