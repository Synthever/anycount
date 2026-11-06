"""
Spike 002: Parallel Tiled Grid + Seam Point-NMS via Gemini VLM.
Splits dense scenes into overlapping tiles, queries Gemini concurrently across all tiles,
projects tile-local [y, x] back to global image coordinates, and applies Core-Region
Filtering + Radius Point-NMS to eliminate seam duplicates.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

API_URL = os.environ.get("GEMINI_API_URL", "https://jr.rkhyg.xyz/v1/chat/completions")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash-high")


def render_count_anything_style(
    image_path: Path,
    text_query: str,
    points_px: List[Dict[str, float]],
    out_path: Path,
    grid_boxes: List[Tuple[int, int, int, int]] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    if grid_boxes:
        for (x0, y0, x1, y1) in grid_boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 90), width=1)

    radius = max(4, round(min(width, height) / 180))
    for pt in points_px:
        x = max(0.0, min(float(pt["x"]), float(width - 1)))
        y = max(0.0, min(float(pt["y"]), float(height - 1)))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(230, 40, 35, 230),
            outline=(255, 255, 255, 255),
            width=max(1, radius // 2),
        )
    label = f"{text_query}: {len(points_px)}"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, width // 45))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 10
    draw.rectangle(
        [0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2],
        fill=(0, 0, 0, 170),
    )
    draw.text((pad, pad), label, fill=(255, 255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


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


def query_tile(
    tile_img: Image.Image,
    tile_info: Dict[str, Any],
    text_query: str,
) -> Dict[str, Any]:
    buf = io.BytesIO()
    tile_img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f'Point to every "{text_query}" in this image.\n'
        "Return one point [y, x] (0 to 1000) per visible instance.\n"
        "IMPORTANT: Do NOT write coordinates in your thinking process. "
        "Keep thinking under 3 sentences, then immediately output the JSON:\n"
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
            "User-Agent": "anycount-spike/1.0",
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

    raw_points = extract_json_points("".join(content_parts))
    tw, th = tile_img.size
    crop_x0, crop_y0, crop_x1, crop_y1 = tile_info["crop_box"]
    core_x0, core_y0, core_x1, core_y1 = tile_info["core_box"]
    seam_tol = tile_info["seam_tolerance_px"]

    global_points = []
    for item in raw_points:
        pt = item.get("point") if isinstance(item, dict) else item
        if not pt or len(pt) < 2:
            continue
        ny, nx = float(pt[0]), float(pt[1])
        gx = crop_x0 + (nx / 1000.0) * tw
        gy = crop_y0 + (ny / 1000.0) * th
        if (
            core_x0 - seam_tol <= gx <= core_x1 + seam_tol
            and core_y0 - seam_tol <= gy <= core_y1 + seam_tol
        ):
            global_points.append({
                "x": round(gx, 2),
                "y": round(gy, 2),
                "tile_id": tile_info["tile_id"],
            })

    return {
        "tile_id": tile_info["tile_id"],
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "points": global_points,
    }


def point_nms(points: List[Dict[str, Any]], min_dist_px: float) -> List[Dict[str, Any]]:
    """Merge duplicate detections across tile seams and within tiles if closer than min_dist_px."""
    kept: List[Dict[str, Any]] = []
    for pt in points:
        merged = False
        for k in kept:
            dist = math.hypot(pt["x"] - k["x"], pt["y"] - k["y"])
            thresh = min_dist_px * (1.35 if pt.get("tile_id") != k.get("tile_id") else 0.75)
            if dist < thresh:
                k["x"] = round((k["x"] + pt["x"]) / 2.0, 2)
                k["y"] = round((k["y"] + pt["y"]) / 2.0, 2)
                merged = True
                break
        if not merged:
            kept.append(dict(pt))
    return kept


def run_tiled_counting(
    image_path: Path,
    text_query: str,
    rows: int = 2,
    cols: int = 2,
    overlap_ratio: float = 0.08,
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    cell_w = w / cols
    cell_h = h / rows
    pad_x = cell_w * overlap_ratio
    pad_y = cell_h * overlap_ratio
    seam_tol = min(cell_w, cell_h) * 0.04

    tiles = []
    core_boxes = []
    for r in range(rows):
        for c in range(cols):
            core_x0 = c * cell_w
            core_y0 = r * cell_h
            core_x1 = (c + 1) * cell_w
            core_y1 = (r + 1) * cell_h
            core_boxes.append((int(core_x0), int(core_y0), int(core_x1), int(core_y1)))

            crop_x0 = max(0, int(round(core_x0 - (pad_x if c > 0 else 0))))
            crop_y0 = max(0, int(round(core_y0 - (pad_y if r > 0 else 0))))
            crop_x1 = min(w, int(round(core_x1 + (pad_x if c < cols - 1 else 0))))
            crop_y1 = min(h, int(round(core_y1 + (pad_y if r < rows - 1 else 0))))

            tile_crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            tiles.append((
                tile_crop,
                {
                    "tile_id": f"r{r}_c{c}",
                    "crop_box": (crop_x0, crop_y0, crop_x1, crop_y1),
                    "core_box": (core_x0, core_y0, core_x1, core_y1),
                    "seam_tolerance_px": seam_tol,
                },
            ))

    t0 = time.perf_counter()
    all_raw_points: List[Dict[str, Any]] = []
    tile_stats = []
    total_prompt_tok = 0
    total_comp_tok = 0
    total_reason_tok = 0

    with ThreadPoolExecutor(max_workers=rows * cols) as pool:
        futs = [pool.submit(query_tile, t_img, t_info, text_query) for t_img, t_info in tiles]
        for fut in as_completed(futs):
            res = fut.result()
            all_raw_points.extend(res["points"])
            u = res.get("usage") or {}
            total_prompt_tok += u.get("prompt_tokens", 0)
            total_comp_tok += u.get("completion_tokens", 0)
            total_reason_tok += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            tile_stats.append({
                "tile_id": res["tile_id"],
                "raw_count": len(res["points"]),
                "elapsed_s": res["elapsed_s"],
            })

    wall_elapsed = time.perf_counter() - t0

    if len(all_raw_points) >= 4:
        nn_dists = []
        for i, p1 in enumerate(all_raw_points):
            dmin = min(
                math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])
                for j, p2 in enumerate(all_raw_points)
                if i != j
            )
            if dmin > 1.0:
                nn_dists.append(dmin)
        nn_dists.sort()
        median_nn = nn_dists[len(nn_dists) // 2] if nn_dists else 20.0
        nms_radius = max(8.0, min(median_nn * 0.55, min(w, h) * 0.06))
    else:
        nms_radius = min(w, h) * 0.04

    dedup_points = point_nms(all_raw_points, min_dist_px=nms_radius)

    return {
        "grid": f"{rows}x{cols}",
        "raw_count": len(all_raw_points),
        "count": len(dedup_points),
        "nms_radius_px": round(nms_radius, 2),
        "wall_elapsed_s": round(wall_elapsed, 2),
        "prompt_tokens": total_prompt_tok,
        "completion_tokens": total_comp_tok,
        "reasoning_tokens": total_reason_tok,
        "tile_stats": sorted(tile_stats, key=lambda x: x["tile_id"]),
        "core_boxes": core_boxes,
        "points": dedup_points,
    }


def main() -> None:
    research_root = Path(__file__).resolve().parents[2]
    repo_root = research_root.parent
    samples_dir = research_root / "samples"
    test_dir = repo_root / "docs" / "image" / "test"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (samples_dir / "coins.jpg", "coins", 24, 2, 2),
        (samples_dir / "containers.jpg", "container", 64, 2, 2),
        (samples_dir / "apples.jpg", "apples", 215, 2, 2),
        (test_dir / "tennis-ball.jpg", "tennis ball", 43, 2, 2),
        (test_dir / "Heidelbeeren.jpg", "blueberry", 113, 2, 2),
        (test_dir / "Heidelbeeren-bnw.jpg", "blueberry", 113, 2, 2),
        (test_dir / "Heidelbeeren-bnw2.jpg", "blueberry", 113, 2, 2),
        (test_dir / "candy-ball.jpg", "candy", 132, 2, 2),
        (test_dir / "candy-bnw.jpg", "candy", 132, 2, 2),
    ]

    summary = []
    for img_path, query, gt, r, c in tasks:
        filename = img_path.name
        slug = f"{img_path.stem}__{query.replace(' ', '_')}__{r}x{c}"
        out_img = out_dir / f"{slug}.jpg"
        out_json = out_dir / f"{slug}.json"
        if out_json.exists() and out_img.exists():
            res = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[002-tiled] Cached {filename} ({r}x{c}) | query={query!r} -> count={res['count']} ({res['wall_elapsed_s']}s)", flush=True)
        else:
            print(f"[002-tiled] Running {filename} ({r}x{c}) | query={query!r} (GT~{gt})...", flush=True)
            res = run_tiled_counting(img_path, query, rows=r, cols=c)
            render_count_anything_style(
                img_path, query, res["points"], out_img, grid_boxes=res["core_boxes"]
            )
            out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

        err = res["count"] - gt
        abs_pct = round(abs(err) / gt * 100, 1) if gt else 0.0
        row = {
            "image": filename,
            "query": query,
            "grid": f"{r}x{c}",
            "gt": gt,
            "raw_count": res["raw_count"],
            "pred_count": res["count"],
            "signed_error": err,
            "mape_pct": abs_pct,
            "wall_elapsed_s": res["wall_elapsed_s"],
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "reasoning_tokens": res["reasoning_tokens"],
            "output_image": str(out_img.relative_to(research_root)),
        }
        summary.append(row)
        print(
            f"  -> raw={res['raw_count']} -> dedup_count={res['count']} "
            f"(err={err:+d}, MAPE={abs_pct}%), wall_time={res['wall_elapsed_s']}s",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
