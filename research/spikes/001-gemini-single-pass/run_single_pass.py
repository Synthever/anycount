"""
Spike 001: Single-Pass Object Counting & Point Localization via Gemini VLM.
Requests normalized [y, x] coordinates (0..1000) for every object matching text_query,
then renders the exact Count-Anything visual overlay style (red dot + white ring + top-left badge).
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image, ImageDraw, ImageFont

API_URL = os.environ.get("GEMINI_API_URL", "https://jr.rkhyg.xyz/v1/chat/completions")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash-high")


def render_count_anything_style(
    image_path: Path,
    text_query: str,
    points_px: List[Dict[str, float]],
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
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


def call_gemini_points(image_path: Path, text_query: str) -> Dict[str, Any]:
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        f'Locate every individual "{text_query}" visible in this image.\n'
        "Place exactly one point [y, x] at the center of each visible instance (normalized 0 to 1000, "
        "where [0,0] is top-left and [1000,1000] is bottom-right).\n"
        "Scan systematically row by row from top-left to bottom-right so you do not miss any or double-count.\n"
        "IMPORTANT: Keep internal thinking very brief (do NOT write out coordinates in your thinking trace; "
        "emit the coordinates directly in the final JSON response).\n"
        "Return ONLY valid JSON with no markdown fences in this exact schema:\n"
        '{"points": [[y1, x1], [y2, x2], ...]}'
    )

    payload = {
        "model": MODEL,
        "reasoning_effort": "low",
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
    with urllib.request.urlopen(req, timeout=180) as resp:
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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    samples_dir = root / "samples"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("cats.jpg", "cats", 2),
        ("cats.jpg", "remote control", 2),
        ("coins.jpg", "coins", 24),
        ("containers.jpg", "container", 64),
        ("apples.jpg", "apples", 215),
    ]

    summary = []
    for filename, query, gt in tasks:
        img_path = samples_dir / filename
        slug = f"{img_path.stem}__{query.replace(' ', '_')}"
        out_img = out_dir / f"{slug}.jpg"
        out_json = out_dir / f"{slug}.json"
        if out_json.exists() and out_img.exists():
            res = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[001-single] Cached {filename} | query={query!r} -> count={res['count']} ({res['elapsed_s']}s)", flush=True)
        else:
            print(f"[001-single] Running {filename} | query={query!r} (GT~{gt})...", flush=True)
            res = call_gemini_points(img_path, query)
            render_count_anything_style(img_path, query, res["points"], out_img)
            out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

        err = res["count"] - gt
        abs_pct = round(abs(err) / gt * 100, 1) if gt else 0.0
        row = {
            "image": filename,
            "query": query,
            "gt": gt,
            "pred_count": res["count"],
            "signed_error": err,
            "mape_pct": abs_pct,
            "elapsed_s": res["elapsed_s"],
            "prompt_tokens": res["usage"].get("prompt_tokens", 0),
            "completion_tokens": res["usage"].get("completion_tokens", 0),
            "reasoning_tokens": (res["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "output_image": str(out_img.relative_to(root)),
        }
        summary.append(row)
        print(f"  -> count={res['count']} (err={err:+d}, MAPE={abs_pct}%), time={res['elapsed_s']}s, reasoning_tok={row['reasoning_tokens']}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
