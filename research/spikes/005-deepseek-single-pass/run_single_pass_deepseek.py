"""
Spike 005 (Metode 5): Single-Pass Object Counting & Point Localization via deepseek-v4.1-flash-1017.
Uses a natural two-step count-then-locate prompt requesting normalized [x, y] coordinates (0..1000)
for every object matching text_query, then renders the exact Count-Anything visual overlay style
(red dot + white ring + top-left badge).
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont

API_URL = os.environ.get(
    "LLM_API_URL",
    os.environ.get("GEMINI_API_URL", "https://jr.rkhyg.xyz/v1/chat/completions"),
)
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
MODEL = os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "deepseek-v4.1-flash-1017"))


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
    # 1. Try fenced code blocks first
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = fenced + [text.strip()]
    for cand in candidates:
        cand_strip = cand.strip()
        idx_brace = cand_strip.find("{")
        if idx_brace != -1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cand_strip[idx_brace:])
                if isinstance(parsed, dict) and "points" in parsed and isinstance(parsed["points"], list):
                    return parsed["points"]
            except Exception:
                pass

    # 2. Fallback: extract [x, y] pairs after "points" key if present
    target = text[text.rfind('"points"') :] if '"points"' in text else text
    pairs = re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", target)
    return [[float(x), float(y)] for x, y in pairs]


def parse_llm_response(raw_body: str) -> Tuple[str, Dict[str, Any]]:
    raw_strip = raw_body.lstrip()
    if raw_strip.startswith("{"):
        try:
            data, _ = json.JSONDecoder().raw_decode(raw_strip)
            choices = data.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or choices[0].get("delta") or {}
                raw_content = msg.get("content") or ""
                if isinstance(raw_content, list):
                    content = "".join(
                        part.get("text", "") for part in raw_content if isinstance(part, dict)
                    )
                else:
                    content = str(raw_content)
            usage = data.get("usage") or {}
            return content.strip(), usage
        except Exception:
            pass

    content_parts: List[str] = []
    usage: Dict[str, Any] = {}
    for raw_line in raw_strip.splitlines():
        line = raw_line.strip()
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
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
        if chunk.get("usage"):
            usage = chunk["usage"]

    return "".join(content_parts).strip(), usage


def call_deepseek_points(image_path: Path, text_query: str) -> Dict[str, Any]:
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        f"How many {text_query} are visible in this image? First count them carefully, "
        f"then output a JSON object with the exact count and the center coordinates [x, y] "
        f"(normalized 0 to 1000, where x is horizontal left-to-right and y is vertical top-to-bottom) for each {text_query}:\n"
        '{"count": <int>, "points": [[x1, y1], [x2, y2], ...]}'
    )

    payload = {
        "model": MODEL,
        "max_tokens": 8192,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    last_err: Exception | None = None
    raw_body = ""
    elapsed = 0.0
    for attempt in range(3):
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
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - t0
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                raise last_err

    full_text, usage = parse_llm_response(raw_body)
    raw_points = extract_json_points(full_text)

    img = Image.open(image_path)
    w, h = img.size
    points_px = []
    for item in raw_points:
        if isinstance(item, dict):
            pt = item.get("point") or [item.get("x", 0), item.get("y", 0)]
        else:
            pt = item
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            nx, ny = float(pt[0]), float(pt[1])
            points_px.append({
                "x": round((nx / 1000.0) * w, 2),
                "y": round((ny / 1000.0) * h, 2),
                "norm_x": nx,
                "norm_y": ny,
            })

    return {
        "model": MODEL,
        "count": len(points_px),
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "points": points_px,
    }


def process_task(
    idx: int,
    img_path: Path,
    query: str,
    gt: int,
    out_dir: Path,
    research_root: Path,
) -> Tuple[int, Dict[str, Any]]:
    filename = img_path.name
    slug = f"{img_path.stem}__{query.replace(' ', '_')}"
    out_img = out_dir / f"{slug}.jpg"
    out_json = out_dir / f"{slug}.json"
    if out_json.exists() and out_img.exists():
        res = json.loads(out_json.read_text(encoding="utf-8"))
        print(
            f"[005-deepseek-single] Cached {filename} | query={query!r} -> count={res['count']} ({res['elapsed_s']}s)",
            flush=True,
        )
    else:
        print(f"[005-deepseek-single] Running {filename} | query={query!r} (GT~{gt})...", flush=True)
        res = call_deepseek_points(img_path, query)
        render_count_anything_style(img_path, query, res["points"], out_img)
        out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    err = res["count"] - gt
    abs_pct = round(abs(err) / gt * 100, 1) if gt else 0.0
    usage = res.get("usage") or {}
    row = {
        "image": filename,
        "query": query,
        "gt": gt,
        "pred_count": res["count"],
        "signed_error": err,
        "mape_pct": abs_pct,
        "elapsed_s": res["elapsed_s"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        "output_image": str(out_img.relative_to(research_root)),
    }
    print(
        f"  -> [{filename} | {query}] count={res['count']} (err={err:+d}, MAPE={abs_pct}%), time={res['elapsed_s']}s",
        flush=True,
    )
    return idx, row


def main() -> None:
    research_root = Path(__file__).resolve().parents[2]
    repo_root = research_root.parent
    samples_dir = research_root / "samples"
    test_dir = repo_root / "docs" / "image" / "test"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (samples_dir / "cats.jpg", "cats", 2),
        (samples_dir / "cats.jpg", "remote control", 2),
        (samples_dir / "coins.jpg", "coins", 24),
        (samples_dir / "containers.jpg", "container", 64),
        (samples_dir / "apples.jpg", "apples", 215),
        (test_dir / "tennis-ball.jpg", "tennis ball", 43),
        (test_dir / "Heidelbeeren.jpg", "blueberry", 113),
        (test_dir / "Heidelbeeren-bnw.jpg", "blueberry", 113),
        (test_dir / "Heidelbeeren-bnw2.jpg", "blueberry", 113),
        (test_dir / "candy-ball.jpg", "candy", 132),
        (test_dir / "candy-bnw.jpg", "candy", 132),
    ]

    indexed_rows: List[Tuple[int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(process_task, idx, img_path, query, gt, out_dir, research_root)
            for idx, (img_path, query, gt) in enumerate(tasks)
        ]
        for fut in as_completed(futures):
            indexed_rows.append(fut.result())

    indexed_rows.sort(key=lambda item: item[0])
    summary = [row for _, row in indexed_rows]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
