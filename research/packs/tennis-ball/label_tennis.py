"""Teacher-label the tennis-ball dataset with Gemini (full pass + 2x2 tiled rescue)."""
from __future__ import annotations

import base64
import io
import json
import math
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

API_URL = "https://jr.rkhyg.xyz/v1/chat/completions"
MODEL = "gemini-3.8-flash-high"
ROOT = Path("/root/workspace/anycount/research/packs/tennis-ball")
RAW = ROOT / "raw"
OUT = ROOT / "labels"
OUT.mkdir(parents=True, exist_ok=True)
MAX_SIDE = 1280
OVERLAP = 0.08


def api_key() -> str:
    con = sqlite3.connect("/var/lib/9router/db/data.sqlite")
    row = con.execute("select key from apiKeys where name='anycount' and isActive=1").fetchone()
    con.close()
    if not row:
        raise SystemExit("no active api key named anycount")
    return row[0]


KEY = api_key()


def _parse_body(body: str) -> dict:
    """Accept plain JSON or SSE-framed ('data: {...}') gateway responses."""
    stripped = body.lstrip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    chunks = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    if not chunks:
        raise SystemExit(f"unparseable gateway body: {body[:200]!r}")
    merged = {"choices": [{"message": {"content": ""}}]}
    parts: list[str] = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    merged["choices"][0]["message"]["content"] = "".join(parts)
    return merged


def ask(img: Image.Image, query: str = "tennis ball") -> list[list[float]]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        f'Point to every "{query}" in this image.\n'
        "Return one point [y, x] (0 to 1000) per visible instance.\n"
        "IMPORTANT: Do NOT write coordinates in your thinking process. "
        "Keep thinking under 3 sentences, then immediately output the JSON:\n"
        '{"points": [[y1, x1], [y2, x2], ...]}'
    )
    payload = {
        "model": MODEL,
        "reasoning_effort": "low",
        "max_tokens": 4096,
        "stream": False,
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
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read().decode("utf-8", "replace")
    data = _parse_body(body)
    text = data["choices"][0]["message"].get("content") or ""
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text)
    text = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    pts = parsed.get("points") or []
    clean = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            y, x = float(p[0]), float(p[1])
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                clean.append([y, x])
    return clean


def nms(points: list[dict], radius: float) -> list[dict]:
    kept: list[dict] = []
    for p in sorted(points, key=lambda d: d.get("score", 1.0), reverse=True):
        if all(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) > radius for q in kept):
            kept.append(p)
    return kept


def label_one(path: Path) -> dict:
    im = Image.open(path).convert("RGB")
    scale = min(1.0, MAX_SIDE / max(im.size))
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    w, h = im.size

    full = ask(im)
    pts = [{"x": x / 1000 * w, "y": y / 1000 * h, "score": 1.0} for y, x in full]

    used_tiles = False
    if len(pts) < 8 and max(w, h) >= 700:
        used_tiles = True
        ow, oh = round(w * OVERLAP), round(h * OVERLAP)
        tiles = []
        for r in range(2):
            for c in range(2):
                x0 = max(0, round(c * w / 2) - ow)
                x1 = min(w, round((c + 1) * w / 2) + ow)
                y0 = max(0, round(r * h / 2) - oh)
                y1 = min(h, round((r + 1) * h / 2) + oh)
                tiles.append((x0, y0, x1, y1))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda t: (t, ask(im.crop(t))), tiles))
        for (x0, y0, x1, y1), tile_pts in results:
            tw, th = x1 - x0, y1 - y0
            seam_x0, seam_x1 = x0 + tw * 0.06, x1 - tw * 0.06
            seam_y0, seam_y1 = y0 + th * 0.06, y1 - th * 0.06
            for ny, nx in tile_pts:
                gx, gy = x0 + nx / 1000 * tw, y0 + ny / 1000 * th
                core = (seam_x0 <= gx <= seam_x1) and (seam_y0 <= gy <= seam_y1)
                if core or len(tile_pts) < 4:
                    pts.append({"x": gx, "y": gy, "score": 0.9})

    merged = nms(pts, radius=max(4.0, min(w, h) * 0.012))
    return {
        "file": path.name,
        "width": w,
        "height": h,
        "mode": "tiled" if used_tiles else "single",
        "full_count": len(full),
        "count": len(merged),
        "points": [{"x": round(p["x"], 2), "y": round(p["y"], 2)} for p in merged],
    }


def main() -> None:
    files = sorted(RAW.glob("*.jpg"))
    print(f"images: {len(files)}")
    done = 0
    kept = 0
    for path in files:
        dest = OUT / f"{path.stem}.json"
        if dest.exists():
            kept += 1
            done += 1
            continue
        try:
            result = label_one(path)
        except Exception as exc:  # noqa: BLE001
            print(f"fail {path.name}: {exc}")
            continue
        dest.write_text(json.dumps(result, indent=1))
        done += 1
        if result["count"] > 0:
            kept += 1
        print(f"{path.name}: {result['count']} ({result['mode']})")
        time.sleep(0.3)
    print(f"done: {done}, with points: {kept}")


if __name__ == "__main__":
    main()
