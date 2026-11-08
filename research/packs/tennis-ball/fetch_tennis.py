"""Fetch permissively licensed tennis-ball photos from Wikimedia Commons."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/root/workspace/anycount/research/packs/tennis-ball/raw")
OUT.mkdir(parents=True, exist_ok=True)
API = "https://commons.wikimedia.org/w/api.php"
UA = "AnyCountResearchBot/1.0 (research prototype; contact rakha@students.amikom.ac.id)"

QUERIES = [
    "tennis ball",
    "tennis balls",
    "tennis balls bucket",
    "tennis balls court",
    "pile of tennis balls",
    "tennis ball hopper",
]
OK_LICENSE = re.compile(r"(cc0|cc[ -]by|public domain|pd-|no restrictions)", re.I)


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def collect() -> list[dict]:
    seen: dict[str, dict] = {}
    for q in QUERIES:
        try:
            data = api(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {q}",
                    "gsrnamespace": "6",
                    "gsrlimit": "40",
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": "1280",
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"search failed {q!r}: {exc}")
            continue
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if int(info.get("width") or 0) < 1000:
                continue
            meta = info.get("extmetadata") or {}
            lic = (meta.get("LicenseShortName") or {}).get("value", "")
            if not OK_LICENSE.search(lic):
                continue
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            seen[page["title"]] = {
                "title": page["title"],
                "url": url,
                "license": lic,
                "artist": re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", ""))[:120],
                "page": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}",
            }
        time.sleep(0.4)
    return list(seen.values())


def main() -> None:
    items = collect()
    print(f"candidates: {len(items)}")
    manifest = []
    for i, item in enumerate(items[:60]):
        name = f"tb_{i:03d}.jpg"
        dest = OUT / name
        if dest.exists():
            manifest.append({**item, "file": name})
            continue
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            manifest.append({**item, "file": name})
        except Exception as exc:  # noqa: BLE001
            print(f"download failed {item['title'][:50]}: {exc}")
        time.sleep(0.2)
    (OUT.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum((OUT / m["file"]).stat().st_size for m in manifest if (OUT / m["file"]).exists())
    print(f"downloaded: {len(manifest)} files, {total/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
