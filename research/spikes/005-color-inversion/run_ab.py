#!/usr/bin/env python3
"""Spike 005 A/B sweep: baseline vs RGB-negative inversion.

Reads ANYCOUNT_PASSCODE from prototype/.env (never printed), unlocks a session on
the local service, then runs each (sample, method, invert, repetition) sequentially
and stores counts in ab_runs.json next to this script.

Usage: python3 run_ab.py [--vlm-reps N]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:2030"
ENV_PATH = Path("/root/workspace/anycount/prototype/.env")
OUT_PATH = Path(__file__).with_name("ab_runs.json")

# Offline packs exist for apple / cat / tennis-ball only; other classes cold-start.
OFFLINE = [("sample_cats", "cats", ["m2", "m3"])]
# VLM methods accept any query; m6 (DeepSeek single pass) is the fast one.
VLM = [
    ("sample_coins", "coins"),
    ("sample_containers", "container"),
    ("sample_blueberry", "blueberry"),
    ("sample_candy", "candy"),
    ("sample_apples", "apples"),
]
TERMINAL = {"completed", "partial", "failed"}


def passcode() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("ANYCOUNT_PASSCODE="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ANYCOUNT_PASSCODE missing from prototype/.env")


def unlock() -> str:
    """The session cookie is flagged Secure, so cookiejar would drop it over
    loopback HTTP; carry the Set-Cookie value explicitly instead."""
    body = json.dumps({"passcode": passcode()}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/auth/unlock",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as res:
        if res.status != 200:
            raise SystemExit(f"unlock failed: {res.status}")
        raw = res.headers.get("Set-Cookie", "")
    if not raw:
        raise SystemExit("unlock returned no session cookie")
    return raw.split(";", 1)[0]


COOKIE = ""


def call(method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Cookie": COOKIE}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


def wait_run(run_id: str, timeout_s: int = 300) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        detail = call("GET", f"/api/runs/{run_id}")
        if detail["status"] in TERMINAL:
            return detail
        time.sleep(2)
    raise TimeoutError(f"run {run_id} did not finish in {timeout_s}s")


def record(sample_id, query, methods, invert, rep):
    created = call(
        "POST",
        "/api/runs",
        {"imageId": sample_id, "query": query, "methods": methods, "invert": invert},
    )
    detail = wait_run(created["runId"])
    return {
        "sample": sample_id,
        "query": query,
        "invert": invert,
        "rep": rep,
        "runId": detail["runId"],
        "methods": [
            {
                "methodId": m["methodId"],
                "status": m["status"],
                "count": m["count"],
                "elapsedMs": m["elapsedMs"],
            }
            for m in detail["methods"]
        ],
    }


def main() -> int:
    global COOKIE
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-reps", type=int, default=3)
    args = ap.parse_args()

    COOKIE = unlock()
    rows = []

    for sample_id, query, methods in OFFLINE:
        for invert in (False, True):
            row = record(sample_id, query, methods, invert, 0)
            rows.append(row)
            print(json.dumps(row), flush=True)

    for sample_id, query in VLM:
        for invert in (False, True):
            for rep in range(1, args.vlm_reps + 1):
                row = record(sample_id, query, ["m6"], invert, rep)
                rows.append(row)
                print(json.dumps(row), flush=True)

    existing = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
        except json.JSONDecodeError:
            existing = []
    OUT_PATH.write_text(json.dumps(existing + rows, indent=2) + "\n")
    print(f"\nwrote {len(rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
