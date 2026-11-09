# AnyCount Studio — Prototype (`prototype/`)

Single-binary Rust (Axum) service that serves a React SPA (bundled into the binary with
`rust-embed`) and runs the AnyCount counting methods:

| Method | Engine | Runs where |
| --- | --- | --- |
| `m1` | Count-Anything / SAM3 checkpoint (`workers/ca_worker.py`) | local Python venv, `Semaphore(1)` guarded |
| `m2` | Offline ONNX pack, single pass 640 (`workers/onnx_worker.py --mode single`) | local Python venv, ~0.15-0.25 s |
| `m3` | Offline ONNX pack, always 2x2 tiled + Point-NMS (`--mode tiled`) | local Python venv, ~0.5 s, denser scenes |
| `m4` / `m5` | Gemini 3.8 Flash single-pass / 2x2 tiled + Point-NMS | JRouter gateway (cloud) |
| `m6` / `m7` | DeepSeek V4.1 Flash single-pass / 2x2 tiled + Point-NMS | JRouter gateway (cloud) |

Bundled packs: `apple` (`research/spikes/004-.../models/apple_v2_distilled_int8.onnx`,
289/289 on `apples.jpg` at `m3`) and `tennis-ball`
(`research/packs/tennis-ball/models/tennis_ball_v1_int8.onnx`, 41/43 on `tennis-ball.jpg`
at `m2`). Other object classes start as `cold_start` until an `.onnx` pack is trained
from the result card or `POST /api/packs/train`.

Live deployment: **https://testing.rkhyg.xyz** → Caddy → `127.0.0.1:2030`
(`anycount-prototype.service`, registered in `vpsdash` so Caddy reloads never drop it).

---

## 1. Configuration

All configuration is environment variables (see `.env.example`); the binary never reads a
config file. `ANYCOUNT_BIND` **must** be a loopback address — the process refuses to bind
`0.0.0.0` (the host-facing TLS termination is Caddy's job).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANYCOUNT_BIND` | `127.0.0.1:2030` | Axum listener (loopback only) |
| `ANYCOUNT_DATA_DIR` | `/var/lib/anycount-prototype` | SQLite DB, uploads, renders, `.onnx` packs |
| `ANYCOUNT_PASSCODE` | `2limanov` | Passcode that unlocks write/compute endpoints |
| `ANYCOUNT_REPO_ROOT` | `/root/workspace/anycount` | Sample images, montages, research `.onnx` weights |
| `VLM_API_URL` | `https://jr.rkhyg.xyz/v1/chat/completions` | OpenAI-compatible gateway |
| `VLM_API_KEY` | *(empty — set in `.env` only)* | Gateway key, never sent to the browser |
| `GEMINI_MODEL` | `gemini-3.8-flash-high` | M4 / M5 / M2-M3 auto-train teacher |
| `DEEPSEEK_MODEL` | `deepseek-v4.1-flash-1017` | M6 / M7 |
| `CA_PYTHON_BIN` | `/root/workspace/count-anything/.venv/bin/python` | M1 interpreter |
| `CA_REPO_DIR` | `/root/workspace/count-anything` | M1 checkpoint/repo |
| `ONNX_PYTHON_BIN` | `/root/workspace/countthis/pilot/.venv/bin/python` | M2 interpreter (`onnxruntime`, `ultralytics`) |
| `YOLO_BASE_WEIGHTS` | `/root/workspace/countthis/pilot/yolo11n.pt` | Base weights for M2 auto-train |

Public read-only endpoints: `/api/health`, `/api/auth/status`, `/api/samples`,
`/api/samples/montages/{filename}`, `/api/packs`, `/api/packs/{slug}/download`,
`/api/docs`, `/api/images/{id}/raw`.
Session-gated (`anycount_session` cookie from `POST /api/auth/unlock`): `POST /api/images`,
`POST /api/runs`, `POST /api/packs/train`, and the per-user run surface
`GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/stream`,
`GET /api/runs/{id}/raw`, `GET /api/runs/{id}/render/{method_id}` (they expose uploaded
photos and their points).

### Provisioning `.env` (VPS)

`.env` is git-ignored and must be `0600`. The `VLM_API_KEY` lives in JRouter's SQLite DB —
copy it without printing it:

```bash
python3 - <<'PY'
import os, sqlite3, pathlib
key = sqlite3.connect("file:/var/lib/9router/db/data.sqlite?mode=ro", uri=True) \
    .execute("SELECT key FROM apiKeys WHERE name='anycount'").fetchone()[0]
env = pathlib.Path("/root/workspace/anycount/prototype/.env")
base = pathlib.Path("/root/workspace/anycount/prototype/.env.example").read_text()
base = base.replace("VLM_API_KEY=\n", f"VLM_API_KEY={key}\n")
env.write_text(base); os.chmod(env, 0o600)
print("provisioned", len(base.splitlines()), "vars")
PY
```

---

## 2. Local development

Two terminals (Vite dev server proxies `/api` to the Rust backend):

```bash
# terminal 1 — backend (http://127.0.0.1:2030)
cd prototype
cp .env.example .env          # then fill VLM_API_KEY for cloud methods
cargo run                     # migrations + sample/pack seeding happen on boot

# terminal 2 — frontend (http://127.0.0.1:5173, hot reload)
cd prototype/web
npm install
npm run dev
```

Tests and typecheck:

```bash
cd prototype/web && npm test && npm run typecheck
cd prototype && cargo test
```

> `m1`, `m2` and `m3` need the two Python venvs from the table above; without them those
> methods return a `failed` row while `m4`–`m7` keep working (they are pure Rust HTTP).

---

## 3. Production build & `systemd` deployment (this VPS)

`deploy/build.sh` builds the SPA first (`web/dist` is embedded by `rust-embed` at compile
time) and then the release binary:

```bash
chmod +x prototype/deploy/build.sh
prototype/deploy/build.sh
# -> prototype/target/release/anycount-prototype
```

Install the unit and start it:

```bash
cp prototype/deploy/anycount-prototype.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now anycount-prototype
systemctl status anycount-prototype --no-pager
journalctl -u anycount-prototype -f
```

The unit runs as `root`, `WorkingDirectory=/root/workspace/anycount/prototype` (the workers
are spawned as `workers/*.py`, i.e. relative to it), loads `EnvironmentFile=.env`, restarts
on failure, caps memory at `MemoryMax=10G` and deliberately **does not** set `PrivateTmp`
so the SAM3/ONNX subprocesses can `mmap` the 3.4 GB checkpoint.

Smoke test on the box: `curl -s http://127.0.0.1:2030/api/health`.

### Reverse proxy + `vpsdash` registration

`/etc/caddy/Caddyfile` is generated by `vpsdash`; hand edits are lost on the next reload.
Register the domain through the DB instead:

```bash
cp -a /etc/caddy/Caddyfile /root/caddyfile.backup.anycount   # safety copy
python3 prototype/deploy/register_caddy_vpsdash.py
# -> Registered and applied hosts: ['jr.rkhyg.xyz', 'lab.rkhyg.xyz', 'testing.rkhyg.xyz', 'vm.rkhyg.xyz']
```

The script takes `vpsdash`'s `caddy.lock`, upserts
`testing.rkhyg.xyz → 127.0.0.1:2030` into `sites`, re-renders the Caddyfile in exact
`vpsdash` format, aborts if `vm.rkhyg.xyz` disappeared from the enabled set, runs
`caddy validate`, atomically replaces the file, `systemctl reload caddy`, and updates the
`caddy_state` SHA-256 hashes. It is idempotent — re-run it after any `vpsdash` change.

Rollback:

```bash
cp -a /root/caddyfile.backup.anycount /etc/caddy/Caddyfile
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("/root/.local/share/vpsdash/vpsdash.db")
c.execute("DELETE FROM sites WHERE host='testing.rkhyg.xyz'"); c.commit()
PY
systemctl reload caddy && systemctl disable --now anycount-prototype
```

---

## 4. Docker (portable, for machines other than this VPS)

Docker is **not** installed on this VPS; the files below exist so the same service can be
run anywhere. Multi-stage build: `node:22-bookworm-slim` (SPA) → `rust:1.88-bookworm`
(binary) → `debian:bookworm-slim` (runtime + `workers/`).

```bash
cd prototype
cp .env.example .env           # fill VLM_API_KEY
docker compose up -d --build
curl -s http://127.0.0.1:2030/api/health
docker compose logs -f
```

`compose.yml` uses `network_mode: host` **on purpose**: the server refuses non-loopback
binds, and a `127.0.0.1:2030:2030` publish cannot reach a socket bound to the container's
own loopback. With host networking, `127.0.0.1:2030` is identical inside and outside the
container, matching the production Caddy setup. State lives in the `anycount-data` volume;
the repo is mounted read-only at `/app/repo` for the sample images, montages and `.onnx`
weights. Override the pinned toolchain images if a future `Cargo.lock` needs a newer MSRV:

```bash
docker build --build-arg RUST_IMAGE=rust:1.99-bookworm -t anycount-prototype .
```

`m4`–`m7` work out of the box (outbound HTTPS to the gateway). `m1`/`m2`/`m3` additionally need
Python venvs with PyTorch/SAM3 and `onnxruntime`/`ultralytics`; mount them and point
`CA_PYTHON_BIN`, `CA_REPO_DIR`, `ONNX_PYTHON_BIN`, `YOLO_BASE_WEIGHTS` at the mounts.

---

## 5. API quick reference

```bash
BASE=https://testing.rkhyg.xyz
curl -s $BASE/api/health                       # {"ok":true,"service":"anycount-prototype"}
curl -s $BASE/api/samples | jq '.samples|length, .montages|length'
curl -s $BASE/api/packs | jq '.packs[0] | {slug,sizeMb,status}'

# unlock (cookie jar) then run
curl -s -c jar -X POST $BASE/api/auth/unlock -H 'content-type: application/json' \
     -d '{"passcode":"2limanov"}'
curl -s -b jar -X POST $BASE/api/runs -H 'content-type: application/json' \
     -d '{"imageId":"sample_apples","query":"apples","methods":["m2"]}'
curl -s $BASE/api/runs/<runId> | jq '.methods[] | {methodId,status,count,elapsedMs}'
curl -s -o /tmp/render.jpg -w '%{http_code}\n' $BASE/api/runs/<runId>/render/m2
```

`POST /api/runs` returns immediately; poll `GET /api/runs/{id}` or subscribe to
`GET /api/runs/{id}/stream` (SSE). Each method reports `status`
(`running`/`queued`/`completed`/`cold_start`/`failed`), `count`, `points`, `gridBoxes`,
`elapsedMs`, token usage and `renderedUrl`.

Add `"invert": true` to the run payload to count on the RGB-negative derivative of the
image (every channel becomes `255 - value`). The derivative is cached per image under
`{ANYCOUNT_DATA_DIR}/derived/{imageId}_invert.jpg`, all seven methods and the renderer
read it, and the run reports `"inverted": true` while `GET /api/runs/{id}/raw` serves
the exact source image that run counted on (session-gated, like the rest of the run
surface).
