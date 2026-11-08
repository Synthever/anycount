"""
Idempotently registers testing.rkhyg.xyz -> 127.0.0.1:2030 in vpsdash SQLite DB
(/root/.local/share/vpsdash/vpsdash.db), renders /etc/caddy/Caddyfile in exact
vpsdash format, validates with `caddy validate`, updates caddy_state SHA-256,
and reloads Caddy.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

DB_PATH = Path("/root/.local/share/vpsdash/vpsdash.db")
LOCK_PATH = Path("/root/.local/share/vpsdash/caddy.lock")
CADDYFILE = Path("/etc/caddy/Caddyfile")
HEADER = "# managed by vpsdash — jangan edit manual, perubahan akan tertimpa"


def escape_header_value(val: str) -> str:
    return val.replace("\\", "\\\\").replace('"', '\\"')


def render_caddyfile(rows: list[tuple]) -> str:
    out = [HEADER]
    for host, kind, upstream, root_path, headers_json in rows:
        block = [f"\n{host} {{"]
        if kind == "static":
            if root_path:
                block.append(f"    root * {root_path}")
                block.append("    encode gzip")
                block.append("    file_server")
        else:
            if upstream:
                block.append(f"    reverse_proxy {upstream}")
        if headers_json:
            try:
                headers = json.loads(headers_json)
                for k, v in headers:
                    block.append(f'    header {k} "{escape_header_value(v)}"')
            except Exception:
                pass
        block.append("}")
        out.append("\n".join(block))
    return "\n".join(out) + "\n"


def main() -> None:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sites (host, type, upstream, root_path, enabled, headers_json, imported, created_at, updated_at)
                VALUES ('testing.rkhyg.xyz', 'proxy', '127.0.0.1:2030', NULL, 1, '[]', 0, ?, ?)
                ON CONFLICT(host) DO UPDATE SET
                    type = 'proxy',
                    upstream = '127.0.0.1:2030',
                    enabled = 1,
                    updated_at = excluded.updated_at
                """,
                (now, now),
            )
            conn.commit()

            cur.execute(
                "SELECT host, type, upstream, root_path, headers_json FROM sites WHERE enabled = 1 ORDER BY host"
            )
            rows = cur.fetchall()
            hosts = [r[0] for r in rows]
            if "vm.rkhyg.xyz" not in hosts:
                raise RuntimeError("Safety abort: vm.rkhyg.xyz missing from enabled sites!")

            rendered = render_caddyfile(rows)
            tmp_caddy = Path("/etc/caddy/Caddyfile.anycount.tmp")
            tmp_caddy.write_text(rendered, encoding="utf-8")

            subprocess.run(
                ["/usr/bin/caddy", "validate", "--config", str(tmp_caddy), "--adapter", "caddyfile"],
                check=True,
            )
            tmp_caddy.replace(CADDYFILE)
            subprocess.run(["systemctl", "reload", "caddy"], check=True)

            digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            cur.execute(
                "UPDATE caddy_state SET rendered_hash = ?, applied_hash = ?, applied_at = ? WHERE id = 1",
                (digest, digest, now),
            )
            cur.execute("UPDATE sites SET last_applied_at = ? WHERE enabled = 1", (now,))
            conn.commit()
            print("Registered and applied hosts:", hosts)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
