# AnyCount Web Prototype (`prototype/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy **AnyCount Studio** inside `prototype/` — an interactive Rust Axum + React web prototype hosted at `https://testing.rkhyg.xyz` to test and compare all 6 object-counting methods (`M1` Count-Anything SAM3, `M2` Offline ONNX + LLM Validator/Auto-Train, `M3` Gemini Single-Pass, `M4` Gemini 2x2 Tiled, `M5` DeepSeek Single-Pass, `M6` DeepSeek 2x2 Tiled) with an interactive point-correction canvas.

**Architecture:** A single-binary Rust Axum (`0.8`) backend serves REST + SSE endpoints and embeds the compiled React 19 + Vite SPA (`web/dist`) via `rust-embed`. Methods `M3`–`M6` run natively in async Rust (`reqwest` + `image` 2x2 cropper + Adaptive Radius Point-NMS), while `M1` (SAM3 3.4 GB) and `M2` (`onnxruntime` / `ultralytics` Auto-Train) execute via isolated Python workers guarded by a `Semaphore::new(1)` so the 14 GB VPS never runs out of memory.

**Tech Stack:** Rust 1.99 (`axum 0.8`, `tokio 1`, `sqlx 0.8` SQLite, `reqwest 0.12`, `image 0.25`, `rust-embed 8`), React 19, TypeScript 5.9, Vite 6, Tailwind CSS v4, Vitest 3, Caddy + `vpsdash` SQLite site registry, `systemd` (`anycount-prototype.service` on `127.0.0.1:2030`) + portable `Dockerfile`/`compose.yml`.

**Spec:** `docs/superpowers/specs/2026-11-07-anycount-prototype-design.md`

## Global Constraints

- **Target Host & Repo:** Build and verify on VPS `root@49.128.191.225` inside `/root/workspace/anycount/prototype` and sync via Git (`origin/main`) to `D:\Github\anycount\prototype`.
- **Bind Address:** Default bind MUST be loopback `127.0.0.1:2030` (`ANYCOUNT_BIND`); never bind `0.0.0.0` directly on the VPS host.
- **Passcode Gate:** Default passcode is `2limanov` (`ANYCOUNT_PASSCODE`). Read-only endpoints (`/api/health`, `/api/auth/status`, `/api/samples`, `/api/packs`, `/api/packs/{slug}/download`, `/api/docs`, `/api/images/{id}/raw`, `/api/runs/{id}/render/{method_id}`) remain public for viewing; live mutation/compute endpoints (`POST /api/images`, `POST /api/runs`, `POST /api/packs/train`) require a valid session cookie (`anycount_session`) obtained via `POST /api/auth/unlock`.
- **Zero Leaked Secrets:** Never hardcode `VLM_API_KEY` in source files, tests, or `.env.example`. Read `VLM_API_KEY` exclusively from `.env` on the VPS (mode `0600`, gitignored).
- **Heavy Process Guard:** `M1` (`ca_worker.py`) and `M2` Auto-Train (`onnx_worker.py train`) MUST acquire `AppState::heavy_semaphore` (`Semaphore::new(1)`) before spawning subprocesses.
- **Axum 0.8 Path Syntax:** All parameterized routes MUST use `{param}` braces (e.g. `/api/runs/{id}/stream`), never legacy `:param`.
- **Caddy & `vpsdash` Safety:** Never hand-edit `/etc/caddy/Caddyfile` without registering `testing.rkhyg.xyz -> 127.0.0.1:2030` into `/root/.local/share/vpsdash/vpsdash.db` (`sites` + `caddy_state` SHA-256 hash) first.
- **Design Fidelity:** Dark-first UI matching `docs/design/image.png` (`#141414` canvas, `#1c1c1c` surface, `#b8a4ed` lavender, `#34d39a` emerald, `#ff4d8b` pink, monospace telemetry footer, floating bottom composer) plus a working Light Mode toggle using `JRouter/docs/DESIGN.md` warm cream (`#fffaf0`) and brand card tokens.

---

### Task 1: Rust Crate Foundation, Config, SQLite Migrations, Error Handling & Passcode Auth (`2limanov`)

**Files:**
- Create: `prototype/Cargo.toml`
- Create: `prototype/build.rs`
- Create: `prototype/.env.example`
- Create: `prototype/migrations/0001_init.sql`
- Create: `prototype/src/config.rs`
- Create: `prototype/src/error.rs`
- Create: `prototype/src/state.rs`
- Create: `prototype/src/auth.rs`
- Create: `prototype/src/assets.rs`
- Create: `prototype/src/routes/mod.rs`
- Create: `prototype/src/routes/auth.rs`
- Create: `prototype/src/lib.rs`
- Create: `prototype/src/main.rs`
- Test: `prototype/tests/auth_and_health.rs`

**Interfaces:**
- Consumes: Environment variables (`ANYCOUNT_BIND`, `ANYCOUNT_DATA_DIR`, `ANYCOUNT_PASSCODE`, `ANYCOUNT_REPO_ROOT`, `VLM_API_URL`, `VLM_API_KEY`, `GEMINI_MODEL`, `DEEPSEEK_MODEL`, `CA_PYTHON_BIN`, `CA_REPO_DIR`, `ONNX_PYTHON_BIN`, `YOLO_BASE_WEIGHTS`).
- Produces:
  - `anycount_prototype::config::Config` (`from_env()`, `for_test(data_dir: &Path)`)
  - `anycount_prototype::error::AppError` (`AuthRequired`, `BadRequest(String)`, `NotFound(String)`, `Conflict(String)`, `Internal(anyhow::Error)`) implementing `IntoResponse` -> `{"error": {"code": "...", "message": "..."}}`
  - `anycount_prototype::state::AppState` (`pool: SqlitePool`, `cfg: Arc<Config>`, `http: reqwest::Client`, `heavy_semaphore: Arc<Semaphore>`, `run_streams: Arc<DashMap<String, broadcast::Sender<String>>>`)
  - `anycount_prototype::auth::RequireSession` (Axum `FromRequestParts<AppState>` extractor verifying `anycount_session` cookie against `sessions` table)
  - `anycount_prototype::router(state: AppState) -> Router`

- [ ] **Step 1: Scaffold `prototype/Cargo.toml`, `build.rs`, `.env.example`, and `migrations/0001_init.sql`**

Write `prototype/Cargo.toml`:
```toml
[package]
name = "anycount-prototype"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
axum = { version = "0.8", features = ["macros", "multipart"] }
axum-extra = { version = "0.10", features = ["cookie"] }
base64 = "0.22"
dashmap = "6"
futures = "0.3"
hex = "0.4"
image = { version = "0.25", default-features = false, features = ["jpeg", "png", "webp"] }
mime_guess = "2"
rand = "0.8"
regex = "1"
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "json", "stream"] }
rust-embed = "8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
sqlx = { version = "0.8", default-features = false, features = ["runtime-tokio", "sqlite", "migrate", "macros"] }
tempfile = "3"
thiserror = "2"
time = { version = "0.3", features = ["serde", "formatting", "macros"] }
tokio = { version = "1", features = ["full"] }
tokio-stream = { version = "0.1", features = ["sync"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "limit"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
http-body-util = "0.1"
tower = { version = "0.5", features = ["util"] }
```

Write `prototype/build.rs` (creates `web/dist/index.html` placeholder if missing so `cargo test` compiles cleanly before Vite runs):
```rust
use std::{fs, path::Path};

fn main() {
    let dist = Path::new("web/dist");
    if !dist.exists() {
        let _ = fs::create_dir_all(dist);
        let _ = fs::write(
            dist.join("index.html"),
            "<!doctype html><html><body>AnyCount Prototype Dev Placeholder</body></html>",
        );
    }
    println!("cargo:rerun-if-changed=web/dist");
    println!("cargo:rerun-if-changed=migrations");
}
```

Write `prototype/.env.example`:
```dotenv
ANYCOUNT_BIND=127.0.0.1:2030
ANYCOUNT_DATA_DIR=/var/lib/anycount-prototype
ANYCOUNT_PASSCODE=2limanov
ANYCOUNT_REPO_ROOT=/root/workspace/anycount
VLM_API_URL=https://jr.rkhyg.xyz/v1/chat/completions
VLM_API_KEY=
GEMINI_MODEL=gemini-3.8-flash-high
DEEPSEEK_MODEL=deepseek-v4.1-flash-1017
CA_PYTHON_BIN=/root/workspace/count-anything/.venv/bin/python
CA_REPO_DIR=/root/workspace/count-anything
ONNX_PYTHON_BIN=/root/workspace/countthis/pilot/.venv/bin/python
YOLO_BASE_WEIGHTS=/root/workspace/countthis/pilot/yolo11n.pt
```

Write `prototype/migrations/0001_init.sql`:
```sql
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    orig_path TEXT NOT NULL,
    work_path TEXT NOT NULL,
    is_sample INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id),
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_methods (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    method_id TEXT NOT NULL,
    status TEXT NOT NULL,
    count INTEGER,
    points_json TEXT,
    grid_boxes_json TEXT,
    rendered_path TEXT,
    elapsed_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model_size_mb REAL,
    meta_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, method_id)
);

CREATE TABLE IF NOT EXISTS model_packs (
    slug TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    onnx_path TEXT NOT NULL,
    size_mb REAL NOT NULL,
    imgsz INTEGER NOT NULL,
    trained_epochs INTEGER NOT NULL,
    val_mape_pct REAL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

- [ ] **Step 2: Write the failing integration test `prototype/tests/auth_and_health.rs`**

```rust
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn health_and_passcode_unlock_flow() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    assert_eq!(cfg.bind_addr.ip().to_string(), "127.0.0.1");
    assert_eq!(cfg.passcode, "2limanov");

    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state);

    // 1. Health is public
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 2. Auth status starts locked
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/auth/status").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], false);

    // 3. Wrong passcode rejected with 401 AUTH_INVALID_PASSCODE
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"wrong"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // 4. Correct passcode "2limanov" sets HttpOnly cookie and unlocks session
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"2limanov"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let set_cookie = res
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(set_cookie.contains("anycount_session="));
    assert!(set_cookie.contains("HttpOnly"));

    let cookie_kv = set_cookie.split(';').next().unwrap();
    let res = app
        .oneshot(
            Request::builder()
                .uri("/api/auth/status")
                .header(header::COOKIE, cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], true);
}
```

- [ ] **Step 3: Implement `config.rs`, `error.rs`, `state.rs`, `auth.rs`, `assets.rs`, `routes/auth.rs`, `lib.rs`, and `main.rs`**

Write `prototype/src/config.rs`:
```rust
use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub data_dir: PathBuf,
    pub passcode: String,
    pub repo_root: PathBuf,
    pub vlm_api_url: String,
    pub vlm_api_key: String,
    pub gemini_model: String,
    pub deepseek_model: String,
    pub ca_python_bin: PathBuf,
    pub ca_repo_dir: PathBuf,
    pub onnx_python_bin: PathBuf,
    pub yolo_base_weights: PathBuf,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind_str = env::var("ANYCOUNT_BIND").unwrap_or_else(|_| "127.0.0.1:2030".to_string());
        let bind_addr: SocketAddr = bind_str.parse()?;
        if !bind_addr.ip().is_loopback() {
            anyhow::bail!("ANYCOUNT_BIND must be a loopback address, got {bind_addr}");
        }
        Ok(Self {
            bind_addr,
            data_dir: PathBuf::from(
                env::var("ANYCOUNT_DATA_DIR")
                    .unwrap_or_else(|_| "/var/lib/anycount-prototype".to_string()),
            ),
            passcode: env::var("ANYCOUNT_PASSCODE").unwrap_or_else(|_| "2limanov".to_string()),
            repo_root: PathBuf::from(
                env::var("ANYCOUNT_REPO_ROOT")
                    .unwrap_or_else(|_| "/root/workspace/anycount".to_string()),
            ),
            vlm_api_url: env::var("VLM_API_URL")
                .unwrap_or_else(|_| "https://jr.rkhyg.xyz/v1/chat/completions".to_string()),
            vlm_api_key: env::var("VLM_API_KEY").unwrap_or_default(),
            gemini_model: env::var("GEMINI_MODEL")
                .unwrap_or_else(|_| "gemini-3.8-flash-high".to_string()),
            deepseek_model: env::var("DEEPSEEK_MODEL")
                .unwrap_or_else(|_| "deepseek-v4.1-flash-1017".to_string()),
            ca_python_bin: PathBuf::from(
                env::var("CA_PYTHON_BIN")
                    .unwrap_or_else(|_| "/root/workspace/count-anything/.venv/bin/python".to_string()),
            ),
            ca_repo_dir: PathBuf::from(
                env::var("CA_REPO_DIR")
                    .unwrap_or_else(|_| "/root/workspace/count-anything".to_string()),
            ),
            onnx_python_bin: PathBuf::from(
                env::var("ONNX_PYTHON_BIN")
                    .unwrap_or_else(|_| "/root/workspace/countthis/pilot/.venv/bin/python".to_string()),
            ),
            yolo_base_weights: PathBuf::from(
                env::var("YOLO_BASE_WEIGHTS")
                    .unwrap_or_else(|_| "/root/workspace/countthis/pilot/yolo11n.pt".to_string()),
            ),
        })
    }

    pub fn for_test(data_dir: &Path) -> Self {
        Self {
            bind_addr: "127.0.0.1:2030".parse().unwrap(),
            data_dir: data_dir.to_path_buf(),
            passcode: "2limanov".to_string(),
            repo_root: data_dir.join("repo"),
            vlm_api_url: "http://127.0.0.1:9/v1/chat/completions".to_string(),
            vlm_api_key: "sk-test-fixture".to_string(),
            gemini_model: "gemini-3.8-flash-high".to_string(),
            deepseek_model: "deepseek-v4.1-flash-1017".to_string(),
            ca_python_bin: PathBuf::from("python3"),
            ca_repo_dir: data_dir.join("ca"),
            onnx_python_bin: PathBuf::from("python3"),
            yolo_base_weights: data_dir.join("yolo11n.pt"),
        }
    }
}
```

Write `prototype/src/error.rs`:
```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Passcode diperlukan untuk menjalankan fitur live ini")]
    AuthRequired,
    #[error("Passcode tidak sesuai")]
    InvalidPasscode,
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("Kesalahan internal server")]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::AuthRequired => "AUTH_REQUIRED",
            Self::InvalidPasscode => "AUTH_INVALID_PASSCODE",
            Self::BadRequest(_) => "BAD_REQUEST",
            Self::NotFound(_) => "NOT_FOUND",
            Self::Conflict(_) => "CONFLICT",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::AuthRequired | Self::InvalidPasscode => StatusCode::UNAUTHORIZED,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        if let Self::Internal(ref err) = self {
            tracing::error!(error = ?err, "internal error");
        }
        let status = self.status();
        let body = json!({
            "error": {
                "code": self.code(),
                "message": self.to_string(),
            }
        });
        (status, Json(body)).into_response()
    }
}
```

Write `prototype/src/state.rs`:
```rust
use std::{fs, sync::Arc, time::Duration};

use dashmap::DashMap;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    SqlitePool,
};
use tokio::sync::{broadcast, Semaphore};

use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    pub pool: SqlitePool,
    pub cfg: Arc<Config>,
    pub http: reqwest::Client,
    pub heavy_semaphore: Arc<Semaphore>,
    pub run_streams: Arc<DashMap<String, broadcast::Sender<String>>>,
}

impl AppState {
    pub async fn init(cfg: Config) -> anyhow::Result<Self> {
        fs::create_dir_all(cfg.data_dir.join("uploads"))?;
        fs::create_dir_all(cfg.data_dir.join("renders"))?;
        fs::create_dir_all(cfg.data_dir.join("packs"))?;
        fs::create_dir_all(cfg.data_dir.join("tmp"))?;

        let db_path = cfg.data_dir.join("anycount.db");
        let opts = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(10));

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await?;

        sqlx::migrate!("./migrations").run(&pool).await?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .user_agent("anycount-prototype/0.1")
            .build()?;

        Ok(Self {
            pool,
            cfg: Arc::new(cfg),
            http,
            heavy_semaphore: Arc::new(Semaphore::new(1)),
            run_streams: Arc::new(DashMap::new()),
        })
    }
}
```

Write `prototype/src/auth.rs`:
```rust
use axum::{extract::FromRequestParts, http::request::Parts};
use axum_extra::extract::CookieJar;
use rand::RngCore;
use sha2::{Digest, Sha256};
use time::{Duration, OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{error::AppError, state::AppState};

pub const SESSION_COOKIE: &str = "anycount_session";

pub fn now_rfc3339() -> String {
    OffsetDateTime::now_utc().format(&Rfc3339).unwrap()
}

pub fn hash_token(raw: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

pub async fn is_session_valid(st: &AppState, jar: &CookieJar) -> bool {
    let Some(c) = jar.get(SESSION_COOKIE) else {
        return false;
    };
    let token_hash = hash_token(c.value());
    let now = now_rfc3339();
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT token_hash FROM sessions WHERE token_hash = ? AND expires_at > ?",
    )
    .bind(token_hash)
    .bind(now)
    .fetch_optional(&st.pool)
    .await
    .ok()
    .flatten();
    row.is_some()
}

pub async fn create_session(st: &AppState) -> anyhow::Result<String> {
    let raw = generate_token();
    let token_hash = hash_token(&raw);
    let now = OffsetDateTime::now_utc();
    let expires = (now + Duration::days(30)).format(&Rfc3339)?;
    sqlx::query("INSERT INTO sessions (token_hash, created_at, expires_at) VALUES (?, ?, ?)")
        .bind(token_hash)
        .bind(now.format(&Rfc3339)?)
        .bind(expires)
        .execute(&st.pool)
        .await?;
    Ok(raw)
}

pub struct RequireSession;

impl FromRequestParts<AppState> for RequireSession {
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        let jar = CookieJar::from_headers(&parts.headers);
        if is_session_valid(state, &jar).await {
            Ok(Self)
        } else {
            Err(AppError::AuthRequired)
        }
    }
}
```

Write `prototype/src/assets.rs`:
```rust
use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    Json,
};
use rust_embed::Embed;
use serde_json::json;

#[derive(Embed)]
#[folder = "web/dist"]
struct WebAssets;

pub async fn static_handler(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    if path.starts_with("api/") || path == "api" {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"code": "NOT_FOUND", "message": "Endpoint API tidak ditemukan"}})),
        )
            .into_response();
    }

    let target = if path.is_empty() { "index.html" } else { path };
    if let Some(file) = WebAssets::get(target) {
        let mime = mime_guess::from_path(target).first_or_octet_stream();
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.data.into_owned()))
            .unwrap();
    }

    if let Some(index) = WebAssets::get("index.html") {
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
            .body(Body::from(index.data.into_owned()))
            .unwrap();
    }

    (StatusCode::NOT_FOUND, "Frontend build missing").into_response()
}
```

Write `prototype/src/routes/mod.rs` and `prototype/src/routes/auth.rs`:
```rust
// prototype/src/routes/mod.rs
pub mod auth;
```

```rust
// prototype/src/routes/auth.rs
use axum::{
    extract::State,
    http::{header, HeaderMap},
    response::IntoResponse,
    Json,
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;
use serde_json::json;

use crate::{
    auth::{create_session, hash_token, is_session_valid, SESSION_COOKIE},
    error::AppError,
    state::AppState,
};

#[derive(Deserialize)]
pub struct UnlockReq {
    pub passcode: String,
}

pub async fn status(State(st): State<AppState>, jar: CookieJar) -> impl IntoResponse {
    Json(json!({ "unlocked": is_session_valid(&st, &jar).await }))
}

pub async fn unlock(
    State(st): State<AppState>,
    Json(req): Json<UnlockReq>,
) -> Result<impl IntoResponse, AppError> {
    if req.passcode.trim() != st.cfg.passcode {
        return Err(AppError::InvalidPasscode);
    }
    let token = create_session(&st).await?;
    let cookie = format!(
        "{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
    );
    let mut headers = HeaderMap::new();
    headers.insert(header::SET_COOKIE, cookie.parse().unwrap());
    Ok((headers, Json(json!({ "unlocked": true }))))
}

pub async fn logout(State(st): State<AppState>, jar: CookieJar) -> impl IntoResponse {
    if let Some(c) = jar.get(SESSION_COOKIE) {
        let _ = sqlx::query("DELETE FROM sessions WHERE token_hash = ?")
            .bind(hash_token(c.value()))
            .execute(&st.pool)
            .await;
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        header::SET_COOKIE,
        format!("{SESSION_COOKIE}=; Path=/; HttpOnly; Max-Age=0")
            .parse()
            .unwrap(),
    );
    (headers, Json(json!({ "unlocked": false })))
}
```

Write `prototype/src/lib.rs` and `prototype/src/main.rs`:
```rust
// prototype/src/lib.rs
pub mod assets;
pub mod auth;
pub mod config;
pub mod error;
pub mod routes;
pub mod state;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "service": "anycount-prototype" }))
}

pub fn router(state: state::AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/auth/status", get(routes::auth::status))
        .route("/api/auth/unlock", post(routes::auth::unlock))
        .route("/api/auth/logout", post(routes::auth::logout))
        .fallback(assets::static_handler)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}
```

```rust
// prototype/src/main.rs
use anycount_prototype::{config::Config, router, state::AppState};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,anycount_prototype=debug".into()),
        )
        .init();

    let cfg = Config::from_env()?;
    let bind_addr = cfg.bind_addr;
    let state = AppState::init(cfg).await?;
    let app = router(state);

    let listener = TcpListener::bind(bind_addr).await?;
    tracing::info!(%bind_addr, "AnyCount Studio listening");
    axum::serve(listener, app).await?;
    Ok(())
}
```

- [ ] **Step 4: Run `cargo test` on VPS to verify Task 1 passes**

Run: `ssh root@49.128.191.225 'export PATH="$HOME/.cargo/bin:$PATH"; cd /root/workspace/anycount/prototype && cargo test --test auth_and_health'`
Expected: `test result: ok. 1 passed; 0 failed`

- [ ] **Step 5: Commit Task 1**

```bash
git add prototype/Cargo.toml prototype/Cargo.lock prototype/build.rs prototype/.env.example prototype/migrations/ prototype/src/ prototype/tests/auth_and_health.rs
git commit -m "feat(prototype): scaffold Rust Axum backend, SQLite migrations, and passcode auth"
```

---

### Task 2: Image Ingestion, Research Sample Seeding, Docs API & Count-Anything Style Renderer

**Files:**
- Create: `prototype/src/services/mod.rs`
- Create: `prototype/src/services/renderer.rs`
- Create: `prototype/src/routes/images.rs`
- Create: `prototype/src/routes/samples.rs`
- Create: `prototype/src/routes/docs.rs`
- Modify: `prototype/src/routes/mod.rs`
- Modify: `prototype/src/lib.rs`
- Modify: `prototype/src/main.rs`
- Test: `prototype/tests/images_and_renderer.rs`

**Interfaces:**
- Consumes: `AppState`, `RequireSession`, `AppError`.
- Produces:
  - `services::renderer::Point2D { pub x: f64, pub y: f64 }`
  - `services::renderer::render_annotated_jpeg(src_path: &Path, points: &[Point2D], grid_boxes: Option<&[[i32; 4]]>, out_path: &Path) -> anyhow::Result<()>`
  - `routes::samples::seed_research_samples(st: &AppState) -> anyhow::Result<()>`
  - Endpoints:
    - `POST /api/images` (requires `RequireSession`, multipart `file`, saves original + resized `<=1280px` JPEG work image, returns `{ id, filename, width, height, url }`)
    - `GET /api/images/{id}/raw` (public image bytes)
    - `GET /api/samples` (returns preset sample photos + pre-computed benchmark summaries & montage URLs)
    - `GET /api/samples/montages/{filename}` (serves montage JPEGs from `research/spikes/003-method-comparison/results/`)
    - `GET /api/docs` (returns structured JSON of the 6 methods + 14-item Indonesian layperson glossary)

- [ ] **Step 1: Write the failing test `prototype/tests/images_and_renderer.rs`**

```rust
use std::io::Cursor;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use image::{ImageFormat, Rgb, RgbImage};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn render_dots_and_serve_docs_and_upload_image() {
    let tmp = TempDir::new().unwrap();
    let src_img = tmp.path().join("sample.jpg");
    let mut img = RgbImage::from_pixel(400, 300, Rgb([40, 80, 120]));
    img.put_pixel(200, 150, Rgb([255, 255, 255]));
    img.save(&src_img).unwrap();

    let out_img = tmp.path().join("rendered.jpg");
    let pts = vec![anycount_prototype::services::renderer::Point2D { x: 200.0, y: 150.0 }];
    anycount_prototype::services::renderer::render_annotated_jpeg(&src_img, &pts, None, &out_img)
        .unwrap();
    assert!(out_img.exists());
    let rendered = image::open(&out_img).unwrap().to_rgb8();
    // Center pixel at (200, 150) must be painted red (~230, 40, 35)
    let px = rendered.get_pixel(200, 150);
    assert!(px[0] > 180 && px[1] < 80, "expected red dot at center, got {:?}", px);

    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state);

    // Public /api/docs returns 6 methods and glossary
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/docs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let docs: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(docs["methods"].as_array().unwrap().len(), 6);
    assert!(docs["glossary"].as_array().unwrap().len() >= 12);

    // Upload without session returns 401
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/images")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=BOUND")
                .body(Body::from("--BOUND--\r\n"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Unlock then upload JPEG
    let unlock = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"2limanov"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let cookie = unlock
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    let mut jpg_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg).unwrap();
    let mut mp = Vec::new();
    mp.extend_from_slice(
        b"--BOUND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n",
    );
    mp.extend_from_slice(&jpg_bytes);
    mp.extend_from_slice(b"\r\n--BOUND--\r\n");

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/images")
                .header(header::COOKIE, cookie)
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=BOUND")
                .body(Body::from(mp))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let uploaded: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(uploaded["width"], 400);
    assert_eq!(uploaded["height"], 300);
}
```

- [ ] **Step 2: Implement `services/renderer.rs`, `routes/images.rs`, `routes/samples.rs`, `routes/docs.rs`, and wire into `lib.rs` & `main.rs`**

Write `prototype/src/services/mod.rs` and `prototype/src/services/renderer.rs`:
```rust
// prototype/src/services/mod.rs
pub mod renderer;
```

```rust
// prototype/src/services/renderer.rs
use std::{fs::File, io::BufWriter, path::Path};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

pub fn save_jpeg(img: &RgbImage, out_path: &Path, quality: u8) -> anyhow::Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);
    let mut enc = JpegEncoder::new_with_quality(&mut writer, quality);
    enc.encode_image(&DynamicImage::ImageRgb8(img.clone()))?;
    Ok(())
}

/// Draws Count-Anything style red dots (230, 40, 35) with crisp white outer ring (255, 255, 255)
/// and optional subtle 2x2 grid lines onto `src_path`, saving JPEG to `out_path`.
pub fn render_annotated_jpeg(
    src_path: &Path,
    points: &[Point2D],
    grid_boxes: Option<&[[i32; 4]]>,
    out_path: &Path,
) -> anyhow::Result<()> {
    let mut img = image::open(src_path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let w_i = w as i32;
    let h_i = h as i32;

    if let Some(boxes) = grid_boxes {
        for &[x0, y0, x1, y1] in boxes {
            for x in x0.max(0)..=x1.min(w_i - 1) {
                for &yy in &[y0, y1] {
                    if yy >= 0 && yy < h_i {
                        blend_px(&mut img, x as u32, yy as u32, [255, 255, 255], 0.40);
                    }
                }
            }
            for y in y0.max(0)..=y1.min(h_i - 1) {
                for &xx in &[x0, x1] {
                    if xx >= 0 && xx < w_i {
                        blend_px(&mut img, xx as u32, y as u32, [255, 255, 255], 0.40);
                    }
                }
            }
        }
    }

    let min_dim = w.min(h) as f64;
    let r_inner = (min_dim / 180.0).round().max(4.0);
    let ring_w = (r_inner / 2.0).floor().max(1.5);
    let r_outer = r_inner + ring_w;
    let pad = (r_outer.ceil() as i32) + 1;

    for pt in points {
        let cx = pt.x.clamp(0.0, (w - 1) as f64);
        let cy = pt.y.clamp(0.0, (h - 1) as f64);
        let x_start = ((cx as i32) - pad).max(0);
        let x_end = ((cx as i32) + pad).min(w_i - 1);
        let y_start = ((cy as i32) - pad).max(0);
        let y_end = ((cy as i32) + pad).min(h_i - 1);

        for py in y_start..=y_end {
            for px in x_start..=x_end {
                let dist = ((px as f64 - cx).powi(2) + (py as f64 - cy).powi(2)).sqrt();
                if dist <= r_inner {
                    blend_px(&mut img, px as u32, py as u32, [230, 40, 35], 0.92);
                } else if dist <= r_outer {
                    img.put_pixel(px as u32, py as u32, Rgb([255, 255, 255]));
                }
            }
        }
    }

    save_jpeg(&img, out_path, 92)
}

fn blend_px(img: &mut RgbImage, x: u32, y: u32, fg: [u8; 3], alpha: f32) {
    let bg = img.get_pixel(x, y).0;
    let out = [
        (fg[0] as f32 * alpha + bg[0] as f32 * (1.0 - alpha)).round() as u8,
        (fg[1] as f32 * alpha + bg[1] as f32 * (1.0 - alpha)).round() as u8,
        (fg[2] as f32 * alpha + bg[2] as f32 * (1.0 - alpha)).round() as u8,
    ];
    img.put_pixel(x, y, Rgb(out));
}
```

Write `prototype/src/routes/images.rs`:
```rust
use std::path::PathBuf;
use axum::{
    body::Body,
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use image::imageops::FilterType;
use serde_json::json;

use crate::{
    auth::{generate_token, now_rfc3339, RequireSession},
    error::AppError,
    services::renderer::save_jpeg,
    state::AppState,
};

pub async fn upload(
    State(st): State<AppState>,
    _session: RequireSession,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, AppError> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = "upload.jpg".to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart tidak valid: {e}")))?
    {
        if field.name() == Some("file") {
            if let Some(name) = field.file_name() {
                filename = name.to_string();
            }
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(format!("Gagal membaca file: {e}")))?;
            if data.len() > 15 * 1024 * 1024 {
                return Err(AppError::BadRequest("Ukuran gambar maksimal 15 MB".into()));
            }
            file_bytes = Some(data.to_vec());
            break;
        }
    }

    let raw = file_bytes.ok_or_else(|| AppError::BadRequest("Field 'file' wajib diisi".into()))?;
    let decoded = image::load_from_memory(&raw)
        .map_err(|e| AppError::BadRequest(format!("Format gambar tidak didukung: {e}")))?;
    let rgb = decoded.to_rgb8();
    let (w, h) = rgb.dimensions();

    let id = format!("img_{}", &generate_token()[..16]);
    let orig_path = st.cfg.data_dir.join("uploads").join(format!("{id}_orig.jpg"));
    let work_path = st.cfg.data_dir.join("uploads").join(format!("{id}_work.jpg"));

    save_jpeg(&rgb, &orig_path, 92)?;
    let max_side = w.max(h);
    if max_side > 1280 {
        let resized = decoded.resize(1280, 1280, FilterType::Lanczos3).to_rgb8();
        save_jpeg(&resized, &work_path, 90)?;
    } else {
        save_jpeg(&rgb, &work_path, 90)?;
    }

    sqlx::query(
        "INSERT INTO images (id, filename, width, height, orig_path, work_path, is_sample, created_at) \
         VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
    )
    .bind(&id)
    .bind(&filename)
    .bind(w as i64)
    .bind(h as i64)
    .bind(orig_path.to_string_lossy().to_string())
    .bind(work_path.to_string_lossy().to_string())
    .bind(now_rfc3339())
    .execute(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    Ok(Json(json!({
        "id": id,
        "filename": filename,
        "width": w,
        "height": h,
        "url": format!("/api/images/{id}/raw"),
    })))
}

pub async fn get_raw(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, AppError> {
    let row: Option<(String,)> =
        sqlx::query_as("SELECT orig_path FROM images WHERE id = ?")
            .bind(&id)
            .fetch_optional(&st.pool)
            .await
            .map_err(|e| AppError::Internal(e.into()))?;

    let (path_str,) = row.ok_or_else(|| AppError::NotFound("Gambar tidak ditemukan".into()))?;
    let bytes = tokio::fs::read(PathBuf::from(&path_str))
        .await
        .map_err(|_| AppError::NotFound("File gambar tidak ditemukan di disk".into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "public, max-age=86400")
        .body(Body::from(bytes))
        .unwrap())
}
```

Write `prototype/src/routes/samples.rs` (seeds the research sample images into `images` table with deterministic IDs `sample_cats`, `sample_coins`, `sample_tennis`, `sample_containers`, `sample_candy`, `sample_candy_bnw`, `sample_blueberry`, `sample_blueberry_bnw`, `sample_blueberry_bnw2`, `sample_apples` and serves pre-computed benchmark montages):
```rust
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use image::imageops::FilterType;
use serde_json::json;

use crate::{auth::now_rfc3339, error::AppError, services::renderer::save_jpeg, state::AppState};

const PRESETS: &[(&str, &str, &str, i64)] = &[
    ("sample_cats", "cats.jpg", "cats", 2),
    ("sample_coins", "coins.jpg", "coins", 24),
    ("sample_tennis", "tennis-ball.jpg", "tennis ball", 43),
    ("sample_containers", "containers.jpg", "container", 64),
    ("sample_blueberry", "Heidelbeeren.jpg", "blueberry", 113),
    ("sample_blueberry_bnw", "Heidelbeeren-bnw.jpg", "blueberry", 113),
    ("sample_blueberry_bnw2", "Heidelbeeren-bnw2.jpg", "blueberry", 113),
    ("sample_candy", "candy-ball.jpg", "candy", 132),
    ("sample_candy_bnw", "candy-bnw.jpg", "candy", 132),
    ("sample_apples", "apples.jpg", "apples", 289),
];

pub async fn seed_research_samples(st: &AppState) -> anyhow::Result<()> {
    let samples_dir = st.cfg.repo_root.join("research/samples");
    let alt_test_dir = st.cfg.repo_root.join("docs/image/test");

    for &(id, filename, _default_query, _gt) in PRESETS {
        let candidate_a = samples_dir.join(filename);
        let candidate_b = alt_test_dir.join(filename);
        let src = if candidate_a.exists() {
            candidate_a
        } else if candidate_b.exists() {
            candidate_b
        } else {
            continue;
        };

        let Ok(decoded) = image::open(&src) else {
            continue;
        };
        let rgb = decoded.to_rgb8();
        let (w, h) = rgb.dimensions();
        let work_path = st.cfg.data_dir.join("uploads").join(format!("{id}_work.jpg"));
        if !work_path.exists() {
            if w.max(h) > 1280 {
                let resized = decoded.resize(1280, 1280, FilterType::Lanczos3).to_rgb8();
                save_jpeg(&resized, &work_path, 90)?;
            } else {
                save_jpeg(&rgb, &work_path, 90)?;
            }
        }

        sqlx::query(
            "INSERT INTO images (id, filename, width, height, orig_path, work_path, is_sample, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, 1, ?) \
             ON CONFLICT(id) DO UPDATE SET orig_path = excluded.orig_path, work_path = excluded.work_path",
        )
        .bind(id)
        .bind(filename)
        .bind(w as i64)
        .bind(h as i64)
        .bind(src.to_string_lossy().to_string())
        .bind(work_path.to_string_lossy().to_string())
        .bind(now_rfc3339())
        .execute(&st.pool)
        .await?;
    }
    Ok(())
}

pub async fn list_samples(State(st): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let rows: Vec<(String, String, i64, i64)> = sqlx::query_as(
        "SELECT id, filename, width, height FROM images WHERE is_sample = 1 ORDER BY id",
    )
    .fetch_all(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let presets: Vec<_> = PRESETS
        .iter()
        .filter_map(|&(id, filename, default_query, gt)| {
            let found = rows.iter().find(|r| r.0 == id)?;
            Some(json!({
                "id": id,
                "filename": filename,
                "defaultQuery": default_query,
                "groundTruth": gt,
                "width": found.2,
                "height": found.3,
                "url": format!("/api/images/{id}/raw"),
            }))
        })
        .collect();

    let montages = vec![
        json!({"id": "cats", "title": "Cats & Open-Vocab Remote (GT: 2)", "url": "/api/samples/montages/montage_cats_and_remotes.jpg", "best": "M5 DeepSeek (3.2s) / M3 Gemini (5.0s)"}),
        json!({"id": "coins", "title": "Ancient Coins 4x6 Grid (GT: 24)", "url": "/api/samples/montages/montage_coins.jpg", "best": "M5 DeepSeek (24/24, 3.1s) & M3 Gemini (24/24)"}),
        json!({"id": "tennis", "title": "Tennis Court Balls (GT: 43)", "url": "/api/samples/montages/montage_tennis_ball.jpg", "best": "M3 Gemini Single (42/43, 7.7s)"}),
        json!({"id": "containers", "title": "Port Shipping Containers (GT: 64)", "url": "/api/samples/montages/montage_containers.jpg", "best": "M4 Gemini 2x2 (65/64) & M6 DeepSeek 2x2 (61/64, 2.5s)"}),
        json!({"id": "candy", "title": "Colorful Candy Pile (GT: ~132)", "url": "/api/samples/montages/montage_candy_ball.jpg", "best": "M3 Gemini Single & M1 SAM3 (128/132, 0 double-dots)"}),
        json!({"id": "candy_bnw", "title": "Black & White Candy Ablation (GT: ~132)", "url": "/api/samples/montages/montage_candy_bnw.jpg", "best": "M6 DeepSeek 2x2 (133/132, 3.6s)"}),
        json!({"id": "blueberry", "title": "Blueberries Color (GT: ~113)", "url": "/api/samples/montages/montage_heidelbeeren.jpg", "best": "M6 DeepSeek 2x2 (121) & M3 Gemini (122)"}),
        json!({"id": "blueberry_bnw", "title": "Blueberries Grayscale (GT: ~113)", "url": "/api/samples/montages/montage_heidelbeeren_bnw.jpg", "best": "M3 Gemini Single (112/113, 0.9% err)"}),
        json!({"id": "blueberry_bnw2", "title": "Blueberries High-Contrast B&W (GT: ~113)", "url": "/api/samples/montages/montage_heidelbeeren_bnw2.jpg", "best": "M4 Gemini 2x2 (127, cleanest centering)"}),
        json!({"id": "apples", "title": "Dense Hundreds of Apples (GT: ~289-339)", "url": "/api/samples/montages/montage_apples.jpg", "best": "M2 Offline ONNX v2 (290 in 0.21s) & M6 DeepSeek 2x2 (336 in 5.6s)"}),
        json!({"id": "m2_flywheel", "title": "Metode 2 Active Learning Flywheel (Apple v1 -> LLM -> v2)", "url": "/api/samples/montages/montage_m2_flywheel_apple_val_2.jpg", "best": "Error dropped from 77.7% to 2.33% (0.14s offline)"}),
    ];

    Ok(Json(json!({ "samples": presets, "montages": montages })))
}

pub async fn get_montage(
    State(st): State<AppState>,
    Path(filename): Path<String>,
) -> Result<Response, AppError> {
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err(AppError::BadRequest("Nama file tidak valid".into()));
    }
    let path = st
        .cfg
        .repo_root
        .join("research/spikes/003-method-comparison/results")
        .join(&filename);
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|_| AppError::NotFound("Montage tidak ditemukan".into()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "public, max-age=86400")
        .body(Body::from(bytes))
        .unwrap())
}
```

Write `prototype/src/routes/docs.rs`:
```rust
use axum::Json;
use serde_json::{json, Value};

pub async fn get_docs() -> Json<Value> {
    Json(json!({
        "methods": [
            {
                "id": "m1",
                "code": "M1",
                "name": "Count-Anything (SAM3 3.4 GB)",
                "badge": "Heavy Baseline",
                "color": "ochre",
                "speed": "14s - 28s",
                "footprint": "3.4 GB Checkpoint (7.2 GB RAM)",
                "summary": "Model riset berbasis Segment Anything 3 (848M parameter). Menempatkan titik di pusat mask tiap benda, namun sangat berat di RAM/GPU dan sensitif terhadap bayangan kontras tinggi.",
                "bestFor": "Pembanding laboratorium untuk foto warna standar."
            },
            {
                "id": "m2",
                "code": "M2",
                "name": "Offline ONNX Pack + LLM Validator",
                "badge": "On-Device + Flywheel",
                "color": "mint",
                "speed": "0.14s - 0.21s (Offline)",
                "footprint": "2.89 MB per objek (.onnx INT8)",
                "summary": "Aplikasi mengunduh bobot ringan .onnx sesuai objek yang dicari (mis. apel 2.89 MB), menghitung instan tanpa internet, lalu divalidasi & dilatih ulang otomatis oleh Guru LLM.",
                "bestFor": "Kategori benda populer yang sering dihitung berulang kali dengan biaya server Rp 0."
            },
            {
                "id": "m3",
                "code": "M3",
                "name": "Gemini 3.8 Flash Single-Pass",
                "badge": "Best Grounding",
                "color": "pink",
                "speed": "5s - 19s",
                "footprint": "0 MB (Cloud API)",
                "summary": "Mengirim foto utuh 1x ke gemini-3.8-flash-high untuk mendapatkan koordinat [y, x] tiap benda. Penempatan titik per-objek paling jujur dan presisi untuk foto jarak dekat hingga 130 benda.",
                "bestFor": "Pencarian kata bebas (open-vocabulary) foto jarak dekat & Validator pembuat label training M2."
            },
            {
                "id": "m4",
                "code": "M4",
                "name": "Gemini 3.8 Flash 2x2 Tiled + NMS",
                "badge": "High-Detail Dense",
                "color": "lavender",
                "speed": "11s - 31s",
                "footprint": "0 MB (4x Parallel Cloud)",
                "summary": "Memotong gambar menjadi 4 petak (overlap 8%), memanggil Gemini paralel, dan menyatukan perbatasan dengan Adaptive Point-NMS. Mengangkat akurasi kontainer jauh dari 29 menjadi 65/64.",
                "bestFor": "Objek kecil/jarak jauh (kontainer pelabuhan) dan foto minim warna/gelap."
            },
            {
                "id": "m5",
                "code": "M5",
                "name": "DeepSeek V4.1 Flash Single-Pass",
                "badge": "Fast Sparse",
                "color": "peach",
                "speed": "2.9s - 5.7s",
                "footprint": "0 MB (Cloud API)",
                "summary": "Menggunakan pola prompt Count-Then-Locate [x, y] pada deepseek-v4.1-flash-1017 tanpa jeda thinking panjang. Sangat cepat untuk jumlah benda <= 45.",
                "bestFor": "Hitung cepat objek sedikit-menengah (kucing, koin, bola tenis)."
            },
            {
                "id": "m6",
                "code": "M6",
                "name": "DeepSeek V4.1 Flash 2x2 Tiled + NMS",
                "badge": "Fastest Dense Cloud",
                "color": "teal",
                "speed": "2.5s - 5.6s (Avg 3.79s)",
                "footprint": "0 MB (4x Parallel Cloud)",
                "summary": "Memadukan kecepatan DeepSeek V4.1 Flash dengan pembagian 2x2 paralel + Point-NMS. Menghilangkan gejala loop pada ratusan apel (336 apel dalam 5.6 detik).",
                "bestFor": "Mode hitung cepat di cloud untuk tumpukan padat ratusan benda."
            }
        ],
        "glossary": [
            {"term": "LLM / VLM (Vision-Language Model)", "plain": "Otak AI pintar di cloud (seperti Gemini dan DeepSeek) yang bisa melihat foto sekaligus memahami perintah teks bebas manusia."},
            {"term": "Open-Vocabulary", "plain": "Kemampuan mencari benda apa saja cukup dengan mengetik namanya tanpa dibatasi pilihan menu kategori yang kaku."},
            {"term": "Spatial Pointing [0..1000]", "plain": "Cara AI menunjuk letak pusat benda memakai skala angka seribu dari kiri ke kanan dan dari atas ke bawah."},
            {"term": "Single-Pass (Foto Utuh)", "plain": "Mengirim satu lembar foto utuh dalam sekali tanya tanpa dipotong-potong."},
            {"term": "Tiled Grid 2x2 (Petak 2x2)", "plain": "Memotong foto menjadi 4 kuadran seperti kaca pembesar agar benda mungil di kejauhan terlihat 2x lebih jelas oleh AI."},
            {"term": "Overlap 8%", "plain": "Lebihan area potong di garis tengah supaya benda yang kebetulan berada pas di garis potong tidak terbelah dua."},
            {"term": "Point-NMS (Penghapus Titik Kembar)", "plain": "Saringan otomatis yang menggabungkan dua titik berdekatan di garis sambungan petak agar satu benda tidak terhitung dua kali."},
            {"term": "ONNX & Kuantisasi INT8", "plain": "Format file model AI ringkas untuk HP/komputer tanpa internet; INT8 memadatkan ukuran model dari 10 MB menjadi ~2.89 MB."},
            {"term": "Cold-Start (Objek Baru di M2)", "plain": "Kondisi saat pengguna mencari benda yang belum punya file .onnx lokal, sehingga sistem meminta bantuan LLM terlebih dahulu."},
            {"term": "Distillation Flywheel (Guru ke Murid)", "plain": "Siklus otomatis di mana titik jawaban dari AI besar (Guru LLM) dipakai melatih model kecil (.onnx) agar ikut menjadi pintar."},
            {"term": "Ground Truth (GT) & MAPE", "plain": "GT adalah jumlah benda asli hasil hitungan teliti manusia; MAPE adalah persentase rata-rata melesetnya hitungan AI (makin mendekati 0% makin akurat)."},
            {"term": "False Positive & False Negative", "plain": "False Positive adalah salah menaruh titik di tempat bukan benda; False Negative adalah benda asli yang terlewat belum ditandai."}
        ]
    }))
}
```

Update `prototype/src/routes/mod.rs`:
```rust
pub mod auth;
pub mod docs;
pub mod images;
pub mod samples;
```

Update `prototype/src/lib.rs` to mount `/api/images`, `/api/images/{id}/raw`, `/api/samples`, `/api/samples/montages/{filename}`, and `/api/docs`, and call `routes::samples::seed_research_samples(&state).await?` inside `prototype/src/main.rs` after `AppState::init`.

- [ ] **Step 3: Run `cargo test` on VPS to verify Task 2 passes**

Run: `ssh root@49.128.191.225 'export PATH="$HOME/.cargo/bin:$PATH"; cd /root/workspace/anycount/prototype && cargo test'`
Expected: `auth_and_health` (1 passed) and `images_and_renderer` (1 passed) both green.

- [ ] **Step 4: Commit Task 2**

```bash
git add prototype/src/ prototype/tests/images_and_renderer.rs
git commit -m "feat(prototype): add Count-Anything dot renderer, image upload, sample seeding, and docs API"
```

---

### Task 3: Native Rust VLM Engine (M3/M4 Gemini & M5/M6 DeepSeek) + 2x2 Tiled Grid & Point-NMS

**Files:**
- Create: `prototype/src/services/tiling.rs`
- Create: `prototype/src/services/vlm.rs`
- Modify: `prototype/src/services/mod.rs`

**Interfaces:**
- Consumes: `Point2D` from `services::renderer`, `Config`, `reqwest::Client`.
- Produces:
  - `services::tiling::TileSpec { pub tile_id: String, pub crop_box: [u32; 4], pub core_box: [f64; 4], pub seam_tol_px: f64, pub jpeg_bytes: Vec<u8>, pub tile_w: u32, pub tile_h: u32 }`
  - `services::tiling::split_2x2(img_path: &Path, overlap_ratio: f64) -> anyhow::Result<(u32, u32, Vec<[i32; 4]>, Vec<TileSpec>)>`
  - `services::tiling::adaptive_point_nms(points: &[TiledPoint], img_w: u32, img_h: u32) -> Vec<Point2D>`
  - `services::vlm::VlmRunOutput { pub points: Vec<Point2D>, pub grid_boxes: Option<Vec<[i32; 4]>>, pub prompt_tokens: i64, pub completion_tokens: i64, pub raw_preview: String }`
  - `services::vlm::run_vlm_method(http: &reqwest::Client, cfg: &Config, method_id: &str, work_img_path: &Path, orig_w: u32, orig_h: u32, query: &str) -> anyhow::Result<VlmRunOutput>` (handles `"m3"`, `"m4"`, `"m5"`, `"m6"`).

- [ ] **Step 1: Implement `services/tiling.rs` and `services/vlm.rs` with unit tests first**

Write `prototype/src/services/tiling.rs` (including inline `#[cfg(test)] mod tests` testing `split_2x2` geometry and `adaptive_point_nms` seam deduplication):
```rust
use std::{io::Cursor, path::Path};
use image::{codecs::jpeg::JpegEncoder, DynamicImage};

use crate::services::renderer::Point2D;

#[derive(Clone, Debug)]
pub struct TileSpec {
    pub tile_id: String,
    pub crop_box: [u32; 4], // [x0, y0, x1, y1]
    pub core_box: [f64; 4], // [x0, y0, x1, y1]
    pub seam_tol_px: f64,
    pub jpeg_bytes: Vec<u8>,
    pub tile_w: u32,
    pub tile_h: u32,
}

#[derive(Clone, Debug)]
pub struct TiledPoint {
    pub x: f64,
    pub y: f64,
    pub tile_id: String,
}

pub fn split_2x2(
    img_path: &Path,
    overlap_ratio: f64,
) -> anyhow::Result<(u32, u32, Vec<[i32; 4]>, Vec<TileSpec>)> {
    let img = image::open(img_path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let rows = 2usize;
    let cols = 2usize;
    let cell_w = w as f64 / cols as f64;
    let cell_h = h as f64 / rows as f64;
    let pad_x = cell_w * overlap_ratio;
    let pad_y = cell_h * overlap_ratio;
    let seam_tol = cell_w.min(cell_h) * 0.04;

    let mut core_boxes = Vec::with_capacity(4);
    let mut tiles = Vec::with_capacity(4);

    for r in 0..rows {
        for c in 0..cols {
            let core_x0 = c as f64 * cell_w;
            let core_y0 = r as f64 * cell_h;
            let core_x1 = (c + 1) as f64 * cell_w;
            let core_y1 = (r + 1) as f64 * cell_h;
            core_boxes.push([
                core_x0 as i32,
                core_y0 as i32,
                core_x1 as i32,
                core_y1 as i32,
            ]);

            let crop_x0 = if c > 0 { (core_x0 - pad_x).round().max(0.0) as u32 } else { 0 };
            let crop_y0 = if r > 0 { (core_y0 - pad_y).round().max(0.0) as u32 } else { 0 };
            let crop_x1 = if c < cols - 1 {
                (core_x1 + pad_x).round().min(w as f64) as u32
            } else {
                w
            };
            let crop_y1 = if r < rows - 1 {
                (core_y1 + pad_y).round().min(h as f64) as u32
            } else {
                h
            };

            let tw = crop_x1.saturating_sub(crop_x0).max(1);
            let th = crop_y1.saturating_sub(crop_y0).max(1);
            let sub = image::imageops::crop_imm(&img, crop_x0, crop_y0, tw, th).to_image();

            let mut jpeg_bytes = Vec::new();
            let mut cursor = Cursor::new(&mut jpeg_bytes);
            let mut enc = JpegEncoder::new_with_quality(&mut cursor, 90);
            enc.encode_image(&DynamicImage::ImageRgb8(sub))?;

            tiles.push(TileSpec {
                tile_id: format!("r{r}_c{c}"),
                crop_box: [crop_x0, crop_y0, crop_x1, crop_y1],
                core_box: [core_x0, core_y0, core_x1, core_y1],
                seam_tol_px: seam_tol,
                jpeg_bytes,
                tile_w: tw,
                tile_h: th,
            });
        }
    }

    Ok((w, h, core_boxes, tiles))
}

pub fn adaptive_point_nms(points: &[TiledPoint], img_w: u32, img_h: u32) -> Vec<Point2D> {
    let min_dim = img_w.min(img_h) as f64;
    let nms_radius = if points.len() >= 4 {
        let mut nn_dists = Vec::new();
        for (i, p1) in points.iter().enumerate() {
            let mut dmin = f64::INFINITY;
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let d = (p1.x - p2.x).hypot(p1.y - p2.y);
                    if d < dmin {
                        dmin = d;
                    }
                }
            }
            if dmin.is_finite() && dmin > 1.0 {
                nn_dists.push(dmin);
            }
        }
        nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_nn = if nn_dists.is_empty() {
            20.0
        } else {
            nn_dists[nn_dists.len() / 2]
        };
        (median_nn * 0.55).clamp(8.0, min_dim * 0.06)
    } else {
        (min_dim * 0.04).max(8.0)
    };

    let mut kept: Vec<TiledPoint> = Vec::new();
    for pt in points {
        let mut merged = false;
        for k in &mut kept {
            let dist = (pt.x - k.x).hypot(pt.y - k.y);
            let factor = if pt.tile_id != k.tile_id { 1.35 } else { 0.75 };
            if dist < nms_radius * factor {
                k.x = (k.x + pt.x) / 2.0;
                k.y = (k.y + pt.y) / 2.0;
                merged = true;
                break;
            }
        }
        if !merged {
            kept.push(pt.clone());
        }
    }

    kept.into_iter()
        .map(|p| Point2D {
            x: (p.x * 100.0).round() / 100.0,
            y: (p.y * 100.0).round() / 100.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use tempfile::TempDir;

    #[test]
    fn split_2x2_and_seam_nms_deduplicates_border_pair() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("grid.jpg");
        RgbImage::from_pixel(640, 480, Rgb([20, 30, 40])).save(&p).unwrap();

        let (w, h, cores, tiles) = split_2x2(&p, 0.08).unwrap();
        assert_eq!((w, h), (640, 480));
        assert_eq!(cores.len(), 4);
        assert_eq!(tiles.len(), 4);
        assert!(tiles[0].tile_w > 320 && tiles[0].tile_h > 240);

        let raw = vec![
            TiledPoint { x: 100.0, y: 100.0, tile_id: "r0_c0".into() },
            TiledPoint { x: 319.0, y: 120.0, tile_id: "r0_c0".into() },
            TiledPoint { x: 322.0, y: 121.0, tile_id: "r0_c1".into() }, // duplicate on vertical seam
            TiledPoint { x: 500.0, y: 380.0, tile_id: "r1_c1".into() },
        ];
        let dedup = adaptive_point_nms(&raw, 640, 480);
        assert_eq!(dedup.len(), 3);
    }
}
```

Write `prototype/src/services/vlm.rs` (including inline `#[cfg(test)] mod tests` verifying SSE parsing + `[y, x]` conversion for Gemini and `JSON + data: [DONE]` parsing + `[x, y]` conversion for DeepSeek):
```rust
use std::path::Path;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use futures::future::join_all;
use regex::Regex;
use serde_json::{json, Value};

use crate::{
    config::Config,
    services::{
        renderer::Point2D,
        tiling::{adaptive_point_nms, split_2x2, TiledPoint},
    },
};

#[derive(Clone, Debug)]
pub struct VlmRunOutput {
    pub points: Vec<Point2D>,
    pub grid_boxes: Option<Vec<[i32; 4]>>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub raw_preview: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VlmFlavor {
    GeminiYx,
    DeepSeekXy,
}

pub fn parse_gateway_response(raw_body: &str) -> (String, i64, i64) {
    let trimmed = raw_body.trim_start();
    if trimmed.starts_with('{') {
        let mut de = serde_json::Deserializer::from_str(trimmed).into_iter::<Value>();
        if let Some(Ok(val)) = de.next() {
            let content = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let p_tok = val["usage"]["prompt_tokens"].as_i64().unwrap_or(0);
            let c_tok = val["usage"]["completion_tokens"].as_i64().unwrap_or(0);
            return (content, p_tok, c_tok);
        }
    }

    let mut parts = Vec::new();
    let mut p_tok = 0;
    let mut c_tok = 0;
    for line in raw_body.lines() {
        let line = line.trim();
        if !line.starts_with("data: ") {
            continue;
        }
        let payload = line[6..].trim();
        if payload == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
            if let Some(s) = chunk["choices"][0]["delta"]["content"].as_str() {
                parts.push(s.to_string());
            }
            if chunk.get("usage").is_some() {
                p_tok = chunk["usage"]["prompt_tokens"].as_i64().unwrap_or(p_tok);
                c_tok = chunk["usage"]["completion_tokens"].as_i64().unwrap_or(c_tok);
            }
        }
    }
    (parts.join(""), p_tok, c_tok)
}

pub fn extract_norm_pairs(text: &str) -> Vec<[f64; 2]> {
    let re_fence = Regex::new(r"(?s)^```(?:json)?\s*|\s*```$").unwrap();
    let cleaned = re_fence.replace_all(text.trim(), "");
    if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
        let arr_opt = if val.is_array() {
            val.as_array()
        } else {
            val.get("points").and_then(|p| p.as_array())
        };
        if let Some(arr) = arr_opt {
            let mut out = Vec::new();
            for item in arr {
                if let Some(pair) = item.as_array() {
                    if pair.len() >= 2 {
                        if let (Some(a), Some(b)) = (pair[0].as_f64(), pair[1].as_f64()) {
                            out.push([a, b]);
                        }
                    }
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
    }

    let re_pair = Regex::new(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]").unwrap();
    re_pair
        .captures_iter(&cleaned)
        .filter_map(|cap| {
            let a = cap.get(1)?.as_str().parse::<f64>().ok()?;
            let b = cap.get(2)?.as_str().parse::<f64>().ok()?;
            Some([a, b])
        })
        .collect()
}

pub fn norm_to_px(pair: [f64; 2], flavor: VlmFlavor, w: f64, h: f64) -> (f64, f64) {
    match flavor {
        VlmFlavor::GeminiYx => {
            let (ny, nx) = (pair[0], pair[1]);
            ((nx / 1000.0) * w, (ny / 1000.0) * h)
        }
        VlmFlavor::DeepSeekXy => {
            let (nx, ny) = (pair[0], pair[1]);
            ((nx / 1000.0) * w, (ny / 1000.0) * h)
        }
    }
}

async fn call_vlm_bytes(
    http: &reqwest::Client,
    cfg: &Config,
    flavor: VlmFlavor,
    jpeg_bytes: &[u8],
    query: &str,
    is_crop: bool,
) -> anyhow::Result<(Vec<[f64; 2]>, i64, i64, String)> {
    if cfg.vlm_api_key.trim().is_empty() {
        anyhow::bail!("VLM_API_KEY belum dikonfigurasi di .env server");
    }
    let b64 = B64.encode(jpeg_bytes);
    let scope = if is_crop { "image crop" } else { "image" };

    let payload = match flavor {
        VlmFlavor::GeminiYx => {
            let prompt = format!(
                "Locate every individual \"{query}\" visible in this {scope}.\n\
                 Place exactly one point [y, x] at the center of each visible instance (normalized 0 to 1000, \
                 where [0,0] is top-left and [1000,1000] is bottom-right).\n\
                 IMPORTANT: Do NOT write coordinates in your thinking process; keep thinking under 3 sentences \
                 and immediately output valid JSON:\n\
                 {{\"points\": [[y1, x1], [y2, x2], ...]}}"
            );
            json!({
                "model": cfg.gemini_model,
                "reasoning_effort": "low",
                "max_tokens": 4096,
                "temperature": 0.1,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            })
        }
        VlmFlavor::DeepSeekXy => {
            let max_tok = if is_crop { 4096 } else { 8192 };
            let prompt = format!(
                "How many {query} are visible in this {scope}? First count them carefully, \
                 then output a JSON object with the exact count and the center coordinates [x, y] \
                 (normalized 0 to 1000, where x is horizontal left-to-right and y is vertical top-to-bottom) \
                 for each visible {query}:\n\
                 {{\"count\": <int>, \"points\": [[x1, y1], [x2, y2], ...]}}"
            );
            json!({
                "model": cfg.deepseek_model,
                "max_tokens": max_tok,
                "temperature": 0.1,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            })
        }
    };

    let resp = http
        .post(&cfg.vlm_api_url)
        .bearer_auth(&cfg.vlm_api_key)
        .json(&payload)
        .send()
        .await?;

    let status = resp.status();
    let raw_body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("Upstream VLM HTTP {status}: {}", &raw_body[..raw_body.len().min(240)]);
    }

    let (content, p_tok, c_tok) = parse_gateway_response(&raw_body);
    let pairs = extract_norm_pairs(&content);
    let preview: String = content.chars().take(500).collect();
    Ok((pairs, p_tok, c_tok, preview))
}

pub async fn run_vlm_method(
    http: &reqwest::Client,
    cfg: &Config,
    method_id: &str,
    work_img_path: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) -> anyhow::Result<VlmRunOutput> {
    let (flavor, tiled) = match method_id {
        "m3" => (VlmFlavor::GeminiYx, false),
        "m4" => (VlmFlavor::GeminiYx, true),
        "m5" => (VlmFlavor::DeepSeekXy, false),
        "m6" => (VlmFlavor::DeepSeekXy, true),
        other => anyhow::bail!("Metode VLM tidak dikenal: {other}"),
    };

    if !tiled {
        let jpeg_bytes = tokio::fs::read(work_img_path).await?;
        let (pairs, p_tok, c_tok, preview) =
            call_vlm_bytes(http, cfg, flavor, &jpeg_bytes, query, false).await?;
        let points = pairs
            .into_iter()
            .map(|pair| {
                let (x, y) = norm_to_px(pair, flavor, orig_w as f64, orig_h as f64);
                Point2D {
                    x: (x * 100.0).round() / 100.0,
                    y: (y * 100.0).round() / 100.0,
                }
            })
            .collect();
        return Ok(VlmRunOutput {
            points,
            grid_boxes: None,
            prompt_tokens: p_tok,
            completion_tokens: c_tok,
            raw_preview: preview,
        });
    }

    let (work_w, work_h, work_cores, tiles) = split_2x2(work_img_path, 0.08)?;
    let scale_x = orig_w as f64 / work_w as f64;
    let scale_y = orig_h as f64 / work_h as f64;

    let futs = tiles.iter().map(|t| async move {
        let res = call_vlm_bytes(http, cfg, flavor, &t.jpeg_bytes, query, true).await?;
        Ok::<_, anyhow::Error>((t, res))
    });

    let results = join_all(futs).await;
    let mut all_tiled_pts = Vec::new();
    let mut total_p_tok = 0;
    let mut total_c_tok = 0;
    let mut previews = Vec::new();

    for item in results {
        let (tile, (pairs, pt, ct, prev)) = item?;
        total_p_tok += pt;
        total_c_tok += ct;
        previews.push(format!("[{}] {}", tile.tile_id, prev));

        let [crop_x0, crop_y0, _, _] = tile.crop_box;
        let [core_x0, core_y0, core_x1, core_y1] = tile.core_box;
        let tol = tile.seam_tol_px;

        for pair in pairs {
            let (lx, ly) = norm_to_px(pair, flavor, tile.tile_w as f64, tile.tile_h as f64);
            let gx = crop_x0 as f64 + lx;
            let gy = crop_y0 as f64 + ly;
            if gx >= core_x0 - tol && gx <= core_x1 + tol && gy >= core_y0 - tol && gy <= core_y1 + tol {
                all_tiled_pts.push(TiledPoint {
                    x: gx * scale_x,
                    y: gy * scale_y,
                    tile_id: tile.tile_id.clone(),
                });
            }
        }
    }

    let dedup = adaptive_point_nms(&all_tiled_pts, orig_w, orig_h);
    let orig_cores: Vec<[i32; 4]> = work_cores
        .into_iter()
        .map(|[x0, y0, x1, y1]| {
            [
                (x0 as f64 * scale_x).round() as i32,
                (y0 as f64 * scale_y).round() as i32,
                (x1 as f64 * scale_x).round() as i32,
                (y1 as f64 * scale_y).round() as i32,
            ]
        })
        .collect();

    Ok(VlmRunOutput {
        points: dedup,
        grid_boxes: Some(orig_cores),
        prompt_tokens: total_p_tok,
        completion_tokens: total_c_tok,
        raw_preview: previews.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemini_sse_yx_and_deepseek_json_done_xy() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"points\\\": [[250, 500]]}\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20}}\n\
                   data: [DONE]\n";
        let (txt, pt, ct) = parse_gateway_response(sse);
        assert_eq!((pt, ct), (100, 20));
        let pairs = extract_norm_pairs(&txt);
        assert_eq!(pairs, vec![[250.0, 500.0]]);
        let (px, py) = norm_to_px(pairs[0], VlmFlavor::GeminiYx, 1000.0, 800.0);
        assert_eq!((px, py), (500.0, 200.0));

        let ds = "{\"choices\":[{\"message\":{\"content\":\"Found 2.\\n```json\\n{\\\"count\\\":2,\\\"points\\\":[[700, 300]]}\\n```\"}}],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":15}}data: [DONE]";
        let (txt2, pt2, ct2) = parse_gateway_response(ds);
        assert_eq!((pt2, ct2), (50, 15));
        let pairs2 = extract_norm_pairs(&txt2);
        let (px2, py2) = norm_to_px(pairs2[0], VlmFlavor::DeepSeekXy, 1000.0, 800.0);
        assert_eq!((px2, py2), (700.0, 240.0));
    }
}
```

Update `prototype/src/services/mod.rs`:
```rust
pub mod renderer;
pub mod tiling;
pub mod vlm;
```

- [ ] **Step 2: Run unit tests on VPS to verify Task 3 passes**

Run: `ssh root@49.128.191.225 'export PATH="$HOME/.cargo/bin:$PATH"; cd /root/workspace/anycount/prototype && cargo test --lib'`
Expected: `2 passed; 0 failed` (`split_2x2_and_seam_nms_deduplicates_border_pair` and `parses_gemini_sse_yx_and_deepseek_json_done_xy`).

- [ ] **Step 3: Commit Task 3**

```bash
git add prototype/src/services/
git commit -m "feat(prototype): implement native Rust VLM clients (Gemini & DeepSeek), 2x2 tiling, and Point-NMS"
```

---

### Task 4: Python Workers (`M1` SAM3 & `M2` ONNX + Auto-Train), Model Packs API & Multi-Method Runs SSE Orchestrator

**Files:**
- Create: `prototype/workers/ca_worker.py`
- Create: `prototype/workers/onnx_worker.py`
- Create: `prototype/src/services/m1_ca.rs`
- Create: `prototype/src/services/m2_onnx.rs`
- Create: `prototype/src/routes/packs.rs`
- Create: `prototype/src/routes/runs.rs`
- Modify: `prototype/src/services/mod.rs`
- Modify: `prototype/src/routes/mod.rs`
- Modify: `prototype/src/lib.rs`
- Modify: `prototype/src/main.rs`
- Test: `prototype/tests/runs_and_packs.rs`

**Interfaces:**
- Consumes: `AppState` (`heavy_semaphore`, `run_streams`), `RequireSession`, `services::vlm`, `services::renderer`.
- Produces:
  - `routes::packs::seed_default_apple_pack(st: &AppState) -> anyhow::Result<()>` (registers `apple_v2_distilled_int8.onnx` `2.89 MB`, aliases `["apple", "apples", "apel", "red apple", "green apple"]`, `val_mape_pct = 2.33`)
  - Endpoints:
    - `GET /api/packs` (lists all registered `.onnx` packs)
    - `GET /api/packs/{slug}/download` (downloads `.onnx` file)
    - `POST /api/packs/train` (`RequireSession`, body `{ imageId, query }`: calls Gemini Validator to obtain teacher points, then spawns `onnx_worker.py train` under `heavy_semaphore` to distill a new `<slug>_int8.onnx` pack and mark it `ready`)
    - `POST /api/runs` (`RequireSession`, body `{ imageId, query, methods: ["m1".."m6"] }`: inserts `runs` + `run_methods`, spawns concurrent tasks per selected method, returns `{ runId, status: "running", methods: [...] }`)
    - `GET /api/runs` (recent 20 runs)
    - `GET /api/runs/{id}` (full run state including per-method points & metadata)
    - `GET /api/runs/{id}/stream` (SSE stream emitting JSON snapshots whenever any method transitions `queued -> running -> completed | cold_start | failed`)
    - `GET /api/runs/{id}/render/{method_id}` (serves the rendered Count-Anything style JPEG)

- [ ] **Step 1: Write Python workers `prototype/workers/ca_worker.py` and `prototype/workers/onnx_worker.py`**

Write `prototype/workers/ca_worker.py`:
```python
"""
Thin worker for Method 1: Count-Anything (SAM3 3.4 GB checkpoint).
Outputs a single JSON line prefixed with ANYCOUNT_JSON: containing pixel coordinates
scaled to target width/height.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--orig-width", type=int, required=True)
    parser.add_argument("--orig-height", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.ca_repo)
    from count_anything import CountAnything

    img_path = Path(args.image).resolve()
    with Image.open(img_path) as im:
        w, h = im.size
    sx = args.orig_width / float(w)
    sy = args.orig_height / float(h)

    t0 = time.perf_counter()
    ckpt = Path(args.ca_repo) / "checkpoints" / "count_anything.pt"
    model = CountAnything(str(ckpt), output_dir=args.work_dir)
    result = model(str(img_path), args.query)[0]
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    pts = [
        {"x": round(float(p["x"]) * sx, 2), "y": round(float(p["y"]) * sy, 2)}
        for p in result.pred_points
    ]
    payload = {
        "count": len(pts),
        "points": pts,
        "elapsed_ms": elapsed_ms,
        "model_size_mb": 3410.0,
    }
    print("ANYCOUNT_JSON:" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
```

Write `prototype/workers/onnx_worker.py` (supports `infer` using adaptive `640px` single-pass + automatic `2x2` tiled ONNX pass for dense scenes, AND `train` to distill a new `.onnx` INT8 pack from LLM teacher points):
```python
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
```

- [ ] **Step 2: Write `prototype/tests/runs_and_packs.rs` testing pack lookup, Cold-Start on M2, and SSE run streaming**

```rust
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use image::{Rgb, RgbImage};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn m2_cold_start_and_run_creation_work() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state.clone());

    // Unlock session
    let unlock = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"2limanov"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let cookie = unlock
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    // Seed a dummy image row
    let img_path = tmp.path().join("uploads/img_1.jpg");
    RgbImage::from_pixel(320, 240, Rgb([10, 20, 30])).save(&img_path).unwrap();
    sqlx::query(
        "INSERT INTO images (id, filename, width, height, orig_path, work_path, is_sample, created_at) \
         VALUES ('img_1', 'test.jpg', 320, 240, ?, ?, 0, '2026-11-07T00:00:00Z')",
    )
    .bind(img_path.to_string_lossy().to_string())
    .bind(img_path.to_string_lossy().to_string())
    .execute(&state.pool)
    .await
    .unwrap();

    // Trigger M2 for an untrained object "pineapple" -> transitions to cold_start
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs")
                .header(header::COOKIE, &cookie)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"imageId":"img_1","query":"pineapple","methods":["m2"]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let created: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let run_id = created["runId"].as_str().unwrap().to_string();

    tokio::time::sleep(std::time::Duration::from_millis(120)).await;

    let res = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/runs/{run_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let detail: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(detail["methods"][0]["methodId"], "m2");
    assert_eq!(detail["methods"][0]["status"], "cold_start");
}
```

- [ ] **Step 3: Implement `services/m1_ca.rs`, `services/m2_onnx.rs`, `routes/packs.rs`, `routes/runs.rs`, and wire into `lib.rs` & `main.rs`**

Write `prototype/src/services/m1_ca.rs`:
```rust
use std::path::Path;
use serde_json::Value;
use tokio::process::Command;

use crate::{services::renderer::Point2D, state::AppState};

pub struct CaOutput {
    pub points: Vec<Point2D>,
    pub elapsed_ms: i64,
    pub model_size_mb: f64,
}

pub async fn run_count_anything(
    st: &AppState,
    work_img: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
    run_id: &str,
) -> anyhow::Result<CaOutput> {
    let _permit = st.heavy_semaphore.acquire().await?;
    let worker_script = Path::new("workers/ca_worker.py");
    let ca_work_dir = st.cfg.data_dir.join("tmp").join(format!("ca_{run_id}"));

    let out = Command::new(&st.cfg.ca_python_bin)
        .env_remove("PYTHONPATH")
        .arg(worker_script)
        .arg("--ca-repo")
        .arg(&st.cfg.ca_repo_dir)
        .arg("--image")
        .arg(work_img)
        .arg("--query")
        .arg(query)
        .arg("--orig-width")
        .arg(orig_w.to_string())
        .arg("--orig-height")
        .arg(orig_h.to_string())
        .arg("--work-dir")
        .arg(&ca_work_dir)
        .output()
        .await?;

    let _ = tokio::fs::remove_dir_all(&ca_work_dir).await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        anyhow::bail!("Count-Anything gagal: {}", &stderr[stderr.len().saturating_sub(400)..]);
    }

    let json_line = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("ANYCOUNT_JSON:"))
        .ok_or_else(|| anyhow::anyhow!("Output JSON Count-Anything tidak ditemukan"))?;
    let val: Value = serde_json::from_str(json_line)?;
    let points: Vec<Point2D> = serde_json::from_value(val["points"].clone())?;

    Ok(CaOutput {
        points,
        elapsed_ms: val["elapsed_ms"].as_i64().unwrap_or(0),
        model_size_mb: val["model_size_mb"].as_f64().unwrap_or(3410.0),
    })
}
```

Write `prototype/src/services/m2_onnx.rs`:
```rust
use std::path::{Path, PathBuf};
use serde_json::Value;
use tokio::process::Command;

use crate::{services::renderer::Point2D, state::AppState};

pub enum OnnxOutcome {
    Ready {
        pack_slug: String,
        points: Vec<Point2D>,
        grid_boxes: Option<Vec<[i32; 4]>>,
        elapsed_ms: i64,
        model_size_mb: f64,
        mode: String,
    },
    ColdStart {
        normalized_query: String,
    },
}

pub fn normalize_query_slug(q: &str) -> String {
    let lower: String = q
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = lower.trim_matches('_');
    trimmed.strip_suffix('s').unwrap_or(trimmed).to_string()
}

pub async fn find_ready_pack(st: &AppState, query: &str) -> anyhow::Result<Option<(String, PathBuf, f64)>> {
    let norm = query.trim().to_lowercase();
    let slug_guess = normalize_query_slug(query);
    let rows: Vec<(String, String, String, f64)> = sqlx::query_as(
        "SELECT slug, aliases_json, onnx_path, size_mb FROM model_packs WHERE status = 'ready'",
    )
    .fetch_all(&st.pool)
    .await?;

    for (slug, aliases_json, onnx_path, size_mb) in rows {
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
        if slug == norm
            || slug == slug_guess
            || aliases.iter().any(|a| a.eq_ignore_ascii_case(&norm) || a.eq_ignore_ascii_case(&slug_guess))
        {
            let p = PathBuf::from(&onnx_path);
            if p.exists() {
                return Ok(Some((slug, p, size_mb)));
            }
        }
    }
    Ok(None)
}

pub async fn run_m2_onnx(
    st: &AppState,
    work_img: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) -> anyhow::Result<OnnxOutcome> {
    let Some((slug, onnx_path, _size_mb)) = find_ready_pack(st, query).await? else {
        return Ok(OnnxOutcome::ColdStart {
            normalized_query: normalize_query_slug(query),
        });
    };

    let out = Command::new(&st.cfg.onnx_python_bin)
        .env_remove("PYTHONPATH")
        .arg("workers/onnx_worker.py")
        .arg("infer")
        .arg("--onnx")
        .arg(&onnx_path)
        .arg("--image")
        .arg(work_img)
        .arg("--orig-width")
        .arg(orig_w.to_string())
        .arg("--orig-height")
        .arg(orig_h.to_string())
        .output()
        .await?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("ONNX inference gagal: {}", &err[err.len().saturating_sub(300)..]);
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("ANYCOUNT_JSON:"))
        .ok_or_else(|| anyhow::anyhow!("Output JSON ONNX tidak ditemukan"))?;
    let val: Value = serde_json::from_str(json_line)?;
    let points: Vec<Point2D> = serde_json::from_value(val["points"].clone())?;
    let grid_boxes: Option<Vec<[i32; 4]>> = serde_json::from_value(val["grid_boxes"].clone()).ok();

    Ok(OnnxOutcome::Ready {
        pack_slug: slug,
        points,
        grid_boxes,
        elapsed_ms: val["elapsed_ms"].as_i64().unwrap_or(0),
        model_size_mb: val["model_size_mb"].as_f64().unwrap_or(2.89),
        mode: val["mode"].as_str().unwrap_or("single_640").to_string(),
    })
}
```

Write `prototype/src/routes/packs.rs` (seeds `apple_v2_distilled_int8.onnx` on startup, serves `/api/packs`, `/api/packs/{slug}/download`, and `/api/packs/train` which calls `run_vlm_method(..., "m3", ...)` to get teacher labels and distills a new `.onnx` pack via `onnx_worker.py train` under `heavy_semaphore`):
```rust
use std::path::PathBuf;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

use crate::{
    auth::{now_rfc3339, RequireSession},
    error::AppError,
    services::{m2_onnx::normalize_query_slug, vlm::run_vlm_method},
    state::AppState,
};

pub async fn seed_default_apple_pack(st: &AppState) -> anyhow::Result<()> {
    let apple_onnx = st
        .cfg
        .repo_root
        .join("research/spikes/004-offline-onnx-llm-validator/models/apple_v2_distilled_int8.onnx");
    if !apple_onnx.exists() {
        return Ok(());
    }
    let size_mb = ((apple_onnx.metadata()?.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
    let aliases = serde_json::to_string(&["apple", "apples", "apel", "red apple", "green apple"])?;

    sqlx::query(
        "INSERT INTO model_packs (slug, display_name, aliases_json, onnx_path, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at) \
         VALUES ('apple', 'Apple Detector v2 (LLM-Distilled INT8)', ?, ?, ?, 640, 35, 2.33, 'ready', ?) \
         ON CONFLICT(slug) DO UPDATE SET onnx_path = excluded.onnx_path, size_mb = excluded.size_mb",
    )
    .bind(aliases)
    .bind(apple_onnx.to_string_lossy().to_string())
    .bind(size_mb)
    .bind(now_rfc3339())
    .execute(&st.pool)
    .await?;
    Ok(())
}

pub async fn list_packs(State(st): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let rows: Vec<(String, String, String, f64, i64, i64, Option<f64>, String, String)> = sqlx::query_as(
        "SELECT slug, display_name, aliases_json, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at \
         FROM model_packs ORDER BY slug",
    )
    .fetch_all(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let packs: Vec<_> = rows
        .into_iter()
        .map(|r| {
            let aliases: Vec<String> = serde_json::from_str(&r.2).unwrap_or_default();
            json!({
                "slug": r.0,
                "displayName": r.1,
                "aliases": aliases,
                "sizeMb": r.3,
                "imgsz": r.4,
                "trainedEpochs": r.5,
                "valMapePct": r.6,
                "status": r.7,
                "updatedAt": r.8,
                "downloadUrl": format!("/api/packs/{}/download", r.0),
            })
        })
        .collect();
    Ok(Json(json!({ "packs": packs })))
}

pub async fn download_pack(
    State(st): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Response, AppError> {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT onnx_path FROM model_packs WHERE slug = ? AND status = 'ready'",
    )
    .bind(&slug)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let (onnx_path,) = row.ok_or_else(|| AppError::NotFound("Model pack tidak ditemukan".into()))?;
    let bytes = tokio::fs::read(&onnx_path)
        .await
        .map_err(|_| AppError::NotFound("File .onnx tidak ditemukan di disk".into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{slug}_int8.onnx\""),
        )
        .body(Body::from(bytes))
        .unwrap())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainPackReq {
    pub image_id: String,
    pub query: String,
}

pub async fn train_pack(
    State(st): State<AppState>,
    _session: RequireSession,
    Json(req): Json<TrainPackReq>,
) -> Result<impl IntoResponse, AppError> {
    let slug = normalize_query_slug(&req.query);
    if slug.is_empty() {
        return Err(AppError::BadRequest("Query objek tidak boleh kosong".into()));
    }

    let img_row: Option<(String, i64, i64)> = sqlx::query_as(
        "SELECT work_path, width, height FROM images WHERE id = ?",
    )
    .bind(&req.image_id)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;
    let (work_path, _orig_w, _orig_h) =
        img_row.ok_or_else(|| AppError::NotFound("Gambar tidak ditemukan".into()))?;

    let aliases = serde_json::to_string(&[req.query.trim().to_lowercase(), slug.clone()]).unwrap();
    let out_onnx = st.cfg.data_dir.join("packs").join(format!("{slug}_int8.onnx"));

    sqlx::query(
        "INSERT INTO model_packs (slug, display_name, aliases_json, onnx_path, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at) \
         VALUES (?, ?, ?, ?, 0.0, 640, 15, NULL, 'training', ?) \
         ON CONFLICT(slug) DO UPDATE SET status = 'training', updated_at = excluded.updated_at",
    )
    .bind(&slug)
    .bind(format!("{} Detector (Auto-Distilled INT8)", req.query.trim()))
    .bind(&aliases)
    .bind(out_onnx.to_string_lossy().to_string())
    .bind(now_rfc3339())
    .execute(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let st_bg = st.clone();
    let slug_bg = slug.clone();
    let query_bg = req.query.clone();
    tokio::spawn(async move {
        let res: anyhow::Result<()> = async {
            let work_img = PathBuf::from(&work_path);
            let dim = image::image_dimensions(&work_img)?;
            // 1. Obtain teacher points from Gemini Single-Pass Validator
            let teacher = run_vlm_method(&st_bg.http, &st_bg.cfg, "m3", &work_img, dim.0, dim.1, &query_bg).await?;
            if teacher.points.is_empty() {
                anyhow::bail!("Validator LLM menemukan 0 objek pada gambar ini");
            }
            let tmp_dir = st_bg.cfg.data_dir.join("tmp").join(format!("train_{slug_bg}"));
            tokio::fs::create_dir_all(&tmp_dir).await?;
            let pts_file = tmp_dir.join("teacher_points.json");
            tokio::fs::write(&pts_file, serde_json::to_vec(&json!({ "points": teacher.points }))?).await?;

            // 2. Distill YOLO11n -> INT8 ONNX under heavy_semaphore
            let _permit = st_bg.heavy_semaphore.acquire().await?;
            let out = Command::new(&st_bg.cfg.onnx_python_bin)
                .env_remove("PYTHONPATH")
                .arg("workers/onnx_worker.py")
                .arg("train")
                .arg("--slug")
                .arg(&slug_bg)
                .arg("--image")
                .arg(&work_img)
                .arg("--points-json")
                .arg(&pts_file)
                .arg("--base-weights")
                .arg(&st_bg.cfg.yolo_base_weights)
                .arg("--out-onnx")
                .arg(&out_onnx)
                .arg("--work-dir")
                .arg(tmp_dir.join("ds"))
                .arg("--epochs")
                .arg("15")
                .output()
                .await?;
            let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
            if !out.status.success() {
                anyhow::bail!("onnx_worker train failed");
            }
            let size_mb = ((tokio::fs::metadata(&out_onnx).await?.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
            sqlx::query("UPDATE model_packs SET size_mb = ?, status = 'ready', updated_at = ? WHERE slug = ?")
                .bind(size_mb)
                .bind(now_rfc3339())
                .bind(&slug_bg)
                .execute(&st_bg.pool)
                .await?;
            Ok(())
        }
        .await;

        if let Err(e) = res {
            tracing::error!(error = ?e, slug = %slug_bg, "auto-train pack failed");
            let _ = sqlx::query("UPDATE model_packs SET status = 'failed', updated_at = ? WHERE slug = ?")
                .bind(now_rfc3339())
                .bind(&slug_bg)
                .execute(&st_bg.pool)
                .await;
        }
    });

    Ok(Json(json!({ "slug": slug, "status": "training" })))
}
```

Write `prototype/src/routes/runs.rs` (orchestrates concurrent execution of any subset of `["m1","m2","m3","m4","m5","m6"]`, saves rendered JPEGs, updates SQLite, and broadcasts live JSON snapshots over `GET /api/runs/{id}/stream` SSE):
```rust
use std::{convert::Infallible, path::PathBuf, time::Instant};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    auth::{generate_token, now_rfc3339, RequireSession},
    error::AppError,
    services::{
        m1_ca::run_count_anything,
        m2_onnx::{run_m2_onnx, OnnxOutcome},
        renderer::{render_annotated_jpeg, Point2D},
        vlm::run_vlm_method,
    },
    state::AppState,
};

const VALID_METHODS: &[&str] = &["m1", "m2", "m3", "m4", "m5", "m6"];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRunReq {
    pub image_id: String,
    pub query: String,
    pub methods: Vec<String>,
}

pub async fn load_run_json(st: &AppState, run_id: &str) -> anyhow::Result<Option<Value>> {
    let run_row: Option<(String, String, String, String, String, i64, i64)> = sqlx::query_as(
        "SELECT r.id, r.image_id, r.query, r.status, r.created_at, i.width, i.height \
         FROM runs r JOIN images i ON i.id = r.image_id WHERE r.id = ?",
    )
    .bind(run_id)
    .fetch_optional(&st.pool)
    .await?;

    let Some((id, image_id, query, status, created_at, width, height)) = run_row else {
        return Ok(None);
    };

    let m_rows: Vec<(
        String,
        String,
        Option<i64>,
        Option<String>,
        Option<String>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<f64>,
        Option<String>,
    )> = sqlx::query_as(
        "SELECT method_id, status, count, points_json, grid_boxes_json, elapsed_ms, \
         prompt_tokens, completion_tokens, model_size_mb, meta_json \
         FROM run_methods WHERE run_id = ? ORDER BY method_id",
    )
    .bind(&id)
    .fetch_all(&st.pool)
    .await?;

    let methods: Vec<Value> = m_rows
        .into_iter()
        .map(|m| {
            let pts: Value = m.3.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!([]));
            let grids: Value = m.4.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
            let meta: Value = m.9.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!({}));
            json!({
                "methodId": m.0,
                "status": m.1,
                "count": m.2,
                "points": pts,
                "gridBoxes": grids,
                "elapsedMs": m.5,
                "promptTokens": m.6,
                "completionTokens": m.7,
                "modelSizeMb": m.8,
                "meta": meta,
                "renderedUrl": format!("/api/runs/{id}/render/{}", m.0),
            })
        })
        .collect();

    Ok(Some(json!({
        "runId": id,
        "imageId": image_id,
        "imageUrl": format!("/api/images/{image_id}/raw"),
        "imageWidth": width,
        "imageHeight": height,
        "query": query,
        "status": status,
        "createdAt": created_at,
        "methods": methods,
    })))
}

async fn emit_snapshot(st: &AppState, run_id: &str) {
    if let Ok(Some(val)) = load_run_json(st, run_id).await {
        if let Some(tx) = st.run_streams.get(run_id) {
            let _ = tx.send(val.to_string());
        }
    }
}

pub async fn create_run(
    State(st): State<AppState>,
    _session: RequireSession,
    Json(req): Json<CreateRunReq>,
) -> Result<impl IntoResponse, AppError> {
    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err(AppError::BadRequest("Query objek wajib diisi".into()));
    }
    let mut methods: Vec<String> = req
        .methods
        .into_iter()
        .filter(|m| VALID_METHODS.contains(&m.as_str()))
        .collect();
    methods.sort();
    methods.dedup();
    if methods.is_empty() {
        return Err(AppError::BadRequest("Pilih minimal satu metode (m1..m6)".into()));
    }

    let img_row: Option<(String, String, i64, i64)> = sqlx::query_as(
        "SELECT orig_path, work_path, width, height FROM images WHERE id = ?",
    )
    .bind(&req.image_id)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;
    let (orig_path, work_path, orig_w, orig_h) =
        img_row.ok_or_else(|| AppError::NotFound("Gambar tidak ditemukan".into()))?;

    let run_id = format!("run_{}", &generate_token()[..16]);
    let now = now_rfc3339();
    sqlx::query("INSERT INTO runs (id, image_id, query, status, created_at) VALUES (?, ?, ?, 'running', ?)")
        .bind(&run_id)
        .bind(&req.image_id)
        .bind(&query)
        .bind(&now)
        .execute(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    for m in &methods {
        let init_status = if m == "m1" && st.heavy_semaphore.available_permits() == 0 {
            "queued"
        } else {
            "running"
        };
        sqlx::query(
            "INSERT INTO run_methods (run_id, method_id, status, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(m)
        .bind(init_status)
        .bind(&now)
        .execute(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    }

    let (tx, _) = broadcast::channel::<String>(32);
    st.run_streams.insert(run_id.clone(), tx);

    let initial_json = load_run_json(&st, &run_id)
        .await
        .map_err(AppError::Internal)?
        .unwrap();

    // Spawn concurrent execution per selected method
    let st_orch = st.clone();
    let run_id_orch = run_id.clone();
    tokio::spawn(async move {
        let mut handles = Vec::new();
        for method_id in methods {
            let st_m = st_orch.clone();
            let rid = run_id_orch.clone();
            let q = query.clone();
            let orig_p = PathBuf::from(&orig_path);
            let work_p = PathBuf::from(&work_path);
            let w = orig_w as u32;
            let h = orig_h as u32;

            handles.push(tokio::spawn(async move {
                execute_single_method(&st_m, &rid, &method_id, &orig_p, &work_p, w, h, &q).await;
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        let _ = sqlx::query("UPDATE runs SET status = 'completed' WHERE id = ?")
            .bind(&run_id_orch)
            .execute(&st_orch.pool)
            .await;
        emit_snapshot(&st_orch, &run_id_orch).await;
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        st_orch.run_streams.remove(&run_id_orch);
    });

    Ok(Json(initial_json))
}

async fn execute_single_method(
    st: &AppState,
    run_id: &str,
    method_id: &str,
    orig_path: &PathBuf,
    work_path: &PathBuf,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) {
    let started = Instant::now();
    let _ = sqlx::query("UPDATE run_methods SET status = 'running', updated_at = ? WHERE run_id = ? AND method_id = ?")
        .bind(now_rfc3339())
        .bind(run_id)
        .bind(method_id)
        .execute(&st.pool)
        .await;
    emit_snapshot(st, run_id).await;

    let render_out = st
        .cfg
        .data_dir
        .join("renders")
        .join(format!("{run_id}_{method_id}.jpg"));

    match method_id {
        "m1" => match run_count_anything(st, work_path, orig_w, orig_h, query, run_id).await {
            Ok(ca) => {
                let _ = render_annotated_jpeg(orig_path, &ca.points, None, &render_out);
                save_success(
                    st,
                    run_id,
                    method_id,
                    &ca.points,
                    None,
                    &render_out,
                    ca.elapsed_ms,
                    0,
                    0,
                    ca.model_size_mb,
                    json!({"engine": "Count-Anything SAM3 (CPU mmap)"}),
                )
                .await;
            }
            Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
        },
        "m2" => match run_m2_onnx(st, work_path, orig_w, orig_h, query).await {
            Ok(OnnxOutcome::Ready {
                pack_slug,
                points,
                grid_boxes,
                elapsed_ms,
                model_size_mb,
                mode,
            }) => {
                let _ = render_annotated_jpeg(orig_path, &points, grid_boxes.as_deref(), &render_out);
                save_success(
                    st,
                    run_id,
                    method_id,
                    &points,
                    grid_boxes,
                    &render_out,
                    elapsed_ms,
                    0,
                    0,
                    model_size_mb,
                    json!({"packSlug": pack_slug, "mode": mode}),
                )
                .await;
            }
            Ok(OnnxOutcome::ColdStart { normalized_query }) => {
                let _ = sqlx::query(
                    "UPDATE run_methods SET status = 'cold_start', elapsed_ms = ?, meta_json = ?, updated_at = ? \
                     WHERE run_id = ? AND method_id = ?",
                )
                .bind(started.elapsed().as_millis() as i64)
                .bind(
                    json!({
                        "normalizedQuery": normalized_query,
                        "message": format!("Model pack .onnx untuk '{}' belum dilatih di katalog offline.", query),
                    })
                    .to_string(),
                )
                .bind(now_rfc3339())
                .bind(run_id)
                .bind(method_id)
                .execute(&st.pool)
                .await;
            }
            Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
        },
        "m3" | "m4" | "m5" | "m6" => {
            match run_vlm_method(&st.http, &st.cfg, method_id, work_path, orig_w, orig_h, query).await {
                Ok(vlm) => {
                    let elapsed_ms = started.elapsed().as_millis() as i64;
                    let _ = render_annotated_jpeg(
                        orig_path,
                        &vlm.points,
                        vlm.grid_boxes.as_deref(),
                        &render_out,
                    );
                    save_success(
                        st,
                        run_id,
                        method_id,
                        &vlm.points,
                        vlm.grid_boxes,
                        &render_out,
                        elapsed_ms,
                        vlm.prompt_tokens,
                        vlm.completion_tokens,
                        0.0,
                        json!({"rawPreview": vlm.raw_preview}),
                    )
                    .await;
                }
                Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
            }
        }
        _ => {}
    }
    emit_snapshot(st, run_id).await;
}

async fn save_success(
    st: &AppState,
    run_id: &str,
    method_id: &str,
    points: &[Point2D],
    grid_boxes: Option<Vec<[i32; 4]>>,
    render_out: &PathBuf,
    elapsed_ms: i64,
    p_tok: i64,
    c_tok: i64,
    model_size_mb: f64,
    meta: Value,
) {
    let _ = sqlx::query(
        "UPDATE run_methods SET status = 'completed', count = ?, points_json = ?, grid_boxes_json = ?, \
         rendered_path = ?, elapsed_ms = ?, prompt_tokens = ?, completion_tokens = ?, model_size_mb = ?, \
         meta_json = ?, updated_at = ? WHERE run_id = ? AND method_id = ?",
    )
    .bind(points.len() as i64)
    .bind(serde_json::to_string(points).unwrap_or_else(|_| "[]".into()))
    .bind(grid_boxes.map(|g| serde_json::to_string(&g).unwrap_or_default()))
    .bind(render_out.to_string_lossy().to_string())
    .bind(elapsed_ms)
    .bind(p_tok)
    .bind(c_tok)
    .bind(model_size_mb)
    .bind(meta.to_string())
    .bind(now_rfc3339())
    .bind(run_id)
    .bind(method_id)
    .execute(&st.pool)
    .await;
}

async fn save_failure(st: &AppState, run_id: &str, method_id: &str, elapsed_ms: i64, msg: &str) {
    let _ = sqlx::query(
        "UPDATE run_methods SET status = 'failed', elapsed_ms = ?, meta_json = ?, updated_at = ? \
         WHERE run_id = ? AND method_id = ?",
    )
    .bind(elapsed_ms)
    .bind(json!({ "error": msg }).to_string())
    .bind(now_rfc3339())
    .bind(run_id)
    .bind(method_id)
    .execute(&st.pool)
    .await;
}

pub async fn get_run(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let val = load_run_json(&st, &id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Run tidak ditemukan".into()))?;
    Ok(Json(val))
}

pub async fn list_runs(State(st): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let ids: Vec<(String,)> = sqlx::query_as("SELECT id FROM runs ORDER BY created_at DESC LIMIT 20")
        .fetch_all(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    let mut items = Vec::new();
    for (id,) in ids {
        if let Ok(Some(v)) = load_run_json(&st, &id).await {
            items.push(v);
        }
    }
    Ok(Json(json!({ "runs": items })))
}

pub async fn stream_run(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, AppError> {
    let initial = load_run_json(&st, &id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Run tidak ditemukan".into()))?;

    let rx_opt = st.run_streams.get(&id).map(|tx| tx.subscribe());
    let init_stream = futures::stream::once(async move {
        Ok::<Event, Infallible>(Event::default().data(initial.to_string()))
    });

    let live_stream = match rx_opt {
        Some(rx) => BroadcastStream::new(rx)
            .filter_map(|msg| async move {
                msg.ok().map(|s| Ok::<Event, Infallible>(Event::default().data(s)))
            })
            .boxed(),
        None => futures::stream::empty().boxed(),
    };

    Ok(Sse::new(init_stream.chain(live_stream)).keep_alive(KeepAlive::default()))
}

pub async fn get_rendered_image(
    State(st): State<AppState>,
    Path((id, method_id)): Path<(String, String)>,
) -> Result<Response, AppError> {
    let row: Option<(Option<String>,)> = sqlx::query_as(
        "SELECT rendered_path FROM run_methods WHERE run_id = ? AND method_id = ?",
    )
    .bind(&id)
    .bind(&method_id)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let path_str = row
        .and_then(|r| r.0)
        .ok_or_else(|| AppError::NotFound("Hasil render belum tersedia".into()))?;
    let bytes = tokio::fs::read(&path_str)
        .await
        .map_err(|_| AppError::NotFound("File render tidak ditemukan".into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(Body::from(bytes))
        .unwrap())
}
```

Update `prototype/src/services/mod.rs`, `prototype/src/routes/mod.rs`, `prototype/src/lib.rs`, and `prototype/src/main.rs` so all routes (`/api/packs`, `/api/packs/{slug}/download`, `/api/packs/train`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/stream`, `/api/runs/{id}/render/{method_id}`) and startup seeders (`seed_research_samples` + `seed_default_apple_pack`) are registered.

- [ ] **Step 4: Run `cargo test` on VPS to verify all backend tests pass**

Run: `ssh root@49.128.191.225 'export PATH="$HOME/.cargo/bin:$PATH"; cd /root/workspace/anycount/prototype && cargo test'`
Expected: All unit and integration tests pass (`auth_and_health`, `images_and_renderer`, `runs_and_packs`, and `lib` tests).

- [ ] **Step 5: Commit Task 4**

```bash
git add prototype/workers/ prototype/src/ prototype/tests/runs_and_packs.rs
git commit -m "feat(prototype): add M1 SAM3 & M2 ONNX workers, model packs API, and SSE run orchestrator"
```
---

### Task 5: React Frontend Shell, JRouter Dual-Theme Tokens (Dark Default + Warm Cream Light), Sidebar, Topbar & Passcode Modal

**Files:**
- Create: `prototype/web/package.json`
- Create: `prototype/web/tsconfig.json`
- Create: `prototype/web/vite.config.ts`
- Create: `prototype/web/index.html`
- Create: `prototype/web/src/styles/tokens.css`
- Create: `prototype/web/src/api/client.ts`
- Create: `prototype/web/src/components/Sidebar.tsx`
- Create: `prototype/web/src/components/Topbar.tsx`
- Create: `prototype/web/src/components/PasscodeModal.tsx`
- Create: `prototype/web/src/App.tsx`
- Create: `prototype/web/src/main.tsx`
- Test: `prototype/web/src/__tests__/shell.test.tsx`

**Interfaces:**
- Consumes: Backend REST endpoints (`/api/auth/status`, `/api/auth/unlock`, `/api/auth/logout`, `/api/samples`, `/api/packs`, `/api/docs`, `/api/images`, `/api/runs`, `/api/runs/{id}/stream`).
- Produces:
  - `web/src/api/client.ts` typed DTOs (`Point2D`, `MethodResult`, `RunDetail`, `SamplePreset`, `MontageItem`, `ModelPack`, `DocsPayload`) and API helpers (`getAuthStatus`, `unlockPasscode`, `uploadImage`, `createRun`, `subscribeRun`, `getSamples`, `getPacks`, `trainPack`, `getDocs`).
  - `Sidebar`, `Topbar`, `PasscodeModal`, and `App` shell with dark-first default (`document.documentElement.classList.contains("dark") === true`) matching `docs/design/image.png` and toggle to `JRouter/docs/DESIGN.md` warm cream (`#fffaf0`).

- [ ] **Step 1: Scaffold `prototype/web/package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, and `src/styles/tokens.css`**

Write `prototype/web/package.json`:
```json
{
  "name": "anycount-prototype-web",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "typecheck": "tsc --noEmit",
    "test": "vitest run"
  },
  "dependencies": {
    "lucide-react": "^0.460.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.1.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.2.0",
    "@testing-library/user-event": "^14.6.1",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "jsdom": "^25.0.1",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.7.2",
    "vite": "^6.0.5",
    "vitest": "^3.0.4"
  }
}
```

Write `prototype/web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "types": ["vitest/globals"]
  },
  "include": ["src"]
}
```

Write `prototype/web/vite.config.ts`:
```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:2030",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
```

Write `prototype/web/index.html`:
```html
<!doctype html>
<html lang="id" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AnyCount Studio — 6-Method Object Counting Prototype</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Write `prototype/web/src/styles/tokens.css` (exact JRouter `DESIGN.md` warm-cream light tokens + JRouter Playground `docs/design/image.png` dark tokens):
```css
@import "tailwindcss";

:root {
  --canvas: #fffaf0;
  --surface-soft: #faf5e8;
  --surface-card: #f5f0e0;
  --surface-strong: #ebe6d6;
  --surface-composer: #ffffff;
  --ink: #0a0a0a;
  --body-text: #3a3a3a;
  --muted-text: #6a6a6a;
  --hairline: rgba(10, 10, 10, 0.09);
  --hairline-strong: rgba(10, 10, 10, 0.16);
  --nav-active-bg: #0a0a0a;
  --nav-active-fg: #fffaf0;

  --brand-pink: #ff4d8b;
  --brand-teal: #1a3a3a;
  --brand-lavender: #b8a4ed;
  --brand-peach: #ffb084;
  --brand-ochre: #e8b94a;
  --brand-mint: #a4d4c5;
  --brand-emerald: #16a34a;
  --status-danger: #ef4444;
}

.dark {
  --canvas: #0e0e10;
  --surface-soft: #141416;
  --surface-card: #18181b;
  --surface-strong: #222226;
  --surface-composer: #18181b;
  --ink: #f4f4f5;
  --body-text: #d4d4d8;
  --muted-text: #8e8e98;
  --hairline: rgba(255, 255, 255, 0.08);
  --hairline-strong: rgba(255, 255, 255, 0.15);
  --nav-active-bg: #f4f4f5;
  --nav-active-fg: #0a0a0a;

  --brand-emerald: #34d399;
}

body {
  margin: 0;
  background-color: var(--canvas);
  color: var(--ink);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}

.font-mono-tech {
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

.display-head {
  font-weight: 500;
  letter-spacing: -0.03em;
}
```

- [ ] **Step 2: Implement `src/api/client.ts`, `Sidebar.tsx`, `Topbar.tsx`, `PasscodeModal.tsx`, `App.tsx` (with route placeholders replaced in Task 6 or clean imported page components)**

Write `prototype/web/src/api/client.ts`:
```ts
export interface Point2D {
  x: number;
  y: number;
}

export interface MethodResult {
  methodId: "m1" | "m2" | "m3" | "m4" | "m5" | "m6";
  status: "queued" | "running" | "completed" | "cold_start" | "failed";
  count: number | null;
  points: Point2D[];
  gridBoxes: [number, number, number, number][] | null;
  elapsedMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  modelSizeMb: number | null;
  meta: Record<string, unknown>;
  renderedUrl: string;
}

export interface RunDetail {
  runId: string;
  imageId: string;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  query: string;
  status: string;
  createdAt: string;
  methods: MethodResult[];
}

export interface SamplePreset {
  id: string;
  filename: string;
  defaultQuery: string;
  groundTruth: number;
  width: number;
  height: number;
  url: string;
}

export interface MontageItem {
  id: string;
  title: string;
  url: string;
  best: string;
}

export interface ModelPack {
  slug: string;
  displayName: string;
  aliases: string[];
  sizeMb: number;
  imgsz: number;
  trainedEpochs: number;
  valMapePct: number | null;
  status: "ready" | "training" | "failed";
  updatedAt: string;
  downloadUrl: string;
}

export interface MethodDocItem {
  id: string;
  code: string;
  name: string;
  badge: string;
  color: string;
  speed: string;
  footprint: string;
  summary: string;
  bestFor: string;
}

export interface GlossaryItem {
  term: string;
  plain: string;
}

async function reqJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message || `HTTP ${res.status}`;
    const err = new Error(msg) as Error & { code?: string; status?: number };
    err.code = data?.error?.code;
    err.status = res.status;
    throw err;
  }
  return data as T;
}

export const api = {
  getAuthStatus: () => reqJson<{ unlocked: boolean }>("/api/auth/status"),
  unlock: (passcode: string) =>
    reqJson<{ unlocked: boolean }>("/api/auth/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passcode }),
    }),
  logout: () => reqJson<{ unlocked: boolean }>("/api/auth/logout", { method: "POST" }),
  uploadImage: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return reqJson<{ id: string; filename: string; width: number; height: number; url: string }>(
      "/api/images",
      { method: "POST", body: fd }
    );
  },
  getSamples: () => reqJson<{ samples: SamplePreset[]; montages: MontageItem[] }>("/api/samples"),
  getPacks: () => reqJson<{ packs: ModelPack[] }>("/api/packs"),
  trainPack: (imageId: string, query: string) =>
    reqJson<{ slug: string; status: string }>("/api/packs/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId, query }),
    }),
  getDocs: () => reqJson<{ methods: MethodDocItem[]; glossary: GlossaryItem[] }>("/api/docs"),
  createRun: (imageId: string, query: string, methods: string[]) =>
    reqJson<RunDetail>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageId, query, methods }),
    }),
  listRuns: () => reqJson<{ runs: RunDetail[] }>("/api/runs"),
  getRun: (runId: string) => reqJson<RunDetail>(`/api/runs/${runId}`),
  subscribeRun: (runId: string, onUpdate: (run: RunDetail) => void): (() => void) => {
    const es = new EventSource(`/api/runs/${runId}/stream`);
    es.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data) as RunDetail;
        onUpdate(parsed);
        if (parsed.status === "completed") {
          es.close();
        }
      } catch {
        // ignore malformed keepalive
      }
    };
    es.onerror = () => es.close();
    return () => es.close();
  },
};
```

Write `prototype/web/src/components/Sidebar.tsx`, `Topbar.tsx`, and `PasscodeModal.tsx`:
- `Sidebar.tsx`: Fixed 248px left navigation matching `docs/design/image.png` — top brand block **AnyCount Studio** + subtitle `PROTOTYPE / v0.1.0`, emerald status card showing active offline `.onnx` pack count (`apple · 2.89 MB Ready`) with quick button to `/packs`, and 4 grouped navigation links (`STUDIO -> Playground`, `BENCHMARK -> Benchmark Gallery`, `ENGINE -> Model Packs (.onnx)`, `REFERENCE -> Method Docs & Glossary`) where active route uses `background: var(--nav-active-bg); color: var(--nav-active-fg)`.
- `Topbar.tsx`: Sticky header bar with sidebar toggle button, breadcrumb (`Studio / <Active Page>`), right-aligned **Theme Toggle** (`Dark` <-> `Warm Cream`), passcode status pill (`Unlocked` emerald badge or `Locked · Enter Passcode` button), and `AC` avatar pill.
- `PasscodeModal.tsx`: Clean modal prompting for passcode (`placeholder="Masukkan kode akses..."`, hint for authorized testers) calling `api.unlock(passcode)` and invoking `onUnlocked()` callback.

- [ ] **Step 3: Write Vitest test `prototype/web/src/__tests__/shell.test.tsx`**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect, vi } from "vitest";
import { Sidebar } from "../components/Sidebar";
import { Topbar } from "../components/Topbar";

describe("AnyCount Studio Shell", () => {
  it("renders all 4 navigation destinations and toggles dark/cream theme", () => {
    document.documentElement.classList.add("dark");
    const onToggleTheme = vi.fn(() => {
      document.documentElement.classList.toggle("dark");
    });

    render(
      <MemoryRouter initialEntries={["/"]}>
        <Sidebar collapsed={false} packCount={1} />
        <Topbar
          pageTitle="Playground"
          dark={true}
          onToggleTheme={onToggleTheme}
          unlocked={false}
          onOpenUnlock={() => {}}
          onToggleSidebar={() => {}}
        />
      </MemoryRouter>
    );

    expect(screen.getByText("AnyCount Studio")).toBeTruthy();
    expect(screen.getAllByText("Playground").length).toBeGreaterThan(0);
    expect(screen.getByText("Benchmark Gallery")).toBeTruthy();
    expect(screen.getByText("Model Packs (.onnx)")).toBeTruthy();
    expect(screen.getByText("Method Docs & Glossary")).toBeTruthy();

    const themeBtn = screen.getByRole("button", { name: /theme/i });
    fireEvent.click(themeBtn);
    expect(onToggleTheme).toHaveBeenCalledTimes(1);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });
});
```

---

### Task 6: Interactive Dot Canvas, Method Result Cards, Floating Composer & All 4 Pages

**Files:**
- Create: `prototype/web/src/components/InteractiveCanvas.tsx`
- Create: `prototype/web/src/components/MethodResultCard.tsx`
- Create: `prototype/web/src/components/Composer.tsx`
- Create: `prototype/web/src/pages/PlaygroundPage.tsx`
- Create: `prototype/web/src/pages/BenchmarkPage.tsx`
- Create: `prototype/web/src/pages/ModelPacksPage.tsx`
- Create: `prototype/web/src/pages/DocsGlossaryPage.tsx`
- Modify: `prototype/web/src/App.tsx`
- Test: `prototype/web/src/__tests__/canvas_and_playground.test.tsx`

**Interfaces:**
- Consumes: `api` client and DTOs from `web/src/api/client.ts`, Shell components from Task 5.
- Produces:
  - `InteractiveCanvas`: Renders image + interactive SVG/Canvas overlay of Count-Anything style red-white dots + top-left dark translucent counter badge (`<query>: <effectiveCount>`). Supports:
    - `editMode`: clicking within `14px` screen distance of an existing dot removes it (`-1`); clicking empty area adds a dot (`+1`) at exact image pixel coordinates; maintains undo history (`Undo` + `Reset` buttons).
    - `showGrid`: toggles `2x2` tile boundary overlay when `gridBoxes` is present.
    - Client-side `Export Edited JPG` button so manual corrections can also be downloaded as a JPG!
  - `MethodResultCard`: JRouter Playground-style response card (`docs/design/image.png`):
    - Header: Method badge (`M1`..`M6`), full method name, live status pill (`Queued` / `Running...` / `Cold-Start` / `12.4s`), and action buttons (`Edit Dots`, `2x2 Grid`, `<> Raw`, `Download JPG`, `Retry`).
    - Body: `InteractiveCanvas` when `completed`; informative Cold-Start panel on `m2` when `status === "cold_start"` with two direct action buttons (**"Run LLM Validator (M3)"** and **"Auto-Train .onnx Pack"**); clear error banner when `failed`.
    - Footer: Monospace telemetry line (`7.71s · 42 objects (+1 manual) · 3,243 in · 470 out · 0 MB local`).
  - `Composer`: Floating bottom prompt bar (`docs/design/image.png`) with drag-drop/upload + paste support, horizontal quick-sample chips, text query input, single-method dropdown OR multi-method `Compare` pill selector (`M1`..`M6` checkboxes + `Fast Compare (M2+M3+M6)` preset + `All 6 Methods` preset), `History` drawer toggle, and circular submit button (`↑`).
  - `BenchmarkPage`, `ModelPacksPage`, and `DocsGlossaryPage` using JRouter saturated brand cards (`pink`, `teal`, `lavender`, `peach`, `ochre`, `cream`) in Light Mode and sleek dark bordered cards in Dark Mode.

- [ ] **Step 1: Implement `InteractiveCanvas.tsx`, `MethodResultCard.tsx`, `Composer.tsx`, and the 4 pages (`PlaygroundPage`, `BenchmarkPage`, `ModelPacksPage`, `DocsGlossaryPage`)**

Ensure `InteractiveCanvas.tsx` exposes pure helper `applyCanvasClick(points: Point2D[], clickX: number, clickY: number, hitRadiusPx: number): { next: Point2D[]; delta: "added" | "removed" }` so unit tests verify exact point addition and deletion deterministically:
```ts
export function applyCanvasClick(
  points: Point2D[],
  clickX: number,
  clickY: number,
  hitRadiusPx: number
): { next: Point2D[]; delta: "added" | "removed" } {
  let bestIdx = -1;
  let bestDist = Infinity;
  points.forEach((pt, idx) => {
    const d = Math.hypot(pt.x - clickX, pt.y - clickY);
    if (d <= hitRadiusPx && d < bestDist) {
      bestDist = d;
      bestIdx = idx;
    }
  });
  if (bestIdx >= 0) {
    return {
      next: points.filter((_, idx) => idx !== bestIdx),
      delta: "removed",
    };
  }
  return {
    next: [...points, { x: Math.round(clickX * 100) / 100, y: Math.round(clickY * 100) / 100 }],
    delta: "added",
  };
}
```

- [ ] **Step 2: Write Vitest test `prototype/web/src/__tests__/canvas_and_playground.test.tsx`**

```tsx
import { describe, it, expect } from "vitest";
import { applyCanvasClick } from "../components/InteractiveCanvas";

describe("InteractiveCanvas dot editing logic", () => {
  it("adds a new point on empty space and removes an existing point when clicked within radius", () => {
    const initial = [
      { x: 100, y: 100 },
      { x: 250, y: 180 },
    ];

    // Click empty area -> +1 point
    const added = applyCanvasClick(initial, 400, 300, 14);
    expect(added.delta).toBe("added");
    expect(added.next.length).toBe(3);

    // Click near (100, 100) -> removes that dot
    const removed = applyCanvasClick(added.next, 105, 98, 14);
    expect(removed.delta).toBe("removed");
    expect(removed.next.length).toBe(2);
    expect(removed.next.some((p) => p.x === 100 && p.y === 100)).toBe(false);
  });
});
```

- [ ] **Step 3: Run `npm test` and `npm run build` on VPS, then rebuild Rust crate so `rust-embed` embeds real `web/dist`**

Run:
```bash
ssh root@49.128.191.225 'export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; export PATH="$HOME/.cargo/bin:$PATH"; cd /root/workspace/anycount/prototype/web && npm install && npm test && npm run build && cd /root/workspace/anycount/prototype && cargo test'
```
Expected: Vitest suites pass (`shell.test.tsx` + `canvas_and_playground.test.tsx`), `vite build` emits `prototype/web/dist/index.html` + assets with zero TypeScript errors, and `cargo test` passes with embedded production assets.

- [ ] **Step 4: Commit Tasks 5 & 6**

```bash
git add prototype/web/
git commit -m "feat(prototype-web): build JRouter-themed React UI, interactive dot canvas, compare playground, and 4 studio pages"
```

---

### Task 7: Deployment Scripts, Portable Docker Config, Live VPS Hosting (`testing.rkhyg.xyz` via Caddy + `vpsdash`) & End-to-End Verification

**Files:**
- Create: `prototype/Dockerfile`
- Create: `prototype/compose.yml`
- Create: `prototype/deploy/build.sh`
- Create: `prototype/deploy/anycount-prototype.service`
- Create: `prototype/deploy/register_caddy_vpsdash.py`
- Create: `prototype/README.md`

**Interfaces:**
- Consumes: Compiled release binary `/root/workspace/anycount/prototype/target/release/anycount-prototype`, VPS `.env` at `/root/workspace/anycount/prototype/.env`, `/root/.local/share/vpsdash/vpsdash.db`, `/etc/caddy/Caddyfile`.
- Produces: Live HTTPS service at `https://testing.rkhyg.xyz` reverse-proxying to `127.0.0.1:2030`, supervised by `systemd` (`anycount-prototype.service`) and registered in `vpsdash` so Caddy reloads never drop the domain.

- [ ] **Step 1: Write `prototype/Dockerfile`, `prototype/compose.yml`, `prototype/deploy/build.sh`, `prototype/deploy/anycount-prototype.service`, `prototype/deploy/register_caddy_vpsdash.py`, and `prototype/README.md`**

Write `prototype/deploy/build.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -s "$HOME/.nvm/nvm.sh" ]; then
  export NVM_DIR="$HOME/.nvm"
  . "$NVM_DIR/nvm.sh"
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "[1/2] Building React frontend (web/dist)..."
cd "$PROTO_DIR/web"
npm ci --prefer-offline || npm install
npm run build

echo "[2/2] Building Rust single-binary release..."
cd "$PROTO_DIR"
cargo build --release
ls -lh "$PROTO_DIR/target/release/anycount-prototype"
```

Write `prototype/deploy/anycount-prototype.service`:
```ini
[Unit]
Description=AnyCount Prototype Web Studio (Rust Axum + React)
After=network-online.target caddy.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/workspace/anycount/prototype
EnvironmentFile=/root/workspace/anycount/prototype/.env
ExecStart=/root/workspace/anycount/prototype/target/release/anycount-prototype
Restart=always
RestartSec=2
MemoryMax=10G
LimitNOFILE=65535
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Write `prototype/deploy/register_caddy_vpsdash.py` (uses Python 3.12 stdlib `sqlite3` + `hashlib` + `fcntl` lock matching `vpsdash/src/domain/caddy.rs` to register `testing.rkhyg.xyz -> 127.0.0.1:2030` atomically and reload Caddy):
```python
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
```

Also write `prototype/Dockerfile` (multi-stage `node:22-bookworm-slim` -> `rust:1.88-bookworm` -> `debian:bookworm-slim`) and `prototype/compose.yml` (`127.0.0.1:2030:2030`) plus `prototype/README.md` documenting local dev, systemd deployment, and Docker usage.

- [ ] **Step 2: Provision `.env` on VPS (mode `0600`, gitignored), build release binary, start `anycount-prototype.service`, and register `testing.rkhyg.xyz` in Caddy + `vpsdash`**

Run on VPS (reading `VLM_API_KEY` from runtime env without committing it):
```bash
chmod +x /root/workspace/anycount/prototype/deploy/build.sh
/root/workspace/anycount/prototype/deploy/build.sh
cp /root/workspace/anycount/prototype/deploy/anycount-prototype.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now anycount-prototype
python3 /root/workspace/anycount/prototype/deploy/register_caddy_vpsdash.py
```

- [ ] **Step 3: Execute Live End-to-End Verification against `https://testing.rkhyg.xyz`**

Verify:
1. `curl -sSf https://testing.rkhyg.xyz/api/health` returns `{"ok":true,"service":"anycount-prototype"}` over valid HTTPS TLS.
2. `curl -sSf https://testing.rkhyg.xyz/api/samples` returns the 10 preset samples and 11 montages; `curl -sSf https://testing.rkhyg.xyz/api/packs` returns the `apple` `.onnx` pack (`2.89 MB`, `status: "ready"`).
3. Unlock session with `POST https://testing.rkhyg.xyz/api/auth/unlock` (`{"passcode":"2limanov"}`), then run live `POST /api/runs`:
   - Test `sample_apples` with `methods: ["m2"]` -> completes via ONNX in `< 500 ms` with ~290 apples and valid rendered JPEG at `/api/runs/{id}/render/m2`.
   - Test `sample_coins` with `methods: ["m2", "m3", "m6"]` -> `m2` returns `cold_start`, `m3` returns `24` coins, `m6` returns ~`24-25` coins, all rendered JPEGs HTTP `200`.
   - Test `sample_cats` with `methods: ["m1"]` -> `Count-Anything` worker completes cleanly under `heavy_semaphore` and returns `2` cats.
4. Open `https://testing.rkhyg.xyz` in real browser (`browser_exec`) to verify Dark Mode layout (`docs/design/image.png` fidelity), Light Cream Mode toggle (`DESIGN.md`), Interactive Canvas dot add/delete, and zero console errors.

- [ ] **Step 4: Commit Task 7, push to `origin/main`, and pull to `D:\Github\anycount`**

```bash
git add prototype/Dockerfile prototype/compose.yml prototype/deploy/ prototype/README.md
git commit -m "feat(prototype): add systemd & Docker deployment, vpsdash Caddy registration, and README"
git push origin main
```
