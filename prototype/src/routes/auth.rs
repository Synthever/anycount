use std::hint::black_box;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use axum_extra::extract::CookieJar;
use dashmap::DashMap;
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

/// Fixed-window limiter: 10 attempts per IP per minute. Successful unlocks clear
/// the bucket so a legitimate operator never locks themselves out.
const UNLOCK_LIMIT: u32 = 10;
const UNLOCK_WINDOW: Duration = Duration::from_secs(60);

static UNLOCK_ATTEMPTS: LazyLock<DashMap<String, (Instant, u32)>> = LazyLock::new(DashMap::new);

fn client_ip(headers: &HeaderMap) -> String {
    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            headers
                .get("x-real-ip")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn unlock_allowed(ip: &str) -> bool {
    let now = Instant::now();
    let mut entry = UNLOCK_ATTEMPTS.entry(ip.to_string()).or_insert((now, 0));
    if now.duration_since(entry.0) >= UNLOCK_WINDOW {
        *entry = (now, 0);
    }
    entry.1 += 1;
    entry.1 <= UNLOCK_LIMIT
}

fn digests_equal_ct(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != 64 || b_bytes.len() != 64 {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..64 {
        diff |= black_box(a_bytes[i]) ^ black_box(b_bytes[i]);
    }
    black_box(diff) == 0
}

pub async fn status(State(st): State<AppState>, jar: CookieJar) -> impl IntoResponse {
    Json(json!({ "unlocked": is_session_valid(&st, &jar).await }))
}

pub async fn unlock(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UnlockReq>,
) -> Result<Response, AppError> {
    let ip = client_ip(&headers);
    if !unlock_allowed(&ip) {
        tracing::warn!(%ip, "unlock throttled: too many attempts in window");
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Terlalu banyak percobaan passcode. Coba lagi dalam 60 detik."
                }
            })),
        )
            .into_response());
    }

    let candidate_digest = hash_token(req.passcode.trim());
    let expected_digest = hash_token(&st.cfg.passcode);
    if !digests_equal_ct(&candidate_digest, &expected_digest) {
        tracing::warn!(%ip, "invalid passcode unlock attempt");
        return Err(AppError::InvalidPasscode);
    }
    UNLOCK_ATTEMPTS.remove(&ip);

    let token = create_session(&st).await?;
    let cookie = format!(
        "{SESSION_COOKIE}={token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000"
    );
    let mut headers = HeaderMap::new();
    headers.insert(header::SET_COOKIE, cookie.parse().unwrap());
    Ok((headers, Json(json!({ "unlocked": true }))).into_response())
}

pub async fn logout(
    State(st): State<AppState>,
    jar: CookieJar,
) -> Result<impl IntoResponse, AppError> {
    if let Some(c) = jar.get(SESSION_COOKIE) {
        sqlx::query("DELETE FROM sessions WHERE token_hash = ?")
            .bind(hash_token(c.value()))
            .execute(&st.pool)
            .await
            .map_err(anyhow::Error::from)?;
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        header::SET_COOKIE,
        format!("{SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0")
            .parse()
            .unwrap(),
    );
    Ok((headers, Json(json!({ "unlocked": false }))))
}
