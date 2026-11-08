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
