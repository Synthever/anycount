use std::hint::black_box;

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
    Json(req): Json<UnlockReq>,
) -> Result<impl IntoResponse, AppError> {
    let candidate_digest = hash_token(req.passcode.trim());
    let expected_digest = hash_token(&st.cfg.passcode);
    if !digests_equal_ct(&candidate_digest, &expected_digest) {
        tracing::warn!("invalid passcode unlock attempt");
        return Err(AppError::InvalidPasscode);
    }
    let token = create_session(&st).await?;
    let cookie =
        format!("{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000");
    let mut headers = HeaderMap::new();
    headers.insert(header::SET_COOKIE, cookie.parse().unwrap());
    Ok((headers, Json(json!({ "unlocked": true }))))
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
        format!("{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
            .parse()
            .unwrap(),
    );
    Ok((headers, Json(json!({ "unlocked": false }))))
}
