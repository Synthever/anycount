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
