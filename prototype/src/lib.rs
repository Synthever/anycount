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
        .route(
            "/api/auth/unlock",
            post(routes::auth::unlock).layer(DefaultBodyLimit::max(4096)),
        )
        .route("/api/auth/logout", post(routes::auth::logout))
        .fallback(assets::static_handler)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}
