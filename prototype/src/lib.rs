pub mod assets;
pub mod auth;
pub mod config;
pub mod error;
pub mod routes;
pub mod services;
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
        .route("/api/images", post(routes::images::upload))
        .route("/api/images/{id}/raw", get(routes::images::get_raw))
        .route("/api/samples", get(routes::samples::list_samples))
        .route(
            "/api/samples/montages/{filename}",
            get(routes::samples::get_montage),
        )
        .route("/api/docs", get(routes::docs::get_docs))
        .fallback(assets::static_handler)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}
