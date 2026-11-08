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
