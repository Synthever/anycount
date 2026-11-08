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
