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
