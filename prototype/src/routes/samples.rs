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
    (
        "sample_blueberry_bnw",
        "Heidelbeeren-bnw.jpg",
        "blueberry",
        113,
    ),
    (
        "sample_blueberry_bnw2",
        "Heidelbeeren-bnw2.jpg",
        "blueberry",
        113,
    ),
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
        let work_path = st
            .cfg
            .data_dir
            .join("uploads")
            .join(format!("{id}_work.jpg"));
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
        json!({"id": "apples", "title": "Dense Hundreds of Apples (GT: ~289-339)", "url": "/api/samples/montages/montage_apples.jpg", "best": "M2 Offline ONNX v2 (289 di GT 289, 0.54s tiled) & M6 DeepSeek 2x2 (336 in 5.6s)"}),
        json!({"id": "m2_flywheel", "title": "Metode 2 Active Learning Flywheel (Apple v1 -> LLM -> v2)", "url": "/api/samples/montages/montage_m2_flywheel_apple_val_2.jpg", "best": "Error dropped from 77.7% to 2.33% (0.54s tiled offline)"}),
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
