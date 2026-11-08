use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::{process::Command, time::timeout};

use crate::{
    auth::{now_rfc3339, RequireSession},
    error::AppError,
    services::{m2_onnx::normalize_query_slug, vlm::run_vlm_method},
    state::AppState,
};

/// Heavy distillation can legitimately take many minutes on CPU; cap it so a
/// wedged worker cannot hold the semaphore forever.
const TRAIN_TIMEOUT_SECS: u64 = 1800;

struct TrainJob;

/// In-flight auto-train jobs keyed by pack slug. Kept outside `model_packs`
/// because a ready pack must keep serving while a new artifact is distilled.
static TRAIN_JOBS: LazyLock<DashMap<String, TrainJob>> = LazyLock::new(DashMap::new);

/// Claims the single train slot for `slug`, or `None` when another call already
/// owns it. The check and the insert happen inside one `DashMap` entry lock, so
/// two concurrent `/api/packs/train` calls can never both win and clobber the
/// same `tmp_onnx`.
fn claim_train_slot(slug: &str) -> Option<TrainJobGuard> {
    match TRAIN_JOBS.entry(slug.to_string()) {
        Entry::Occupied(_) => None,
        Entry::Vacant(slot) => {
            slot.insert(TrainJob);
            Some(TrainJobGuard {
                slug: slug.to_string(),
            })
        }
    }
}

/// Releases the train slot on every exit path — early `?`, task completion or
/// panic. Without it an aborted request would leave the slug `409` forever.
struct TrainJobGuard {
    slug: String,
}

impl Drop for TrainJobGuard {
    fn drop(&mut self) {
        TRAIN_JOBS.remove(&self.slug);
    }
}

/// Removes a per-job scratch directory when the job ends, whatever the outcome
/// (success, worker error, timeout or panic). A timed-out train used to leave
/// `data/tmp/train_<slug>/` behind for good.
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let path = self.0.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _ = tokio::fs::remove_dir_all(&path).await;
            });
        } else if !path.as_os_str().is_empty() {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

pub async fn seed_default_apple_pack(st: &AppState) -> anyhow::Result<()> {
    let apple_onnx = st
        .cfg
        .repo_root
        .join("research/spikes/004-offline-onnx-llm-validator/models/apple_v2_distilled_int8.onnx");
    if !apple_onnx.exists() {
        return Ok(());
    }
    let size_mb =
        ((apple_onnx.metadata()?.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
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

/// Seeds any additional bundled packs (currently the tennis-ball detector trained
/// from public Wikimedia Commons photos with LLM pseudo-labels).
pub async fn seed_bundled_object_packs(st: &AppState) -> anyhow::Result<()> {
    let tennis_onnx = st
        .cfg
        .repo_root
        .join("research/packs/tennis-ball/models/tennis_ball_v1_int8.onnx");
    if !tennis_onnx.exists() {
        return Ok(());
    }
    let size_mb =
        ((tennis_onnx.metadata()?.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
    let aliases =
        serde_json::to_string(&["tennis ball", "tennis balls", "bola tenis", "bola tennis"])?;

    sqlx::query(
        "INSERT INTO model_packs (slug, display_name, aliases_json, onnx_path, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at) \
         VALUES ('tennis-ball', 'Tennis Ball Detector v1 (LLM-Distilled INT8)', ?, ?, ?, 640, 35, 4.7, 'ready', ?) \
         ON CONFLICT(slug) DO UPDATE SET onnx_path = excluded.onnx_path, size_mb = excluded.size_mb",
    )
    .bind(aliases)
    .bind(tennis_onnx.to_string_lossy().to_string())
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
                "trainingInFlight": TRAIN_JOBS.contains_key(&r.0),
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
    let row: Option<(String,)> =
        sqlx::query_as("SELECT onnx_path FROM model_packs WHERE slug = ? AND status = 'ready'")
            .bind(&slug)
            .fetch_optional(&st.pool)
            .await
            .map_err(|e| AppError::Internal(e.into()))?;

    let (onnx_path,) =
        row.ok_or_else(|| AppError::NotFound("Model pack tidak ditemukan".into()))?;
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
        return Err(AppError::BadRequest(
            "Query objek tidak boleh kosong".into(),
        ));
    }
    // Atomic check-and-claim: the second concurrent request for the same slug
    // gets a 409 instead of spawning a second worker over the same tmp file.
    let Some(job_guard) = claim_train_slot(&slug) else {
        return Err(AppError::Conflict(
            "Auto-train untuk pack ini masih berjalan".into(),
        ));
    };

    let img_row: Option<(String, i64, i64)> =
        sqlx::query_as("SELECT work_path, width, height FROM images WHERE id = ?")
            .bind(&req.image_id)
            .fetch_optional(&st.pool)
            .await
            .map_err(|e| AppError::Internal(e.into()))?;
    let (work_path, _orig_w, _orig_h) =
        img_row.ok_or_else(|| AppError::NotFound("Gambar tidak ditemukan".into()))?;

    let existing: Option<(String,)> =
        sqlx::query_as("SELECT status FROM model_packs WHERE slug = ?")
            .bind(&slug)
            .fetch_optional(&st.pool)
            .await
            .map_err(|e| AppError::Internal(e.into()))?;

    let aliases = serde_json::to_string(&[req.query.trim().to_lowercase(), slug.clone()]).unwrap();
    let out_onnx = st
        .cfg
        .data_dir
        .join("packs")
        .join(format!("{slug}_int8.onnx"));
    let tmp_onnx = st
        .cfg
        .data_dir
        .join("packs")
        .join(format!("{slug}_int8.tmp.onnx"));

    let previous_status = existing.as_ref().map(|(status,)| status.clone());
    let keep_serving = previous_status.as_deref() == Some("ready");

    if existing.is_none() {
        sqlx::query(
            "INSERT INTO model_packs (slug, display_name, aliases_json, onnx_path, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at) \
             VALUES (?, ?, ?, ?, 0.0, 640, 15, NULL, 'training', ?)",
        )
        .bind(&slug)
        .bind(format!("{} Detector (Auto-Distilled INT8)", req.query.trim()))
        .bind(&aliases)
        .bind(out_onnx.to_string_lossy().to_string())
        .bind(now_rfc3339())
        .execute(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    } else if !keep_serving {
        sqlx::query("UPDATE model_packs SET status = 'training', updated_at = ? WHERE slug = ?")
            .bind(now_rfc3339())
            .bind(&slug)
            .execute(&st.pool)
            .await
            .map_err(|e| AppError::Internal(e.into()))?;
    }

    let st_bg = st.clone();
    let slug_bg = slug.clone();
    let query_bg = req.query.clone();
    tokio::spawn(async move {
        // Ownership of the slot moves into the task: it is released when the
        // task ends, including on error or panic.
        let _job_guard = job_guard;
        let res: anyhow::Result<()> = async {
            let work_img = PathBuf::from(&work_path);
            let dim = image::image_dimensions(&work_img)?;
            // 1. Obtain teacher points from the Gemini 2x2 tiled Validator (M5)
            let teacher = run_vlm_method(&st_bg.http, &st_bg.cfg, "m5", &work_img, dim.0, dim.1, &query_bg).await?;
            if teacher.points.is_empty() {
                anyhow::bail!("Validator LLM menemukan 0 objek pada gambar ini");
            }
            let tmp_dir = st_bg.cfg.data_dir.join("tmp").join(format!("train_{slug_bg}"));
            tokio::fs::create_dir_all(&tmp_dir).await?;
            // Guard, not a plain call: a timeout, worker failure or panic must
            // not leave the dataset directory behind under data/tmp.
            let _tmp_guard = TempDirGuard(tmp_dir.clone());
            let pts_file = tmp_dir.join("teacher_points.json");
            tokio::fs::write(&pts_file, serde_json::to_vec(&json!({ "points": teacher.points }))?).await?;

            // 2. Distill YOLO11n -> INT8 ONNX under heavy_semaphore. The artifact
            // is written to a temp path and only promoted after it exists.
            let _permit = st_bg.heavy_semaphore.acquire().await?;
            let out = timeout(
                Duration::from_secs(TRAIN_TIMEOUT_SECS),
                Command::new(&st_bg.cfg.onnx_python_bin)
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
                    .arg(&tmp_onnx)
                    .arg("--work-dir")
                    .arg(tmp_dir.join("ds"))
                    .arg("--epochs")
                    .arg("15")
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "onnx_worker train timeout setelah {TRAIN_TIMEOUT_SECS} detik (proses dibunuh)"
                )
            })??;
            if !out.status.success() {
                anyhow::bail!("onnx_worker train failed");
            }
            let meta = tokio::fs::metadata(&tmp_onnx).await?;
            if meta.len() == 0 {
                anyhow::bail!("onnx_worker menghasilkan artefak .onnx kosong");
            }
            tokio::fs::rename(&tmp_onnx, &out_onnx).await?;
            let size_mb = ((tokio::fs::metadata(&out_onnx).await?.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
            sqlx::query(
                "UPDATE model_packs SET onnx_path = ?, size_mb = ?, trained_epochs = 15, status = 'ready', updated_at = ? WHERE slug = ?",
            )
            .bind(out_onnx.to_string_lossy().to_string())
            .bind(size_mb)
            .bind(now_rfc3339())
            .bind(&slug_bg)
            .execute(&st_bg.pool)
            .await?;
            Ok(())
        }
        .await;

        match res {
            Ok(()) => {
                tracing::info!(slug = %slug_bg, "auto-train pack ready");
            }
            Err(e) => {
                tracing::error!(error = ?e, slug = %slug_bg, "auto-train pack failed");
                let _ = tokio::fs::remove_file(&tmp_onnx).await;
                let restored = match previous_status.as_deref() {
                    Some(prev) => sqlx::query(
                        "UPDATE model_packs SET status = ?, updated_at = ? WHERE slug = ?",
                    )
                    .bind(prev)
                    .bind(now_rfc3339())
                    .bind(&slug_bg)
                    .execute(&st_bg.pool)
                    .await
                    .map(|_| prev.to_string()),
                    None => sqlx::query(
                        "UPDATE model_packs SET status = 'failed', updated_at = ? WHERE slug = ?",
                    )
                    .bind(now_rfc3339())
                    .bind(&slug_bg)
                    .execute(&st_bg.pool)
                    .await
                    .map(|_| "failed".to_string()),
                };
                if let Err(db_err) = restored {
                    tracing::error!(error = ?db_err, slug = %slug_bg, "failed to restore pack status");
                }
            }
        }
    });

    Ok(Json(
        json!({ "slug": slug, "status": "training", "packStatusUnchanged": keep_serving }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_train_slot_is_insert_if_absent_and_released_on_drop() {
        let slug = "unit_claim_slot_pack";
        let first = claim_train_slot(slug).expect("first claim wins an empty slot");
        assert!(claim_train_slot(slug).is_none(), "second claim must lose");
        drop(first);
        assert!(
            claim_train_slot(slug).is_some(),
            "slot must be reusable after the guard drops"
        );
    }

    #[tokio::test]
    async fn concurrent_train_claims_yield_exactly_one_winner() {
        let slug = "concurrent_claim_slot_pack";
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let barrier = barrier.clone();
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                // Hold the claimed slot until both callers have attempted:
                // otherwise the first winner releases it before the second
                // even asks, and the test would prove nothing.
                let guard = claim_train_slot(slug);
                let won = guard.is_some();
                barrier.wait().await;
                won
            }));
        }
        let mut winners = 0;
        for handle in handles {
            if handle.await.unwrap() {
                winners += 1;
            }
        }
        assert_eq!(winners, 1, "exactly one concurrent caller may own the slot");
    }

    #[tokio::test]
    async fn temp_dir_guard_removes_the_dataset_directory() {
        let root = tempfile::TempDir::new().unwrap();
        let dir = root.path().join("tmp/train_guard_test");
        tokio::fs::create_dir_all(dir.join("ds/images")).await.unwrap();
        tokio::fs::write(dir.join("ds/labels.txt"), b"x").await.unwrap();

        drop(TempDirGuard(dir.clone()));

        // The guard schedules the removal on the runtime; wait for it to land.
        for _ in 0..50 {
            if !dir.exists() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("dataset directory still present after the guard dropped");
    }
}
