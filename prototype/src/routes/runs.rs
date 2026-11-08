use std::{convert::Infallible, path::PathBuf, time::Instant};
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::{
    auth::{generate_token, now_rfc3339, RequireSession},
    error::AppError,
    services::{
        m1_ca::run_count_anything,
        m2_onnx::{run_m2_onnx, OnnxOutcome},
        renderer::{render_annotated_jpeg, Point2D},
        vlm::run_vlm_method,
    },
    state::AppState,
};

const VALID_METHODS: &[&str] = &["m1", "m2", "m3", "m4", "m5", "m6"];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRunReq {
    pub image_id: String,
    pub query: String,
    pub methods: Vec<String>,
}

pub async fn load_run_json(st: &AppState, run_id: &str) -> anyhow::Result<Option<Value>> {
    let run_row: Option<(String, String, String, String, String, i64, i64)> = sqlx::query_as(
        "SELECT r.id, r.image_id, r.query, r.status, r.created_at, i.width, i.height \
         FROM runs r JOIN images i ON i.id = r.image_id WHERE r.id = ?",
    )
    .bind(run_id)
    .fetch_optional(&st.pool)
    .await?;

    let Some((id, image_id, query, status, created_at, width, height)) = run_row else {
        return Ok(None);
    };

    let m_rows: Vec<(
        String,
        String,
        Option<i64>,
        Option<String>,
        Option<String>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<f64>,
        Option<String>,
    )> = sqlx::query_as(
        "SELECT method_id, status, count, points_json, grid_boxes_json, elapsed_ms, \
         prompt_tokens, completion_tokens, model_size_mb, meta_json \
         FROM run_methods WHERE run_id = ? ORDER BY method_id",
    )
    .bind(&id)
    .fetch_all(&st.pool)
    .await?;

    let methods: Vec<Value> = m_rows
        .into_iter()
        .map(|m| {
            let pts: Value = m.3.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!([]));
            let grids: Value = m.4.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(Value::Null);
            let meta: Value = m.9.as_deref().and_then(|s| serde_json::from_str(s).ok()).unwrap_or(json!({}));
            json!({
                "methodId": m.0,
                "status": m.1,
                "count": m.2,
                "points": pts,
                "gridBoxes": grids,
                "elapsedMs": m.5,
                "promptTokens": m.6,
                "completionTokens": m.7,
                "modelSizeMb": m.8,
                "meta": meta,
                "renderedUrl": format!("/api/runs/{id}/render/{}", m.0),
            })
        })
        .collect();

    Ok(Some(json!({
        "runId": id,
        "imageId": image_id,
        "imageUrl": format!("/api/images/{image_id}/raw"),
        "imageWidth": width,
        "imageHeight": height,
        "query": query,
        "status": status,
        "createdAt": created_at,
        "methods": methods,
    })))
}

async fn emit_snapshot(st: &AppState, run_id: &str) {
    if let Ok(Some(val)) = load_run_json(st, run_id).await {
        if let Some(tx) = st.run_streams.get(run_id) {
            let _ = tx.send(val.to_string());
        }
    }
}

pub async fn create_run(
    State(st): State<AppState>,
    _session: RequireSession,
    Json(req): Json<CreateRunReq>,
) -> Result<impl IntoResponse, AppError> {
    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err(AppError::BadRequest("Query objek wajib diisi".into()));
    }
    let mut methods: Vec<String> = req
        .methods
        .into_iter()
        .filter(|m| VALID_METHODS.contains(&m.as_str()))
        .collect();
    methods.sort();
    methods.dedup();
    if methods.is_empty() {
        return Err(AppError::BadRequest("Pilih minimal satu metode (m1..m6)".into()));
    }

    let img_row: Option<(String, String, i64, i64)> = sqlx::query_as(
        "SELECT orig_path, work_path, width, height FROM images WHERE id = ?",
    )
    .bind(&req.image_id)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;
    let (orig_path, work_path, orig_w, orig_h) =
        img_row.ok_or_else(|| AppError::NotFound("Gambar tidak ditemukan".into()))?;

    let run_id = format!("run_{}", &generate_token()[..16]);
    let now = now_rfc3339();
    sqlx::query("INSERT INTO runs (id, image_id, query, status, created_at) VALUES (?, ?, ?, 'running', ?)")
        .bind(&run_id)
        .bind(&req.image_id)
        .bind(&query)
        .bind(&now)
        .execute(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    for m in &methods {
        let init_status = if m == "m1" && st.heavy_semaphore.available_permits() == 0 {
            "queued"
        } else {
            "running"
        };
        sqlx::query(
            "INSERT INTO run_methods (run_id, method_id, status, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(m)
        .bind(init_status)
        .bind(&now)
        .execute(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    }

    let (tx, _) = broadcast::channel::<String>(32);
    st.run_streams.insert(run_id.clone(), tx);

    let initial_json = load_run_json(&st, &run_id)
        .await
        .map_err(AppError::Internal)?
        .unwrap();

    // Spawn concurrent execution per selected method
    let st_orch = st.clone();
    let run_id_orch = run_id.clone();
    tokio::spawn(async move {
        let mut handles = Vec::new();
        for method_id in methods {
            let st_m = st_orch.clone();
            let rid = run_id_orch.clone();
            let q = query.clone();
            let orig_p = PathBuf::from(&orig_path);
            let work_p = PathBuf::from(&work_path);
            let w = orig_w as u32;
            let h = orig_h as u32;

            handles.push(tokio::spawn(async move {
                execute_single_method(&st_m, &rid, &method_id, &orig_p, &work_p, w, h, &q).await;
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        let _ = sqlx::query("UPDATE runs SET status = 'completed' WHERE id = ?")
            .bind(&run_id_orch)
            .execute(&st_orch.pool)
            .await;
        emit_snapshot(&st_orch, &run_id_orch).await;
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        st_orch.run_streams.remove(&run_id_orch);
    });

    Ok(Json(initial_json))
}

async fn execute_single_method(
    st: &AppState,
    run_id: &str,
    method_id: &str,
    orig_path: &PathBuf,
    work_path: &PathBuf,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) {
    let started = Instant::now();
    let _ = sqlx::query("UPDATE run_methods SET status = 'running', updated_at = ? WHERE run_id = ? AND method_id = ?")
        .bind(now_rfc3339())
        .bind(run_id)
        .bind(method_id)
        .execute(&st.pool)
        .await;
    emit_snapshot(st, run_id).await;

    let render_out = st
        .cfg
        .data_dir
        .join("renders")
        .join(format!("{run_id}_{method_id}.jpg"));

    match method_id {
        "m1" => match run_count_anything(st, work_path, orig_w, orig_h, query, run_id).await {
            Ok(ca) => {
                let _ = render_annotated_jpeg(orig_path, &ca.points, None, &render_out);
                save_success(
                    st,
                    run_id,
                    method_id,
                    &ca.points,
                    None,
                    &render_out,
                    ca.elapsed_ms,
                    0,
                    0,
                    ca.model_size_mb,
                    json!({"engine": "Count-Anything SAM3 (CPU mmap)"}),
                )
                .await;
            }
            Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
        },
        "m2" => match run_m2_onnx(st, work_path, orig_w, orig_h, query).await {
            Ok(OnnxOutcome::Ready {
                pack_slug,
                points,
                grid_boxes,
                elapsed_ms,
                model_size_mb,
                mode,
            }) => {
                let _ = render_annotated_jpeg(orig_path, &points, grid_boxes.as_deref(), &render_out);
                save_success(
                    st,
                    run_id,
                    method_id,
                    &points,
                    grid_boxes,
                    &render_out,
                    elapsed_ms,
                    0,
                    0,
                    model_size_mb,
                    json!({"packSlug": pack_slug, "mode": mode}),
                )
                .await;
            }
            Ok(OnnxOutcome::ColdStart { normalized_query }) => {
                let _ = sqlx::query(
                    "UPDATE run_methods SET status = 'cold_start', elapsed_ms = ?, meta_json = ?, updated_at = ? \
                     WHERE run_id = ? AND method_id = ?",
                )
                .bind(started.elapsed().as_millis() as i64)
                .bind(
                    json!({
                        "normalizedQuery": normalized_query,
                        "message": format!("Model pack .onnx untuk '{}' belum dilatih di katalog offline.", query),
                    })
                    .to_string(),
                )
                .bind(now_rfc3339())
                .bind(run_id)
                .bind(method_id)
                .execute(&st.pool)
                .await;
            }
            Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
        },
        "m3" | "m4" | "m5" | "m6" => {
            match run_vlm_method(&st.http, &st.cfg, method_id, work_path, orig_w, orig_h, query).await {
                Ok(vlm) => {
                    let elapsed_ms = started.elapsed().as_millis() as i64;
                    let _ = render_annotated_jpeg(
                        orig_path,
                        &vlm.points,
                        vlm.grid_boxes.as_deref(),
                        &render_out,
                    );
                    save_success(
                        st,
                        run_id,
                        method_id,
                        &vlm.points,
                        vlm.grid_boxes,
                        &render_out,
                        elapsed_ms,
                        vlm.prompt_tokens,
                        vlm.completion_tokens,
                        0.0,
                        json!({"rawPreview": vlm.raw_preview}),
                    )
                    .await;
                }
                Err(e) => save_failure(st, run_id, method_id, started.elapsed().as_millis() as i64, &e.to_string()).await,
            }
        }
        _ => {}
    }
    emit_snapshot(st, run_id).await;
}

async fn save_success(
    st: &AppState,
    run_id: &str,
    method_id: &str,
    points: &[Point2D],
    grid_boxes: Option<Vec<[i32; 4]>>,
    render_out: &PathBuf,
    elapsed_ms: i64,
    p_tok: i64,
    c_tok: i64,
    model_size_mb: f64,
    meta: Value,
) {
    let _ = sqlx::query(
        "UPDATE run_methods SET status = 'completed', count = ?, points_json = ?, grid_boxes_json = ?, \
         rendered_path = ?, elapsed_ms = ?, prompt_tokens = ?, completion_tokens = ?, model_size_mb = ?, \
         meta_json = ?, updated_at = ? WHERE run_id = ? AND method_id = ?",
    )
    .bind(points.len() as i64)
    .bind(serde_json::to_string(points).unwrap_or_else(|_| "[]".into()))
    .bind(grid_boxes.map(|g| serde_json::to_string(&g).unwrap_or_default()))
    .bind(render_out.to_string_lossy().to_string())
    .bind(elapsed_ms)
    .bind(p_tok)
    .bind(c_tok)
    .bind(model_size_mb)
    .bind(meta.to_string())
    .bind(now_rfc3339())
    .bind(run_id)
    .bind(method_id)
    .execute(&st.pool)
    .await;
}

async fn save_failure(st: &AppState, run_id: &str, method_id: &str, elapsed_ms: i64, msg: &str) {
    let _ = sqlx::query(
        "UPDATE run_methods SET status = 'failed', elapsed_ms = ?, meta_json = ?, updated_at = ? \
         WHERE run_id = ? AND method_id = ?",
    )
    .bind(elapsed_ms)
    .bind(json!({ "error": msg }).to_string())
    .bind(now_rfc3339())
    .bind(run_id)
    .bind(method_id)
    .execute(&st.pool)
    .await;
}

pub async fn get_run(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let val = load_run_json(&st, &id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Run tidak ditemukan".into()))?;
    Ok(Json(val))
}

pub async fn list_runs(State(st): State<AppState>) -> Result<impl IntoResponse, AppError> {
    let ids: Vec<(String,)> = sqlx::query_as("SELECT id FROM runs ORDER BY created_at DESC LIMIT 20")
        .fetch_all(&st.pool)
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    let mut items = Vec::new();
    for (id,) in ids {
        if let Ok(Some(v)) = load_run_json(&st, &id).await {
            items.push(v);
        }
    }
    Ok(Json(json!({ "runs": items })))
}

pub async fn stream_run(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, AppError> {
    let initial = load_run_json(&st, &id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Run tidak ditemukan".into()))?;

    let rx_opt = st.run_streams.get(&id).map(|tx| tx.subscribe());
    let init_stream = futures::stream::once(async move {
        Ok::<Event, Infallible>(Event::default().data(initial.to_string()))
    });

    let live_stream = match rx_opt {
        Some(rx) => BroadcastStream::new(rx)
            .filter_map(|msg| async move {
                msg.ok().map(|s| Ok::<Event, Infallible>(Event::default().data(s)))
            })
            .boxed(),
        None => futures::stream::empty().boxed(),
    };

    Ok(Sse::new(init_stream.chain(live_stream)).keep_alive(KeepAlive::default()))
}

pub async fn get_rendered_image(
    State(st): State<AppState>,
    Path((id, method_id)): Path<(String, String)>,
) -> Result<Response, AppError> {
    let row: Option<(Option<String>,)> = sqlx::query_as(
        "SELECT rendered_path FROM run_methods WHERE run_id = ? AND method_id = ?",
    )
    .bind(&id)
    .bind(&method_id)
    .fetch_optional(&st.pool)
    .await
    .map_err(|e| AppError::Internal(e.into()))?;

    let path_str = row
        .and_then(|r| r.0)
        .ok_or_else(|| AppError::NotFound("Hasil render belum tersedia".into()))?;
    let bytes = tokio::fs::read(&path_str)
        .await
        .map_err(|_| AppError::NotFound("File render tidak ditemukan".into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .body(Body::from(bytes))
        .unwrap())
}
