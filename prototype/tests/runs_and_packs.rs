use std::time::Duration;

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use image::{Rgb, RgbImage};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

async fn unlock_cookie(app: &Router) -> String {
    let unlock = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"2limanov"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unlock.status(), StatusCode::OK);
    unlock
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn m2_cold_start_and_run_creation_work() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg)
        .await
        .unwrap();
    let app = anycount_prototype::router(state.clone());

    let cookie = unlock_cookie(&app).await;

    // Seed a dummy image row
    let img_path = tmp.path().join("uploads/img_1.jpg");
    RgbImage::from_pixel(320, 240, Rgb([10, 20, 30]))
        .save(&img_path)
        .unwrap();
    sqlx::query(
        "INSERT INTO images (id, filename, width, height, orig_path, work_path, is_sample, created_at) \
         VALUES ('img_1', 'test.jpg', 320, 240, ?, ?, 0, '2026-11-07T00:00:00Z')",
    )
    .bind(img_path.to_string_lossy().to_string())
    .bind(img_path.to_string_lossy().to_string())
    .execute(&state.pool)
    .await
    .unwrap();

    // Trigger M2 for an untrained object "pineapple" -> transitions to cold_start
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/runs")
                .header(header::COOKIE, &cookie)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"imageId":"img_1","query":"pineapple","methods":["m2"]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let created: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let run_id = created["runId"].as_str().unwrap().to_string();

    tokio::time::sleep(Duration::from_millis(120)).await;

    let res = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/runs/{run_id}"))
                .header(header::COOKIE, &cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let detail: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(detail["methods"][0]["methodId"], "m2");
    assert_eq!(detail["methods"][0]["status"], "cold_start");
    // Spec section 4: a cold-start row makes the run partial, not completed.
    assert_eq!(detail["status"], "partial");
}

#[tokio::test]
async fn run_endpoints_require_session() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg)
        .await
        .unwrap();
    let app = anycount_prototype::router(state);

    for uri in [
        "/api/runs",
        "/api/runs/run_missing",
        "/api/runs/run_missing/stream",
        "/api/runs/run_missing/render/m2",
    ] {
        let res = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::UNAUTHORIZED,
            "anonymous {uri} must be 401"
        );
        let body: Value =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(body["error"]["code"], "AUTH_REQUIRED");
    }
}

#[tokio::test]
async fn failed_auto_train_keeps_ready_pack_and_m2_answers() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg)
        .await
        .unwrap();
    let app = anycount_prototype::router(state.clone());

    let cookie = unlock_cookie(&app).await;

    let img_path = tmp.path().join("uploads/img_2.jpg");
    RgbImage::from_pixel(96, 96, Rgb([200, 30, 30]))
        .save(&img_path)
        .unwrap();
    sqlx::query(
        "INSERT INTO images (id, filename, width, height, orig_path, work_path, is_sample, created_at) \
         VALUES ('img_2', 'apples.jpg', 96, 96, ?, ?, 0, '2026-11-07T00:00:00Z')",
    )
    .bind(img_path.to_string_lossy().to_string())
    .bind(img_path.to_string_lossy().to_string())
    .execute(&state.pool)
    .await
    .unwrap();

    // A ready apple pack whose onnx_path exists must keep serving while a
    // re-train runs, and must survive that re-train failing.
    let onnx_fixture = tmp.path().join("apple_fixture.onnx");
    std::fs::write(&onnx_fixture, b"onnx-fixture").unwrap();
    sqlx::query(
        "INSERT INTO model_packs (slug, display_name, aliases_json, onnx_path, size_mb, imgsz, trained_epochs, val_mape_pct, status, updated_at) \
         VALUES ('apple', 'Apple fixture', '[\"apple\",\"apples\"]', ?, 2.89, 640, 35, 2.33, 'ready', '2026-11-07T00:00:00Z')",
    )
    .bind(onnx_fixture.to_string_lossy().to_string())
    .execute(&state.pool)
    .await
    .unwrap();

    // The VLM teacher points at 127.0.0.1:9, so the background job fails fast.
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/packs/train")
                .header(header::COOKIE, &cookie)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"imageId":"img_2","query":"apples"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["packStatusUnchanged"], true);

    // Wait until the in-flight marker clears (job finished, success or failure).
    let mut finished = false;
    for _ in 0..100 {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/packs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let packs: Value =
            serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
        let in_flight = packs["packs"][0]["trainingInFlight"]
            .as_bool()
            .unwrap_or(true);
        if !in_flight {
            finished = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(finished, "auto-train job did not finish");

    let status: (String,) = sqlx::query_as("SELECT status FROM model_packs WHERE slug = 'apple'")
        .fetch_one(&state.pool)
        .await
        .unwrap();
    assert_eq!(
        status.0, "ready",
        "failed re-train must restore the ready status"
    );

    let pack = anycount_prototype::services::m2_onnx::find_ready_pack(&state, "apples")
        .await
        .unwrap();
    assert!(
        pack.is_some(),
        "M2 must still find the apple pack after a failed re-train"
    );
}
