use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use image::{Rgb, RgbImage};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn m2_cold_start_and_run_creation_work() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state.clone());

    // Unlock session
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
    let cookie = unlock
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    // Seed a dummy image row
    let img_path = tmp.path().join("uploads/img_1.jpg");
    RgbImage::from_pixel(320, 240, Rgb([10, 20, 30])).save(&img_path).unwrap();
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
    let created: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let run_id = created["runId"].as_str().unwrap().to_string();

    tokio::time::sleep(std::time::Duration::from_millis(120)).await;

    let res = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/runs/{run_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let detail: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(detail["methods"][0]["methodId"], "m2");
    assert_eq!(detail["methods"][0]["status"], "cold_start");
}
