use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn health_and_passcode_unlock_flow() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    assert_eq!(cfg.bind_addr.ip().to_string(), "127.0.0.1");
    assert_eq!(cfg.passcode, "2limanov");

    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state);

    // 1. Health is public
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 2. Auth status starts locked
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/auth/status").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], false);

    // 3. Wrong passcode rejected with 401 AUTH_INVALID_PASSCODE
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"passcode":"wrong"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // 4. Correct passcode "2limanov" sets HttpOnly cookie and unlocks session
    let res = app
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
    assert_eq!(res.status(), StatusCode::OK);
    let set_cookie = res
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(set_cookie.contains("anycount_session="));
    assert!(set_cookie.contains("HttpOnly"));

    let cookie_kv = set_cookie.split(';').next().unwrap();
    let res = app
        .oneshot(
            Request::builder()
                .uri("/api/auth/status")
                .header(header::COOKIE, cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], true);
}
