use std::sync::Mutex;

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    routing::get,
    Json, Router,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

static ENV_LOCK: Mutex<()> = Mutex::new(());

async fn protected_probe(_session: anycount_prototype::auth::RequireSession) -> Json<Value> {
    Json(json!({ "ok": true }))
}

#[tokio::test]
async fn health_and_passcode_unlock_flow() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    assert_eq!(cfg.bind_addr.ip().to_string(), "127.0.0.1");
    assert_eq!(cfg.passcode, "2limanov");
    assert_eq!(cfg.vlm_api_key, "test-fixture-key");

    let cfg_debug = format!("{cfg:?}");
    assert!(cfg_debug.contains("[REDACTED]"));
    assert!(!cfg_debug.contains("2limanov"));
    assert!(!cfg_debug.contains("test-fixture-key"));

    let state = anycount_prototype::state::AppState::init(cfg)
        .await
        .unwrap();
    let test_protected = Router::new()
        .route("/api/test/protected", get(protected_probe))
        .with_state(state.clone());
    let app = anycount_prototype::router(state.clone()).merge(test_protected);

    // 1. Health is public
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 2. Auth status starts locked
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/auth/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], false);

    // 3. Protected route rejects unauthenticated calls with 401 AUTH_REQUIRED
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/test/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "AUTH_REQUIRED");

    // 4. Wrong passcode rejected with 401 AUTH_INVALID_PASSCODE JSON body
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
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "AUTH_INVALID_PASSCODE");

    // 5. Oversized unlock payload (>4096 bytes) rejected with 413 Payload Too Large
    let oversized = format!(r#"{{"passcode":"{}"}}"#, "a".repeat(5000));
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/unlock")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(oversized))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);

    // 6. Correct passcode "2limanov" sets HttpOnly cookie and unlocks session
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
    assert!(set_cookie.contains("Secure"));
    assert!(set_cookie.contains("SameSite=Lax"));

    let cookie_kv = set_cookie.split(';').next().unwrap().to_string();

    // 7. Status is unlocked and protected route returns 200 with valid session cookie
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/auth/status")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], true);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/test/protected")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["ok"], true);

    // 8. Logout revokes DB session, clears cookie with SameSite=Lax, and re-locks protected route
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/logout")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let clear_cookie = res
        .headers()
        .get(header::SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(
        clear_cookie,
        "anycount_session=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"
    );
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], false);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/auth/status")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["unlocked"], false);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/test/protected")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "AUTH_REQUIRED");

    // 9. Logout propagates DB error instead of ignoring it
    state.pool.close().await;
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/auth/logout")
                .header(header::COOKIE, &cookie_kv)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "INTERNAL_ERROR");
}

#[tokio::test]
async fn unlock_is_rate_limited_per_ip() {
    let tmp = TempDir::new().unwrap();
    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg)
        .await
        .unwrap();
    let app = anycount_prototype::router(state);

    let attempt = |passcode: &'static str, ip: &'static str| {
        let app = app.clone();
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/auth/unlock")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("x-forwarded-for", ip)
                    .body(Body::from(format!(r#"{{"passcode":"{passcode}"}}"#)))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    };

    // Fixed window: 10 failed attempts allowed, the 11th is throttled.
    for i in 0..10 {
        let res = attempt("definitely-wrong", "203.0.113.9").await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "attempt {i}");
    }
    let res = attempt("definitely-wrong", "203.0.113.9").await;
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "RATE_LIMITED");

    // Even the right passcode is refused while the IP is throttled.
    let res = attempt("2limanov", "203.0.113.9").await;
    assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);

    // A different client IP is unaffected.
    let res = attempt("2limanov", "198.51.100.7").await;
    assert_eq!(res.status(), StatusCode::OK);
}

#[test]
fn config_from_env_trims_and_rejects_empty_passcode() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev_bind = std::env::var("ANYCOUNT_BIND").ok();
    let prev_pass = std::env::var("ANYCOUNT_PASSCODE").ok();

    std::env::remove_var("ANYCOUNT_BIND");
    std::env::set_var("ANYCOUNT_PASSCODE", "   ");
    let err = anycount_prototype::config::Config::from_env().unwrap_err();
    assert!(
        err.to_string()
            .contains("ANYCOUNT_PASSCODE cannot be empty"),
        "unexpected error: {err}"
    );

    std::env::set_var("ANYCOUNT_PASSCODE", "  2limanov  ");
    let cfg = anycount_prototype::config::Config::from_env().unwrap();
    assert_eq!(cfg.passcode, "2limanov");

    match prev_bind {
        Some(v) => std::env::set_var("ANYCOUNT_BIND", v),
        None => std::env::remove_var("ANYCOUNT_BIND"),
    }
    match prev_pass {
        Some(v) => std::env::set_var("ANYCOUNT_PASSCODE", v),
        None => std::env::remove_var("ANYCOUNT_PASSCODE"),
    }
}
