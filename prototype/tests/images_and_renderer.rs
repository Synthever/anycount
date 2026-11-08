use std::io::Cursor;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use image::{ImageFormat, Rgb, RgbImage};
use serde_json::Value;
use tempfile::TempDir;
use tower::ServiceExt;

#[tokio::test]
async fn render_dots_and_serve_docs_and_upload_image() {
    let tmp = TempDir::new().unwrap();
    let src_img = tmp.path().join("sample.jpg");
    let mut img = RgbImage::from_pixel(400, 300, Rgb([40, 80, 120]));
    img.put_pixel(200, 150, Rgb([255, 255, 255]));
    img.save(&src_img).unwrap();

    let out_img = tmp.path().join("rendered.jpg");
    let pts = vec![anycount_prototype::services::renderer::Point2D { x: 200.0, y: 150.0 }];
    anycount_prototype::services::renderer::render_annotated_jpeg(&src_img, &pts, None, &out_img)
        .unwrap();
    assert!(out_img.exists());
    let rendered = image::open(&out_img).unwrap().to_rgb8();
    // Center pixel at (200, 150) must be painted red (~230, 40, 35)
    let px = rendered.get_pixel(200, 150);
    assert!(px[0] > 180 && px[1] < 80, "expected red dot at center, got {:?}", px);

    let cfg = anycount_prototype::config::Config::for_test(tmp.path());
    let state = anycount_prototype::state::AppState::init(cfg).await.unwrap();
    let app = anycount_prototype::router(state);

    // Public /api/docs returns 6 methods and glossary
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/api/docs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let docs: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(docs["methods"].as_array().unwrap().len(), 6);
    assert!(docs["glossary"].as_array().unwrap().len() >= 12);

    // Upload without session returns 401
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/images")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=BOUND")
                .body(Body::from("--BOUND--\r\n"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

    // Unlock then upload JPEG
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

    let mut jpg_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut jpg_bytes), ImageFormat::Jpeg).unwrap();
    let mut mp = Vec::new();
    mp.extend_from_slice(
        b"--BOUND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n",
    );
    mp.extend_from_slice(&jpg_bytes);
    mp.extend_from_slice(b"\r\n--BOUND--\r\n");

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/images")
                .header(header::COOKIE, cookie)
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=BOUND")
                .body(Body::from(mp))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let uploaded: Value = serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(uploaded["width"], 400);
    assert_eq!(uploaded["height"], 300);
}
