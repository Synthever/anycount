use base64::{engine::general_purpose::STANDARD as B64, Engine};
use futures::future::join_all;
use regex::Regex;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::LazyLock;

use crate::{
    config::Config,
    services::{
        renderer::Point2D,
        tiling::{adaptive_point_nms, split_2x2, TiledPoint},
    },
};

#[derive(Clone, Debug)]
pub struct VlmRunOutput {
    pub points: Vec<Point2D>,
    pub grid_boxes: Option<Vec<[i32; 4]>>,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub raw_preview: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VlmFlavor {
    GeminiYx,
    DeepSeekXy,
}

#[derive(Clone, Debug)]
pub struct VlmEmptyReply {
    pub preview: String,
}

impl std::fmt::Display for VlmEmptyReply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Balasan VLM kosong atau tidak bisa diurai menjadi titik objek"
        )
    }
}

impl std::error::Error for VlmEmptyReply {}

pub fn parse_gateway_response(raw_body: &str) -> (String, i64, i64) {
    let trimmed = raw_body.trim_start();
    if trimmed.starts_with('{') {
        let mut de = serde_json::Deserializer::from_str(trimmed).into_iter::<Value>();
        if let Some(Ok(val)) = de.next() {
            let content = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let p_tok = val["usage"]["prompt_tokens"].as_i64().unwrap_or(0);
            let c_tok = val["usage"]["completion_tokens"].as_i64().unwrap_or(0);
            return (content, p_tok, c_tok);
        }
    }

    let mut parts = Vec::new();
    let mut p_tok = 0;
    let mut c_tok = 0;
    for line in raw_body.lines() {
        let line = line.trim();
        if !line.starts_with("data: ") {
            continue;
        }
        let payload = line[6..].trim();
        if payload == "[DONE]" {
            break;
        }
        if let Ok(chunk) = serde_json::from_str::<Value>(payload) {
            if let Some(s) = chunk["choices"][0]["delta"]["content"].as_str() {
                parts.push(s.to_string());
            }
            if chunk.get("usage").is_some() {
                p_tok = chunk["usage"]["prompt_tokens"].as_i64().unwrap_or(p_tok);
                c_tok = chunk["usage"]["completion_tokens"]
                    .as_i64()
                    .unwrap_or(c_tok);
            }
        }
    }
    (parts.join(""), p_tok, c_tok)
}

static RE_FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)^```(?:json)?\s*|\s*```$").unwrap());
static RE_PAIR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]").unwrap());

pub fn extract_norm_pairs(text: &str) -> Vec<[f64; 2]> {
    let cleaned = RE_FENCE.replace_all(text.trim(), "");
    if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
        let arr_opt = if val.is_array() {
            val.as_array()
        } else {
            val.get("points").and_then(|p| p.as_array())
        };
        if let Some(arr) = arr_opt {
            let mut out = Vec::new();
            for item in arr {
                if let Some(pair) = item.as_array() {
                    if pair.len() >= 2 {
                        if let (Some(a), Some(b)) = (pair[0].as_f64(), pair[1].as_f64()) {
                            out.push([a, b]);
                        }
                    }
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
    }

    let re_pair = &*RE_PAIR;
    re_pair
        .captures_iter(&cleaned)
        .filter_map(|cap| {
            let a = cap.get(1)?.as_str().parse::<f64>().ok()?;
            let b = cap.get(2)?.as_str().parse::<f64>().ok()?;
            Some([a, b])
        })
        .collect()
}

pub fn norm_to_px(pair: [f64; 2], flavor: VlmFlavor, w: f64, h: f64) -> (f64, f64) {
    match flavor {
        VlmFlavor::GeminiYx => {
            let (ny, nx) = (pair[0], pair[1]);
            ((nx / 1000.0) * w, (ny / 1000.0) * h)
        }
        VlmFlavor::DeepSeekXy => {
            let (nx, ny) = (pair[0], pair[1]);
            ((nx / 1000.0) * w, (ny / 1000.0) * h)
        }
    }
}

async fn call_vlm_bytes(
    http: &reqwest::Client,
    cfg: &Config,
    flavor: VlmFlavor,
    jpeg_bytes: &[u8],
    query: &str,
    is_crop: bool,
) -> anyhow::Result<(Vec<[f64; 2]>, i64, i64, String)> {
    if cfg.vlm_api_key.trim().is_empty() {
        anyhow::bail!("VLM_API_KEY belum dikonfigurasi di .env server");
    }
    let b64 = B64.encode(jpeg_bytes);
    let scope = if is_crop { "image crop" } else { "image" };

    let payload = match flavor {
        VlmFlavor::GeminiYx => {
            let prompt = format!(
                "Locate every individual \"{query}\" visible in this {scope}.\n\
                 Place exactly one point [y, x] at the center of each visible instance (normalized 0 to 1000, \
                 where [0,0] is top-left and [1000,1000] is bottom-right).\n\
                 IMPORTANT: Do NOT write coordinates in your thinking process; keep thinking under 3 sentences \
                 and immediately output valid JSON:\n\
                 {{\"points\": [[y1, x1], [y2, x2], ...]}}"
            );
            json!({
                "model": cfg.gemini_model,
                "reasoning_effort": "low",
                "max_tokens": 4096,
                "temperature": 0.1,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            })
        }
        VlmFlavor::DeepSeekXy => {
            let max_tok = if is_crop { 4096 } else { 8192 };
            let prompt = format!(
                "How many {query} are visible in this {scope}? First count them carefully, \
                 then output a JSON object with the exact count and the center coordinates [x, y] \
                 (normalized 0 to 1000, where x is horizontal left-to-right and y is vertical top-to-bottom) \
                 for each visible {query}:\n\
                 {{\"count\": <int>, \"points\": [[x1, y1], [x2, y2], ...]}}"
            );
            json!({
                "model": cfg.deepseek_model,
                "max_tokens": max_tok,
                "temperature": 0.1,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            })
        }
    };

    let resp = http
        .post(&cfg.vlm_api_url)
        .bearer_auth(&cfg.vlm_api_key)
        .json(&payload)
        .send()
        .await?;

    let status = resp.status();
    let raw_body = resp.text().await?;
    if !status.is_success() {
        let tail: String = raw_body.chars().take(240).collect();
        anyhow::bail!("Upstream VLM HTTP {status}: {tail}");
    }

    let (content, p_tok, c_tok) = parse_gateway_response(&raw_body);
    let pairs = extract_norm_pairs(&content);
    let preview: String = content.chars().take(500).collect();
    if pairs.is_empty() && !reply_has_structure(&content) {
        // An empty or unparseable reply is an execution failure, not "0 objects":
        // silently reporting count 0 poisons runs, renders and the flywheel.
        return Err(anyhow::Error::new(VlmEmptyReply {
            preview: preview.clone(),
        }));
    }
    Ok((pairs, p_tok, c_tok, preview))
}

/// True when the model reply is a JSON point payload (even an empty one) or at
/// least contains parseable `[x, y]` pairs.
fn reply_has_structure(text: &str) -> bool {
    let cleaned = RE_FENCE.replace_all(text.trim(), "");
    if let Ok(val) = serde_json::from_str::<Value>(&cleaned) {
        if val.is_array() {
            return true;
        }
        if let Some(points) = val.get("points") {
            return points.is_array();
        }
    }
    RE_PAIR.is_match(&cleaned)
}

pub async fn run_vlm_method(
    http: &reqwest::Client,
    cfg: &Config,
    method_id: &str,
    work_img_path: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) -> anyhow::Result<VlmRunOutput> {
    let (flavor, tiled) = match method_id {
        "m4" => (VlmFlavor::GeminiYx, false),
        "m5" => (VlmFlavor::GeminiYx, true),
        "m6" => (VlmFlavor::DeepSeekXy, false),
        "m7" => (VlmFlavor::DeepSeekXy, true),
        other => anyhow::bail!("Metode VLM tidak dikenal: {other}"),
    };

    if !tiled {
        let jpeg_bytes = tokio::fs::read(work_img_path).await?;
        let (pairs, p_tok, c_tok, preview) =
            call_vlm_bytes(http, cfg, flavor, &jpeg_bytes, query, false).await?;
        let points: Vec<Point2D> = pairs
            .into_iter()
            .map(|pair| {
                let (x, y) = norm_to_px(pair, flavor, orig_w as f64, orig_h as f64);
                Point2D {
                    x: (x * 100.0).round() / 100.0,
                    y: (y * 100.0).round() / 100.0,
                }
            })
            .collect();
        // `call_vlm_bytes` already refuses unparseable/prose replies, so an
        // empty vector here means the model returned a structurally valid
        // zero-object answer (`{"points": []}`). That is a successful count of
        // 0 for a photo without the queried object, not an execution failure.
        return Ok(VlmRunOutput {
            points,
            grid_boxes: None,
            prompt_tokens: p_tok,
            completion_tokens: c_tok,
            raw_preview: preview,
        });
    }

    let (work_w, work_h, work_cores, tiles) = split_2x2(work_img_path, 0.08)?;
    let scale_x = orig_w as f64 / work_w as f64;
    let scale_y = orig_h as f64 / work_h as f64;

    let futs = tiles.iter().map(|t| async move {
        let res = call_vlm_bytes(http, cfg, flavor, &t.jpeg_bytes, query, true).await?;
        Ok::<_, anyhow::Error>((t, res))
    });

    let results = join_all(futs).await;
    let mut all_tiled_pts = Vec::new();
    let mut total_p_tok = 0;
    let mut total_c_tok = 0;
    let mut previews = Vec::new();
    let mut tiles_ok = 0usize;
    let mut tile_errors = Vec::new();

    for item in results {
        // A single bad tile (upstream 5xx, timeout, unparseable prose) must not
        // discard the other three: it contributes zero points and the rest of
        // the tiles still count. The method only fails when every tile fails.
        let (tile, (pairs, pt, ct, prev)) = match item {
            Ok(v) => v,
            Err(e) => {
                tile_errors.push(e.to_string());
                continue;
            }
        };
        tiles_ok += 1;
        total_p_tok += pt;
        total_c_tok += ct;
        previews.push(format!("[{}] {}", tile.tile_id, prev));

        let [crop_x0, crop_y0, _, _] = tile.crop_box;
        let [core_x0, core_y0, core_x1, core_y1] = tile.core_box;
        let tol = tile.seam_tol_px;

        for pair in pairs {
            let (lx, ly) = norm_to_px(pair, flavor, tile.tile_w as f64, tile.tile_h as f64);
            let gx = crop_x0 as f64 + lx;
            let gy = crop_y0 as f64 + ly;
            if gx >= core_x0 - tol
                && gx <= core_x1 + tol
                && gy >= core_y0 - tol
                && gy <= core_y1 + tol
            {
                all_tiled_pts.push(TiledPoint {
                    x: gx * scale_x,
                    y: gy * scale_y,
                    tile_id: tile.tile_id.clone(),
                });
            }
        }
    }

    for err in &tile_errors {
        previews.push(format!("[tile-error] {err}"));
    }

    if tiles_ok == 0 {
        return Err(anyhow::Error::new(VlmEmptyReply {
            preview: previews.join("\n"),
        }));
    }
    // At least one tile answered structurally; zero points then means "no such
    // object in this photo", which is a valid count of 0.

    let dedup = adaptive_point_nms(&all_tiled_pts, orig_w, orig_h);
    let orig_cores: Vec<[i32; 4]> = work_cores
        .into_iter()
        .map(|[x0, y0, x1, y1]| {
            [
                (x0 as f64 * scale_x).round() as i32,
                (y0 as f64 * scale_y).round() as i32,
                (x1 as f64 * scale_x).round() as i32,
                (y1 as f64 * scale_y).round() as i32,
            ]
        })
        .collect();

    Ok(VlmRunOutput {
        points: dedup,
        grid_boxes: Some(orig_cores),
        prompt_tokens: total_p_tok,
        completion_tokens: total_c_tok,
        raw_preview: previews.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn find_header_end(buf: &[u8]) -> Option<usize> {
        buf.windows(4).position(|w| w == b"\r\n\r\n")
    }

    /// Minimal stand-in for the LLM gateway: each request gets the next
    /// scripted `(status, message content)` reply; the last entry repeats.
    async fn spawn_vlm_mock(replies: Vec<(u16, String)>) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let replies = replies.clone();
                let counter = counter.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 2048];
                    let header_end = loop {
                        match sock.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => buf.extend_from_slice(&chunk[..n]),
                        }
                        if let Some(pos) = find_header_end(&buf) {
                            break pos;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_lowercase();
                    let content_len: usize = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    let already = buf.len() - (header_end + 4);
                    if content_len > already {
                        let mut rest = vec![0u8; content_len - already];
                        if sock.read_exact(&mut rest).await.is_err() {
                            return;
                        }
                    }
                    let idx = counter.fetch_add(1, Ordering::SeqCst);
                    let (status, content) = replies[idx.min(replies.len() - 1)].clone();
                    let body = if status == 200 {
                        json!({
                            "choices": [{"message": {"content": content}}],
                            "usage": {"prompt_tokens": 7, "completion_tokens": 3}
                        })
                        .to_string()
                    } else {
                        content
                    };
                    let resp = format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        (format!("http://{addr}/v1/chat/completions"), hits)
    }

    fn test_cfg(dir: &Path, url: String) -> Config {
        let mut cfg = Config::for_test(dir);
        cfg.vlm_api_url = url;
        cfg
    }

    #[tokio::test]
    async fn valid_empty_points_is_a_zero_count_not_an_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let (url, _hits) = spawn_vlm_mock(vec![(200, r#"{"points": []}"#.to_string())]).await;
        let img = dir.path().join("work.jpg");
        std::fs::write(&img, b"any-bytes-work-for-the-single-pass").unwrap();

        let out = run_vlm_method(&reqwest::Client::new(), &test_cfg(dir.path(), url), "m4", &img, 1000, 800, "apple")
            .await
            .expect("a structurally valid zero-object reply must succeed");
        assert!(out.points.is_empty());
        assert_eq!(out.raw_preview, r#"{"points": []}"#);
        assert_eq!(out.prompt_tokens, 7);
    }

    #[tokio::test]
    async fn prose_reply_still_fails_the_single_pass() {
        let dir = tempfile::TempDir::new().unwrap();
        let (url, _hits) =
            spawn_vlm_mock(vec![(200, "I could not find any apples here.".to_string())]).await;
        let img = dir.path().join("work.jpg");
        std::fs::write(&img, b"any-bytes").unwrap();

        let err = run_vlm_method(&reqwest::Client::new(), &test_cfg(dir.path(), url), "m4", &img, 1000, 800, "apple")
            .await
            .expect_err("unparseable replies must stay errors");
        assert!(err.downcast_ref::<VlmEmptyReply>().is_some());
    }

    fn write_test_jpeg(dir: &Path) -> std::path::PathBuf {
        let img = dir.join("work.jpg");
        image::RgbImage::from_pixel(400, 400, image::Rgb([90, 120, 150]))
            .save(&img)
            .unwrap();
        img
    }

    #[tokio::test]
    async fn one_failing_tile_keeps_the_other_three() {
        // The first tile request hits an upstream 500; the remaining three
        // answer with one centred point each. The method must still succeed.
        let dir = tempfile::TempDir::new().unwrap();
        let (url, hits) = spawn_vlm_mock(vec![
            (500, "upstream boom".to_string()),
            (200, r#"{"points": [[500, 500]]}"#.to_string()),
        ])
        .await;
        let img = write_test_jpeg(dir.path());

        let out = run_vlm_method(&reqwest::Client::new(), &test_cfg(dir.path(), url), "m5", &img, 400, 400, "apple")
            .await
            .expect("one failed tile must not fail M5");
        assert_eq!(hits.load(Ordering::SeqCst), 4);
        assert_eq!(out.points.len(), 3);
        assert!(out.raw_preview.contains("tile-error"));
        assert_eq!(out.grid_boxes.as_ref().unwrap().len(), 4);
    }

    #[tokio::test]
    async fn every_tile_failing_still_fails_the_method() {
        let dir = tempfile::TempDir::new().unwrap();
        let (url, hits) = spawn_vlm_mock(vec![(500, "upstream boom".to_string())]).await;
        let img = write_test_jpeg(dir.path());

        let err = run_vlm_method(&reqwest::Client::new(), &test_cfg(dir.path(), url), "m7", &img, 400, 400, "apple")
            .await
            .expect_err("all tiles failing must fail the method");
        assert!(err.downcast_ref::<VlmEmptyReply>().is_some());
        assert_eq!(hits.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn m3_belongs_to_the_offline_tiled_path_not_the_vlm_dispatcher() {
        let dir = tempfile::TempDir::new().unwrap();
        let img = write_test_jpeg(dir.path());
        let err = run_vlm_method(
            &reqwest::Client::new(),
            &test_cfg(dir.path(), "http://127.0.0.1:9/v1/chat/completions".into()),
            "m3",
            &img,
            400,
            400,
            "apple",
        )
        .await
        .expect_err("m3 must be rejected by the VLM dispatcher");
        assert!(
            err.to_string().contains("tidak dikenal"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn detects_empty_and_unparseable_replies() {
        assert!(!reply_has_structure(""));
        assert!(!reply_has_structure("I could not find any apples here."));
        assert!(reply_has_structure(r#"{"points": []}"#));
        assert!(reply_has_structure("[[250, 500]]"));
        assert!(reply_has_structure(
            "Found 2.\n```json\n{\"count\":2,\"points\":[[700, 300]]}\n```"
        ));
    }

    #[test]
    fn parses_gemini_sse_yx_and_deepseek_json_done_xy() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"points\\\": [[250, 500]]}\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20}}\n\
                   data: [DONE]\n";
        let (txt, pt, ct) = parse_gateway_response(sse);
        assert_eq!((pt, ct), (100, 20));
        let pairs = extract_norm_pairs(&txt);
        assert_eq!(pairs, vec![[250.0, 500.0]]);
        let (px, py) = norm_to_px(pairs[0], VlmFlavor::GeminiYx, 1000.0, 800.0);
        assert_eq!((px, py), (500.0, 200.0));

        let ds = "{\"choices\":[{\"message\":{\"content\":\"Found 2.\\n```json\\n{\\\"count\\\":2,\\\"points\\\":[[700, 300]]}\\n```\"}}],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":15}}data: [DONE]";
        let (txt2, pt2, ct2) = parse_gateway_response(ds);
        assert_eq!((pt2, ct2), (50, 15));
        let pairs2 = extract_norm_pairs(&txt2);
        let (px2, py2) = norm_to_px(pairs2[0], VlmFlavor::DeepSeekXy, 1000.0, 800.0);
        assert_eq!((px2, py2), (700.0, 240.0));
    }
}
