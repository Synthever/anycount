use std::path::{Path, PathBuf};
use serde_json::Value;
use tokio::process::Command;

use crate::{services::renderer::Point2D, state::AppState};

pub enum OnnxOutcome {
    Ready {
        pack_slug: String,
        points: Vec<Point2D>,
        grid_boxes: Option<Vec<[i32; 4]>>,
        elapsed_ms: i64,
        model_size_mb: f64,
        mode: String,
    },
    ColdStart {
        normalized_query: String,
    },
}

pub fn normalize_query_slug(q: &str) -> String {
    let lower: String = q
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = lower.trim_matches('_');
    trimmed.strip_suffix('s').unwrap_or(trimmed).to_string()
}

pub async fn find_ready_pack(st: &AppState, query: &str) -> anyhow::Result<Option<(String, PathBuf, f64)>> {
    let norm = query.trim().to_lowercase();
    let slug_guess = normalize_query_slug(query);
    let rows: Vec<(String, String, String, f64)> = sqlx::query_as(
        "SELECT slug, aliases_json, onnx_path, size_mb FROM model_packs WHERE status = 'ready'",
    )
    .fetch_all(&st.pool)
    .await?;

    for (slug, aliases_json, onnx_path, size_mb) in rows {
        let aliases: Vec<String> = serde_json::from_str(&aliases_json).unwrap_or_default();
        if slug == norm
            || slug == slug_guess
            || aliases.iter().any(|a| a.eq_ignore_ascii_case(&norm) || a.eq_ignore_ascii_case(&slug_guess))
        {
            let p = PathBuf::from(&onnx_path);
            if p.exists() {
                return Ok(Some((slug, p, size_mb)));
            }
        }
    }
    Ok(None)
}

pub async fn run_m2_onnx(
    st: &AppState,
    work_img: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
) -> anyhow::Result<OnnxOutcome> {
    let Some((slug, onnx_path, _size_mb)) = find_ready_pack(st, query).await? else {
        return Ok(OnnxOutcome::ColdStart {
            normalized_query: normalize_query_slug(query),
        });
    };

    let out = Command::new(&st.cfg.onnx_python_bin)
        .env_remove("PYTHONPATH")
        .arg("workers/onnx_worker.py")
        .arg("infer")
        .arg("--onnx")
        .arg(&onnx_path)
        .arg("--image")
        .arg(work_img)
        .arg("--orig-width")
        .arg(orig_w.to_string())
        .arg("--orig-height")
        .arg(orig_h.to_string())
        .output()
        .await?;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("ONNX inference gagal: {}", &err[err.len().saturating_sub(300)..]);
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_line = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("ANYCOUNT_JSON:"))
        .ok_or_else(|| anyhow::anyhow!("Output JSON ONNX tidak ditemukan"))?;
    let val: Value = serde_json::from_str(json_line)?;
    let points: Vec<Point2D> = serde_json::from_value(val["points"].clone())?;
    let grid_boxes: Option<Vec<[i32; 4]>> = serde_json::from_value(val["grid_boxes"].clone()).ok();

    Ok(OnnxOutcome::Ready {
        pack_slug: slug,
        points,
        grid_boxes,
        elapsed_ms: val["elapsed_ms"].as_i64().unwrap_or(0),
        model_size_mb: val["model_size_mb"].as_f64().unwrap_or(2.89),
        mode: val["mode"].as_str().unwrap_or("single_640").to_string(),
    })
}
