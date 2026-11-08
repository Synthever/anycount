use std::path::Path;
use serde_json::Value;
use tokio::process::Command;

use crate::{services::renderer::Point2D, state::AppState};

pub struct CaOutput {
    pub points: Vec<Point2D>,
    pub elapsed_ms: i64,
    pub model_size_mb: f64,
}

pub async fn run_count_anything(
    st: &AppState,
    work_img: &Path,
    orig_w: u32,
    orig_h: u32,
    query: &str,
    run_id: &str,
) -> anyhow::Result<CaOutput> {
    let _permit = st.heavy_semaphore.acquire().await?;
    let worker_script = Path::new("workers/ca_worker.py");
    let ca_work_dir = st.cfg.data_dir.join("tmp").join(format!("ca_{run_id}"));

    let out = Command::new(&st.cfg.ca_python_bin)
        .env_remove("PYTHONPATH")
        .arg(worker_script)
        .arg("--ca-repo")
        .arg(&st.cfg.ca_repo_dir)
        .arg("--image")
        .arg(work_img)
        .arg("--query")
        .arg(query)
        .arg("--orig-width")
        .arg(orig_w.to_string())
        .arg("--orig-height")
        .arg(orig_h.to_string())
        .arg("--work-dir")
        .arg(&ca_work_dir)
        .output()
        .await?;

    let _ = tokio::fs::remove_dir_all(&ca_work_dir).await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        anyhow::bail!("Count-Anything gagal: {}", &stderr[stderr.len().saturating_sub(400)..]);
    }

    let json_line = stdout
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("ANYCOUNT_JSON:"))
        .ok_or_else(|| anyhow::anyhow!("Output JSON Count-Anything tidak ditemukan"))?;
    let val: Value = serde_json::from_str(json_line)?;
    let points: Vec<Point2D> = serde_json::from_value(val["points"].clone())?;

    Ok(CaOutput {
        points,
        elapsed_ms: val["elapsed_ms"].as_i64().unwrap_or(0),
        model_size_mb: val["model_size_mb"].as_f64().unwrap_or(3410.0),
    })
}
