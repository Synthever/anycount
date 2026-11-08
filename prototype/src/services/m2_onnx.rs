use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

use crate::{services::renderer::Point2D, state::AppState};

/// M1 and train are both bounded; M2/M3 must be too, or a wedged ONNX runtime
/// keeps an HTTP request (and a Python process) alive forever.
const M2_INFER_TIMEOUT_SECS: u64 = 120;

/// M2 and M3 are the same offline pack at two explicit inference shapes:
/// M2 = one 640 pass, M3 = always 2x2 tiled + Point-NMS. Nothing escalates
/// automatically any more, so the caller states the mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxMode {
    Single,
    Tiled,
}

impl OnnxMode {
    pub fn flag(self) -> &'static str {
        match self {
            OnnxMode::Single => "single",
            OnnxMode::Tiled => "tiled",
        }
    }

    /// Fallback label when an older worker build omits `mode` in its JSON.
    pub fn default_label(self) -> &'static str {
        match self {
            OnnxMode::Single => "single_640",
            OnnxMode::Tiled => "tiled_2x2_640+point_nms",
        }
    }
}

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

pub async fn find_ready_pack(
    st: &AppState,
    query: &str,
) -> anyhow::Result<Option<(String, PathBuf, f64)>> {
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
            || aliases
                .iter()
                .any(|a| a.eq_ignore_ascii_case(&norm) || a.eq_ignore_ascii_case(&slug_guess))
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
    mode: OnnxMode,
) -> anyhow::Result<OnnxOutcome> {
    let Some((slug, onnx_path, _size_mb)) = find_ready_pack(st, query).await? else {
        return Ok(OnnxOutcome::ColdStart {
            normalized_query: normalize_query_slug(query),
        });
    };

    let out = tokio::time::timeout(
        Duration::from_secs(M2_INFER_TIMEOUT_SECS),
        Command::new(&st.cfg.onnx_python_bin)
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
            .arg("--mode")
            .arg(mode.flag())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "ONNX inference timeout setelah {M2_INFER_TIMEOUT_SECS} detik (proses dibunuh)"
        )
    })??;

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let count = err.chars().count();
        let tail: String = err.chars().skip(count.saturating_sub(300)).collect();
        anyhow::bail!("ONNX inference gagal: {tail}");
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
        mode: val["mode"]
            .as_str()
            .unwrap_or_else(|| mode.default_label())
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::OnnxMode;

    #[test]
    fn worker_flags_and_fallback_labels_are_stable() {
        assert_eq!(OnnxMode::Single.flag(), "single");
        assert_eq!(OnnxMode::Tiled.flag(), "tiled");
        assert_eq!(OnnxMode::Single.default_label(), "single_640");
        assert_eq!(
            OnnxMode::Tiled.default_label(),
            "tiled_2x2_640+point_nms"
        );
    }
}
