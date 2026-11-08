use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub data_dir: PathBuf,
    pub passcode: String,
    pub repo_root: PathBuf,
    pub vlm_api_url: String,
    pub vlm_api_key: String,
    pub gemini_model: String,
    pub deepseek_model: String,
    pub ca_python_bin: PathBuf,
    pub ca_repo_dir: PathBuf,
    pub onnx_python_bin: PathBuf,
    pub yolo_base_weights: PathBuf,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind_str = env::var("ANYCOUNT_BIND").unwrap_or_else(|_| "127.0.0.1:2030".to_string());
        let bind_addr: SocketAddr = bind_str.parse()?;
        if !bind_addr.ip().is_loopback() {
            anyhow::bail!("ANYCOUNT_BIND must be a loopback address, got {bind_addr}");
        }
        Ok(Self {
            bind_addr,
            data_dir: PathBuf::from(
                env::var("ANYCOUNT_DATA_DIR")
                    .unwrap_or_else(|_| "/var/lib/anycount-prototype".to_string()),
            ),
            passcode: env::var("ANYCOUNT_PASSCODE").unwrap_or_else(|_| "2limanov".to_string()),
            repo_root: PathBuf::from(
                env::var("ANYCOUNT_REPO_ROOT")
                    .unwrap_or_else(|_| "/root/workspace/anycount".to_string()),
            ),
            vlm_api_url: env::var("VLM_API_URL")
                .unwrap_or_else(|_| "https://jr.rkhyg.xyz/v1/chat/completions".to_string()),
            vlm_api_key: env::var("VLM_API_KEY").unwrap_or_default(),
            gemini_model: env::var("GEMINI_MODEL")
                .unwrap_or_else(|_| "gemini-3.8-flash-high".to_string()),
            deepseek_model: env::var("DEEPSEEK_MODEL")
                .unwrap_or_else(|_| "deepseek-v4.1-flash-1017".to_string()),
            ca_python_bin: PathBuf::from(
                env::var("CA_PYTHON_BIN")
                    .unwrap_or_else(|_| "/root/workspace/count-anything/.venv/bin/python".to_string()),
            ),
            ca_repo_dir: PathBuf::from(
                env::var("CA_REPO_DIR")
                    .unwrap_or_else(|_| "/root/workspace/count-anything".to_string()),
            ),
            onnx_python_bin: PathBuf::from(
                env::var("ONNX_PYTHON_BIN")
                    .unwrap_or_else(|_| "/root/workspace/countthis/pilot/.venv/bin/python".to_string()),
            ),
            yolo_base_weights: PathBuf::from(
                env::var("YOLO_BASE_WEIGHTS")
                    .unwrap_or_else(|_| "/root/workspace/countthis/pilot/yolo11n.pt".to_string()),
            ),
        })
    }

    pub fn for_test(data_dir: &Path) -> Self {
        Self {
            bind_addr: "127.0.0.1:2030".parse().unwrap(),
            data_dir: data_dir.to_path_buf(),
            passcode: "2limanov".to_string(),
            repo_root: data_dir.join("repo"),
            vlm_api_url: "http://127.0.0.1:9/v1/chat/completions".to_string(),
            vlm_api_key: "sk-test-fixture".to_string(),
            gemini_model: "gemini-3.8-flash-high".to_string(),
            deepseek_model: "deepseek-v4.1-flash-1017".to_string(),
            ca_python_bin: PathBuf::from("python3"),
            ca_repo_dir: data_dir.join("ca"),
            onnx_python_bin: PathBuf::from("python3"),
            yolo_base_weights: data_dir.join("yolo11n.pt"),
        }
    }
}
