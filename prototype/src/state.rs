use std::{fs, sync::Arc, time::Duration};

use dashmap::DashMap;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions},
    SqlitePool,
};
use tokio::sync::{broadcast, Semaphore};

use crate::config::Config;

#[derive(Clone)]
pub struct AppState {
    pub pool: SqlitePool,
    pub cfg: Arc<Config>,
    pub http: reqwest::Client,
    pub heavy_semaphore: Arc<Semaphore>,
    pub run_streams: Arc<DashMap<String, broadcast::Sender<String>>>,
}

impl AppState {
    pub async fn init(cfg: Config) -> anyhow::Result<Self> {
        fs::create_dir_all(cfg.data_dir.join("uploads"))?;
        fs::create_dir_all(cfg.data_dir.join("renders"))?;
        fs::create_dir_all(cfg.data_dir.join("packs"))?;
        fs::create_dir_all(cfg.data_dir.join("tmp"))?;

        let db_path = cfg.data_dir.join("anycount.db");
        let opts = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(10));

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await?;

        sqlx::migrate!("./migrations").run(&pool).await?;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .user_agent("anycount-prototype/0.1")
            .build()?;

        Ok(Self {
            pool,
            cfg: Arc::new(cfg),
            http,
            heavy_semaphore: Arc::new(Semaphore::new(1)),
            run_streams: Arc::new(DashMap::new()),
        })
    }
}
