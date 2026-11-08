use anycount_prototype::{config::Config, router, routes, state::AppState};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,anycount_prototype=debug".into()),
        )
        .init();

    let cfg = Config::from_env()?;
    let bind_addr = cfg.bind_addr;
    let state = AppState::init(cfg).await?;
    routes::samples::seed_research_samples(&state).await?;
    let app = router(state);

    let listener = TcpListener::bind(bind_addr).await?;
    tracing::info!(%bind_addr, "AnyCount Studio listening");
    axum::serve(listener, app).await?;
    Ok(())
}
