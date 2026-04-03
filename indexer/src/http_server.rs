//! HTTP server for the indexer daemon.
//!
//! Exposes two endpoints:
//!   POST /rpc   — {"method": "...", "params": {...}} → {"result": ...}
//!   GET  /ping  — health check, returns "pong"
//!
//! The bound port is written to ~/.opencode/indexer.port so clients can
//! discover it without knowing the port in advance.

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde_json::Value;

use crate::daemon::Dispatcher;

/// Start the HTTP server on `127.0.0.1:{port}` (pass 0 for a random port).
///
/// Writes the actual bound port to `~/.opencode/indexer.port` then blocks
/// serving requests until the process exits.
pub async fn serve(dispatch: Dispatcher, port: u16) -> anyhow::Result<()> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual = listener.local_addr()?.port();

    write_port(actual).await;

    println!(
        "{}",
        serde_json::json!({"type": "http_ready", "port": actual})
    );

    let app = Router::new()
        .route("/rpc", post(handle_rpc))
        .route("/ping", get(ping))
        .with_state(dispatch);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn ping() -> &'static str {
    "pong"
}

async fn handle_rpc(State(dispatch): State<Dispatcher>, Json(body): Json<Value>) -> Json<Value> {
    let method = body["method"].as_str().unwrap_or("").to_string();
    let params = body.get("params").cloned().unwrap_or(Value::Null);
    let result = dispatch(method, params).await;
    Json(serde_json::json!({"result": result}))
}

async fn write_port(port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    let dir = home.join(".opencode");
    let _ = tokio::fs::create_dir_all(&dir).await;
    let _ = tokio::fs::write(dir.join("indexer.port"), port.to_string()).await;
}
