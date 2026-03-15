use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{fmt, EnvFilter};

use opencode_indexer::cli;

// Limit thread pools to prevent excessive thread creation.
// LanceDB creates its own multi-threaded tokio runtime, so we need to cap all thread pools
// BEFORE any runtime starts (env vars only work if set before runtime initialization).
fn limit_thread_pools() {
    // SAFETY: Setting env vars is safe when done before spawning threads.
    // We call this at program start before any async runtime is created.
    unsafe {
        if std::env::var("TOKIO_WORKER_THREADS").is_err() {
            std::env::set_var("TOKIO_WORKER_THREADS", "4");
        }
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            std::env::set_var("RAYON_NUM_THREADS", "4");
        }
        if std::env::var("OMP_NUM_THREADS").is_err() {
            std::env::set_var("OMP_NUM_THREADS", "2");
        }
        if std::env::var("MKL_NUM_THREADS").is_err() {
            std::env::set_var("MKL_NUM_THREADS", "2");
        }
        if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
            std::env::set_var("OPENBLAS_NUM_THREADS", "2");
        }
    }
}

// Use synchronous main to set env vars BEFORE tokio runtime starts.
// This is critical because LanceDB creates its own runtime and reads TOKIO_WORKER_THREADS.
fn main() -> Result<()> {
    // Set thread limits FIRST, before ANY runtime is created
    limit_thread_pools();

    // Initialize logging
    fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("opencode_indexer=info".parse()?),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    // Create single-threaded runtime manually
    // Benefits: event-loop style, ~0% CPU when idle, simpler debugging
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Run the CLI
    let args = cli::Args::parse();
    rt.block_on(cli::run(args))
}
