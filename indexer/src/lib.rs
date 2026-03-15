//! OpenCode Indexer library
//!
//! This library exposes internal modules for testing purposes.
//! The main entry point is the `opencode-indexer` binary.

pub mod actor;
pub mod cleaner;
pub mod cli;
pub mod config;
pub mod daemon;
pub mod discover;
pub mod hardware;
pub mod http_server;
pub mod model_client;
pub mod storage;
pub mod watcher;
