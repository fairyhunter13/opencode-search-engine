//! Integration tests for the Rust indexer with real Python model server.
//!
//! These tests spawn the real Python model server and test the full pipeline.

mod python_server;

use assert_cmd::prelude::*;
use predicates::str::contains;
use std::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_shows_files_and_chunks_count() {
    let server = python_server::PythonServer::start().await.expect("start Python server");

    let root = tempfile::TempDir::new().expect("temp root");
    
    // Create multiple files
    std::fs::write(root.path().join("file1.txt"), "content one\nline two\n").expect("write file1");
    std::fs::write(root.path().join("file2.txt"), "content two\nline two\n").expect("write file2");

    let db = root.path().join(".lancedb-status-test");
    let home = server.home_path().to_path_buf();
    let envs = server.env_vars();

    // Index both files
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home)
        .arg("--root")
        .arg(root.path())
        .arg("--db")
        .arg(&db)
        .arg("--tier")
        .arg("budget")
        .arg("--dimensions")
        .arg("256");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    let home_clone = home.clone();
    let db_clone = db.clone();
    let root_path = root.path().to_path_buf();
    tokio::task::spawn_blocking(move || {
        cmd.assert().success();
    })
    .await
    .unwrap();

    // Check status output includes Files: and Chunks: lines
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home_clone)
        .arg("--root")
        .arg(&root_path)
        .arg("--db")
        .arg(&db_clone)
        .arg("--dimensions")
        .arg("256")
        .arg("--status");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("Files: 2"))
            .stdout(contains("Chunks:"))
            .stdout(contains("Tier: budget"))
            .stdout(contains("Dimensions: 256"))
            .stdout(contains("Status: idle"));
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn indexes_single_file_and_searches_in_tiny_project() {
    let server = python_server::PythonServer::start().await.expect("start Python server");

    let root = tempfile::TempDir::new().expect("temp root");
    let file = root.path().join("hello.txt");
    std::fs::write(
        &file,
        "hello world\nthis is tiny\njust a test\n",
    )
    .expect("write file");

    let db = root.path().join(".lancedb-test");
    let home = server.home_path().to_path_buf();
    let envs = server.env_vars();

    // 1) First run indexes and embeds
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home)
        .arg("--root")
        .arg(root.path())
        .arg("--db")
        .arg(&db)
        .arg("--tier")
        .arg("budget")
        .arg("--dimensions")
        .arg("256")
        .arg("--file")
        .arg(&file)
        .arg("--verbose");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("Indexed:"));
    })
    .await
    .unwrap();

    // 2) Second run detects unchanged and does not re-embed
    let home_clone = home.clone();
    let db_clone = db.clone();
    let file_clone = file.clone();
    let root_path = root.path().to_path_buf();
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home_clone)
        .arg("--root")
        .arg(&root_path)
        .arg("--db")
        .arg(&db_clone)
        .arg("--tier")
        .arg("budget")
        .arg("--dimensions")
        .arg("256")
        .arg("--file")
        .arg(&file_clone)
        .arg("--verbose");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("Indexed:"));
    })
    .await
    .unwrap();

    // 3) Search uses embed_query + rerank through the model server
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home)
        .arg("--root")
        .arg(root.path())
        .arg("--db")
        .arg(&db)
        .arg("--tier")
        .arg("budget")
        .arg("--dimensions")
        .arg("256")
        .arg("--search")
        .arg("hello");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("Search results for: hello"))
            .stdout(contains("hello.txt"));
    })
    .await
    .unwrap();
}

/// Test that async WriteQueue correctly handles multiple files with concurrent indexing.
/// This verifies that the non-blocking write queue doesn't lose data and properly
/// persists all files to storage.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn async_write_queue_indexes_multiple_files_concurrently() {
    let server = python_server::PythonServer::start().await.expect("start Python server");

    let root = tempfile::TempDir::new().expect("temp root");
    
    // Create many files to stress the async write queue
    for i in 0..20 {
        let content = format!(
            "file {} content\nline two of file {}\nmore text here {}\n",
            i, i, i
        );
        std::fs::write(root.path().join(format!("file{}.txt", i)), content).expect("write file");
    }

    let db = root.path().join(".lancedb-async-queue-test");
    let home = server.home_path().to_path_buf();
    let envs = server.env_vars();

    // Index all files with verbose output to see progress
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home)
        .arg("--root")
        .arg(root.path())
        .arg("--db")
        .arg(&db)
        .arg("--tier")
        .arg("budget")
        .arg("--dimensions")
        .arg("256")
        .arg("--verbose");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    let home_clone = home.clone();
    let db_clone = db.clone();
    let root_path = root.path().to_path_buf();
    
    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("20 files processed"));
    })
    .await
    .unwrap();

    // Verify status shows all 20 files were indexed
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("opencode-indexer"));
    cmd.env("HOME", &home_clone)
        .arg("--root")
        .arg(&root_path)
        .arg("--db")
        .arg(&db_clone)
        .arg("--dimensions")
        .arg("256")
        .arg("--status");

    for (k, v) in &envs {
        cmd.env(k, v);
    }

    tokio::task::spawn_blocking(move || {
        cmd.assert()
            .success()
            .stdout(contains("Files: 20"))
            .stdout(contains("Status: idle"));
    })
    .await
    .unwrap();
}
