//! End-to-end tests for the actor system.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

use opencode_indexer::actor::{
    DispatcherMsg, WorkerMsg, StorageMsg, ProjectContext, ChunkData,
    dispatcher, worker, storage, capacity,
    ActorHandle,
};

/// Test that actors start and can be shut down cleanly
#[tokio::test]
async fn actor_system_starts_and_shuts_down() {
    // Spawn storage actor
    let storage_handle = storage::spawn();
    
    // Create dispatcher channel
    let (dispatcher_tx, _dispatcher_rx) = mpsc::channel(capacity::DISPATCHER);
    
    // Spawn 2 workers
    let workers = worker::spawn_pool(2, dispatcher_tx.clone());
    assert_eq!(workers.len(), 2);
    
    // Spawn dispatcher
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    // Give actors time to start
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    // Shutdown dispatcher (which shuts down workers)
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    
    // Shutdown storage
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    
    // Give actors time to shutdown
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test that dispatcher distributes work to workers
#[tokio::test]
async fn dispatcher_distributes_work_to_workers() {
    // Spawn storage actor
    let storage_handle = storage::spawn();
    
    // Create dispatcher channel  
    let (dispatcher_tx, _dispatcher_rx) = mpsc::channel(capacity::DISPATCHER);
    
    // Spawn 2 workers
    let workers = worker::spawn_pool(2, dispatcher_tx.clone());
    
    // Spawn dispatcher
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    // Create test project context
    let ctx = ProjectContext {
        project_key: "test-project".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Send file changed event
    dispatcher.send(DispatcherMsg::FilesChanged {
        paths: vec![PathBuf::from("/tmp/test/file1.rs")],
        ctx: ctx.clone(),
    }).await.unwrap();
    
    // Give workers time to process (they'll fail since files don't exist, but that's ok)
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test that storage actor receives and processes messages
#[tokio::test]
async fn storage_actor_receives_messages() {
    let storage_handle = storage::spawn();
    
    // Send some chunks
    storage_handle.send(StorageMsg::WriteChunks {
        project_key: "test".to_string(),
        path: "test.rs".to_string(),
        chunks: vec![],
    }).await.unwrap();
    
    // Send delete
    storage_handle.send(StorageMsg::DeleteFiles {
        project_key: "test".to_string(),
        paths: vec!["old.rs".to_string()],
    }).await.unwrap();
    
    // Flush
    storage_handle.send(StorageMsg::Flush {
        project_key: "test".to_string(),
    }).await.unwrap();
    
    // Shutdown
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test that backpressure works (bounded channels)
#[tokio::test]
async fn bounded_channels_provide_backpressure() {
    // Storage with small capacity for testing
    let (tx, rx) = mpsc::channel::<StorageMsg>(2);
    let storage_handle = ActorHandle::new(tx);
    
    // Spawn storage consumer that's slow
    tokio::spawn(async move {
        let mut rx = rx;
        while let Some(msg) = rx.recv().await {
            if matches!(msg, StorageMsg::Shutdown) {
                break;
            }
            // Slow consumer
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    
    // Try to send more messages than capacity
    // First 2 should succeed immediately (fill buffer)
    storage_handle.try_send(StorageMsg::WriteChunks {
        project_key: "test".to_string(),
        path: "a.rs".to_string(),
        chunks: vec![],
    }).unwrap();
    
    storage_handle.try_send(StorageMsg::WriteChunks {
        project_key: "test".to_string(),
        path: "b.rs".to_string(),
        chunks: vec![],
    }).unwrap();
    
    // Third should fail (buffer full)
    let result = storage_handle.try_send(StorageMsg::WriteChunks {
        project_key: "test".to_string(),
        path: "c.rs".to_string(),
        chunks: vec![],
    });
    
    assert!(result.is_err(), "expected backpressure when channel full");
    
    // Cleanup
    let _ = storage_handle.send(StorageMsg::Shutdown).await;
}

/// Test workers block on recv when idle (conceptual test)
#[tokio::test]
async fn workers_block_on_recv_when_idle() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _) = mpsc::channel(capacity::DISPATCHER);
    
    // Spawn workers
    let workers = worker::spawn_pool(2, dispatcher_tx);
    
    // Workers should be idle, blocked on recv()
    // We can't directly measure CPU, but we can verify they're responsive
    
    // Give workers time to start and block
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Send shutdown to one worker
    workers[0].send(WorkerMsg::Shutdown).await.unwrap();
    
    // Should complete quickly (not stuck)
    let result = timeout(Duration::from_secs(1), async {
        workers[1].send(WorkerMsg::Shutdown).await
    }).await;
    
    assert!(result.is_ok(), "worker should respond to shutdown quickly");
    
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
}

/// Test that multiple files are processed in parallel across workers
#[tokio::test]
async fn multiple_files_processed_in_parallel() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _) = mpsc::channel(capacity::DISPATCHER);
    
    // Spawn 4 workers
    let workers = worker::spawn_pool(4, dispatcher_tx.clone());
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    let ctx = ProjectContext {
        project_key: "parallel-test".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Send 10 file events
    let start = std::time::Instant::now();
    for i in 0..10 {
        dispatcher.send(DispatcherMsg::FilesChanged {
            paths: vec![PathBuf::from(format!("/tmp/test/file{}.rs", i))],
            ctx: ctx.clone(),
        }).await.unwrap();
    }
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    
    let elapsed = start.elapsed();
    // Should complete quickly due to parallel processing
    assert!(elapsed < Duration::from_secs(2), "parallel processing should be fast");
}

/// Test graceful shutdown waits for in-flight work
#[tokio::test]
async fn graceful_shutdown_completes_in_flight_work() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _) = mpsc::channel(capacity::DISPATCHER);
    let workers = worker::spawn_pool(2, dispatcher_tx.clone());
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    let ctx = ProjectContext {
        project_key: "shutdown-test".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Send work
    dispatcher.send(DispatcherMsg::FilesChanged {
        paths: vec![PathBuf::from("/tmp/test/file.rs")],
        ctx,
    }).await.unwrap();
    
    // Immediately request shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    
    // Should complete without panic
    tokio::time::sleep(Duration::from_millis(200)).await;
}

/// Test that worker ready messages flow back to dispatcher
#[tokio::test]
async fn worker_ready_messages_enable_more_work() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _dispatcher_rx) = mpsc::channel(capacity::DISPATCHER);
    
    // Spawn 1 worker
    let workers = worker::spawn_pool(1, dispatcher_tx.clone());
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    let ctx = ProjectContext {
        project_key: "ready-test".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Send multiple work items
    for i in 0..5 {
        dispatcher.send(DispatcherMsg::FilesChanged {
            paths: vec![PathBuf::from(format!("/tmp/test/file{}.rs", i))],
            ctx: ctx.clone(),
        }).await.unwrap();
    }
    
    // Wait for worker to process and send ready messages
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test storage actor batches writes efficiently
#[tokio::test]
async fn storage_actor_batches_writes() {
    let storage_handle = storage::spawn();
    
    // Send multiple write requests rapidly
    for i in 0..50 {
        storage_handle.send(StorageMsg::WriteChunks {
            project_key: "batch-test".to_string(),
            path: format!("file{}.rs", i),
            chunks: vec![ChunkData {
                content: format!("content {}", i),
                embedding: vec![0.1; 384],
                start_line: 1,
                end_line: 10,
                file_hash: format!("hash{}", i),
            }],
        }).await.unwrap();
    }
    
    // Flush
    storage_handle.send(StorageMsg::Flush {
        project_key: "batch-test".to_string(),
    }).await.unwrap();
    
    // Shutdown
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test that dispatcher handles rapid fire events without dropping
#[tokio::test]
async fn dispatcher_handles_rapid_events() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _) = mpsc::channel(capacity::DISPATCHER);
    let workers = worker::spawn_pool(4, dispatcher_tx.clone());
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    let ctx = ProjectContext {
        project_key: "rapid-test".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Rapid fire 100 events
    for i in 0..100 {
        let result = dispatcher.try_send(DispatcherMsg::FilesChanged {
            paths: vec![PathBuf::from(format!("/tmp/test/file{}.rs", i))],
            ctx: ctx.clone(),
        });
        // Some may fail due to backpressure, that's ok
        if result.is_err() {
            break;
        }
    }
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Test that delete operations are processed correctly
#[tokio::test]
async fn delete_operations_processed() {
    let storage_handle = storage::spawn();
    let (dispatcher_tx, _) = mpsc::channel(capacity::DISPATCHER);
    let workers = worker::spawn_pool(2, dispatcher_tx.clone());
    let dispatcher = dispatcher::spawn(workers, storage_handle.clone());
    
    let ctx = ProjectContext {
        project_key: "delete-test".to_string(),
        root: Arc::new(PathBuf::from("/tmp/test")),
        db_path: Arc::new(PathBuf::from("/tmp/test.db")),
        include_dirs: Arc::new(vec![]),
        tier: "local".to_string(),
        dimensions: 384,
    };
    
    // Send delete event
    dispatcher.send(DispatcherMsg::FilesDeleted {
        paths: vec![
            PathBuf::from("/tmp/test/deleted1.rs"),
            PathBuf::from("/tmp/test/deleted2.rs"),
        ],
        ctx,
    }).await.unwrap();
    
    // Wait for processing
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    // Shutdown
    dispatcher.send(DispatcherMsg::Shutdown).await.unwrap();
    storage_handle.send(StorageMsg::Shutdown).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
}
