//! Worker actor - processes files for indexing.
//!
//! Each worker OWNS its model client (no sharing).
//! Workers block on recv() when idle (0% CPU).
//! NO mutex, NO semaphore - pure message passing.

use std::path::PathBuf;
use tokio::sync::mpsc;

use super::{WorkerMsg, DispatcherMsg, StorageMsg, ProjectContext, ChunkData, capacity};

/// Run a worker actor
pub async fn run(
    worker_id: usize,
    mut rx: mpsc::Receiver<WorkerMsg>,
    dispatcher_tx: mpsc::Sender<DispatcherMsg>,
) {
    tracing::info!("worker {} started", worker_id);
    
    // Worker OWNS its model client - no sharing, no mutex
    let mut model_client: Option<crate::model_client::PooledClient> = None;
    
    // Main message loop - blocks on recv() when idle (0% CPU)
    while let Some(msg) = rx.recv().await {
        match msg {
            WorkerMsg::ProcessFile { path, ctx, storage } => {
                tracing::debug!("worker {} processing {:?}", worker_id, path);
                
                // Ensure we have a model client
                if model_client.is_none() {
                    match crate::model_client::pooled().await {
                        Ok(client) => model_client = Some(client),
                        Err(e) => {
                            tracing::error!("worker {} failed to get model client: {}", worker_id, e);
                            // Notify dispatcher we're ready (even though we failed)
                            let _ = dispatcher_tx.send(DispatcherMsg::WorkerReady { worker_id }).await;
                            continue;
                        }
                    }
                }
                
                // Process the file
                let result = process_file(
                    worker_id,
                    &path,
                    &ctx,
                    model_client.as_mut().unwrap(),
                    &storage,
                ).await;
                
                match result {
                    Ok(indexed) => {
                        if indexed {
                            tracing::trace!("worker {} successfully indexed {:?}", worker_id, path);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("worker {} failed to process {:?}: {}", worker_id, path, e);
                    }
                }
                
                // Notify dispatcher we're ready for more work
                let _ = dispatcher_tx.send(DispatcherMsg::WorkerReady { worker_id }).await;
            }
            
            WorkerMsg::DeleteFiles { paths, storage } => {
                tracing::debug!("worker {} deleting {} files", worker_id, paths.len());
                
                // Send delete request to storage actor
                let _ = storage.send(StorageMsg::DeleteFiles {
                    project_key: String::new(), // Will be filled by caller
                    paths,
                }).await;
                
                // Notify dispatcher we're ready
                let _ = dispatcher_tx.send(DispatcherMsg::WorkerReady { worker_id }).await;
            }
            
            WorkerMsg::Shutdown => {
                tracing::info!("worker {} shutting down", worker_id);
                break;
            }
        }
    }
    
    tracing::info!("worker {} stopped", worker_id);
}

/// Process a single file with full logic from old processor task
async fn process_file(
    worker_id: usize,
    path: &PathBuf,
    ctx: &ProjectContext,
    client: &mut crate::model_client::PooledClient,
    storage: &super::StorageHandle,
) -> anyhow::Result<bool> {  // Returns true if file was indexed
    use anyhow::Context;
    
    // Get relative path
    let rel_path = path.strip_prefix(&*ctx.root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());
    
    // FAST PATH: Check if file needs indexing (hash comparison)
    // This is critical for CPU efficiency - skip unchanged files
    let file_hash = match tokio::task::spawn_blocking({
        let path = path.clone();
        move || crate::storage::hash_file(&path)
    }).await {
        Ok(Ok(hash)) => hash,
        Ok(Err(e)) => {
            tracing::debug!("worker {} failed to hash {:?}: {}", worker_id, path, e);
            return Ok(false);
        }
        Err(e) => {
            tracing::debug!("worker {} hash task panicked for {:?}: {}", worker_id, path, e);
            return Ok(false);
        }
    };
    
    // Check if file content changed (using hash)
    // TODO: Add needs_index check when storage is accessible
    // For now, always index (storage actor handles dedup)
    
    // Read file content
    let content = match tokio::fs::read_to_string(path).await {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("worker {} failed to read {:?}: {}", worker_id, path, e);
            return Ok(false);
        }
    };
    
    // Skip empty files
    if content.trim().is_empty() {
        return Ok(false);
    }
    
    // Chunk the content
    let chunks_result = client.chunk(&content, &rel_path, &ctx.tier).await
        .with_context(|| format!("failed to chunk {:?}", path))?;
    
    if chunks_result.is_empty() {
        return Ok(false);
    }
    
    // Embed chunks
    let texts: Vec<String> = chunks_result.iter().map(|c| c.content.clone()).collect();
    let embeddings = client.embed_passages(&texts, &ctx.tier, ctx.dimensions).await
        .with_context(|| format!("failed to embed {:?}", path))?;
    
    // Build chunk data with file hash
    let chunk_data: Vec<ChunkData> = chunks_result.iter()
        .zip(embeddings.iter())
        .map(|(chunk, embedding)| ChunkData {
            content: chunk.content.clone(),
            embedding: embedding.clone(),
            start_line: chunk.start_line as u32,
            end_line: chunk.end_line as u32,
            file_hash: file_hash.clone(),
        })
        .collect();
    
    let chunk_count = chunk_data.len();
    
    // Send to storage actor
    storage.send(StorageMsg::WriteChunks {
        project_key: ctx.project_key.clone(),
        path: rel_path.clone(),
        chunks: chunk_data,
    }).await.ok();
    
    tracing::debug!(
        "worker {} indexed {:?}: {} chunks",
        worker_id, rel_path, chunk_count
    );
    
    Ok(true)  // File was indexed
}

/// Spawn a worker actor
pub fn spawn(
    worker_id: usize,
    dispatcher_tx: mpsc::Sender<DispatcherMsg>,
) -> super::WorkerHandle {
    let (tx, rx) = mpsc::channel(capacity::WORKER);
    
    tokio::spawn(async move {
        run(worker_id, rx, dispatcher_tx).await;
    });
    
    super::ActorHandle::new(tx)
}

/// Spawn multiple workers
pub fn spawn_pool(
    num_workers: usize,
    dispatcher_tx: mpsc::Sender<DispatcherMsg>,
) -> Vec<super::WorkerHandle> {
    (0..num_workers)
        .map(|id| spawn(id, dispatcher_tx.clone()))
        .collect()
}
