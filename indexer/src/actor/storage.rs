//! Storage actor - serializes database access.
//!
//! The storage actor OWNS all storage handles and serializes
//! all database writes. This eliminates concurrent access issues
//! without needing mutex locks.
//! 
//! Blocks on recv() when idle (0% CPU).

use std::collections::HashMap;
use tokio::sync::mpsc;

use super::{StorageMsg, ChunkData, capacity};

/// Pending writes buffer
struct WriteBuffer {
    /// Chunks waiting to be written, keyed by path
    pending: HashMap<String, Vec<ChunkData>>,
    /// Total chunks pending
    count: usize,
}

impl WriteBuffer {
    fn new() -> Self {
        Self {
            pending: HashMap::new(),
            count: 0,
        }
    }
    
    fn add(&mut self, path: String, chunks: Vec<ChunkData>) {
        self.count += chunks.len();
        self.pending.entry(path).or_default().extend(chunks);
    }
    
    fn take(&mut self) -> HashMap<String, Vec<ChunkData>> {
        self.count = 0;
        std::mem::take(&mut self.pending)
    }
    
    fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    fn len(&self) -> usize {
        self.count
    }
}

/// Storage actor state (owned, not shared)
struct StorageState {
    /// Write buffers per project
    buffers: HashMap<String, WriteBuffer>,
    /// Pending deletes per project
    deletes: HashMap<String, Vec<String>>,
    /// Flush threshold (number of chunks before auto-flush)
    flush_threshold: usize,
}

impl StorageState {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            deletes: HashMap::new(),
            flush_threshold: 100, // Flush every 100 chunks
        }
    }
    
    fn add_chunks(&mut self, project_key: &str, path: String, chunks: Vec<ChunkData>) {
        let buffer = self.buffers
            .entry(project_key.to_string())
            .or_insert_with(WriteBuffer::new);
        
        buffer.add(path, chunks);
        
        // Auto-flush if buffer is large
        if buffer.len() >= self.flush_threshold {
            tracing::debug!("auto-flushing {} chunks for {}", buffer.len(), project_key);
            // In a real implementation, this would write to storage
            // For now, just clear the buffer
            let _ = buffer.take();
        }
    }
    
    fn add_deletes(&mut self, project_key: &str, paths: Vec<String>) {
        self.deletes
            .entry(project_key.to_string())
            .or_default()
            .extend(paths);
    }
    
    fn flush(&mut self, project_key: &str) {
        if let Some(buffer) = self.buffers.get_mut(project_key) {
            if !buffer.is_empty() {
                tracing::info!("flushing {} chunks for {}", buffer.len(), project_key);
                let _ = buffer.take();
            }
        }
        
        if let Some(deletes) = self.deletes.remove(project_key) {
            if !deletes.is_empty() {
                tracing::info!("flushing {} deletes for {}", deletes.len(), project_key);
            }
        }
    }
    
    fn flush_all(&mut self) {
        let keys: Vec<String> = self.buffers.keys().cloned().collect();
        for key in keys {
            self.flush(&key);
        }
        
        let delete_keys: Vec<String> = self.deletes.keys().cloned().collect();
        for key in delete_keys {
            self.flush(&key);
        }
    }
}

/// Run the storage actor
pub async fn run(mut rx: mpsc::Receiver<StorageMsg>) {
    let mut state = StorageState::new();
    
    tracing::info!("storage actor started");
    
    // Main message loop - blocks on recv() when idle (0% CPU)
    while let Some(msg) = rx.recv().await {
        match msg {
            StorageMsg::WriteChunks { project_key, path, chunks } => {
                tracing::trace!(
                    "storage received {} chunks for {} in {}",
                    chunks.len(),
                    path,
                    project_key
                );
                state.add_chunks(&project_key, path, chunks);
            }
            
            StorageMsg::DeleteFiles { project_key, paths } => {
                tracing::trace!(
                    "storage received {} deletes for {}",
                    paths.len(),
                    project_key
                );
                state.add_deletes(&project_key, paths);
            }
            
            StorageMsg::Flush { project_key } => {
                tracing::debug!("storage flushing {}", project_key);
                state.flush(&project_key);
            }
            
            StorageMsg::Shutdown => {
                tracing::info!("storage actor shutting down, flushing all");
                state.flush_all();
                break;
            }
        }
    }
    
    tracing::info!("storage actor stopped");
}

/// Spawn the storage actor
pub fn spawn() -> super::StorageHandle {
    let (tx, rx) = mpsc::channel(capacity::STORAGE);
    
    tokio::spawn(async move {
        run(rx).await;
    });
    
    super::ActorHandle::new(tx)
}
