//! Actor Model infrastructure for event-driven file indexing.
//!
//! This module implements a pure Actor Model where:
//! - Each actor has its own dedicated channel (mailbox)
//! - NO shared mutable state - actors own their state
//! - Communication ONLY via message passing
//! - Each actor processes messages sequentially
//!
//! This eliminates ALL mutex and semaphore patterns.

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

// Re-export actor types
pub mod dispatcher;
pub mod worker;
pub mod storage;

/// Channel capacity for actor mailboxes
pub mod capacity {
    pub const DISPATCHER: usize = 256;
    pub const WORKER: usize = 8;
    pub const STORAGE: usize = 512;
}

/// Project context passed with work items
#[derive(Debug, Clone)]
pub struct ProjectContext {
    pub project_key: String,
    pub root: Arc<PathBuf>,
    pub db_path: Arc<PathBuf>,
    pub include_dirs: Arc<Vec<PathBuf>>,
    pub tier: String,
    pub dimensions: u32,
}

/// Messages sent to the Dispatcher actor
#[derive(Debug)]
pub enum DispatcherMsg {
    /// Files changed, need indexing
    FilesChanged {
        paths: Vec<PathBuf>,
        ctx: ProjectContext,
    },
    /// Files deleted, need removal from index
    FilesDeleted {
        paths: Vec<PathBuf>,
        ctx: ProjectContext,
    },
    /// Worker is ready for more work
    WorkerReady { worker_id: usize },
    /// Shutdown the dispatcher
    Shutdown,
}

/// Messages sent to Worker actors
#[derive(Debug)]
pub enum WorkerMsg {
    /// Process a single file
    ProcessFile {
        path: PathBuf,
        ctx: ProjectContext,
        storage: StorageHandle,
    },
    /// Delete files from index
    DeleteFiles {
        paths: Vec<String>,
        storage: StorageHandle,
    },
    /// Shutdown the worker
    Shutdown,
}

/// Messages sent to the Storage actor
#[derive(Debug)]
pub enum StorageMsg {
    /// Write chunks to storage
    WriteChunks {
        project_key: String,
        path: String,
        chunks: Vec<ChunkData>,
    },
    /// Delete files from storage
    DeleteFiles {
        project_key: String,
        paths: Vec<String>,
    },
    /// Flush pending writes
    Flush { project_key: String },
    /// Shutdown the storage actor
    Shutdown,
}

/// Chunk data for storage
#[derive(Debug, Clone)]
pub struct ChunkData {
    pub content: String,
    pub embedding: Vec<f32>,
    pub start_line: u32,
    pub end_line: u32,
    pub file_hash: String,
}

/// Handle to send messages to an actor
pub struct ActorHandle<M> {
    tx: mpsc::Sender<M>,
}

impl<M> Clone for ActorHandle<M> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<M> std::fmt::Debug for ActorHandle<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorHandle")
            .field("tx", &"<mpsc::Sender>")
            .finish()
    }
}

impl<M> ActorHandle<M> {
    /// Create a new actor handle
    pub fn new(tx: mpsc::Sender<M>) -> Self {
        Self { tx }
    }

    /// Send a message to the actor (async, waits if mailbox full)
    pub async fn send(&self, msg: M) -> Result<(), mpsc::error::SendError<M>> {
        self.tx.send(msg).await
    }

    /// Try to send without waiting (returns error if mailbox full)
    pub fn try_send(&self, msg: M) -> Result<(), mpsc::error::TrySendError<M>> {
        self.tx.try_send(msg)
    }
}

/// Spawn an actor and return its handle
pub fn spawn_actor<M, F, Fut>(
    name: &'static str,
    capacity: usize,
    mut actor_fn: F,
) -> ActorHandle<M>
where
    M: Send + 'static,
    F: FnMut(mpsc::Receiver<M>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let (tx, rx) = mpsc::channel(capacity);
    
    tokio::spawn(async move {
        tracing::info!("actor '{}' started", name);
        actor_fn(rx).await;
        tracing::info!("actor '{}' stopped", name);
    });
    
    ActorHandle::new(tx)
}

/// Type alias for dispatcher handle
pub type DispatcherHandle = ActorHandle<DispatcherMsg>;

/// Type alias for worker handle
pub type WorkerHandle = ActorHandle<WorkerMsg>;

/// Type alias for storage handle
pub type StorageHandle = ActorHandle<StorageMsg>;
