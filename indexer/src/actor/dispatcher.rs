//! Dispatcher actor - distributes work to workers.
//!
//! The dispatcher receives file change events and distributes them
//! to available workers using round-robin scheduling.
//! NO mutex, NO semaphore - pure message passing.

use std::collections::VecDeque;
use tokio::sync::mpsc;

use super::{DispatcherMsg, WorkerMsg, WorkerHandle, StorageHandle, ProjectContext, capacity};

/// Dispatcher actor state (owned, not shared)
struct DispatcherState {
    /// Worker handles (each worker has dedicated channel)
    workers: Vec<WorkerHandle>,
    /// Storage handle for workers to send writes
    storage: StorageHandle,
    /// Queue of ready worker IDs
    ready_workers: VecDeque<usize>,
    /// Pending work when all workers busy
    pending_work: VecDeque<PendingWork>,
}

#[derive(Debug)]
enum PendingWork {
    Index { path: std::path::PathBuf, ctx: ProjectContext },
    Delete { paths: Vec<String> },
}

impl DispatcherState {
    fn new(workers: Vec<WorkerHandle>, storage: StorageHandle) -> Self {
        let num_workers = workers.len();
        // Initially all workers are ready
        let ready_workers: VecDeque<usize> = (0..num_workers).collect();
        
        Self {
            workers,
            storage,
            ready_workers,
            pending_work: VecDeque::new(),
        }
    }
    
    /// Get next available worker (round-robin among ready workers)
    fn next_ready_worker(&mut self) -> Option<usize> {
        self.ready_workers.pop_front()
    }
    
    /// Mark worker as ready
    fn worker_ready(&mut self, worker_id: usize) {
        if !self.ready_workers.contains(&worker_id) {
            self.ready_workers.push_back(worker_id);
        }
        
        // If there's pending work, dispatch it
        self.dispatch_pending();
    }
    
    /// Dispatch pending work to ready workers
    fn dispatch_pending(&mut self) {
        while let Some(worker_id) = self.ready_workers.front().copied() {
            if let Some(work) = self.pending_work.pop_front() {
                self.ready_workers.pop_front();
                self.dispatch_to_worker(worker_id, work);
            } else {
                break;
            }
        }
    }
    
    /// Dispatch work to a specific worker
    fn dispatch_to_worker(&self, worker_id: usize, work: PendingWork) {
        let worker = &self.workers[worker_id];
        let storage = self.storage.clone();
        
        let msg = match work {
            PendingWork::Index { path, ctx } => WorkerMsg::ProcessFile {
                path,
                ctx,
                storage,
            },
            PendingWork::Delete { paths } => WorkerMsg::DeleteFiles {
                paths,
                storage,
            },
        };
        
        // Non-blocking send - worker mailbox should have capacity
        if let Err(e) = worker.try_send(msg) {
            tracing::warn!("failed to send to worker {}: {:?}", worker_id, e);
            // Put worker back in ready queue
            // (in practice this shouldn't happen with proper capacity)
        }
    }
    
    /// Handle incoming files changed message
    fn handle_files_changed(&mut self, paths: Vec<std::path::PathBuf>, ctx: ProjectContext) {
        for path in paths {
            let work = PendingWork::Index { path, ctx: ctx.clone() };
            
            if let Some(worker_id) = self.next_ready_worker() {
                self.dispatch_to_worker(worker_id, work);
            } else {
                // All workers busy, queue the work
                self.pending_work.push_back(work);
            }
        }
    }
    
    /// Handle incoming files deleted message
    fn handle_files_deleted(&mut self, paths: Vec<std::path::PathBuf>, ctx: ProjectContext) {
        // Convert to relative paths
        let rel_paths: Vec<String> = paths
            .iter()
            .filter_map(|p| {
                p.strip_prefix(&*ctx.root)
                    .ok()
                    .map(|r| r.to_string_lossy().to_string())
            })
            .collect();
        
        if rel_paths.is_empty() {
            return;
        }
        
        let work = PendingWork::Delete { paths: rel_paths };
        
        if let Some(worker_id) = self.next_ready_worker() {
            self.dispatch_to_worker(worker_id, work);
        } else {
            self.pending_work.push_back(work);
        }
    }
    
    /// Shutdown all workers
    async fn shutdown_workers(&self) {
        for worker in &self.workers {
            let _ = worker.send(WorkerMsg::Shutdown).await;
        }
    }
}

/// Run the dispatcher actor
pub async fn run(
    mut rx: mpsc::Receiver<DispatcherMsg>,
    workers: Vec<WorkerHandle>,
    storage: StorageHandle,
) {
    let mut state = DispatcherState::new(workers, storage);
    
    tracing::info!(
        "dispatcher started with {} workers, {} initially ready",
        state.workers.len(),
        state.ready_workers.len()
    );
    
    // Main message loop - blocks on recv() when idle (0% CPU)
    while let Some(msg) = rx.recv().await {
        match msg {
            DispatcherMsg::FilesChanged { paths, ctx } => {
                tracing::debug!(
                    "dispatcher received {} changed files for {}",
                    paths.len(),
                    ctx.project_key
                );
                state.handle_files_changed(paths, ctx);
            }
            
            DispatcherMsg::FilesDeleted { paths, ctx } => {
                tracing::debug!(
                    "dispatcher received {} deleted files for {}",
                    paths.len(),
                    ctx.project_key
                );
                state.handle_files_deleted(paths, ctx);
            }
            
            DispatcherMsg::WorkerReady { worker_id } => {
                tracing::trace!("worker {} ready", worker_id);
                state.worker_ready(worker_id);
            }
            
            DispatcherMsg::Shutdown => {
                tracing::info!("dispatcher shutting down");
                state.shutdown_workers().await;
                break;
            }
        }
    }
    
    tracing::info!(
        "dispatcher stopped, {} pending work items dropped",
        state.pending_work.len()
    );
}

/// Spawn the dispatcher actor
pub fn spawn(
    workers: Vec<WorkerHandle>,
    storage: StorageHandle,
) -> super::DispatcherHandle {
    let (tx, rx) = mpsc::channel(capacity::DISPATCHER);
    
    tokio::spawn(async move {
        run(rx, workers, storage).await;
    });
    
    super::ActorHandle::new(tx)
}
