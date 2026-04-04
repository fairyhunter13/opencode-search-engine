# opencode-indexer

Rust daemon that watches the filesystem and coordinates semantic indexing via the Python embedder.

## Architecture: Event-Driven, NOT Polling

The indexer uses native OS filesystem events — it does **not** poll or scan directories on a timer.

| Platform | Mechanism               | Notes                                    |
| -------- | ----------------------- | ---------------------------------------- |
| Linux    | `inotify`               | Kernel-level file event notifications    |
| macOS    | `FSEvents`              | Core Services event stream               |
| Windows  | `ReadDirectoryChangesW` | Win32 API directory change notifications |

Implemented via the `notify` crate (`src/watcher.rs`).

### Why This Matters

- **~0% CPU when idle** — the kernel blocks until a filesystem event occurs
- No wakeups on quiet directories
- Sub-millisecond event latency when files change

### inotify Watch Limits (Linux)

The indexer manually walks directories and adds `NonRecursive` watches per directory
(see `add_watches` in `watcher.rs`). This avoids exhausting `/proc/sys/fs/inotify/max_user_watches`.

If you see `Failed to watch directory` errors:

```bash
# Check current limit
cat /proc/sys/fs/inotify/max_user_watches

# Increase temporarily
sudo sysctl fs.inotify.max_user_watches=524288

# Increase permanently
echo 'fs.inotify.max_user_watches=524288' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## CPU vs GPU Responsibilities

### Indexer (this binary) — CPU / I/O bound

All work is I/O-bound or lightweight CPU:

| Operation                           | Notes                                            |
| ----------------------------------- | ------------------------------------------------ |
| Filesystem watching                 | Kernel-level inotify/FSEvents — ~0 CPU when idle |
| File discovery / directory walking  | I/O bound                                        |
| File deduplication via content hash | CPU, fast (blake3)                               |
| LanceDB vector storage              | CPU + disk I/O                                   |
| HTTP client to embedder             | Network I/O                                      |
| Debouncing + rate limiting          | Timer-based, negligible CPU                      |

The indexer is intentionally CPU-conservative: thread pools are capped via env vars
(`TOKIO_WORKER_THREADS=4`, `RAYON_NUM_THREADS=4`) set before the async runtime starts.

### Embedder (Python sidecar) — GPU inference

All heavy compute is delegated to the Python embedder process:

| Operation                              | Hardware                                                           |
| -------------------------------------- | ------------------------------------------------------------------ |
| Text embedding (jina-embeddings-v2-\*) | GPU (ONNX Runtime: TensorRT/CUDA/MIGraphX)                         |
| Reranking (ms-marco-MiniLM)            | GPU (same ONNX provider chain)                                     |
| Semantic chunking (potion-base-32M)    | CPU (model2vec static model — intentional, see embedder/CLAUDE.md) |
| Tree-sitter AST chunking               | CPU (C library, no GPU path)                                       |

## Data Flow

```
Filesystem event (inotify/FSEvents)
    ↓
watcher.rs  →  debounce (500ms) + rate limit (2s)
    ↓
worker.rs   →  hash check (skip unchanged files)
    ↓
model_client.rs  →  HTTP POST /chunk  →  Python embedder (chunking)
    ↓
model_client.rs  →  HTTP POST /embed  →  Python embedder (GPU inference)
    ↓
storage.rs  →  LanceDB upsert (vectors + metadata)
```

## Storage (LanceDB)

- Vectors stored per-chunk with `chunk_id = "{path}#{chunk_index}"`
- Upsert logic: delete all chunks for a path, then insert new chunks atomically
- Prevents chunk_id collisions on restart (see mem-457)
- Index compaction runs after batch operations to keep search performance optimal
