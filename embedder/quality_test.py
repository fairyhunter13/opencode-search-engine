#!/usr/bin/env python3
"""Quality comparison: full tokens vs truncated embeddings.

Tests:
1. Cosine similarity between full vs truncated embeddings
2. Retrieval accuracy: does the right query still find the right code?
3. Different truncation lengths as quality/memory tradeoffs
"""

import gc
import numpy as np


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rss():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) / 1024
    return 0


# --- Prepare test data ---
# Realistic code chunks of varying sizes

CHUNK_SMALL = (
    "def fibonacci(n):\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
)

CHUNK_MEDIUM = (
    "class UserService:\n"
    "    def __init__(self, db):\n"
    "        self.db = db\n"
    "        self.cache = {}\n"
    "\n"
    "    def get_user(self, user_id: str):\n"
    "        if user_id in self.cache:\n"
    "            return self.cache[user_id]\n"
    "        user = self.db.query('SELECT * FROM users WHERE id = ?', user_id)\n"
    "        self.cache[user_id] = user\n"
    "        return user\n"
    "\n"
    "    def create_user(self, name: str, email: str):\n"
    "        user = {'name': name, 'email': email}\n"
    "        self.db.execute('INSERT INTO users (name, email) VALUES (?, ?)', name, email)\n"
    "        return user\n"
    "\n"
    "    def delete_user(self, user_id: str):\n"
    "        self.db.execute('DELETE FROM users WHERE id = ?', user_id)\n"
    "        if user_id in self.cache:\n"
    "            del self.cache[user_id]\n"
)

CHUNK_LARGE = (
    "import os\nimport sys\nfrom pathlib import Path\nfrom typing import List, Dict, Optional\n\n"
    "class FileIndexer:\n"
    "    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.go', '.rs', '.java'}\n\n"
    "    def __init__(self, root: Path, db_path: Path):\n"
    "        self.root = root\n"
    "        self.db_path = db_path\n"
    "        self.index = {}\n"
    "        self.file_hashes = {}\n\n"
    "    def scan_directory(self) -> List[Path]:\n"
    "        files = []\n"
    "        for ext in self.SUPPORTED_EXTENSIONS:\n"
    "            files.extend(self.root.rglob(f'*{ext}'))\n"
    "        return sorted(files)\n\n"
    "    def compute_hash(self, path: Path) -> str:\n"
    "        import hashlib\n"
    "        content = path.read_bytes()\n"
    "        return hashlib.sha256(content).hexdigest()\n\n"
    "    def needs_update(self, path: Path) -> bool:\n"
    "        current_hash = self.compute_hash(path)\n"
    "        stored_hash = self.file_hashes.get(str(path))\n"
    "        return current_hash != stored_hash\n\n"
    "    def index_file(self, path: Path) -> Dict:\n"
    "        content = path.read_text(encoding='utf-8', errors='replace')\n"
    "        lines = content.splitlines()\n"
    "        symbols = []\n"
    "        for i, line in enumerate(lines):\n"
    "            stripped = line.strip()\n"
    "            if stripped.startswith('def '):\n"
    "                name = stripped[4:stripped.index('(')]\n"
    "                symbols.append({'type': 'function', 'name': name, 'line': i+1})\n"
    "            elif stripped.startswith('class '):\n"
    "                name = stripped[6:].split('(')[0].split(':')[0].strip()\n"
    "                symbols.append({'type': 'class', 'name': name, 'line': i+1})\n"
    "        return {'path': str(path), 'content': content, 'symbols': symbols}\n\n"
    "    def build_index(self) -> int:\n"
    "        files = self.scan_directory()\n"
    "        count = 0\n"
    "        for path in files:\n"
    "            if self.needs_update(path):\n"
    "                entry = self.index_file(path)\n"
    "                self.index[str(path)] = entry\n"
    "                count += 1\n"
    "        return count\n\n"
    "    def search(self, query: str, limit: int = 10) -> List[Dict]:\n"
    "        results = []\n"
    "        query_lower = query.lower()\n"
    "        for path, entry in self.index.items():\n"
    "            score = 0\n"
    "            if query_lower in entry['content'].lower():\n"
    "                score += 1\n"
    "            for sym in entry['symbols']:\n"
    "                if query_lower in sym['name'].lower():\n"
    "                    score += 5\n"
    "            if score > 0:\n"
    "                results.append({'path': path, 'score': score})\n"
    "        results.sort(key=lambda x: -x['score'])\n"
    "        return results[:limit]\n"
) * 2  # ~4000 chars

CHUNK_XLARGE = (
    "class EmbeddingPipeline:\n"
    "    def __init__(self, model_name, batch_size=32):\n"
    "        self.model_name = model_name\n"
    "        self.batch_size = batch_size\n"
    "        self.embedder = None\n"
    "        self.cache = {}\n\n"
    "    def load_model(self):\n"
    "        from fastembed import TextEmbedding\n"
    "        self.embedder = TextEmbedding(model_name=self.model_name)\n\n"
    "    def embed_batch(self, texts):\n"
    "        results = []\n"
    "        for start in range(0, len(texts), self.batch_size):\n"
    "            batch = texts[start:start + self.batch_size]\n"
    "            prefixed = [f'passage: {t}' for t in batch]\n"
    "            for vec in self.embedder.embed(prefixed):\n"
    "                results.append(vec.tolist())\n"
    "        return results\n\n"
    "    def search(self, query, documents, top_k=5):\n"
    "        query_vec = list(self.embedder.embed([f'query: {query}']))[0]\n"
    "        doc_vecs = self.embed_batch(documents)\n"
    "        scores = []\n"
    "        for i, dv in enumerate(doc_vecs):\n"
    "            sim = sum(a*b for a, b in zip(query_vec, dv))\n"
    "            scores.append((i, sim))\n"
    "        scores.sort(key=lambda x: -x[1])\n"
    "        return scores[:top_k]\n"
) * 5  # ~8000 chars

CHUNK_HUGE = (
    "class DatabaseManager:\n"
    "    def __init__(self, connection_string):\n"
    "        self.connection_string = connection_string\n"
    "        self.pool = None\n"
    "        self.migrations = []\n\n"
    "    def connect(self):\n"
    "        import sqlite3\n"
    "        self.pool = sqlite3.connect(self.connection_string)\n"
    "        self.pool.execute('PRAGMA journal_mode=WAL')\n"
    "        self.pool.execute('PRAGMA synchronous=NORMAL')\n\n"
    "    def migrate(self):\n"
    "        cursor = self.pool.cursor()\n"
    "        cursor.execute('CREATE TABLE IF NOT EXISTS migrations '\n"
    "                       '(id INTEGER PRIMARY KEY, name TEXT NOT NULL)')\n"
    "        applied = set(row[0] for row in cursor.execute('SELECT name FROM migrations'))\n"
    "        for migration in self.migrations:\n"
    "            if migration.name not in applied:\n"
    "                migration.apply(cursor)\n"
    "                cursor.execute('INSERT INTO migrations (name) VALUES (?)', (migration.name,))\n"
    "        self.pool.commit()\n\n"
    "    def query(self, sql, params=None):\n"
    "        cursor = self.pool.cursor()\n"
    "        if params:\n"
    "            cursor.execute(sql, params)\n"
    "        else:\n"
    "            cursor.execute(sql)\n"
    "        return cursor.fetchall()\n\n"
    "    def execute(self, sql, params=None):\n"
    "        cursor = self.pool.cursor()\n"
    "        if params:\n"
    "            cursor.execute(sql, params)\n"
    "        else:\n"
    "            cursor.execute(sql)\n"
    "        self.pool.commit()\n"
    "        return cursor.rowcount\n"
) * 10  # ~16000 chars

code_chunks = [CHUNK_SMALL, CHUNK_MEDIUM, CHUNK_LARGE, CHUNK_XLARGE, CHUNK_HUGE]

# Queries that should match specific chunks
queries = [
    ("fibonacci recursive function", 0),
    ("user database query cache service", 1),
    ("file indexer search directory symbols", 2),
    ("embedding pipeline batch model", 3),
    ("database migration connection pool", 4),
]


def main():
    print("=" * 80)
    print("QUALITY COMPARISON: Full vs Truncated Embeddings")
    print("=" * 80)

    for i, chunk in enumerate(code_chunks):
        print(f"  Chunk {i}: {len(chunk):>6} chars")

    from fastembed import TextEmbedding

    truncation_lengths = [None, 8192, 2048, 1024, 512, 256]
    all_embeddings = {}

    for trunc in truncation_lengths:
        label = f"trunc={trunc}" if trunc else "full (8192 max)"
        print(f"\n--- {label} ---")

        embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

        if trunc is not None:
            embedder.model.tokenizer.enable_truncation(max_length=trunc)

        gc.collect()
        before = rss()

        # Embed all chunks
        chunk_texts = [f"passage: {c}" for c in code_chunks]
        chunk_vecs = [v.tolist() for v in embedder.embed(chunk_texts, batch_size=1)]

        # Embed all queries
        query_texts = [f"query: {q}" for q, _ in queries]
        query_vecs = [v.tolist() for v in embedder.embed(query_texts, batch_size=1)]

        gc.collect()
        after = rss()

        all_embeddings[trunc] = {"chunks": chunk_vecs, "queries": query_vecs}
        print(f"  Memory: +{after - before:.0f} MB workspace")

        # Retrieval test: for each query, rank chunks by cosine similarity
        print(f"  Retrieval results:")
        correct = 0
        for qi, (query_text, expected_idx) in enumerate(queries):
            qv = query_vecs[qi]
            scores = []
            for ci, cv in enumerate(chunk_vecs):
                sim = cosine_sim(qv, cv)
                scores.append((ci, sim))
            scores.sort(key=lambda x: -x[1])
            top_idx = scores[0][0]
            match = "OK" if top_idx == expected_idx else f"WRONG (got {top_idx})"
            if top_idx == expected_idx:
                correct += 1
            ranking = [f"c{ci}:{sim:.4f}" for ci, sim in scores]
            print(f"    Q: '{query_text}' -> top={top_idx} ({match})  [{', '.join(ranking)}]")

        print(f"  Accuracy: {correct}/{len(queries)} ({100*correct/len(queries):.0f}%)")

        del embedder
        gc.collect()

    # --- Compare embedding similarity between full and truncated ---
    print("\n" + "=" * 80)
    print("EMBEDDING SIMILARITY: Full vs Truncated")
    print("=" * 80)

    full_chunks = all_embeddings[None]["chunks"]
    full_queries = all_embeddings[None]["queries"]

    for trunc in truncation_lengths:
        if trunc is None:
            continue
        trunc_chunks = all_embeddings[trunc]["chunks"]
        trunc_queries = all_embeddings[trunc]["queries"]

        print(f"\n  trunc={trunc}:")
        for i in range(len(code_chunks)):
            sim = cosine_sim(full_chunks[i], trunc_chunks[i])
            chars = len(code_chunks[i])
            print(f"    Chunk {i} ({chars:>6} chars): cosine_sim(full, trunc) = {sim:.6f}")

        for i, (q, _) in enumerate(queries):
            sim = cosine_sim(full_queries[i], trunc_queries[i])
            print(f"    Query '{q}': cosine_sim = {sim:.6f}")


if __name__ == "__main__":
    main()
