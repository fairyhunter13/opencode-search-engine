"""Tests for embeddings configuration and batch size settings."""

import os


def test_batch_size_configuration():
    """Verify that batch sizes are set based on hardware detection."""
    from opencode_embedder import embeddings

    # Verify hardware-adaptive, memory-efficient settings
    # Sub-batch sizes vary by hardware tier:
    # LOW_END:  ≤4 CPUs or ≤8 GB RAM  -> sub_batch=64
    # STANDARD: >4 CPUs and >8 GB RAM -> sub_batch=96
    # HIGH_END: ≥16 CPUs and ≥32 GB RAM -> sub_batch=128

    if embeddings._LOW_END:
        assert embeddings._EMBED_SUB_BATCH == 64
    elif embeddings._HIGH_END:
        assert embeddings._EMBED_SUB_BATCH == 128
    else:
        assert embeddings._EMBED_SUB_BATCH == 96

    # Verify max tokens is set
    assert embeddings._MAX_TOKENS == 1024


def test_batch_size_is_greater_than_one():
    """Verify ONNX batch size is appropriate for hardware."""
    from opencode_embedder import embeddings

    # Batch size is now auto-scaled based on GPU/CPU:
    # - CPU: 8 (optimal for cache utilization)
    # - GPU: 8-16 depending on VRAM (capped for concurrent request stability)
    batch_size = embeddings.get_onnx_batch_size()
    assert batch_size >= 8, f"ONNX batch size should be >= 8 for performance, got {batch_size}"
    assert batch_size <= 16, f"ONNX batch size should be <= 16 for OOM stability, got {batch_size}"


def test_embed_passages_returns_correct_count():
    """Test that embed_passages returns one vector per input text."""
    from opencode_embedder.embeddings import embed_passages

    # Use a small model for testing
    texts = ["hello world", "foo bar", "test text"]
    model = "jinaai/jina-embeddings-v2-small-en"
    dimensions = 512

    vectors = embed_passages(texts, model=model, dimensions=dimensions)

    assert len(vectors) == len(texts), f"Expected {len(texts)} vectors, got {len(vectors)}"
    for i, vec in enumerate(vectors):
        assert len(vec) == dimensions, f"Vector {i} has wrong dimension: {len(vec)} vs {dimensions}"
        assert all(isinstance(v, float) for v in vec), f"Vector {i} contains non-floats"


def test_embed_passages_empty_input():
    """Test that embed_passages handles empty input."""
    from opencode_embedder.embeddings import embed_passages

    vectors = embed_passages([], model="jinaai/jina-embeddings-v2-small-en", dimensions=512)
    assert vectors == []


def test_normalize_produces_unit_vectors():
    """Test that vectors are normalized to unit length."""
    import math
    from opencode_embedder.embeddings import embed_passages

    texts = ["test normalization"]
    model = "jinaai/jina-embeddings-v2-small-en"
    dimensions = 512

    vectors = embed_passages(texts, model=model, dimensions=dimensions)

    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.01, f"Vector not normalized: norm={norm}"


def test_ram_detection_is_positive():
    """Test that RAM detection returns a positive value."""
    from opencode_embedder import embeddings

    # _ram_mb should always be positive (detected or default 8GB)
    assert embeddings._ram_mb > 0, f"RAM should be positive, got {embeddings._ram_mb}"
    # Should be at least 1GB (sanity check)
    assert embeddings._ram_mb >= 1024, f"RAM should be >= 1GB, got {embeddings._ram_mb}MB"
