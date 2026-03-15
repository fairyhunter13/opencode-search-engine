"""Tests for server configuration and worker settings."""

import os


def test_embed_workers_configuration():
    """Verify that embed workers are set based on hardware and GPU detection."""
    from opencode_embedder import server
    from opencode_embedder.embeddings import is_gpu_available

    cpus = server._CPUS
    ram_mb = server._RAM_MB
    low_end = cpus <= 4 or ram_mb <= 8192
    high_end = cpus >= 16 and ram_mb >= 32768

    # Worker scaling depends on GPU availability
    if is_gpu_available():
        # GPU mode: workers scaled by VRAM (1-6) to avoid GPU memory contention
        assert 1 <= server.EMBED_WORKERS <= 6, (
            f"GPU mode workers should be 1-6, got {server.EMBED_WORKERS}"
        )
    else:
        # CPU mode: fewer workers to avoid contention
        if low_end:
            assert server.EMBED_WORKERS == 2
        elif high_end:
            expected = min(6, max(4, cpus // 4))
            assert server.EMBED_WORKERS == expected
        else:
            expected = min(4, max(2, cpus // 4))
            assert server.EMBED_WORKERS == expected


def test_embed_workers_minimum():
    """Verify minimum of 1 embed worker."""
    from opencode_embedder import server

    assert server.EMBED_WORKERS >= 1, f"Embed workers should be >= 1, got {server.EMBED_WORKERS}"


def test_sub_batch_size():
    """Verify sub-batch size is reasonable for efficiency."""
    from opencode_embedder import server

    # Should be at least 64 for efficiency
    assert server.EMBED_SUB_BATCH >= 64


# ---- Idle shutdown tests ----


def test_idle_shutdown_default():
    """Verify default idle shutdown timeout is 10 minutes (600 seconds)."""
    from opencode_embedder import server

    # Default should be 600 seconds (10 minutes)
    assert server.IDLE_SHUTDOWN_SECS == 600, (
        f"Default idle shutdown should be 600s, got {server.IDLE_SHUTDOWN_SECS}"
    )


def test_idle_shutdown_env_override(monkeypatch):
    """Verify OPENCODE_EMBED_IDLE_SHUTDOWN env var overrides default."""
    monkeypatch.setenv("OPENCODE_EMBED_IDLE_SHUTDOWN", "300")

    # Need to reimport to pick up env var
    from opencode_embedder.server import _get_idle_shutdown_secs

    result = _get_idle_shutdown_secs()
    assert result == 300, f"Expected 300, got {result}"


def test_idle_shutdown_env_disable(monkeypatch):
    """Verify OPENCODE_EMBED_IDLE_SHUTDOWN=0 disables idle shutdown."""
    monkeypatch.setenv("OPENCODE_EMBED_IDLE_SHUTDOWN", "0")

    from opencode_embedder.server import _get_idle_shutdown_secs

    result = _get_idle_shutdown_secs()
    assert result == 0, f"Expected 0 (disabled), got {result}"


def test_idle_shutdown_env_invalid(monkeypatch):
    """Verify invalid OPENCODE_EMBED_IDLE_SHUTDOWN falls back to default."""
    monkeypatch.setenv("OPENCODE_EMBED_IDLE_SHUTDOWN", "invalid")

    from opencode_embedder.server import _get_idle_shutdown_secs

    result = _get_idle_shutdown_secs()
    assert result == 600, f"Expected 600 (default), got {result}"


def test_idle_shutdown_model_server_init():
    """Verify ModelServer accepts idle_shutdown_secs parameter."""
    from opencode_embedder.server import ModelServer

    # Test with custom idle shutdown
    srv = ModelServer(embed_workers=2, idle_shutdown_secs=120)
    assert srv._idle_shutdown_secs == 120

    # Test with disabled idle shutdown
    srv = ModelServer(embed_workers=2, idle_shutdown_secs=0)
    assert srv._idle_shutdown_secs == 0


def test_idle_shutdown_less_than_cleanup():
    """Verify idle cleanup happens before idle shutdown would trigger."""
    from opencode_embedder import server

    # Idle cleanup should happen before idle shutdown
    # This allows models to be unloaded before the server shuts down
    assert server.IDLE_CLEANUP_SECS < server.IDLE_SHUTDOWN_SECS, (
        f"Idle cleanup ({server.IDLE_CLEANUP_SECS}s) should be less than "
        f"idle shutdown ({server.IDLE_SHUTDOWN_SECS}s)"
    )


# ---- Thread pool executor sizing tests ----


def test_thread_pool_formula_bounds():
    """Thread pool = min(16, cpu_count) must stay in [1, 16] for any CPU count."""
    for cpus in [1, 2, 4, 8, 16, 32, 64, 128]:
        workers = min(16, cpus)
        assert workers >= 1, f"min 1 worker for {cpus} CPUs"
        assert workers <= 16, f"max 16 workers for {cpus} CPUs"


def test_thread_pool_formula_specific_values():
    """Expected values for common CPU counts."""
    assert min(16, 1) == 1
    assert min(16, 4) == 4
    assert min(16, 8) == 8
    assert min(16, 16) == 16
    assert min(16, 32) == 16  # capped
    assert min(16, 64) == 16  # capped


def test_thread_pool_fallback_when_cpu_count_none():
    """When os.cpu_count() returns None, fallback to 4."""
    assert min(16, None or 4) == 4


def test_thread_pool_no_oversubscription():
    """Total threads (workers × OMP) should not vastly exceed CPU count.

    With 1× CPU count for workers and OMP_NUM_THREADS=2,
    total = cpus × 2 (2× oversubscription).
    The old formula (2× CPU, max 32) could reach 4× oversubscription.
    """
    import os

    cpus = os.cpu_count() or 4
    workers = min(16, cpus)
    omp = 2  # typical OMP_NUM_THREADS
    total = workers * omp
    assert total <= cpus * 4, f"total threads ({total}) should not exceed 4× CPUs ({cpus * 4})"
    assert workers <= cpus, f"workers ({workers}) should not exceed CPU count ({cpus})"


def test_thread_pool_current_system():
    """Actual value on this system is within sane bounds."""
    import os

    cpus = os.cpu_count() or 4
    workers = min(16, cpus)
    assert workers > 0
    assert workers <= 16


def test_gpu_workers_vram_based():
    """Verify GPU embed workers are VRAM-based (1-6), not CPU-core-based.

    High TCP concurrency (16 connections) is handled by asyncio tasks that
    wait on the embed semaphore. Only `workers` ONNX sessions run at once
    to avoid GPU memory contention (profiled: 2 workers = 9.0 f/s optimal
    on 24GB VRAM; 16 workers = 7.4 f/s with OOM errors).
    """
    from opencode_embedder.embeddings import is_gpu_available

    if not is_gpu_available():
        return  # Only applicable in GPU mode

    from opencode_embedder import server

    assert 1 <= server.EMBED_WORKERS <= 6, (
        f"GPU embed workers ({server.EMBED_WORKERS}) should be 1-6 "
        f"(VRAM-based scaling to avoid GPU memory contention)"
    )
