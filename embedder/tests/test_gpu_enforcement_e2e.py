"""E2E test: assert GPU is actually being used for inference.

Skipped automatically when OPENCODE_ONNX_PROVIDER=cpu (intentional CPU mode).
"""

import os

import pytest

GPU_PROVIDERS = {"tensorrt", "cuda", "migraphx", "rocm"}


def _cpu_mode() -> bool:
    return os.environ.get("OPENCODE_ONNX_PROVIDER", "").lower() == "cpu"


@pytest.mark.skipif(_cpu_mode(), reason="OPENCODE_ONNX_PROVIDER=cpu — intentional CPU mode")
def test_gpu_enforcement():
    """Health endpoint must report an active GPU provider."""
    from opencode_embedder.server import ModelServer

    health = ModelServer()._handle_health()
    gpu = health.get("gpu", {})
    provider = gpu.get("provider", "unknown")
    is_gpu = gpu.get("is_gpu", False)
    degraded = gpu.get("degraded", False)

    assert is_gpu, (
        f"GPU enforcement failed: is_gpu={is_gpu!r}  provider={provider!r}\n"
        f"  → GPU inference is required in production.\n"
        f"  → Set OPENCODE_ONNX_PROVIDER=cpu to skip this check intentionally.\n"
        f"  → Full gpu stats: {gpu}"
    )

    assert provider in GPU_PROVIDERS, (
        f"GPU enforcement failed: provider={provider!r} is not a recognised GPU provider.\n"
        f"  → Expected one of: {sorted(GPU_PROVIDERS)}\n"
        f"  → Full gpu stats: {gpu}"
    )

    assert not degraded, (
        f"GPU degraded: provider={provider!r} fell back to CPU (driver/library mismatch).\n"
        f"  → Check ONNX Runtime GPU shared libraries and CUDA/ROCm driver versions.\n"
        f"  → Full gpu stats: {gpu}"
    )


@pytest.mark.skipif(_cpu_mode(), reason="OPENCODE_ONNX_PROVIDER=cpu — intentional CPU mode")
def test_gpu_provider_is_consistent():
    """get_active_provider() and health endpoint must agree."""
    from opencode_embedder.embeddings import get_active_provider, is_gpu_available
    from opencode_embedder.server import ModelServer

    health = ModelServer()._handle_health()
    gpu = health.get("gpu", {})

    assert gpu.get("provider") == get_active_provider(), (
        f"Provider mismatch: health reports {gpu.get('provider')!r} "
        f"but get_active_provider() returns {get_active_provider()!r}"
    )

    assert gpu.get("is_gpu") == is_gpu_available(), (
        f"is_gpu mismatch: health reports {gpu.get('is_gpu')!r} "
        f"but is_gpu_available() returns {is_gpu_available()!r}"
    )


def test_cpu_mode_is_explicit():
    """When OPENCODE_ONNX_PROVIDER=cpu, server still reports provider correctly (not a surprise)."""
    if not _cpu_mode():
        pytest.skip("only runs in explicit CPU mode (OPENCODE_ONNX_PROVIDER=cpu)")

    from opencode_embedder.server import ModelServer

    health = ModelServer()._handle_health()
    gpu = health.get("gpu", {})

    assert gpu.get("provider") == "cpu", (
        f"Expected provider=cpu in CPU mode, got {gpu.get('provider')!r}"
    )
    assert gpu.get("is_gpu") is False, (
        f"Expected is_gpu=False in CPU mode, got {gpu.get('is_gpu')!r}"
    )
