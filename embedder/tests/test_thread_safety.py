"""Tests for thread safety fixes.

Tests cover:
- B4: Embedder Cache Thread Safety - Concurrent _embedder() calls
- M13: Resource Leak in _test_provider - Session cleanup
- Rerank empty scores handling
"""

import gc
import threading

import pytest

# Check if fastembed is available
try:
    import fastembed

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False


class TestEmbedderThreadSafety:
    """Tests for B4: Embedder Cache Thread Safety"""

    @pytest.mark.skipif(not HAS_FASTEMBED, reason="fastembed not installed")
    def test_embedder_cache_thread_safe(self):
        """Concurrent _embedder() calls should not cause race conditions"""
        from opencode_embedder.embeddings import _embedder

        results = []
        errors = []

        def get_embedder():
            try:
                emb = _embedder("jinaai/jina-embeddings-v2-small-en")
                results.append(id(emb))
            except Exception as e:
                errors.append(e)

        # Run 10 concurrent calls
        threads = [threading.Thread(target=get_embedder) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All should get same embedder instance (cached)
        assert len(set(results)) == 1, "Should return same cached instance"

    def test_ops_counter_thread_safe(self):
        """Counter increments should be thread-safe"""
        import opencode_embedder.embeddings as emb_module

        # Reset counter for test isolation
        with emb_module._ops_lock:
            initial_gpu = emb_module._gpu_ops_count
            initial_cpu = emb_module._cpu_ops_count
            # Set to known state
            emb_module._gpu_ops_count = 0
            emb_module._cpu_ops_count = 0

        def increment_many():
            for _ in range(100):
                emb_module._increment_gpu_ops()

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with emb_module._ops_lock:
            final_count = emb_module._gpu_ops_count
            # Should have exactly 1000 increments
            assert final_count == 1000, f"Expected 1000, got {final_count}"
            # Restore original counts
            emb_module._gpu_ops_count = initial_gpu + 1000
            emb_module._cpu_ops_count = initial_cpu


class TestResourceCleanup:
    """Tests for M13: Resource Leak in _test_provider"""

    def test_test_provider_has_cleanup_code(self):
        """_test_provider should have cleanup code in finally block"""
        import inspect

        from opencode_embedder.embeddings import _test_provider

        source = inspect.getsource(_test_provider)

        # Verify the function has proper cleanup
        assert "finally:" in source, "Should have finally block for cleanup"
        assert "del session" in source, "Should delete session in cleanup"
        assert "gc.collect()" in source, "Should call gc.collect() in cleanup"

    def test_test_provider_cleans_up_on_success(self):
        """_test_provider should clean up ONNX session even on success"""
        from opencode_embedder.embeddings import _test_provider

        # Just verify it completes without error
        # The cleanup is verified by inspecting the code structure
        result = _test_provider("CPUExecutionProvider")
        # Result can be True or False depending on environment
        assert isinstance(result, bool)

    def test_test_provider_cleans_up_on_failure(self):
        """_test_provider should clean up ONNX session even on failure"""
        from opencode_embedder.embeddings import _test_provider

        # Invalid provider should fail but cleanup properly
        result = _test_provider("InvalidProviderThatDoesNotExist")
        assert result is False

    def test_test_provider_handles_invalid_provider(self):
        """_test_provider should handle invalid providers gracefully"""
        from opencode_embedder.embeddings import _test_provider

        # Invalid provider should return False without crashing
        result = _test_provider("InvalidProviderThatDoesNotExist")
        assert result is False


class TestRerank:
    """Tests for empty scores handling"""

    def test_rerank_empty_docs(self):
        """rerank should handle empty docs gracefully"""
        from opencode_embedder.embeddings import rerank

        result = rerank("query", [], model="Xenova/ms-marco-MiniLM-L-6-v2", top_k=5)
        assert result == []

    def test_rerank_zero_top_k(self):
        """rerank should handle top_k=0 gracefully"""
        from opencode_embedder.embeddings import rerank

        result = rerank("query", ["doc1", "doc2"], model="Xenova/ms-marco-MiniLM-L-6-v2", top_k=0)
        assert result == []

    def test_rerank_has_empty_scores_check(self):
        """rerank should have check for empty scores to prevent ValueError"""
        import inspect

        from opencode_embedder.embeddings import rerank

        source = inspect.getsource(rerank)

        # Verify the function checks for empty scores before min/max
        assert "if not scores:" in source or "if len(scores)" in source, (
            "Should check for empty scores"
        )

    @pytest.mark.skipif(not HAS_FASTEMBED, reason="fastembed not installed")
    def test_rerank_returns_scores(self):
        """rerank should return normalized scores"""
        from opencode_embedder.embeddings import rerank

        docs = ["Python is a programming language", "Dogs are animals", "Cats are pets"]
        result = rerank("programming", docs, model="Xenova/ms-marco-MiniLM-L-6-v2", top_k=2)

        # Should return top 2 results
        assert len(result) <= 2
        # Each result is (index, score)
        for idx, score in result:
            assert 0 <= idx < len(docs)
            assert 0.0 <= score <= 1.0
