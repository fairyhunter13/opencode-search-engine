from opencode_embedder.tokenizer import count_tokens_for_tier, ensure_tokenizer_for_tier


def test_count_tokens_for_tier():
    text = "Hello, world! This is a test."

    assert count_tokens_for_tier(text, "premium") > 0
    assert count_tokens_for_tier(text, "balanced") > 0
    assert count_tokens_for_tier(text, "budget") > 0


def test_count_tokens_for_tier_empty():
    assert count_tokens_for_tier("", "premium") == 0


def test_ensure_tokenizer_for_tier_noop():
    ensure_tokenizer_for_tier("premium")
