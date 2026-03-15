"""Tokenizer utilities (local-only).

The embedder runs fully locally, so we only need a lightweight, deterministic
token estimate for chunk sizing.

This module intentionally avoids network access and heavy model downloads.
"""

from __future__ import annotations


def count_tokens_for_tier(text: str, tier: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def ensure_tokenizer_for_tier(tier: str) -> None:
    # Kept for backwards compatibility with older code paths.
    return
