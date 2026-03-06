"""Ranking utilities for memory retrieval."""

from collections import defaultdict


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int) -> dict[str, float]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
    fused: dict[str, float] = defaultdict(float)
    safe_k = max(1, int(k))
    for ranked_ids in rank_lists:
        for index, item_id in enumerate(ranked_ids, start=1):
            fused[item_id] += 1.0 / (safe_k + index)
    return dict(fused)

