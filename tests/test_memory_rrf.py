"""Tests for memory RRF ranking helpers."""

from Panda_Dive.memory.ranking import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_combines_two_lists():
    """RRF should prefer items that rank high across channels."""
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ],
        k=60,
    )
    assert fused["b"] > fused["c"]
    assert fused["a"] > fused["d"]


def test_reciprocal_rank_fusion_single_list():
    """RRF should work with a single available ranking channel."""
    fused = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert fused["x"] > fused["y"]
    assert "z" not in fused

