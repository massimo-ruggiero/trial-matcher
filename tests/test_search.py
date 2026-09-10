from src.retrieve.search import rrf


def test_agreement_beats_a_single_top_hit():
    # b is third in both rankings, a is first in one and absent from the other.
    # With a large k the steady performer wins: that is the point of fusion.
    fused = dict(rrf([["a", "x", "b"], ["y", "z", "b"]], k=60))
    assert fused["b"] > fused["a"]


def test_small_k_loses_that_property():
    # Qdrant's server-side k=1 is steep enough that the same case ties exactly:
    # 1/2 for the single top hit against 1/4 + 1/4 for the agreed-on one. Ties
    # like this are why more than half of a server-side fused run shares scores.
    fused = dict(rrf([["a", "x", "b"], ["y", "z", "b"]], k=1))
    assert fused["a"] == fused["b"]


def test_score_is_the_sum_over_rankings():
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert fused["a"] == 2 / 61


def test_result_is_sorted_by_score():
    fused = rrf([["a", "b", "c"], ["a", "b", "c"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b", "c"]
