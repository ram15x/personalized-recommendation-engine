import numpy as np
import pytest

from scripts.candidate_ranker_inference import CandidateRanker


def test_ranker_scores_known_items_and_rejects_unknown_user():
    def model(inputs, training=False):
        return inputs["item_id"].astype(np.float32) + inputs["numeric"].reshape(-1)

    ranker = CandidateRanker(model, [11, 22], [10, 20], [0.0], [1.0])
    scored = ranker.score(11, [20, 99, 10], [0.0, 1.0, 0.0])
    assert scored == [2.0, None, 1.0]
    with pytest.raises(ValueError, match="Unknown ranker user"):
        ranker.score(12, [10], [0.0])
    with pytest.raises(ValueError, match="aligned and unique"):
        ranker.score(11, [10, 10], [0.0, 1.0])


def test_ranker_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="normalization"):
        CandidateRanker(None, [11], [10], [0.0], [0.0])


def test_untrained_vocabulary_entries_do_not_use_random_embeddings():
    def model(inputs, training=False):
        return inputs["item_id"].astype(np.float32)

    ranker = CandidateRanker(model, [11, 22], [10, 20], [0.0], [1.0], [11], [10])
    assert ranker.score(11, [10, 20], [0.5, 0.5]) == [1.0, None]
    assert ranker.rank_or_fallback(11, [10, 20], [0.5, 0.5])["reason"] == "unknown_items"
    assert ranker.rank_or_fallback(22, [10], [0.5])["reason"] == "unknown_user"
    with pytest.raises(ValueError, match="Trained ID support"):
        CandidateRanker(model, [11], [10], [0.0], [1.0], [22], [10])


def test_complete_list_reranks_and_unknowns_preserve_retrieval_order():
    def model(inputs, training=False):
        return inputs["item_id"].astype(np.float32)

    ranker = CandidateRanker(model, [11], [10, 20], [0.0], [1.0])
    assert ranker.rank_or_fallback(11, [10, 20], [0.8, 0.3]) == {
        "item_ids": [20, 10], "strategy": "experimental_dcn", "reason": None,
    }
    assert ranker.rank_or_fallback(11, [10, 99], [0.8, 0.3])["reason"] == "unknown_items"
    result = ranker.rank_or_fallback(12, [10, 20], [0.8, 0.3])
    assert result["item_ids"] == [10, 20] and result["reason"] == "unknown_user"
    with pytest.raises(ValueError, match="aligned and unique"):
        ranker.rank_or_fallback(12, [10, 10], [0.8, 0.3])
