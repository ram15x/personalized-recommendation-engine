import math

import pandas as pd
import pytest

from baseline import evaluate, popularity_order, ranking_scores, split_by_time


def test_metrics_match_hand_calculation():
    recall, ndcg = ranking_scores([10, 20, 30], {20, 40}, 3)
    assert recall == 0.5
    assert ndcg == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_time_boundaries_do_not_overlap_when_timestamps_repeat():
    frame = pd.DataFrame({'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11]})
    train, val, test, _, _ = split_by_time(frame)
    assert train.timestamp.max() < val.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    assert len(train) + len(val) + len(test) == len(frame)


def test_unseen_items_count_as_misses_and_seen_movies_are_excluded():
    history = pd.DataFrame({'userId': [1, 2, 2], 'movieId': [10, 20, 30], 'rating': [5, 5, 2]})
    heldout = pd.DataFrame({'userId': [1, 1, 3], 'movieId': [20, 99, 20], 'rating': [5, 5, 5]})
    ranked = popularity_order(history).movieId.tolist()
    metrics, rows = evaluate(ranked, history, heldout, k=2)
    assert rows[rows.userId == 1].iloc[0].recommendations == '20 30'
    assert rows[rows.userId == 1].iloc[0].recall == 0.5
    assert metrics['cold_users']['users'] == 1
    assert metrics['unseen_catalog_positive_fraction'] == pytest.approx(1 / 3)


def test_no_positive_targets_is_an_explicit_error():
    data = pd.DataFrame({'userId': [1], 'movieId': [1], 'rating': [1]})
    with pytest.raises(ValueError, match='No users'):
        evaluate([1], data, data)
