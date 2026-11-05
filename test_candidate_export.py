import gzip
import json

import pytest

from scripts.candidate_export import (
    build_candidate_record,
    observed_user_ids,
    write_candidate_records,
)
from scripts.evaluate_itemknn_50_60 import metric_at_k as metric_at_k_50_60
from scripts.evaluate_itemknn_60_70_replication import metric_at_k as metric_at_k_60_70


def test_candidate_record_keeps_unrated_items_unknown():
    row = build_candidate_record(
        7, [10, 11, 12], [0.8, 0.2, 0.0], {1, 2}, {10: 4.5, 11: 2.0}
    )
    assert row["observed_ratings"] == {"10": 4.5, "11": 2.0}
    assert row["unrated_candidate_outcome"] == "unknown; not a negative label"
    assert row["candidate_item_ids"] == [10, 11, 12]


@pytest.mark.parametrize(
    "items,scores,seen,ratings",
    [
        ([10, 10], [0.5, 0.2], set(), {}),
        ([10], [float("nan")], set(), {}),
        ([10], [0.5], {10}, {}),
        ([10], [0.5], set(), {10: 5.5}),
    ],
)
def test_candidate_record_rejects_invalid_rows(items, scores, seen, ratings):
    with pytest.raises(ValueError):
        build_candidate_record(7, items, scores, seen, ratings)


def test_write_candidate_records_is_gzip_jsonl_and_refuses_overwrite(tmp_path):
    path = tmp_path / "candidate_pairs.jsonl.gz"
    record = build_candidate_record(7, [10], [0.5], set(), {10: 4.0})
    assert write_candidate_records(path, [record]) == 1
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline()) == record
    with pytest.raises(FileExistsError):
        write_candidate_records(path, [record])


def test_observed_user_cohort_does_not_filter_on_rating_outcome():
    # The caller supplies every user observed in the held-out interval,
    # including users whose ratings are below the positive threshold.
    assert observed_user_ids([7, 8, 8, 9]) == {7, 8, 9}


def test_observed_user_cohort_rejects_invalid_ids():
    with pytest.raises(ValueError, match="positive MovieLens IDs"):
        observed_user_ids([7, 0])


@pytest.mark.parametrize("metric", [metric_at_k_50_60, metric_at_k_60_70])
def test_users_without_positive_targets_get_zero_recall_and_ndcg(metric):
    assert metric([10, 11, 12], set(), 10) == (0.0, 0.0)
