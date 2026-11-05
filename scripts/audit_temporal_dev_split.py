"""Audit an earlier global-time development split without fitting a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def find_timestamp_cutoff(counts: Counter[int], rank: int) -> int:
    """Return the first timestamp whose cumulative row count passes rank."""
    seen = 0
    for timestamp in sorted(counts):
        seen += counts[timestamp]
        if seen > rank:
            return int(timestamp)
    raise ValueError("Could not determine a temporal cutoff from the ratings.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path,
                        default=Path("data/ml-25m/ratings.csv"))
    parser.add_argument("--baseline-report", type=Path,
                        default=Path("reports/ml-25m/metrics.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/development/temporal-split-audit.json"))
    parser.add_argument("--positive-rating", type=float, default=4.0)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    old_train_end = int(baseline["split"]["train_before_timestamp"])
    old_test_start = int(baseline["split"]["test_from_timestamp"])
    old_train_rows = int(baseline["split"]["train"])

    timestamp_counts: Counter[int] = Counter()
    for frame in pd.read_csv(args.ratings, usecols=["timestamp"], chunksize=500_000):
        before = frame.loc[frame.timestamp < old_train_end, "timestamp"]
        timestamp_counts.update(before.value_counts().to_dict())

    counted_old_train = sum(timestamp_counts.values())
    if counted_old_train != old_train_rows:
        raise ValueError(
            f"Baseline says {old_train_rows:,} training rows but timestamp scan found "
            f"{counted_old_train:,}; refusing to proceed."
        )

    # The existing baseline's first 80% period is subdivided into an earlier
    # 70% train window and a following 10% development window. Timestamp ties
    # remain together, so realized percentages can differ slightly.
    dev_train_cutoff = find_timestamp_cutoff(
        timestamp_counts, int(old_train_rows * 0.875)
    )
    split = {
        name: {"rows": 0, "positive_rows": 0, "users": set(), "items": set(),
               "positive_users": set(), "positive_items": set()}
        for name in ("train", "validation", "previous_validation", "previous_test")
    }
    columns = ["userId", "movieId", "rating", "timestamp"]
    for frame in pd.read_csv(args.ratings, usecols=columns, chunksize=500_000):
        train = frame.timestamp < dev_train_cutoff
        validation = ((frame.timestamp >= dev_train_cutoff) &
                      (frame.timestamp < old_train_end))
        previous_validation = ((frame.timestamp >= old_train_end) &
                               (frame.timestamp < old_test_start))
        previous_test = frame.timestamp >= old_test_start
        for name, mask in (("train", train), ("validation", validation),
                           ("previous_validation", previous_validation),
                           ("previous_test", previous_test)):
            part = frame.loc[mask]
            entry = split[name]
            entry["rows"] += len(part)
            entry["positive_rows"] += int((part.rating >= args.positive_rating).sum())
            entry["users"].update(map(int, part.userId.unique()))
            entry["items"].update(map(int, part.movieId.unique()))
            positive = part.loc[part.rating >= args.positive_rating]
            entry["positive_users"].update(map(int, positive.userId.unique()))
            entry["positive_items"].update(map(int, positive.movieId.unique()))

    train_users = split["train"]["users"]
    validation_users = split["validation"]["users"]
    train_items = split["train"]["items"]
    validation_items = split["validation"]["items"]
    if split["previous_validation"]["rows"] != int(baseline["split"]["validation"]):
        raise ValueError("Recounted prior validation rows do not match the saved baseline report.")
    if split["previous_test"]["rows"] != int(baseline["split"]["test"]):
        raise ValueError("Recounted prior test rows do not match the saved baseline report.")
    result = {
        "purpose": "temporal split feasibility audit; no training or scoring",
        "dataset": "MovieLens 25M already present locally",
        "ratings_path": str(args.ratings),
        "historical_baseline_cutoff_timestamp": old_train_end,
        "historical_validation_end_timestamp": old_test_start,
        "new_train_cutoff_timestamp": dev_train_cutoff,
        "positive_rating_threshold": args.positive_rating,
        "timestamp_ties_kept_together": True,
        "split_definition": {
            "train": f"timestamp < {dev_train_cutoff}",
            "validation": f"{dev_train_cutoff} <= timestamp < {old_train_end}",
            "previous_validation": f"{old_train_end} <= timestamp < {old_test_start}; not used by this audit",
            "previous_test": f"timestamp >= {old_test_start}; not used by this audit",
        },
        "splits": {
            name: {
                "rows": value["rows"],
                "positive_rows": value["positive_rows"],
                "unique_users": len(value["users"]),
                "unique_items": len(value["items"]),
                "positive_users": len(value["positive_users"]),
                "positive_items": len(value["positive_items"]),
            }
            for name, value in split.items()
        },
        "validation_support": {
            "users_seen_in_train": len(validation_users & train_users),
            "users_absent_from_train": len(validation_users - train_users),
            "items_seen_in_train": len(validation_items & train_items),
            "validation_items_absent_from_train": len(validation_items - train_items),
            "positive_target_users": len(split["validation"]["positive_users"]),
            "positive_target_users_seen_in_train": len(
                split["validation"]["positive_users"] & train_users
            ),
            "positive_target_users_absent_from_train": len(
                split["validation"]["positive_users"] - train_users
            ),
            "positive_target_items": len(split["validation"]["positive_items"]),
            "positive_target_items_seen_in_train_catalog": len(
                split["validation"]["positive_items"] & train_items
            ),
            "positive_target_items_absent_from_train_catalog": len(
                split["validation"]["positive_items"] - train_items
            ),
        },
        "warnings": [
            "This validation window is temporally earlier than the old baseline validation window.",
            "Historical retrieval checkpoints trained through the old 80% cutoff and must not be scored here as if held out.",
            "The old 80-90% validation window was reused; the old final 10% already has a recorded baseline score. Neither is an untouched independent benchmark.",
            "MovieLens ratings are explicit ratings used as implicit preference proxies; they are not clicks or CTR.",
        ],
        "no_model_training": True,
        "previous_test_period_scored_by_this_audit": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
