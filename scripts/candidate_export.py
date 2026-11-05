"""Write candidate-level rows for a time-safe second-stage ranker."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def observed_user_ids(user_ids: Iterable[int]) -> set[int]:
    """Return users with any observed event, without inspecting outcomes."""
    users = {int(user_id) for user_id in user_ids}
    if any(user_id <= 0 for user_id in users):
        raise ValueError("Observed user IDs must be positive MovieLens IDs.")
    return users


def build_candidate_record(
    user_id: int,
    candidate_item_ids: Sequence[int],
    retrieval_scores: Sequence[float],
    seen_item_ids: set[int],
    observed_ratings: Mapping[int, float],
) -> dict:
    """Create an auditable candidate row without inventing labels for unrated items."""
    items = [int(item) for item in candidate_item_ids]
    scores = [float(score) for score in retrieval_scores]
    seen = {int(item) for item in seen_item_ids}
    ratings = {int(item): float(rating) for item, rating in observed_ratings.items()}

    if int(user_id) <= 0:
        raise ValueError("user_id must be a positive MovieLens ID.")
    if not items or len(items) != len(scores):
        raise ValueError("Candidate IDs and retrieval scores must be nonempty and aligned.")
    if any(item <= 0 for item in items) or len(set(items)) != len(items):
        raise ValueError("Candidate IDs must be unique positive MovieLens IDs.")
    if set(items) & seen:
        raise ValueError("Candidate list contains an item already present in user history.")
    if any(not math.isfinite(score) or score < 0 for score in scores):
        raise ValueError("Retrieval scores must be finite and non-negative.")
    if any(not math.isfinite(rating) or not 0.5 <= rating <= 5.0
           for rating in ratings.values()):
        raise ValueError("Observed MovieLens ratings must be finite and within [0.5, 5.0].")

    observed = {str(item): ratings[item] for item in items if item in ratings}
    return {
        "user_id": int(user_id),
        "candidate_item_ids": items,
        "retrieval_scores": scores,
        "observed_ratings": observed,
        "unrated_candidate_outcome": "unknown; not a negative label",
    }


def write_candidate_records(path: Path, records: Iterable[dict]) -> int:
    """Write deterministic gzip JSONL, refusing to overwrite an existing file."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for record in records:
                payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
                compressed.write((payload + "\n").encode("utf-8"))
                count += 1
    return count
