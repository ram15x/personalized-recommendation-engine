"""Compare fixed ranker fallback rules on an existing development export."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

from scripts.candidate_ranker_inference import CandidateRanker


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(order, relevant):
    hits = [position for position, item in enumerate(order[:10]) if item in relevant]
    if not relevant:
        return 0.0, 0.0
    recall = len(hits) / len(relevant)
    dcg = sum(1.0 / np.log2(position + 2) for position in hits)
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(10, len(relevant))))
    return float(recall), float(dcg / ideal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    ranker = CandidateRanker.load(args.ranker)
    rows = []
    with gzip.open(args.candidates, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            uid = int(row["user_id"])
            pos = int(np.searchsorted(ranker.users, uid))
            if pos == len(ranker.users) or ranker.users[pos] != uid:
                continue
            ids = np.asarray(row["candidate_item_ids"][:100], dtype=np.int64)
            values = np.asarray(row["retrieval_scores"][:100], dtype=np.float32)
            if len(ids) != 100 or values.shape != ids.shape or len(np.unique(ids)) != len(ids):
                raise ValueError("Expected 100 distinct aligned candidates for each selected user")
            relevant = {int(item) for item, rating in row["observed_ratings"].items() if float(rating) >= 4}
            rows.append((uid, ids, values, relevant))
    if len(rows) != 386:
        raise ValueError(f"Expected 386 known users, found {len(rows)}")
    outcomes = []
    for uid, ids, values, relevant in rows:
        # Preserve the original comparison cohort. Unsupported users receive
        # the unmodified retrieval order rather than untrained embeddings.
        scored = (ranker.score(uid, ids, values)
                  if uid in ranker.trained_users else [None] * len(ids))
        known_positions = [i for i, value in enumerate(scored) if value is not None]
        ranked_known = sorted(known_positions, key=lambda i: (-scored[i], int(ids[i])))
        partial = ids.copy()
        partial[known_positions] = ids[ranked_known]
        strict = partial if len(known_positions) == len(ids) else ids
        outcomes.append({"user_id": uid, "positive_count": len(relevant),
                         "known_items": len(known_positions),
                         "itemknn": metrics(ids, relevant),
                         "strict": metrics(strict, relevant),
                         "known_slots": metrics(partial, relevant)})
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(outcomes), size=(2000, len(outcomes)))
    result = {"protocol": "docs/fallback-evaluation-protocol.md", "users": len(outcomes),
              "candidate_export_sha256": digest(args.candidates),
              "ranker_manifest_sha256": digest(args.ranker / "artifact_manifest.json"),
              "zero_positive_users": sum(row["positive_count"] == 0 for row in outcomes),
              "strict_reranked_users": sum(row["known_items"] == 100 for row in outcomes),
              "known_slots_reranked_users": sum(row["known_items"] > 1 for row in outcomes),
              "metrics": {}, "paired_bootstrap": {}}
    for name in ("itemknn", "strict", "known_slots"):
        arr = np.asarray([row[name] for row in outcomes], dtype=np.float64)
        result["metrics"][name] = {"recall@10": float(arr[:, 0].mean()), "ndcg@10": float(arr[:, 1].mean())}
        if name != "itemknn":
            base = np.asarray([row["itemknn"] for row in outcomes], dtype=np.float64)
            for i, metric in enumerate(("recall@10", "ndcg@10")):
                draws = (arr[:, i] - base[:, i])[indices].mean(axis=1)
                result["paired_bootstrap"][name + "_minus_itemknn_" + metric] = {
                    "median": float(np.median(draws)),
                    "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
