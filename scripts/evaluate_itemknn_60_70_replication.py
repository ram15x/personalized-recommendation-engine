"""Replicate fixed item-item cosine retrieval on the MovieLens 60/70 window."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
if __package__:
    from scripts.candidate_export import build_candidate_record, observed_user_ids, write_candidate_records
    from scripts.itemknn import build_top_neighbors, sample_evaluation_user_ids
else:  # Support both ``python -m scripts...`` and direct file invocation.
    from candidate_export import build_candidate_record, observed_user_ids, write_candidate_records
    from itemknn import build_top_neighbors, sample_evaluation_user_ids

EXPECTED_SHA256 = "9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385"
COLS = ["userId", "movieId", "rating", "timestamp"]
POSITIVE_RATING = 4.0
NEIGHBORS = 100
CHUNK_SIZE = 500_000
KNOWN_70_CUTOFF = 1_421_944_985
HISTORICAL_80_CUTOFF = 1_466_837_400
HISTORICAL_80_ROWS = 20_000_076


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_cutoff(timestamp_counts: Counter[int], rank: int) -> int:
    seen = 0
    for ts in sorted(timestamp_counts):
        seen += timestamp_counts[ts]
        if seen > rank:
            return int(ts)
    raise ValueError("Could not locate requested temporal boundary.")


def metric_at_k(items: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    hits = [i for i, item in enumerate(items[:k]) if int(item) in relevant]
    recall = len(hits) / len(relevant) if relevant else 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i in hits)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(recall), float(dcg / ideal if ideal else 0.0)


def paired_bootstrap(frame: pd.DataFrame, left: str, right: str) -> dict:
    delta = frame[left].to_numpy(np.float64) - frame[right].to_numpy(np.float64)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(delta), size=(2_000, len(delta)), dtype=np.int32)
    means = delta[draws].mean(axis=1)
    return {
        "left": left, "right": right,
        "median_delta": float(np.median(means)),
        "ci95_low": float(np.quantile(means, .025)),
        "ci95_high": float(np.quantile(means, .975)),
        "users": int(len(delta)), "resamples": 2_000, "seed": 42,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True,
                        help="Existing MovieLens 25M ratings.csv; no download is attempted.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, previously nonexistent output directory.")
    parser.add_argument("--export-candidate-lists", action="store_true",
                        help="Also save top-500 candidates and only their observed future ratings.")
    parser.add_argument("--max-users", type=int, default=2_000,
                        help="Uniform sample cap drawn from training-positive users; 0 evaluates all.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--neighbor-block-size", type=int, default=64,
                        help="Item rows per co-occurrence block; lower values reduce peak memory.")
    args = parser.parse_args()
    if args.max_users < 0:
        parser.error("--max-users must be zero or positive")
    if args.neighbor_block_size < 1:
        parser.error("--neighbor-block-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output}")
    if not args.ratings.is_file():
        raise FileNotFoundError(f"Existing ratings file required: {args.ratings}")
    ratings_hash = sha256(args.ratings)
    if ratings_hash != EXPECTED_SHA256:
        raise ValueError(f"Ratings hash mismatch: {ratings_hash}")

    old_train_end = HISTORICAL_80_CUTOFF
    old_train_rows = HISTORICAL_80_ROWS

    # Determine global 50% and 60% boundaries from the first 80% rows.
    timestamp_counts: Counter[int] = Counter()
    for frame in pd.read_csv(args.ratings, usecols=["timestamp"], chunksize=CHUNK_SIZE):
        before = frame.loc[frame.timestamp < old_train_end, "timestamp"]
        timestamp_counts.update(before.value_counts().to_dict())
    if sum(timestamp_counts.values()) != old_train_rows:
        raise ValueError("Historical training row count mismatch; refusing to score.")
    train_cutoff = find_cutoff(timestamp_counts, int(old_train_rows * .75))
    validation_end = find_cutoff(timestamp_counts, int(old_train_rows * .875))
    if validation_end != KNOWN_70_CUTOFF:
        raise ValueError(
            f"Computed 70% cutoff {validation_end} differs from audited {KNOWN_70_CUTOFF}."
        )

    positive_parts = []
    catalog: set[int] = set()
    popularity: Counter[int] = Counter()
    validation_users: set[int] = set()
    train_rows = validation_rows = 0
    for frame in pd.read_csv(args.ratings, usecols=COLS, chunksize=CHUNK_SIZE):
        train = frame.loc[frame.timestamp < train_cutoff]
        valid = frame.loc[(frame.timestamp >= train_cutoff) &
                          (frame.timestamp < validation_end)]
        train_rows += len(train)
        validation_rows += len(valid)
        catalog.update(map(int, train.movieId.unique()))
        pos = train.loc[train.rating >= POSITIVE_RATING, ["userId", "movieId"]]
        if not pos.empty:
            positive_parts.append(pos)
            popularity.update(map(int, pos.movieId))
        # Select users by observed feedback availability, not by future
        # positive outcome. Positive targets are handled separately below.
        validation_users.update(observed_user_ids(valid.userId.unique()))

    positives = pd.concat(positive_parts, ignore_index=True)
    if positives.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate positive training pairs; refusing to score.")
    user_ids = np.sort(positives.userId.unique().astype(np.int64))
    positive_item_ids = np.sort(positives.movieId.unique().astype(np.int64))
    user_idx = np.searchsorted(user_ids, positives.userId.to_numpy(np.int64)).astype(np.int32)
    item_idx = np.searchsorted(
        positive_item_ids, positives.movieId.to_numpy(np.int64)
    ).astype(np.int32)
    matrix = sparse.csr_matrix(
        (np.ones(len(user_idx), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(user_ids), len(positive_item_ids)), dtype=np.float32,
    )

    observed_validation_user_count = len(validation_users)
    sampled_training_users = sample_evaluation_user_ids(
        user_ids, validation_users, args.max_users, args.sample_seed
    )
    sampled_training_user_ids = set(map(int, sampled_training_users))
    cohort = sampled_training_user_ids
    histories: dict[int, set[int]] = defaultdict(set)
    targets: dict[int, set[int]] = defaultdict(set)
    observed_ratings: dict[int, dict[int, float]] = defaultdict(dict)
    for frame in pd.read_csv(args.ratings, usecols=COLS, chunksize=CHUNK_SIZE):
        selected = frame.loc[frame.userId.isin(cohort)]
        prior = selected.loc[selected.timestamp < train_cutoff]
        for user, items in prior.groupby("userId").movieId.agg(set).items():
            histories[int(user)].update(map(int, items))
        held_window = selected.loc[(selected.timestamp >= train_cutoff) &
                                   (selected.timestamp < validation_end)]
        for user, ratings in held_window.groupby("userId"):
            observed_ratings[int(user)].update({
                int(row.movieId): float(row.rating) for row in ratings.itertuples(index=False)
            })
        held = held_window.loc[held_window.rating >= POSITIVE_RATING]
        for user, items in held.groupby("userId").movieId.agg(set).items():
            targets[int(user)].update(map(int, items))

    similarity = build_top_neighbors(matrix, NEIGHBORS, args.neighbor_block_size)
    catalog_ids = np.asarray(sorted(catalog), dtype=np.int64)
    catalog_lookup = {int(item): i for i, item in enumerate(catalog_ids)}
    positive_catalog_positions = np.asarray(
        [catalog_lookup[int(item)] for item in positive_item_ids], dtype=np.int32
    )
    popularity_counts = np.asarray(
        [popularity.get(int(item), 0) for item in catalog_ids], dtype=np.int64
    )
    popularity_order = np.lexsort((catalog_ids, -popularity_counts))

    rows = []
    candidate_records = []
    warm_users = sorted(cohort & set(map(int, user_ids)))
    for user in warm_users:
        relevant = targets[user] - histories[user]
        user_row = int(np.searchsorted(user_ids, user))
        scores_sparse = matrix.getrow(user_row) @ similarity
        scores = np.zeros(len(catalog_ids), dtype=np.float32)
        if sparse.issparse(scores_sparse):
            scores[positive_catalog_positions] = scores_sparse.toarray().ravel()
        else:
            scores[positive_catalog_positions] = np.asarray(scores_sparse).ravel()
        for item in histories[user]:
            position = catalog_lookup.get(item)
            if position is not None:
                scores[position] = -np.inf
        order = np.lexsort((catalog_ids, -popularity_counts, -scores))
        knn = catalog_ids[order[:500]].astype(int).tolist()
        if args.export_candidate_lists:
            candidate_records.append(build_candidate_record(
                user, knn,
                [float(scores[catalog_lookup[item]]) for item in knn],
                histories[user], observed_ratings[user],
            ))
        pop = [int(catalog_ids[i]) for i in popularity_order
               if int(catalog_ids[i]) not in histories[user]][:500]
        target_catalog = relevant & catalog
        r10, n10 = metric_at_k(knn, relevant, 10)
        p10, pn10 = metric_at_k(pop, relevant, 10)
        r10m, n10m = metric_at_k(knn, target_catalog, 10)
        p10m, pn10m = metric_at_k(pop, target_catalog, 10)
        r100, _ = metric_at_k(knn, relevant, 100)
        p100, _ = metric_at_k(pop, relevant, 100)
        r100m, _ = metric_at_k(knn, target_catalog, 100)
        p100m, _ = metric_at_k(pop, target_catalog, 100)
        r500, _ = metric_at_k(knn, relevant, 500)
        rows.append({
            "userId": user,
            "eligible_targets": len(relevant),
            "targets_in_catalog": len(target_catalog),
            "itemknn_recall@10": r10, "itemknn_ndcg@10": n10,
            "popularity_recall@10": p10, "popularity_ndcg@10": pn10,
            "itemknn_matched_recall@10": r10m,
            "itemknn_matched_ndcg@10": n10m,
            "popularity_matched_recall@10": p10m,
            "popularity_matched_ndcg@10": pn10m,
            "itemknn_recall@100": r100, "popularity_recall@100": p100,
            "itemknn_matched_recall@100": r100m,
            "popularity_matched_recall@100": p100m,
            "itemknn_recall@500": r500,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != len(warm_users):
        raise ValueError("Warm-user support mismatch.")
    if frame.empty or not (frame.eligible_targets > 0).any():
        raise ValueError("No warm users have positive held-out targets; metrics are undefined.")
    summary = {
        "run": "ItemKNN cosine retrieval, MovieLens 60/70 temporal replication",
        "ratings_sha256": ratings_hash,
        "train_cutoff_timestamp": train_cutoff,
        "validation_end_timestamp": validation_end,
        "train_rows": train_rows, "validation_rows": validation_rows,
        "positive_rating_threshold": POSITIVE_RATING,
        "positive_training_pairs": int(len(positives)),
        "training_users": int(len(user_ids)),
        "positive_items_indexed": int(len(positive_item_ids)),
        "full_training_catalog_items": int(len(catalog_ids)),
        "item_neighbors_per_item": NEIGHBORS,
        "similarity": "cosine-normalized binary positive co-rating counts",
        "user_score": "sum of item-item similarities from positive training history",
        "all_validation_users_with_observed_ratings": observed_validation_user_count,
        "sampled_evaluation_warm_users": len(sampled_training_user_ids),
        "sampled_validation_users_with_observed_ratings": len(cohort),
        "evaluation_sample": {
            "max_evaluation_users": args.max_users,
            "seed": args.sample_seed,
            "selection": "uniform without replacement from training-positive users with any held-out observed rating; does not inspect rating values",
            "neighbor_block_size": args.neighbor_block_size,
        },
        "warm_users_with_observed_ratings": len(frame),
        "warm_users_with_positive_targets": int((frame.eligible_targets > 0).sum()),
        "warm_users_without_positive_targets": int((frame.eligible_targets == 0).sum()),
        "eligible_target_pairs": int(frame.eligible_targets.sum()),
        "target_pairs_in_catalog": int(frame.targets_in_catalog.sum()),
        "target_pair_catalog_coverage": float(
            frame.targets_in_catalog.sum() / frame.eligible_targets.sum()
        ) if frame.eligible_targets.sum() else 0.0,
        "metric_convention": {
            "positive_target_cohort": "macro user means among warm users with at least one unseen observed rating >= 4",
            "all_observed_warm_user_cohort": "macro user means over all warm users with any held-out rating; users without a >=4 target receive zero recall/NDCG",
            "unrated_candidate_outcomes": "unknown; excluded from labels",
        },
        "metrics": {
            col: float(frame.loc[frame.eligible_targets > 0, col].mean()) for col in [
                "itemknn_recall@10", "itemknn_ndcg@10",
                "popularity_recall@10", "popularity_ndcg@10",
                "itemknn_matched_recall@10", "itemknn_matched_ndcg@10",
                "popularity_matched_recall@10", "popularity_matched_ndcg@10",
                "itemknn_recall@100", "popularity_recall@100",
                "itemknn_matched_recall@100", "popularity_matched_recall@100",
                "itemknn_recall@500",
            ]
        },
        "metrics_all_observed_warm_users": {
            col: float(frame[col].mean()) for col in [
                "itemknn_recall@10", "itemknn_ndcg@10",
                "popularity_recall@10", "popularity_ndcg@10",
                "itemknn_matched_recall@10", "itemknn_matched_ndcg@10",
                "popularity_matched_recall@10", "popularity_matched_ndcg@10",
                "itemknn_recall@100", "popularity_recall@100",
                "itemknn_matched_recall@100", "popularity_matched_recall@100",
                "itemknn_recall@500",
            ]
        },
        "paired_user_bootstrap": {
            "matched_recall@100": paired_bootstrap(
                frame.loc[frame.eligible_targets > 0], "itemknn_matched_recall@100", "popularity_matched_recall@100"
            ),
            "matched_recall@10": paired_bootstrap(
                frame.loc[frame.eligible_targets > 0], "itemknn_matched_recall@10", "popularity_matched_recall@10"
            ),
            "matched_ndcg@10": paired_bootstrap(
                frame.loc[frame.eligible_targets > 0], "itemknn_matched_ndcg@10", "popularity_matched_ndcg@10"
            ),
        },
        "paired_user_bootstrap_all_observed_warm_users": {
            "matched_recall@100": paired_bootstrap(
                frame, "itemknn_matched_recall@100", "popularity_matched_recall@100"
            ),
            "matched_recall@10": paired_bootstrap(
                frame, "itemknn_matched_recall@10", "popularity_matched_recall@10"
            ),
            "matched_ndcg@10": paired_bootstrap(
                frame, "itemknn_matched_ndcg@10", "popularity_matched_ndcg@10"
            ),
        },
        "official_final_10_percent_used": False,
        "historical_80_90_window_used": False,
        "no_model_training": True,
        "limitations": [
            "This fixed method was selected after inspecting the 50/60 development result; this 60/70 temporal replication is not an independent final test.",
            "This evaluator selects all positive-history warm users with any observed validation rating; positive-target metrics are reported separately. The prior saved run used a future-positive-conditioned cohort and is unchanged.",
            "Rating >= 4 is a preference proxy, not clicks or CTR.",
            "Unrated items may be unknown preferences.",
            "This experiment evaluates candidate retrieval only; no second-stage ranker or online experiment is measured.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    if args.export_candidate_lists:
        candidate_path = args.output / "candidate_pairs.jsonl.gz"
        written = write_candidate_records(candidate_path, candidate_records)
        summary["candidate_pairs_export"] = {
            "file": candidate_path.name,
            "sha256": sha256(candidate_path),
            "rows": written,
            "candidates_per_row": 500,
            "observed_rating_target": "future explicit rating for candidate items only",
            "unrated_candidate_handling": "unknown; excluded from ranker loss, not a negative",
            "user_cohort": "positive-history warm users with any held-out observed rating; not filtered by future rating >= 4",
            "training_cutoff_timestamp": train_cutoff,
            "label_window_end_timestamp": validation_end,
        }
    similarity_path = args.output / "item_similarity_top100.npz"
    item_ids_path = args.output / "positive_item_ids.npy"
    sparse.save_npz(similarity_path, similarity)
    np.save(item_ids_path, positive_item_ids, allow_pickle=False)
    artifact_manifest = {
        "schema_version": 1,
        "ratings_sha256": ratings_hash,
        "train_cutoff_timestamp": train_cutoff,
        "positive_rating_threshold": POSITIVE_RATING,
        "neighbors_per_item": NEIGHBORS,
        "similarity_shape": list(similarity.shape),
        "similarity_nnz": int(similarity.nnz),
        "similarity_sha256": sha256(similarity_path),
        "item_ids_count": int(len(positive_item_ids)),
        "item_ids_sha256": sha256(item_ids_path),
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + chr(10), encoding="utf-8"
    )
    frame.to_csv(args.output / "per_user_metrics.csv", index=False)
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(summary, indent=2) + chr(10), encoding="utf-8")
    summary["metrics_json_sha256"] = sha256(report_path)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved new run:", args.output, flush=True)


if __name__ == "__main__":
    main()
