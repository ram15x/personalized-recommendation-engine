"""Train one no-download BPR comparison against the corrected two-tower run.

This script repeats the fixed BPR comparison on the preregistered 60/70 temporal window.
It uses only the existing MovieLens ratings CSV and a new output path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


TRAIN_CUTOFF = 1_288_571_552
VALIDATION_END = 1_421_944_985
POSITIVE_RATING = 4.0
SEED = 42
BATCH_SIZE = 512
EPOCHS = 3
DIMENSIONS = 32
CHUNK_SIZE = 500_000
EXPECTED_RATINGS_SHA256 = "9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385"
COLS = ["userId", "movieId", "rating", "timestamp"]


class BPRTwoTower(tfrs.Model):
    """Same ID towers as the comparator, trained with pairwise BPR loss."""

    def __init__(self, user_count: int, item_count: int):
        super().__init__()
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_count, DIMENSIONS),
            tf.keras.layers.Dense(DIMENSIONS, activation="relu"),
            tf.keras.layers.Dense(DIMENSIONS),
        ])
        self.item_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(item_count, DIMENSIONS),
            tf.keras.layers.Dense(DIMENSIONS, activation="relu"),
            tf.keras.layers.Dense(DIMENSIONS),
        ])

    def compute_loss(self, features, training=False):
        query = self.user_tower(features["user"])
        positive = self.item_tower(features["positive_item"])
        negative = self.item_tower(features["negative_item"])
        positive_score = tf.reduce_sum(query * positive, axis=1)
        negative_score = tf.reduce_sum(query * negative, axis=1)
        user_factors = self.user_tower.layers[0](features["user"])
        positive_factors = self.item_tower.layers[0](features["positive_item"])
        negative_factors = self.item_tower.layers[0](features["negative_item"])
        regularization = 1e-6 * (
            tf.reduce_sum(tf.square(user_factors), axis=1)
            + tf.reduce_sum(tf.square(positive_factors), axis=1)
            + tf.reduce_sum(tf.square(negative_factors), axis=1)
        )
        return tf.reduce_mean(tf.nn.softplus(negative_score - positive_score) + regularization)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_at_k(recommendations: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    hits = [rank for rank, item in enumerate(recommendations[:k]) if int(item) in relevant]
    recall = len(hits) / len(relevant)
    dcg = sum(1.0 / np.log2(rank + 2) for rank in hits)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return float(recall), float(dcg / ideal if ideal else 0.0)


def top_unseen(ordered_items, seen: set[int], limit: int,
               allowed: set[int] | None = None) -> list[int]:
    if limit <= 0:
        return []
    result = []
    for raw_item in ordered_items:
        item = int(raw_item)
        if item in seen or (allowed is not None and item not in allowed):
            continue
        result.append(item)
        if len(result) >= limit:
            break
    return result


def fixed_negative_samples(user_idx: np.ndarray, item_count: int,
                           sorted_seen_keys: np.ndarray, seed: int) -> np.ndarray:
    """Sample one catalog item per positive, rejecting all training-rated pairs."""
    rng = np.random.default_rng(seed)
    negatives = rng.integers(0, item_count, size=len(user_idx), dtype=np.int32)
    while True:
        keys = user_idx.astype(np.int64) * item_count + negatives
        positions = np.searchsorted(sorted_seen_keys, keys)
        collision = positions < len(sorted_seen_keys)
        valid = np.flatnonzero(collision)
        collision[valid] = sorted_seen_keys[positions[valid]] == keys[valid]
        bad = np.flatnonzero(collision)
        if bad.size == 0:
            return negatives
        negatives[bad] = rng.integers(0, item_count, size=bad.size, dtype=np.int32)


def build_data(ratings_path: Path):
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Existing ratings CSV required; no download attempted: {ratings_path}")
    ratings_hash = sha256(ratings_path)
    if ratings_hash != EXPECTED_RATINGS_SHA256:
        raise ValueError(f"Ratings SHA-256 mismatch: expected {EXPECTED_RATINGS_SHA256}, got {ratings_hash}")

    positive_parts = []
    popularity = Counter()
    catalog: set[int] = set()
    validation_users: set[int] = set()
    train_rows = validation_rows = 0
    for chunk in pd.read_csv(ratings_path, usecols=COLS, chunksize=CHUNK_SIZE):
        train = chunk.loc[chunk.timestamp < TRAIN_CUTOFF]
        valid = chunk.loc[(chunk.timestamp >= TRAIN_CUTOFF) &
                          (chunk.timestamp < VALIDATION_END)]
        train_rows += len(train)
        validation_rows += len(valid)
        catalog.update(map(int, train.movieId.unique()))
        positive = train.loc[train.rating >= POSITIVE_RATING, ["userId", "movieId"]]
        if not positive.empty:
            positive_parts.append(positive)
            popularity.update(map(int, positive.movieId))
        validation_users.update(map(int, valid.loc[
            valid.rating >= POSITIVE_RATING, "userId"].unique()))

    positives = pd.concat(positive_parts, ignore_index=True)
    if train_rows <= 0 or validation_rows <= 0:
        raise ValueError(f"Empty temporal split: train={train_rows}, validation={validation_rows}")
    if positives.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate positive training pairs; refusing to train.")

    user_ids = np.sort(positives.userId.unique().astype(np.int64))
    item_ids = np.sort(positives.movieId.unique().astype(np.int64))
    user_idx = np.searchsorted(user_ids, positives.userId.to_numpy(np.int64)).astype(np.int32)
    item_idx = np.searchsorted(item_ids, positives.movieId.to_numpy(np.int64)).astype(np.int32)
    user_lookup = {int(value): idx for idx, value in enumerate(user_ids)}

    cohort = validation_users
    histories = defaultdict(set)
    targets = defaultdict(set)
    rated_key_parts = []
    for chunk in pd.read_csv(ratings_path, usecols=COLS, chunksize=CHUNK_SIZE):
        train = chunk.loc[(chunk.timestamp < TRAIN_CUTOFF) &
                          chunk.userId.isin(user_ids) & chunk.movieId.isin(item_ids),
                          ["userId", "movieId"]]
        if not train.empty:
            u = np.searchsorted(user_ids, train.userId.to_numpy(np.int64)).astype(np.int64)
            i = np.searchsorted(item_ids, train.movieId.to_numpy(np.int64)).astype(np.int64)
            rated_key_parts.append(u * len(item_ids) + i)

        selected = chunk.loc[chunk.userId.isin(cohort)]
        prior = selected.loc[selected.timestamp < TRAIN_CUTOFF]
        for user, items in prior.groupby("userId").movieId.agg(set).items():
            histories[int(user)].update(map(int, items))
        held = selected.loc[(selected.timestamp >= TRAIN_CUTOFF) &
                            (selected.timestamp < VALIDATION_END) &
                            (selected.rating >= POSITIVE_RATING)]
        for user, items in held.groupby("userId").movieId.agg(set).items():
            targets[int(user)].update(map(int, items))

    seen_keys = np.unique(np.concatenate(rated_key_parts))
    return {
        "ratings_hash": ratings_hash, "train_rows": train_rows,
        "validation_rows": validation_rows, "positives": positives,
        "user_ids": user_ids, "item_ids": item_ids, "user_idx": user_idx,
        "item_idx": item_idx, "user_lookup": user_lookup, "catalog": catalog,
        "popularity": popularity, "histories": histories, "targets": targets,
        "validation_users": cohort, "seen_keys": seen_keys,
    }


def evaluate(model, data):
    user_ids, item_ids = data["user_ids"], data["item_ids"]
    user_vectors = model.user_tower(np.arange(len(user_ids), dtype=np.int32)).numpy().astype(np.float32)
    item_vectors = model.item_tower(np.arange(len(item_ids), dtype=np.int32)).numpy().astype(np.float32)
    if not np.isfinite(user_vectors).all() or not np.isfinite(item_vectors).all():
        raise ValueError("Non-finite model vectors; refusing to score or save.")

    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(item_vectors)
    valid_warm = sorted(set(data["validation_users"]) & set(data["user_lookup"]))
    query = np.stack([user_vectors[data["user_lookup"][user]] for user in valid_warm])
    _, positions = index.search(query, min(index.ntotal, 1000))
    popularity_ids = sorted(data["catalog"],
                            key=lambda item: (-data["popularity"].get(item, 0), item))
    indexed = set(map(int, item_ids))

    rows = []
    for row_index, user in enumerate(valid_warm):
        relevant = data["targets"][user] - data["histories"][user]
        if not relevant:
            continue
        retrieved = [int(item_ids[pos]) for pos in positions[row_index]
                     if pos >= 0 and int(item_ids[pos]) not in data["histories"][user]]
        pop_full = top_unseen(popularity_ids, data["histories"][user], 500)
        pop_matched = top_unseen(popularity_ids, data["histories"][user], 500,
                                 allowed=indexed)
        r10, n10 = metric_at_k(retrieved, relevant, 10)
        p10, pn10 = metric_at_k(pop_full, relevant, 10)
        r10_matched, n10_matched = metric_at_k(retrieved, relevant & indexed, 10)
        p10_matched, pn10_matched = metric_at_k(pop_matched, relevant & indexed, 10)
        r100, _ = metric_at_k(retrieved, relevant, 100)
        r500, _ = metric_at_k(retrieved, relevant, 500)
        mr100, _ = metric_at_k(retrieved, relevant & indexed, 100)
        mp100, _ = metric_at_k(pop_matched, relevant & indexed, 100)
        rows.append({
            "userId": user, "eligible_targets": len(relevant),
            "targets_in_index": len(relevant & indexed),
            "bpr_recall@10": r10, "bpr_ndcg@10": n10,
            "popularity_recall@10": p10, "popularity_ndcg@10": pn10,
            "bpr_indexed_target_recall@10": r10_matched,
            "bpr_indexed_target_ndcg@10": n10_matched,
            "popularity_indexed_target_recall@10": p10_matched,
            "popularity_indexed_target_ndcg@10": pn10_matched,
            "bpr_recall@100": r100, "bpr_recall@500": r500,
            "bpr_matched_recall@100": mr100,
            "popularity_matched_recall@100": mp100,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != len(valid_warm):
        raise ValueError("Validation warm-user support changed unexpectedly.")
    metrics = {
        "warm_users": len(frame),
        "eligible_target_pairs": int(frame.eligible_targets.sum()),
        "target_pairs_in_index": int(frame.targets_in_index.sum()),
        "target_pair_index_coverage": float(frame.targets_in_index.sum() /
                                              frame.eligible_targets.sum()),
        "bpr_recall@10": float(frame["bpr_recall@10"].mean()),
        "bpr_ndcg@10": float(frame["bpr_ndcg@10"].mean()),
        "popularity_recall@10": float(frame["popularity_recall@10"].mean()),
        "popularity_ndcg@10": float(frame["popularity_ndcg@10"].mean()),
        "bpr_candidate_recall@100": float(frame["bpr_recall@100"].mean()),
        "bpr_candidate_recall@500": float(frame["bpr_recall@500"].mean()),
        "bpr_indexed_target_recall@100": float(frame["bpr_matched_recall@100"].mean()),
        "popularity_indexed_target_recall@100": float(frame["popularity_matched_recall@100"].mean()),
    }
    return metrics, frame, user_vectors, item_vectors, index


def paired_bootstrap(frame: pd.DataFrame, left: str, right: str,
                     seed: int = 42, resamples: int = 2_000) -> dict:
    differences = (frame[left].to_numpy(np.float64)
                   - frame[right].to_numpy(np.float64))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(resamples, len(differences)),
                         dtype=np.int32)
    means = differences[draws].mean(axis=1)
    return {"left": left, "right": right,
            "median_delta": float(np.median(means)),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "users": int(len(differences)), "resamples": resamples, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True,
                        help="Existing MovieLens 25M ratings.csv; no download is attempted.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New, previously nonexistent output directory.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {args.output}")
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError as exc:
        # Colab may have initialized TensorFlow in an earlier notebook cell.
        print(f"TensorFlow thread pools already initialized; keeping runtime defaults: {exc}", flush=True)
    faiss.omp_set_num_threads(4)
    output = args.output

    data = build_data(args.ratings)
    model = BPRTwoTower(len(data["user_ids"]), len(data["item_ids"]))
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    losses = []
    for epoch in range(EPOCHS):
        negatives = fixed_negative_samples(data["user_idx"], len(data["item_ids"]),
                                           data["seen_keys"], SEED + epoch)
        features = {"user": data["user_idx"], "positive_item": data["item_idx"],
                    "negative_item": negatives}
        dataset = (tf.data.Dataset.from_tensor_slices(features)
                   .shuffle(100_000, seed=SEED + epoch, reshuffle_each_iteration=False)
                   .batch(BATCH_SIZE).prefetch(1))
        history = model.fit(dataset, epochs=epoch + 1, initial_epoch=epoch, verbose=2)
        losses.append(float(history.history["loss"][-1]))
    training_seconds = time.time() - start

    metrics, per_user, user_vectors, item_vectors, index = evaluate(model, data)
    report = {
        "run": "BPR two-tower temporal replication on the preregistered 60/70 window",
        "training_script_sha256": sha256(Path(__file__)),
        "ratings_sha256": data["ratings_hash"],
        "train_cutoff_timestamp": TRAIN_CUTOFF,
        "validation_end_timestamp": VALIDATION_END,
        "train_rows": data["train_rows"], "validation_rows": data["validation_rows"],
        "positive_rating_threshold": POSITIVE_RATING,
        "positive_training_pairs": int(len(data["positives"])),
        "training_users": int(len(data["user_ids"])),
        "indexed_items": int(len(data["item_ids"])),
        "full_training_catalog_items": int(len(data["catalog"])),
        "negative_sampling": "one uniform negative per positive per epoch (seeds 42, 43, 44); reject every item that user rated in training",
        "model": {"same_tower_architecture_as_corrected_run": True,
                  "dimensions": DIMENSIONS, "batch_size": BATCH_SIZE,
                  "epochs": EPOCHS, "optimizer": "Adagrad(lr=0.05)",
                  "objective": "mean softplus(score_negative - score_positive) plus 1e-6 L2 on input ID embeddings",
                  "loss_by_epoch": losses,
                  "training_seconds": training_seconds},
        "metrics": metrics,
        "paired_user_bootstrap_vs_popularity_same_indexed_targets": {
            "recall@100": paired_bootstrap(per_user, "bpr_matched_recall@100",
                                            "popularity_matched_recall@100"),
            "recall@10": paired_bootstrap(per_user, "bpr_indexed_target_recall@10",
                                           "popularity_indexed_target_recall@10"),
            "ndcg@10": paired_bootstrap(per_user, "bpr_indexed_target_ndcg@10",
                                          "popularity_indexed_target_ndcg@10"),
        },
        "official_final_10_percent_used": False,
        "historical_80_90_window_used": False,
        "development_only": True,
        "limitations": [
            "The BPR objective was selected after viewing the 70/80 development result; this 60/70 temporal replication is not an independent test.",
            "Rating >= 4 is a preference proxy, not click-through data.",
            "Unrated items sampled as negatives may be unknown rather than disliked.",
            "This changes the pairwise objective and negative sampling while retaining the tower architecture; a result does not establish generalization.",
        ],
    }
    np.savez_compressed(output / "embeddings.npz", user_ids=data["user_ids"],
                        item_ids=data["item_ids"], user_vectors=user_vectors,
                        item_vectors=item_vectors)
    faiss.write_index(index, str(output / "items.faiss"))
    model.user_tower.save(output / "user_tower.keras")
    model.item_tower.save(output / "item_tower.keras")
    per_user.to_csv(output / "per_user_metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("Saved new run:", output, flush=True)


if __name__ == "__main__":
    main()
