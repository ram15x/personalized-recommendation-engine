"""Fit the existing DCN scorer with a pairwise candidate-ranking objective.

Reads only saved ItemKNN candidate exports, never raw ratings. Training pairs
compare observed >=4-star candidate ratings with observed <4-star ratings;
unrated candidates remain unknown and are never sampled as negatives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.dcn_ranker import DCNRatingRanker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric_at_k(ranked: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    hits = [i for i, item in enumerate(ranked[:k]) if item in relevant]
    recall = len(hits) / len(relevant)
    dcg = sum(1.0 / np.log2(pos + 2) for pos in hits)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(k, len(relevant))))
    return float(recall), float(dcg / ideal if ideal else 0.0)


def paired_bootstrap(users: list[dict], metric: str, left: str, right: str) -> dict:
    delta = np.asarray([row[left][metric] - row[right][metric] for row in users], dtype=np.float64)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(delta), size=(2_000, len(delta)), dtype=np.int32)
    means = delta[draws].mean(axis=1)
    return {
        "median_difference": float(np.median(means)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "users": len(delta), "resamples": 2_000, "seed": 42,
    }


class PairwiseScorer(tf.keras.Model):
    """Shared DCN scores for a positive and lower-rated item of one user."""

    def __init__(self, scorer: DCNRatingRanker, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scorer = scorer

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        user_id = inputs["user_id"]
        positive = {
            "user_id": user_id, "item_id": inputs["positive_item_id"],
            "numeric": inputs["positive_numeric"],
        }
        negative = {
            "user_id": user_id, "item_id": inputs["negative_item_id"],
            "numeric": inputs["negative_numeric"],
        }
        return self.scorer(positive, training=training) - self.scorer(negative, training=training)


def make_pairs(rows: list[dict], user_to_idx: dict[int, int], item_to_idx: dict[int, int],
               score_mean: float, score_std: float, cap_per_user: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    users, positive_items, negative_items, positive_scores, negative_scores = [], [], [], [], []
    eligible_users = possible_pairs = 0
    positive_observations = negative_observations = 0
    for row in rows:
        uid = int(row["user_id"])
        item_ids = np.asarray(row["candidate_item_ids"], dtype=np.int64)
        raw_scores = np.asarray(row["retrieval_scores"], dtype=np.float32)
        score_by_item = {int(item): float(score) for item, score in zip(item_ids, raw_scores)}
        ratings = {int(item): float(rating) for item, rating in row["observed_ratings"].items()}
        positive_ids = np.asarray(sorted(item for item, rating in ratings.items() if rating >= 4.0), dtype=np.int64)
        negative_ids = np.asarray(sorted(item for item, rating in ratings.items() if rating < 4.0), dtype=np.int64)
        positive_observations += len(positive_ids)
        negative_observations += len(negative_ids)
        if not len(positive_ids) or not len(negative_ids):
            continue
        eligible_users += 1
        pair_count = len(positive_ids) * len(negative_ids)
        possible_pairs += pair_count
        flat_positions = np.arange(pair_count, dtype=np.int64)
        if pair_count > cap_per_user:
            flat_positions = np.sort(rng.choice(flat_positions, size=cap_per_user, replace=False))
        pos_ix = flat_positions // len(negative_ids)
        neg_ix = flat_positions % len(negative_ids)
        pos_ids = positive_ids[pos_ix]
        neg_ids = negative_ids[neg_ix]
        users.extend([user_to_idx[uid]] * len(flat_positions))
        positive_items.extend(item_to_idx.get(int(item), 0) for item in pos_ids)
        negative_items.extend(item_to_idx.get(int(item), 0) for item in neg_ids)
        positive_scores.extend((np.log1p([score_by_item[int(item)] for item in pos_ids]) - score_mean) / score_std)
        negative_scores.extend((np.log1p([score_by_item[int(item)] for item in neg_ids]) - score_mean) / score_std)
    return {
        "user_id": np.asarray(users, dtype=np.int32),
        "positive_item_id": np.asarray(positive_items, dtype=np.int32),
        "negative_item_id": np.asarray(negative_items, dtype=np.int32),
        "positive_numeric": np.asarray(positive_scores, dtype=np.float32).reshape(-1, 1),
        "negative_numeric": np.asarray(negative_scores, dtype=np.float32).reshape(-1, 1),
        "eligible_training_users": eligible_users,
        "possible_observed_pairs": possible_pairs,
        "sampled_training_pairs": len(users),
        "observed_positive_labels": positive_observations,
        "observed_lower_rating_labels": negative_observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--eval-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-pairs-per-user", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if min(args.epochs, args.batch_size, args.max_pairs_per_user) < 1:
        parser.error("epochs, batch size and pair cap must be positive")
    for path in (args.train_candidates, args.eval_candidates):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    started = time.perf_counter()
    train_rows = read_rows(args.train_candidates)
    eval_rows = read_rows(args.eval_candidates)
    train_users = sorted({int(row["user_id"]) for row in train_rows})
    user_to_idx = {uid: index + 1 for index, uid in enumerate(train_users)}
    train_items = sorted({int(item) for row in train_rows for item in row["candidate_item_ids"]})
    item_to_idx = {item: index + 1 for index, item in enumerate(train_items)}

    # Fit the one numeric feature transform over observed training candidates only.
    labeled_log_scores = []
    rating_sum: dict[int, float] = {}
    rating_count: dict[int, int] = {}
    item_sum: dict[int, float] = {}
    item_count: dict[int, int] = {}
    rating_total = 0.0
    rating_total_count = 0
    for row in train_rows:
        uid = int(row["user_id"])
        scores = dict(zip(map(int, row["candidate_item_ids"]), map(float, row["retrieval_scores"])))
        for item_text, value in row["observed_ratings"].items():
            item, rating = int(item_text), float(value)
            labeled_log_scores.append(np.log1p(scores[item]))
            rating_sum[uid] = rating_sum.get(uid, 0.0) + rating
            rating_count[uid] = rating_count.get(uid, 0) + 1
            item_sum[item] = item_sum.get(item, 0.0) + rating
            item_count[item] = item_count.get(item, 0) + 1
            rating_total += rating
            rating_total_count += 1
    if not labeled_log_scores:
        raise ValueError("No observed candidate ratings in training export")
    numeric_mean = float(np.mean(labeled_log_scores))
    numeric_std = float(np.std(labeled_log_scores))
    if numeric_std < 1e-6:
        numeric_std = 1.0

    pairs = make_pairs(train_rows, user_to_idx, item_to_idx, numeric_mean, numeric_std,
                       args.max_pairs_per_user, args.seed)
    if not pairs["sampled_training_pairs"]:
        raise ValueError("No users had both observed high and lower ratings among candidates")
    targets = np.ones(pairs["sampled_training_pairs"], dtype=np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((
        {key: pairs[key] for key in ("user_id", "positive_item_id", "negative_item_id",
                                      "positive_numeric", "negative_numeric")},
        targets,
    )).shuffle(len(targets), seed=args.seed, reshuffle_each_iteration=True).batch(args.batch_size)

    scorer = DCNRatingRanker(len(user_to_idx) + 1, len(item_to_idx) + 1, 1,
                             embedding_dim=32, cross_layers=3,
                             deep_units=(128, 64), dropout=0.1)
    pair_model = PairwiseScorer(scorer)
    pair_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="pair_accuracy")],
    )
    fit = pair_model.fit(train_ds, epochs=args.epochs, verbose=0)

    eval_by_user = {int(row["user_id"]): row for row in eval_rows}
    eval_user_ids = sorted(set(train_users) & set(eval_by_user))
    if not eval_user_ids:
        raise ValueError("No shared user IDs in the two candidate exports")
    global_rating_mean = rating_total / rating_total_count
    per_user = []
    candidate_total = oov_item_count = 0
    for uid in eval_user_ids:
        row = eval_by_user[uid]
        item_ids = list(map(int, row["candidate_item_ids"]))
        raw_scores = np.asarray(row["retrieval_scores"], dtype=np.float32)
        item_indices = np.asarray([item_to_idx.get(item, 0) for item in item_ids], dtype=np.int32)
        numeric = ((np.log1p(raw_scores) - numeric_mean) / numeric_std).reshape(-1, 1).astype(np.float32)
        dcn_scores = np.asarray(scorer({
            "user_id": np.full(len(item_ids), user_to_idx[uid], dtype=np.int32),
            "item_id": item_indices,
            "numeric": numeric,
        }, training=False)).reshape(-1)
        score_lookup = dict(zip(item_ids, map(float, dcn_scores)))
        ratings = {int(item): float(rating) for item, rating in row["observed_ratings"].items()}
        relevant = {item for item, rating in ratings.items() if rating >= 4.0}
        rank_order = sorted(item_ids, key=lambda item: (-score_lookup[item], item))
        mean_order = []
        train_um = rating_sum.get(uid, 0.0) / rating_count[uid] if uid in rating_count else global_rating_mean
        for item in item_ids:
            im = item_sum.get(item, 0.0) / item_count[item] if item in item_count else global_rating_mean
            mean_order.append((item, float(np.clip(train_um + im - global_rating_mean, 0.5, 5.0))))
        mean_order = [item for item, _ in sorted(mean_order, key=lambda pair: (-pair[1], pair[0]))]
        per_user.append({
            "user_id": uid,
            "observed_candidate_ratings": len(ratings),
            "positive_candidate_ratings": len(relevant),
            "itemknn": dict(zip(("recall@10", "ndcg@10"), metric_at_k(item_ids, relevant, 10))),
            "mean_baseline": dict(zip(("recall@10", "ndcg@10"), metric_at_k(mean_order, relevant, 10))),
            "pairwise_dcn": dict(zip(("recall@10", "ndcg@10"), metric_at_k(rank_order, relevant, 10))),
        })
        candidate_total += len(item_ids)
        oov_item_count += int(np.count_nonzero(item_indices == 0))

    def macro(name: str, metric: str) -> float:
        return float(np.mean([row[name][metric] for row in per_user]))

    results = {
        model: {"macro_recall@10": macro(model, "recall@10"),
                "macro_ndcg@10": macro(model, "ndcg@10")}
        for model in ("itemknn", "mean_baseline", "pairwise_dcn")
    }
    bootstraps = {
        f"pairwise_dcn_minus_{baseline}_{metric}": paired_bootstrap(per_user, metric, "pairwise_dcn", baseline)
        for baseline in ("itemknn", "mean_baseline")
        for metric in ("recall@10", "ndcg@10")
    }
    args.output.mkdir(parents=True)
    scorer.save(args.output / "dcn_pairwise_candidate_ranker.keras")
    np.savez_compressed(
        args.output / "feature_mapping.npz",
        user_ids=np.asarray(train_users, dtype=np.int64),
        item_ids=np.asarray(train_items, dtype=np.int64),
        numeric_mean=np.asarray([numeric_mean], dtype=np.float32),
        numeric_std=np.asarray([numeric_std], dtype=np.float32),
    )
    with (args.output / "per_user_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in per_user:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    observed_ratings = sum(len(row["observed_ratings"]) for row in train_rows)
    observed_eval = sum(len(eval_by_user[user]["observed_ratings"]) for user in eval_user_ids)
    positive_eval = sum(sum(float(value) >= 4 for value in eval_by_user[user]["observed_ratings"].values())
                        for user in eval_user_ids)
    report = {
        "run": "Pairwise DCN candidate ranker, MovieLens 50/60 train to 60/70 sampled development evaluation",
        "train_export_sha256": sha256(args.train_candidates),
        "eval_export_sha256": sha256(args.eval_candidates),
        "training_export_rows": len(train_rows),
        "training_users": len(train_users),
        "training_observed_candidate_ratings": observed_ratings,
        "pair_sampling": {
            "positive": "observed rating >= 4.0",
            "lower_rated": "observed rating < 4.0",
            "unrated": "unknown; excluded from pairs",
            "users_with_both_classes": pairs["eligible_training_users"],
            "possible_observed_pairs": pairs["possible_observed_pairs"],
            "max_pairs_per_user": args.max_pairs_per_user,
            "sampled_pairs": pairs["sampled_training_pairs"],
            "observed_positive_labels": pairs["observed_positive_labels"],
            "observed_lower_rating_labels": pairs["observed_lower_rating_labels"],
            "seed": args.seed,
        },
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                   "architecture": "same DCN-v1 scorer as pointwise check",
                   "embedding_dim": 32, "cross_layers": 3, "deep_units": [128, 64],
                   "dropout": 0.1, "optimizer": "Adam(1e-3)",
                   "objective": "binary cross entropy from logits of score(positive)-score(lower-rated)",
                   "numeric_feature": "training-standardized log1p(ItemKNN retrieval score)"},
        "epoch_pairwise_loss": [float(value) for value in fit.history["loss"]],
        "epoch_pair_accuracy": [float(value) for value in fit.history["pair_accuracy"]],
        "evaluation_users": len(eval_user_ids),
        "evaluation_observed_candidate_ratings": observed_eval,
        "evaluation_positive_candidate_ratings": int(positive_eval),
        "evaluation_candidates": candidate_total,
        "evaluation_oov_item_candidates": oov_item_count,
        "metrics": results,
        "paired_user_bootstrap_2000": bootstraps,
        "official_final_10_percent_used": False,
        "raw_ratings_csv_read": False,
        "itemknn_api_or_default_changed": False,
        "limitations": [
            "Both sampled temporal windows were reused during development and are not independent test data.",
            "The evaluation has only the 386 IDs shared by the two sampled candidate exports.",
            "Positive and lower-rated pairs use only observed explicit ratings; unrated items are not negatives.",
            "Top-k metrics treat unrated candidates as non-hits, so MovieLens exposure/selection bias remains.",
            "Ratings are not clicks or CTR; no online A/B test is measured.",
            "This experiment changes only the DCN objective and observed pair sampling relative to the pointwise candidate-ranker run; it is not evidence of production value.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}
    (args.output / "artifact_manifest.json").write_text(json.dumps({"files_sha256": files}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": results,
                      "bootstrap": bootstraps, "support": {"users": len(eval_user_ids),
                      "train_pairs": pairs["sampled_training_pairs"],
                      "eval_labels": observed_eval, "eval_positives": int(positive_eval)},
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
