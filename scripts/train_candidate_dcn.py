"""Train a small DCN rating scorer on saved 50/60 ItemKNN candidates and
evaluate it on the overlapping users in the saved 60/70 candidate export.

This script reads only the two compressed candidate exports. It never opens
the raw MovieLens CSV and never treats unrated candidates as negatives.
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


def bootstrap(users: list[dict], metric: str, left: str, right: str) -> dict:
    delta = np.asarray([row[left][metric] - row[right][metric] for row in users], dtype=np.float64)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(delta), size=(2000, len(delta)), dtype=np.int32)
    means = delta[draws].mean(axis=1)
    return {
        "left_minus_right_median": float(np.median(means)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "users": int(len(delta)), "resamples": 2000, "seed": 42,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--eval-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    for path in (args.train_candidates, args.eval_candidates):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep the local CPU footprint predictable. No device download is needed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    start = time.perf_counter()
    train_rows = read_rows(args.train_candidates)
    eval_rows = read_rows(args.eval_candidates)
    train_user_ids = sorted({int(row["user_id"]) for row in train_rows})
    user_to_idx = {user_id: i + 1 for i, user_id in enumerate(train_user_ids)}
    train_item_ids = sorted({int(item) for row in train_rows for item in row["candidate_item_ids"]})
    item_to_idx = {item_id: i + 1 for i, item_id in enumerate(train_item_ids)}

    train_user, train_item, train_num, train_y = [], [], [], []
    user_sum: dict[int, float] = {}
    user_count: dict[int, int] = {}
    item_sum: dict[int, float] = {}
    item_count: dict[int, int] = {}
    for row in train_rows:
        uid = int(row["user_id"])
        score_by_item = dict(zip(map(int, row["candidate_item_ids"]), map(float, row["retrieval_scores"])))
        for item_text, rating_value in row["observed_ratings"].items():
            item_id, rating = int(item_text), float(rating_value)
            train_user.append(user_to_idx[uid])
            train_item.append(item_to_idx[item_id])
            train_num.append([np.log1p(score_by_item[item_id])])
            train_y.append(rating)
            user_sum[uid] = user_sum.get(uid, 0.0) + rating
            user_count[uid] = user_count.get(uid, 0) + 1
            item_sum[item_id] = item_sum.get(item_id, 0.0) + rating
            item_count[item_id] = item_count.get(item_id, 0) + 1
    if not train_y:
        raise ValueError("No observed candidate ratings in the training export")

    x_num = np.asarray(train_num, dtype=np.float32)
    numeric_mean = x_num.mean(axis=0)
    numeric_std = x_num.std(axis=0)
    numeric_std[numeric_std < 1e-6] = 1.0
    x_num = (x_num - numeric_mean) / numeric_std
    y = np.asarray(train_y, dtype=np.float32)
    x_user = np.asarray(train_user, dtype=np.int32)
    x_item = np.asarray(train_item, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((
        {"user_id": x_user, "item_id": x_item, "numeric": x_num}, y
    )).shuffle(len(y), seed=args.seed, reshuffle_each_iteration=True).batch(args.batch_size)

    model = DCNRatingRanker(
        num_users=len(user_to_idx) + 1,
        num_items=len(item_to_idx) + 1,
        numeric_features=1,
        embedding_dim=32,
        cross_layers=3,
        deep_units=(128, 64),
        dropout=0.1,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    history = model.fit(dataset, epochs=args.epochs, verbose=0)

    eval_user_ids = {int(row["user_id"]) for row in eval_rows}
    shared_user_ids = sorted(set(train_user_ids) & eval_user_ids)
    eval_by_user = {int(row["user_id"]): row for row in eval_rows}
    if not shared_user_ids:
        raise ValueError("No shared user IDs between training and evaluation exports")

    global_mean = float(y.mean())
    per_user = []
    observed_true, observed_pred_dcn, observed_pred_mean = [], [], []
    oov_user_candidates = oov_item_candidates = total_candidates = 0
    for uid in shared_user_ids:
        row = eval_by_user[uid]
        item_ids = list(map(int, row["candidate_item_ids"]))
        retrieval_scores = list(map(float, row["retrieval_scores"]))
        ratings = {int(item): float(value) for item, value in row["observed_ratings"].items()}
        numeric = (np.log1p(np.asarray(retrieval_scores, dtype=np.float32)[:, None]) - numeric_mean) / numeric_std
        known_user = uid in user_to_idx
        uidx = user_to_idx.get(uid, 0)
        item_idx = np.asarray([item_to_idx.get(item, 0) for item in item_ids], dtype=np.int32)
        oov_user_candidates += 0 if known_user else len(item_ids)
        oov_item_candidates += int(np.count_nonzero(item_idx == 0))
        total_candidates += len(item_ids)
        dcn_scores = []
        for left in range(0, len(item_ids), 8192):
            right = min(left + 8192, len(item_ids))
            batch = {
                "user_id": np.full(right - left, uidx, dtype=np.int32),
                "item_id": item_idx[left:right],
                "numeric": numeric[left:right],
            }
            dcn_scores.extend(np.asarray(model(batch, training=False)).reshape(-1).tolist())
        mean_scores = []
        for item in item_ids:
            um = user_sum.get(uid, 0.0) / user_count[uid] if uid in user_count else global_mean
            im = item_sum.get(item, 0.0) / item_count[item] if item in item_count else global_mean
            mean_scores.append(float(np.clip(um + im - global_mean, 0.5, 5.0)))
        retrieval_order = item_ids  # Export order is ItemKNN score descending.
        mean_order = [item_ids[i] for i in np.lexsort((np.asarray(item_ids), -np.asarray(mean_scores)))]
        dcn_order = [item_ids[i] for i in np.lexsort((np.asarray(item_ids), -np.asarray(dcn_scores)))]
        relevant = {item for item, rating in ratings.items() if rating >= 4.0}
        per_user.append({
            "user_id": uid,
            "observed_candidate_ratings": len(ratings),
            "positive_candidate_ratings": len(relevant),
            "itemknn": dict(zip(("recall@10", "ndcg@10"), metric_at_k(retrieval_order, relevant, 10))),
            "mean_baseline": dict(zip(("recall@10", "ndcg@10"), metric_at_k(mean_order, relevant, 10))),
            "dcn": dict(zip(("recall@10", "ndcg@10"), metric_at_k(dcn_order, relevant, 10))),
        })
        for item, rating in ratings.items():
            i = item_ids.index(item)
            observed_true.append(rating)
            observed_pred_dcn.append(float(np.clip(dcn_scores[i], 0.5, 5.0)))
            observed_pred_mean.append(mean_scores[i])

    def mean_metric(model_name: str, metric: str) -> float:
        return float(np.mean([row[model_name][metric] for row in per_user]))

    true = np.asarray(observed_true, dtype=np.float64)
    dcn_pred = np.asarray(observed_pred_dcn, dtype=np.float64)
    mean_pred = np.asarray(observed_pred_mean, dtype=np.float64)
    metrics = {
        name: {
            "macro_recall@10": mean_metric(name, "recall@10"),
            "macro_ndcg@10": mean_metric(name, "ndcg@10"),
        }
        for name in ("itemknn", "mean_baseline", "dcn")
    }
    metrics["observed_candidate_rating_error"] = {
        "support": int(len(true)),
        "dcn_rmse": float(np.sqrt(np.mean((dcn_pred - true) ** 2))),
        "dcn_mae": float(np.mean(np.abs(dcn_pred - true))),
        "mean_baseline_rmse": float(np.sqrt(np.mean((mean_pred - true) ** 2))),
        "mean_baseline_mae": float(np.mean(np.abs(mean_pred - true))),
    }
    bootstrap_results = {
        f"dcn_vs_{baseline}_{metric}": bootstrap(per_user, metric, "dcn", baseline)
        for baseline in ("itemknn", "mean_baseline")
        for metric in ("recall@10", "ndcg@10")
    }
    args.output.mkdir(parents=True)
    model_path = args.output / "dcn_candidate_ranker.keras"
    model.save(model_path)
    np.savez_compressed(
        args.output / "feature_mapping.npz",
        user_ids=np.asarray(train_user_ids, dtype=np.int64),
        item_ids=np.asarray(train_item_ids, dtype=np.int64),
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
    )
    with (args.output / "per_user_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_user:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "run": "DCN candidate-conditioned rating reranker; MovieLens 50/60 train to 60/70 development eval",
        "training_export": str(args.train_candidates),
        "training_export_sha256": sha256(args.train_candidates),
        "evaluation_export": str(args.eval_candidates),
        "evaluation_export_sha256": sha256(args.eval_candidates),
        "training_rows": len(train_rows),
        "training_observed_candidate_labels": int(len(y)),
        "training_users": len(train_user_ids),
        "training_items_in_candidate_vocabulary": len(train_item_ids),
        "training_rating_mean": global_mean,
        "epoch_losses_mse": [float(x) for x in history.history["loss"]],
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                   "embedding_dim": 32, "cross_layers": 3, "deep_units": [128, 64],
                   "dropout": 0.1, "optimizer": "Adam(1e-3)", "loss": "MSE",
                   "numeric_feature": "training-standardized log1p(ItemKNN retrieval score)"},
        "evaluation_users": len(shared_user_ids),
        "evaluation_shared_users": shared_user_ids,
        "evaluation_observed_candidate_rating_support": len(observed_true),
        "evaluation_positive_candidate_rating_support": int(sum(row["positive_candidate_ratings"] for row in per_user)),
        "evaluation_candidate_count": total_candidates,
        "evaluation_oov_user_candidate_count": oov_user_candidates,
        "evaluation_oov_item_candidate_count": oov_item_candidates,
        "metrics": metrics,
        "paired_user_bootstrap_2000": bootstrap_results,
        "limitations": [
            "The two sampled development windows were already used in model development and are not independent test data.",
            "Only users shared by the two sampled exports are evaluated; the overlap is a small, selected development subset.",
            "Unrated candidate outcomes are unknown. Ranking metrics count unobserved items in the top-k as non-hits, so exposure/label bias remains.",
            "MovieLens ratings are explicit preferences, not clicks or CTR.",
            "The 50/60 candidate retrieval scores and the 60/70 evaluation scores come from time-specific ItemKNN fits; only the log-score feature is passed to the ranker.",
            "This is a candidate-ranker diagnostic, not an online or production evaluation. The API default is unchanged.",
        ],
        "official_final_10_percent_used": False,
        "raw_ratings_csv_read": False,
        "elapsed_seconds": time.perf_counter() - start,
    }
    report_path = args.output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    support = {
        "training_export_sha256": report["training_export_sha256"],
        "user_ids": sorted(user_count),
        "item_ids": sorted(item_count),
    }
    (args.output / "trained_support.json").write_text(
        json.dumps(support, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    hashes = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()}
    (args.output / "artifact_manifest.json").write_text(
        json.dumps({"files_sha256": hashes}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "metrics": metrics,
                      "bootstrap": bootstrap_results,
                      "users": len(shared_user_ids), "labels": len(observed_true),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
