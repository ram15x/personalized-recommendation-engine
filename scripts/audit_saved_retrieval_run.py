"""Read-only audit of saved two-tower/FAISS artifacts in a Colab runtime.

This script never trains, applies gradients, downloads data, or writes model
weights. It checks serialization/index consistency, item-ID mapping on one
fixed training batch, the custom loss's gradient state, and saved cohort metrics.
"""
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf


def _norm_summary(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {"min": float(norms.min()), "median": float(np.median(norms)), "max": float(norms.max())}


def _cohort_summary(frame: pd.DataFrame) -> dict:
    groups = []
    metrics = [
        "hybrid_recall@10", "hybrid_ndcg@10", "popularity_recall@10",
        "popularity_ndcg@10", "retrieval_recall@100", "retrieval_recall@500",
    ]
    for has_embedding in (True, False):
        for has_history in (True, False):
            group = frame[
                (frame["has_embedding"] == has_embedding)
                & (frame["has_prior_history"] == has_history)
            ]
            if group.empty:
                continue
            eligible = int(group["eligible_targets"].sum())
            indexed = int(group["targets_in_index"].sum())
            groups.append({
                "has_embedding": has_embedding,
                "has_prior_history": has_history,
                "users": int(len(group)),
                "eligible_targets": eligible,
                "targets_in_index": indexed,
                "target_pair_index_coverage": indexed / eligible if eligible else None,
                **{metric: float(group[metric].mean()) for metric in metrics},
            })
    eligible = int(frame["eligible_targets"].sum())
    indexed = int(frame["targets_in_index"].sum())
    return {
        "groups": groups,
        "all_users": int(len(frame)),
        "eligible_targets": eligible,
        "targets_in_index": indexed,
        "target_pair_index_coverage": indexed / eligible if eligible else None,
    }


def audit(output_dir: Path, ratings_path: Path, training_script: Path) -> dict:
    vectors = np.load(output_dir / "embeddings.npz")
    user_ids = vectors["user_ids"]
    item_ids = vectors["item_ids"]
    saved_users = vectors["user_vectors"]
    saved_items = vectors["item_vectors"]

    user_tower = tf.keras.models.load_model(output_dir / "user_tower.keras")
    item_tower = tf.keras.models.load_model(output_dir / "item_tower.keras")
    user_output = user_tower(np.arange(len(user_ids), dtype=np.int32)).numpy().astype(np.float32)
    item_output = item_tower(np.arange(len(item_ids), dtype=np.int32)).numpy().astype(np.float32)

    index = faiss.read_index(str(output_dir / "items.faiss"))
    index_vectors = index.reconstruct_n(0, index.ntotal)

    module = runpy.run_path(str(training_script), run_name="audit_definitions")
    model = module["TwoTower"](len(user_ids), len(item_ids))
    model.user_tower(tf.constant([0], dtype=tf.int32))
    model.item_tower(tf.constant([0], dtype=tf.int32))
    model.user_tower.set_weights(user_tower.get_weights())
    model.item_tower.set_weights(item_tower.get_weights())

    sample = None
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        chunksize=500_000,
    ):
        positive = chunk.loc[
            (chunk["timestamp"] < module["TRAIN_CUTOFF"])
            & (chunk["rating"] >= module["POSITIVE_RATING"])
        ]
        if len(positive) >= 512:
            sample = positive.sample(n=512, random_state=42)
            break
    if sample is None:
        raise RuntimeError("No 512-row positive training sample in the first CSV chunk")

    user_positions = np.searchsorted(user_ids, sample["userId"].to_numpy(np.int64))
    item_positions = np.searchsorted(item_ids, sample["movieId"].to_numpy(np.int64))
    user_mapping_exact = np.array_equal(user_ids[user_positions], sample["userId"].to_numpy(np.int64))
    item_mapping_exact = np.array_equal(item_ids[item_positions], sample["movieId"].to_numpy(np.int64))
    if not user_mapping_exact or not item_mapping_exact:
        raise ValueError("Sampled raw IDs do not round-trip through the saved ID arrays")

    features = {
        "user": tf.constant(user_positions, dtype=tf.int32),
        "item": tf.constant(item_positions, dtype=tf.int32),
    }
    weights_before = [weight.numpy().copy() for weight in model.weights]
    with tf.GradientTape() as tape:
        loss = model.compute_loss(features, training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_values = [
        gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
        for gradient in gradients if gradient is not None
    ]
    finite_gradients = all(
        bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()) for gradient in gradient_values
    )
    nonzero_gradient_tensors = sum(
        bool(tf.reduce_any(gradient != 0).numpy()) for gradient in gradient_values
    )
    gradient_norm = float(tf.linalg.global_norm([g for g in gradients if g is not None]).numpy())

    query = model.user_tower(features["user"])
    candidate = model.item_tower(features["item"])
    scores = tf.matmul(query, candidate, transpose_b=True)
    same_user = tf.equal(features["user"][:, None], features["user"][None, :])
    same_item = tf.equal(features["item"][:, None], features["item"][None, :])
    diagonal = tf.eye(512, dtype=tf.bool)
    mask = tf.logical_and(tf.logical_or(same_user, same_item), tf.logical_not(diagonal))
    masked_scores = tf.where(mask, tf.cast(-1e9, scores.dtype), scores)
    top1 = float(tf.reduce_mean(tf.cast(
        tf.argmax(masked_scores, axis=1, output_type=tf.int32) == tf.range(512), tf.float32
    )).numpy())
    hardest_negative = tf.reduce_max(
        tf.where(diagonal, tf.cast(-1e9, scores.dtype), masked_scores), axis=1
    )
    positive_margin = float(tf.reduce_mean(tf.linalg.diag_part(scores) - hardest_negative).numpy())
    weights_unchanged = all(
        np.array_equal(before, after.numpy()) for before, after in zip(weights_before, model.weights)
    )

    predictions = pd.read_csv(output_dir / "per_user_predictions.csv")
    return {
        "audit_type": "read-only; no training or weight updates",
        "artifacts": {
            "user_vector_max_abs_diff": float(np.max(np.abs(user_output - saved_users))),
            "item_vector_max_abs_diff": float(np.max(np.abs(item_output - saved_items))),
            "faiss_vector_max_abs_diff": float(np.max(np.abs(index_vectors - saved_items))),
            "faiss_items": int(index.ntotal),
            "vectors_finite": bool(np.isfinite(saved_users).all() and np.isfinite(saved_items).all()),
            "user_norms": _norm_summary(saved_users),
            "item_norms": _norm_summary(saved_items),
        },
        "sampled_batch": {
            "selection": "512 rating>=4 training rows sampled (seed 42) from the first 500,000-row CSV chunk",
            "rows": int(len(sample)),
            "unique_users": int(sample["userId"].nunique()),
            "unique_items": int(sample["movieId"].nunique()),
            "user_mapping_roundtrip_exact": user_mapping_exact,
            "item_mapping_roundtrip_exact": item_mapping_exact,
            "loss": float(loss.numpy()),
            "gradient_tensors": int(len(gradients)),
            "all_gradients_finite": finite_gradients,
            "nonzero_gradient_tensors": int(nonzero_gradient_tensors),
            "gradient_global_norm": gradient_norm,
            "mean_positive_minus_hardest_negative_logit": positive_margin,
            "masked_top1_accuracy": top1,
            "weights_unchanged": weights_unchanged,
        },
        "saved_prediction_cohorts": _cohort_summary(predictions),
        "limitations": [
            "The one-batch gradient and ranking diagnostic is not representative of the full training distribution.",
            "The CSV and model artifacts are expected in the specified Colab paths; this script does not download them.",
            "This audit checks artifact consistency and one mapped batch, not generalization or convergence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/nexvia_aiml03_dev_70_80_fullcatalog"))
    parser.add_argument("--ratings", type=Path, default=Path("/content/ml-25m/ratings.csv"))
    parser.add_argument("--training-script", type=Path, default=Path("/content/train_dev_retrieval.py"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.output_dir, args.ratings, args.training_script)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
