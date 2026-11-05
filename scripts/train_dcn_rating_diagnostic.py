"""Train a bounded DCN rating diagnostic on the existing MovieLens file.

This is not end-to-end retrieval evaluation: validation ranking is restricted to
items each selected user actually rated during the future window. The official
final 10% is never read as training or validation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.dcn_ranker import DCNRatingRanker


EXPECTED_RATINGS_SHA256 = "9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385"
TRAIN_BEFORE = 1_421_944_985  # audited global timestamp boundary at 70%
VALIDATION_BEFORE = 1_466_837_400  # audited global timestamp boundary at 80%
COLS = ["userId", "movieId", "rating", "timestamp"]
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grow(values: np.ndarray, size: int) -> np.ndarray:
    if size <= len(values):
        return values
    return np.pad(values, (0, size - len(values)))


def update_stats(counts: np.ndarray, sums: np.ndarray, ids: np.ndarray,
                 ratings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    needed = int(ids.max()) + 1 if len(ids) else len(counts)
    counts = grow(counts, needed)
    sums = grow(sums, needed)
    counts += np.bincount(ids, minlength=len(counts)).astype(np.int64)
    sums += np.bincount(ids, weights=ratings, minlength=len(sums))
    return counts, sums


def keep_random_sample(current: pd.DataFrame, rows: pd.DataFrame,
                       cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if rows.empty:
        return current
    rows = rows.loc[:, ["userId", "movieId", "rating"]].copy()
    rows["_priority"] = rng.random(len(rows))
    merged = rows if current.empty else pd.concat([current, rows], ignore_index=True)
    if len(merged) > cap:
        merged = merged.nsmallest(cap, "_priority")
    return merged.reset_index(drop=True)


def id_indices(ids: np.ndarray, sorted_known: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_known, ids)
    valid = positions < len(sorted_known)
    matched = np.zeros(len(ids), dtype=bool)
    matched[valid] = sorted_known[positions[valid]] == ids[valid]
    # Embedding row zero is reserved for IDs absent from training.
    return np.where(matched, positions + 1, 0).astype(np.int32)


def build_genres(movies_path: Path, item_ids: np.ndarray) -> np.ndarray:
    table = pd.read_csv(movies_path, usecols=["movieId", "genres"])
    lookup = {name: i for i, name in enumerate(GENRES)}
    max_movie_id = max(int(item_ids.max()), int(table.movieId.max()))
    matrix = np.zeros((max_movie_id + 1, len(GENRES)), dtype=np.float32)
    for row in table.itertuples(index=False):
        movie_id = int(row.movieId)
        for genre in str(row.genres).split("|"):
            if genre in lookup:
                matrix[movie_id, lookup[genre]] = 1.0
    return matrix


def make_features(rows: pd.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray,
                  user_counts: np.ndarray, user_sums: np.ndarray,
                  item_counts: np.ndarray, item_sums: np.ndarray,
                  genre_matrix: np.ndarray, global_mean: float) -> tuple[dict, np.ndarray]:
    users = rows.userId.to_numpy(np.int64)
    items = rows.movieId.to_numpy(np.int64)
    user_idx = id_indices(users, user_ids)
    item_idx = id_indices(items, item_ids)
    safe_user = np.minimum(users, len(user_counts) - 1)
    safe_item = np.minimum(items, len(item_counts) - 1)
    uc = user_counts[safe_user].astype(np.float32)
    ic = item_counts[safe_item].astype(np.float32)
    user_known = (users < len(user_counts)) & (uc > 0)
    item_known = (items < len(item_counts)) & (ic > 0)
    user_mean = np.divide(
        user_sums[safe_user], uc, out=np.full(len(uc), global_mean), where=uc > 0
    )
    item_mean = np.divide(
        item_sums[safe_item], ic, out=np.full(len(ic), global_mean), where=ic > 0
    )
    uc = np.where(user_known, uc, 0.0)
    ic = np.where(item_known, ic, 0.0)
    user_mean = np.where(user_known, user_mean, global_mean)
    item_mean = np.where(item_known, item_mean, global_mean)
    genre_values = np.zeros((len(rows), len(GENRES)), dtype=np.float32)
    genre_values[item_known] = genre_matrix[safe_item[item_known]]
    numeric = np.column_stack([
        user_mean, np.log1p(uc), item_mean, np.log1p(ic), genre_values
    ]).astype(np.float32)
    x = {"user_id": user_idx, "item_id": item_idx, "numeric": numeric}
    return x, rows.rating.to_numpy(np.float32)


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred.astype(np.float64) - y.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "rows": int(len(y))}


def ranking_metrics(rows: pd.DataFrame, predictions: np.ndarray, k: int = 10) -> pd.DataFrame:
    frame = rows[["userId", "rating"]].copy()
    frame["score"] = predictions
    metrics = []
    for user, group in frame.groupby("userId", sort=True):
        relevant = (group.rating.to_numpy() >= 4.0)
        if not relevant.any():
            continue
        order = np.argsort(-group.score.to_numpy(), kind="stable")[:k]
        hits = relevant[order]
        dcg = np.sum(hits / np.log2(np.arange(len(hits), dtype=np.float64) + 2.0))
        ideal_n = min(int(relevant.sum()), k)
        idcg = np.sum(1.0 / np.log2(np.arange(ideal_n, dtype=np.float64) + 2.0))
        metrics.append({
            "userId": int(user), "candidate_ratings": int(len(group)),
            "positive_ratings": int(relevant.sum()),
            "recall_at_10": float(hits.sum() / relevant.sum()),
            "ndcg_at_10": float(dcg / idcg),
        })
    return pd.DataFrame(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Must be a new directory; existing paths are refused.")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--validation-users", type=int, default=2_000)
    parser.add_argument("--min-validation-ratings", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_train_rows < 1 or args.validation_users < 1 or args.epochs < 1:
        parser.error("sample sizes and epochs must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    for path in (args.ratings, args.movies):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_hash = sha256(args.ratings)
    if data_hash != EXPECTED_RATINGS_SHA256:
        raise ValueError(f"Unexpected MovieLens ratings SHA-256: {data_hash}")

    rng = np.random.default_rng(args.seed)
    user_counts = np.zeros(1, dtype=np.int64)
    user_sums = np.zeros(1, dtype=np.float64)
    item_counts = np.zeros(1, dtype=np.int64)
    item_sums = np.zeros(1, dtype=np.float64)
    train_sample = pd.DataFrame(columns=["userId", "movieId", "rating", "_priority"])
    validation_user_counts = np.zeros(1, dtype=np.int64)
    input_rows = train_rows = validation_rows = 0

    for chunk in pd.read_csv(args.ratings, usecols=COLS, chunksize=500_000):
        input_rows += len(chunk)
        train = chunk.loc[chunk.timestamp < TRAIN_BEFORE]
        validation = chunk.loc[(chunk.timestamp >= TRAIN_BEFORE) &
                               (chunk.timestamp < VALIDATION_BEFORE)]
        train_rows += len(train)
        validation_rows += len(validation)
        if len(train):
            u = train.userId.to_numpy(np.int64)
            i = train.movieId.to_numpy(np.int64)
            r = train.rating.to_numpy(np.float64)
            user_counts, user_sums = update_stats(user_counts, user_sums, u, r)
            item_counts, item_sums = update_stats(item_counts, item_sums, i, r)
            train_sample = keep_random_sample(train_sample, train, args.max_train_rows, rng)
        if len(validation):
            ids = validation.userId.to_numpy(np.int64)
            validation_user_counts = grow(validation_user_counts, int(ids.max()) + 1)
            validation_user_counts += np.bincount(
                ids, minlength=len(validation_user_counts)
            ).astype(np.int64)

    if input_rows == 0 or train_rows == 0 or validation_rows == 0:
        raise ValueError("The fixed 70/80 temporal windows were empty or unavailable.")
    eligible_users = np.flatnonzero(
        validation_user_counts >= args.min_validation_ratings
    )
    if len(eligible_users) < 100:
        raise ValueError(f"Too few validation users meet the support rule: {len(eligible_users)}")
    selected_users = np.sort(rng.choice(
        eligible_users, size=min(args.validation_users, len(eligible_users)), replace=False
    ))
    selected_set = set(map(int, selected_users))
    validation_parts = []
    for chunk in pd.read_csv(args.ratings, usecols=COLS, chunksize=500_000):
        selected = chunk.loc[
            (chunk.timestamp >= TRAIN_BEFORE) &
            (chunk.timestamp < VALIDATION_BEFORE) &
            chunk.userId.isin(selected_set), COLS[:-1]
        ]
        if len(selected):
            validation_parts.append(selected)
    validation = pd.concat(validation_parts, ignore_index=True)
    validation = validation.sort_values(["userId", "movieId"], kind="stable")
    user_ids = np.flatnonzero(user_counts > 0).astype(np.int64)
    item_ids = np.flatnonzero(item_counts > 0).astype(np.int64)
    global_mean = float(user_sums.sum() / user_counts.sum())
    genre_matrix = build_genres(args.movies, item_ids)

    train_x, train_y = make_features(
        train_sample, user_ids, item_ids, user_counts, user_sums,
        item_counts, item_sums, genre_matrix, global_mean
    )
    valid_x, valid_y = make_features(
        validation, user_ids, item_ids, user_counts, user_sums,
        item_counts, item_sums, genre_matrix, global_mean
    )
    center = train_x["numeric"].mean(axis=0)
    scale = train_x["numeric"].std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x["numeric"] = (train_x["numeric"] - center) / scale
    valid_x["numeric"] = (valid_x["numeric"] - center) / scale

    tf.keras.utils.set_random_seed(args.seed)
    model = DCNRatingRanker(
        num_users=len(user_ids) + 1, num_items=len(item_ids) + 1,
        numeric_features=train_x["numeric"].shape[1], embedding_dim=32,
        cross_layers=3, deep_units=(128, 64), dropout=0.1,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    history = model.fit(
        train_x, train_y, validation_data=(valid_x, valid_y),
        epochs=args.epochs, batch_size=args.batch_size, shuffle=True, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True
        )],
    )
    dcn_raw_predictions = model.predict(valid_x, batch_size=args.batch_size, verbose=0).ravel()
    dcn_predictions = np.clip(dcn_raw_predictions, 0.5, 5.0)
    valid_user_raw = validation.userId.to_numpy(np.int64)
    valid_item_raw = validation.movieId.to_numpy(np.int64)
    valid_user_safe = np.minimum(valid_user_raw, len(user_counts) - 1)
    valid_item_safe = np.minimum(valid_item_raw, len(item_counts) - 1)
    user_avg = np.full(len(validation), global_mean, dtype=np.float64)
    item_avg = np.full(len(validation), global_mean, dtype=np.float64)
    known_user = (valid_user_raw < len(user_counts)) & (user_counts[valid_user_safe] > 0)
    known_item = (valid_item_raw < len(item_counts)) & (item_counts[valid_item_safe] > 0)
    user_avg[known_user] = (
        user_sums[valid_user_safe[known_user]] / user_counts[valid_user_safe[known_user]]
    )
    item_avg[known_item] = (
        item_sums[valid_item_safe[known_item]] / item_counts[valid_item_safe[known_item]]
    )
    # Missing users/items independently fall back to the training global mean.
    baseline_predictions = np.clip(user_avg + item_avg - global_mean, 0.5, 5.0)
    baseline_metrics = regression_metrics(valid_y, baseline_predictions)
    dcn_raw_metrics = regression_metrics(valid_y, dcn_raw_predictions)
    dcn_metrics = regression_metrics(valid_y, dcn_predictions)
    baseline_rank = ranking_metrics(validation, baseline_predictions.ravel())
    dcn_rank = ranking_metrics(validation, dcn_predictions.ravel())
    rank_join = baseline_rank.merge(dcn_rank, on="userId", suffixes=("_baseline", "_dcn"))
    validation_user_ids = validation.userId.to_numpy(np.int64)
    validation_item_ids = validation.movieId.to_numpy(np.int64)
    warm_user_mask = (
        (validation_user_ids < len(user_counts)) &
        (user_counts[np.minimum(validation_user_ids, len(user_counts) - 1)] > 0)
    )
    known_item_mask = (
        (validation_item_ids < len(item_counts)) &
        (item_counts[np.minimum(validation_item_ids, len(item_counts) - 1)] > 0)
    )
    rank_user_ids = rank_join.userId.to_numpy(np.int64)
    rank_user_safe = np.minimum(rank_user_ids, len(user_counts) - 1)
    rank_join["warm_user"] = (
        (rank_user_ids < len(user_counts)) & (user_counts[rank_user_safe] > 0)
    )

    def rank_summary(frame: pd.DataFrame) -> dict:
        return {
            "users": int(len(frame)),
            "baseline_macro_recall_at_10": float(frame.recall_at_10_baseline.mean()) if len(frame) else None,
            "dcn_macro_recall_at_10": float(frame.recall_at_10_dcn.mean()) if len(frame) else None,
            "baseline_macro_ndcg_at_10": float(frame.ndcg_at_10_baseline.mean()) if len(frame) else None,
            "dcn_macro_ndcg_at_10": float(frame.ndcg_at_10_dcn.mean()) if len(frame) else None,
        }

    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.output / "dcn_rating_ranker.keras"
    model.save(model_path)
    np.save(args.output / "training_user_ids.npy", user_ids, allow_pickle=False)
    np.save(args.output / "training_item_ids.npy", item_ids, allow_pickle=False)
    np.save(args.output / "numeric_feature_mean.npy", center, allow_pickle=False)
    np.save(args.output / "numeric_feature_scale.npy", scale, allow_pickle=False)
    per_user = rank_join[[
        "userId", "warm_user", "candidate_ratings_baseline", "positive_ratings_baseline",
        "recall_at_10_baseline", "recall_at_10_dcn",
        "ndcg_at_10_baseline", "ndcg_at_10_dcn",
    ]]
    per_user.to_csv(args.output / "per_user_metrics.csv", index=False)
    report = {
        "run": "DCN-v1 explicit-rating diagnostic, fixed MovieLens 70/80 window",
        "ratings_sha256": data_hash,
        "split": {"train_before_timestamp": TRAIN_BEFORE,
                  "validation_before_timestamp": VALIDATION_BEFORE,
                  "train_rows": train_rows, "validation_rows": validation_rows,
                  "train_sample_rows": int(len(train_sample)),
                  "validation_sample_users": int(len(selected_users)),
                  "validation_sample_rows": int(len(validation)),
                  "min_validation_ratings_per_user": args.min_validation_ratings},
        "seed": args.seed, "epochs_requested": args.epochs,
        "epochs_run": len(history.history["loss"]), "batch_size": args.batch_size,
        "training_objective": "MSE on observed 0.5-5.0 star ratings",
        "baseline": "clipped user mean + item mean - training global mean",
        "rating_metrics": {
            "baseline_clipped": baseline_metrics,
            "dcn_clipped_primary": dcn_metrics,
            "dcn_raw_sensitivity": dcn_raw_metrics,
        },
        "rating_ranking_on_observed_validation_items_only": {
            "relevance": "observed rating >= 4",
            "users_with_at_least_one_positive": int(len(rank_join)),
            "all": rank_summary(rank_join),
            "warm": rank_summary(rank_join.loc[rank_join.warm_user]),
            "cold": rank_summary(rank_join.loc[~rank_join.warm_user]),
            "validation_users_with_training_history": int(
                validation.loc[warm_user_mask, "userId"].nunique()
            ),
            "validation_users_without_training_history": int(
                validation.loc[~warm_user_mask, "userId"].nunique()
            ),
            "validation_ratings_for_items_seen_in_training": int(known_item_mask.sum()),
            "validation_ratings_for_unseen_items": int((~known_item_mask).sum()),
        },
        "limitations": [
            "The 70/80 development window was reused in earlier Project 2 work.",
            "Ranking is among items users actually rated in validation, not retrieved unseen candidates.",
            "Observed ratings are selection-biased and unrated items are unknown.",
            "This run does not establish candidate retrieval, end-to-end recommendation quality, CTR, or online lift.",
            "The official final 10% was not used.",
        ],
        "training_history": history.history,
        "checkpoint_sha256": sha256(model_path),
        "code_sha256": {
            "training_script": sha256(Path(__file__).resolve()),
            "ranker_module": sha256(Path(__file__).with_name("dcn_ranker.py")),
        },
        "saved_artifacts": {
            name: sha256(args.output / name) for name in [
                "training_user_ids.npy", "training_item_ids.npy",
                "numeric_feature_mean.npy", "numeric_feature_scale.npy",
            ]
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved new diagnostic to {args.output}", flush=True)


if __name__ == "__main__":
    main()
