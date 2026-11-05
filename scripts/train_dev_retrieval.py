"""Train and evaluate one fixed two-tower baseline on the 70/80 dev split.

Intended for a temporary Colab T4 runtime. It downloads the official MovieLens
archive only into that runtime, verifies the ratings checksum, and writes the
model and metrics under /content. It never reads the historical 80-90% panel
or the final 10% period.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
EXPECTED_RATINGS_SHA256 = "9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385"
TRAIN_CUTOFF = 1421944985
VALIDATION_END = 1466837400
POSITIVE_RATING = 4.0
SEED = 42
BATCH_SIZE = 512
EPOCHS = 3
DIMENSIONS = 32
CHUNK_SIZE = 500_000
ROOT = Path("/content")
DATA_DIR = ROOT / "ml-25m"
RATINGS_PATH = DATA_DIR / "ratings.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"
OUTPUT = ROOT / "nexvia_aiml03_dev_70_80_fullcatalog"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset() -> str:
    """Download only in ephemeral Colab storage and verify before parsing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RATINGS_PATH.is_file() or not MOVIES_PATH.is_file():
        archive = ROOT / "ml-25m.zip"
        if not archive.is_file():
            print("Downloading official MovieLens 25M archive into Colab /content...")
            request = urllib.request.Request(URL, headers={"User-Agent": "Project2-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as target:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
        with zipfile.ZipFile(archive) as bundle:
            for member in ("ml-25m/ratings.csv", "ml-25m/movies.csv"):
                bundle.extract(member, ROOT)
    actual = sha256(RATINGS_PATH)
    if actual != EXPECTED_RATINGS_SHA256:
        raise ValueError(f"Ratings checksum mismatch: expected {EXPECTED_RATINGS_SHA256}, got {actual}")
    return actual


class TwoTower(tfrs.Model):
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
        users, items = features["user"], features["item"]
        query = self.user_tower(users)
        candidate = self.item_tower(items)
        scores = tf.matmul(query, candidate, transpose_b=True)
        same_user = tf.equal(users[:, None], users[None, :])
        same_item = tf.equal(items[:, None], items[None, :])
        diagonal = tf.eye(tf.shape(scores)[0], dtype=tf.bool)
        masked = tf.logical_and(tf.logical_or(same_user, same_item), tf.logical_not(diagonal))
        scores = tf.where(masked, tf.cast(-1e9, scores.dtype), scores)
        labels = tf.range(tf.shape(scores)[0])
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=scores))


def summarize(rows: list[dict], mask: np.ndarray, prefix: str) -> dict:
    selected = [row for row, keep in zip(rows, mask) if keep and row["eligible_targets"] > 0]
    if not selected:
        return {"users": 0, "recall@10": None, "ndcg@10": None,
                "candidate_recall@100": None, "candidate_recall@500": None}
    result = {
        "users": len(selected),
        "eligible_target_pairs": int(sum(row["eligible_targets"] for row in selected)),
        "recall@10": float(np.mean([row[f"{prefix}_recall@10"] for row in selected])),
        "ndcg@10": float(np.mean([row[f"{prefix}_ndcg@10"] for row in selected])),
    }
    return result


def summarize_candidates(rows: list[dict], mask: np.ndarray) -> dict:
    selected = [row for row, keep in zip(rows, mask) if keep and row["eligible_targets"] > 0]
    if not selected:
        return {"users": 0, "candidate_recall@100": None, "candidate_recall@500": None}
    return {
        "users": len(selected),
        "eligible_target_pairs": int(sum(row["eligible_targets"] for row in selected)),
        "candidate_recall@100": float(np.mean([row["retrieval_recall@100"] for row in selected])),
        "candidate_recall@500": float(np.mean([row["retrieval_recall@500"] for row in selected])),
    }


def metric_at_k(recommendations: list[int], relevant: set[int], k: int) -> tuple[float, float]:
    if not relevant:
        return 0.0, 0.0
    top = recommendations[:k]
    hits = [index for index, item in enumerate(top) if int(item) in relevant]
    recall = len(hits) / len(relevant)
    dcg = sum(1.0 / np.log2(rank + 2) for rank in hits)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return float(recall), float(dcg / ideal if ideal else 0.0)


def top_unseen(candidates, seen: set[int], limit: int, allowed: set[int] | None = None,
               excluded: set[int] | None = None) -> list[int]:
    """Take a bounded ordered prefix without scanning the full catalog into memory."""
    if limit <= 0:
        return []
    selected = []
    excluded = excluded or set()
    for candidate in candidates:
        item = int(candidate)
        if item in seen or item in excluded or (allowed is not None and item not in allowed):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    faiss.omp_set_num_threads(4)
    OUTPUT.mkdir(parents=True, exist_ok=False)

    data_hash = prepare_dataset()
    print("ratings SHA-256:", data_hash)
    print("TensorFlow:", tf.__version__)
    print("TFRS:", tfrs.__version__)
    print("FAISS:", faiss.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("Fixed split: train timestamp <", TRAIN_CUTOFF,
          "; validation:", TRAIN_CUTOFF, "<= timestamp <", VALIDATION_END)

    columns = ["userId", "movieId", "rating", "timestamp"]
    positive_parts = []
    popularity_counts = Counter()
    training_catalog = set()
    validation_users = set()
    train_rows = validation_rows = 0
    for chunk in pd.read_csv(RATINGS_PATH, usecols=columns, chunksize=CHUNK_SIZE):
        train = chunk.loc[chunk.timestamp < TRAIN_CUTOFF]
        validation = chunk.loc[(chunk.timestamp >= TRAIN_CUTOFF) &
                               (chunk.timestamp < VALIDATION_END)]
        train_rows += len(train)
        validation_rows += len(validation)
        training_catalog.update(map(int, train.movieId.unique()))
        positive = train.loc[train.rating >= POSITIVE_RATING, ["userId", "movieId"]]
        if not positive.empty:
            positive_parts.append(positive)
            popularity_counts.update({int(item): int(count)
                                      for item, count in positive.movieId.value_counts().items()})
        validation_users.update(map(int, validation.loc[
            validation.rating >= POSITIVE_RATING, "userId"].unique()))

    train_positive = pd.concat(positive_parts, ignore_index=True)
    if train_rows != 17_500_066 or validation_rows != 2_500_010:
        raise ValueError(f"Unexpected split rows: train={train_rows}, validation={validation_rows}")
    if train_positive.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate positive user-item pairs found; refusing to train.")
    user_ids = np.sort(train_positive.userId.unique().astype(np.int64))
    item_ids = np.sort(train_positive.movieId.unique().astype(np.int64))
    user_lookup = {int(user): position for position, user in enumerate(user_ids)}
    item_lookup = {int(item): position for position, item in enumerate(item_ids)}
    train_user_ids = set(map(int, user_ids))
    # Keep the catalog definition aligned with baseline.py: every item rated
    # in training is eligible, including items with zero rating>=4 positives.
    full_catalog = training_catalog

    cohort = set(map(int, validation_users))
    histories = {user: set() for user in cohort}
    targets = {user: set() for user in cohort}
    positive_target_items = set()
    for chunk in pd.read_csv(RATINGS_PATH, usecols=columns, chunksize=CHUNK_SIZE):
        selected = chunk.loc[chunk.userId.isin(cohort)]
        prior = selected.loc[selected.timestamp < TRAIN_CUTOFF]
        for user, items in prior.groupby("userId").movieId.agg(set).items():
            histories[int(user)].update(map(int, items))
        held = selected.loc[(selected.timestamp >= TRAIN_CUTOFF) &
                            (selected.timestamp < VALIDATION_END) &
                            (selected.rating >= POSITIVE_RATING)]
        positive_target_items.update(map(int, held.movieId.unique()))
        for user, items in held.groupby("userId").movieId.agg(set).items():
            targets[int(user)].update(map(int, items))

    train_user_idx = np.searchsorted(user_ids, train_positive.userId.to_numpy(np.int64)).astype(np.int32)
    train_item_idx = np.searchsorted(item_ids, train_positive.movieId.to_numpy(np.int64)).astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices({"user": train_user_idx, "item": train_item_idx})
    dataset = dataset.shuffle(min(len(train_positive), 100_000), seed=SEED).batch(BATCH_SIZE).prefetch(1)
    model = TwoTower(len(user_ids), len(item_ids))
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
    print(f"Training pairs={len(train_positive):,}; users={len(user_ids):,}; indexed items={len(item_ids):,}")
    start = time.time()
    history = model.fit(dataset, epochs=EPOCHS, verbose=2)
    training_seconds = time.time() - start

    user_vectors = model.user_tower(np.arange(len(user_ids), dtype=np.int32)).numpy().astype(np.float32)
    item_vectors = model.item_tower(np.arange(len(item_ids), dtype=np.int32)).numpy().astype(np.float32)
    if not np.isfinite(user_vectors).all() or not np.isfinite(item_vectors).all():
        raise ValueError("Non-finite embeddings; refusing to evaluate or save this run.")
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(item_vectors)
    popular_ids = np.array(sorted(full_catalog,
                                  key=lambda item: (-popularity_counts.get(item, 0), item)), dtype=np.int64)
    indexed_set = set(map(int, item_ids))
    warm_users = [user for user in sorted(cohort) if user in user_lookup]
    warm_vectors = np.stack([user_vectors[user_lookup[user]] for user in warm_users])
    search_k = min(index.ntotal, 1000)
    _, positions = index.search(warm_vectors, search_k)
    retrieval_lists = {}
    for row, user in enumerate(warm_users):
        seen = histories[user]
        retrieval_lists[user] = [int(item_ids[pos]) for pos in positions[row]
                                 if pos >= 0 and int(item_ids[pos]) not in seen]

    rows = []
    recommendations = []
    for user in sorted(cohort):
        seen = histories[user]
        relevant = targets[user] - seen
        if not relevant:
            continue
        has_embedding = user in user_lookup
        raw = retrieval_lists.get(user, [])[:500]
        hybrid = list(raw)
        hybrid.extend(top_unseen(popular_ids, seen, max(0, 500 - len(hybrid)),
                                 excluded=set(hybrid)))
        pop = top_unseen(popular_ids, seen, 500)
        r10, n10 = metric_at_k(hybrid, relevant, 10)
        p10, pn10 = metric_at_k(pop, relevant, 10)
        r100, _ = metric_at_k(raw, relevant, 100)
        r500, _ = metric_at_k(raw, relevant, 500)
        same_catalog_pop = top_unseen(popular_ids, seen, 500, allowed=indexed_set)
        sp100, _ = metric_at_k(same_catalog_pop, relevant & indexed_set, 100)
        mr100, _ = metric_at_k(raw, relevant & indexed_set, 100)
        rows.append({"userId": user, "has_embedding": has_embedding,
                     "has_prior_history": bool(seen), "eligible_targets": len(relevant),
                     "targets_in_index": len(relevant & indexed_set),
                     "hybrid_recall@10": r10, "hybrid_ndcg@10": n10,
                     "popularity_recall@10": p10, "popularity_ndcg@10": pn10,
                     "retrieval_recall@100": r100, "retrieval_recall@500": r500,
                     "matched_retrieval_recall@100": mr100,
                     "matched_popularity_recall@100": sp100})
        recommendations.append({"userId": user, "has_embedding": has_embedding,
                                "retrieval_top10": hybrid[:10], "popularity_top10": pop[:10]})

    row_frame = pd.DataFrame(rows)
    embedded = row_frame.has_embedding.to_numpy(dtype=bool)
    all_users = np.ones(len(row_frame), dtype=bool)
    results = {
        "run": "two-tower baseline, audited 70/80 time split",
        "data_sha256": data_hash,
        "train_cutoff_timestamp": TRAIN_CUTOFF,
        "validation_end_timestamp": VALIDATION_END,
        "positive_rating_threshold": POSITIVE_RATING,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "training_positive_pairs": len(train_positive),
        "training_users_with_positive_pairs": len(user_ids),
        "training_positive_items_indexed": len(item_ids),
        "full_training_catalog_items": len(full_catalog),
        "catalog_coverage": len(indexed_set) / len(full_catalog),
        "positive_target_items": len(positive_target_items),
        "positive_target_items_in_full_training_catalog": len(positive_target_items & full_catalog),
        "positive_target_items_absent_from_full_training_catalog": len(positive_target_items - full_catalog),
        "validation_positive_users": len(cohort),
        "evaluated_users_with_unseen_positive_targets": len(row_frame),
        "warm_users": int(embedded.sum()),
        "cold_or_unembedded_users": int((~embedded).sum()),
        "warm": {"retrieval_then_popularity_fill": summarize(rows, embedded, "hybrid"),
                 "candidate_generation": summarize_candidates(rows, embedded),
                 "popularity": summarize(rows, embedded, "popularity")},
        "all": {"retrieval_then_popularity_fill": summarize(rows, all_users, "hybrid"),
                "candidate_generation": summarize_candidates(rows, all_users),
                "popularity": summarize(rows, all_users, "popularity")},
        "warm_same_indexed_catalog_candidate_recall@100": {
            "retrieval": float(row_frame.loc[embedded, "matched_retrieval_recall@100"].mean()),
            "popularity": float(row_frame.loc[embedded, "matched_popularity_recall@100"].mean()),
            "users": int(embedded.sum()),
            "target_pairs": int(row_frame.loc[embedded, "targets_in_index"].sum()),
        },
        "model": {"dimensions": DIMENSIONS, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                  "optimizer": "Adagrad(lr=0.05)", "training_seconds": training_seconds,
                  "loss_by_epoch": [float(x) for x in history.history["loss"]]},
        "official_final_10_percent_used": False,
        "old_80_90_validation_used": False,
        "limitations": [
            "This is one development run on the audited 70/80 validation window, not an independent external test.",
            "The target is rating >= 4, a preference proxy rather than clicks or CTR.",
            "The model is ID-only; users/items absent from training use popularity fallback.",
            "FAISS IndexFlatIP is exact inner-product search.",
        ],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    row_frame.to_csv(OUTPUT / "per_user_predictions.csv", index=False)
    (OUTPUT / "top10_recommendations.json").write_text(json.dumps(recommendations), encoding="utf-8")
    np.savez_compressed(OUTPUT / "embeddings.npz", user_ids=user_ids, item_ids=item_ids,
                        user_vectors=user_vectors, item_vectors=item_vectors)
    faiss.write_index(index, str(OUTPUT / "items.faiss"))
    model.user_tower.save(OUTPUT / "user_tower.keras")
    model.item_tower.save(OUTPUT / "item_tower.keras")
    print(json.dumps(results, indent=2))
    print("Saved artifacts:", OUTPUT)


if __name__ == "__main__":
    main()
