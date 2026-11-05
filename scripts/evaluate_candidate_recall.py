"""Measure saved retrieval candidates at ranking-stage cutoffs; no training."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panel_data(ratings_path, panel_users, train_cutoff, validation_cutoff):
    """Read only the selected users' histories and relevant validation items."""
    histories = defaultdict(set)
    targets = defaultdict(set)
    columns = ["userId", "movieId", "rating", "timestamp"]

    for chunk in pd.read_csv(ratings_path, usecols=columns, chunksize=500_000):
        selected = chunk[chunk.userId.isin(panel_users)]
        history = selected[selected.timestamp < train_cutoff]
        for user, items in history.groupby("userId").movieId.agg(set).items():
            histories[int(user)].update(map(int, items))

        relevant = selected[(selected.timestamp >= train_cutoff) &
                            (selected.timestamp < validation_cutoff) &
                            (selected.rating >= 4.0)]
        for user, items in relevant.groupby("userId").movieId.agg(set).items():
            targets[int(user)].update(map(int, items))

    relevant_by_user = {}
    for user in panel_users:
        unseen = targets[int(user)] - histories[int(user)]
        if unseen:
            relevant_by_user[int(user)] = unseen
    return histories, relevant_by_user


def top_unseen(items, seen, limit):
    selected = []
    for item in items:
        item = int(item)
        if item not in seen:
            selected.append(item)
            if len(selected) == limit:
                break
    return selected


def summarize(recommendations, relevant_by_user, cutoffs):
    users = list(relevant_by_user)
    summary = {"users": len(users),
               "relevant_user_item_pairs": int(sum(len(v) for v in relevant_by_user.values()))}
    for cutoff in cutoffs:
        per_user = [len(set(recommendations[user][:cutoff]) & relevant_by_user[user]) /
                    len(relevant_by_user[user]) for user in users]
        summary[f"recall@{cutoff}"] = float(np.mean(per_user)) if per_user else None
    return summary


def summarize_index_target_coverage(rows):
    """Report how much of the positive target set is even in the saved index."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"users": 0, "positive_pairs": 0,
                "macro_user_average": None, "positive_pair_weighted": None}
    positive_pairs = int(frame.unseen_positive_count.sum())
    weighted = (frame.retrieval_index_positive_coverage *
                frame.unseen_positive_count).sum() / positive_pairs
    return {"users": int(len(frame)), "positive_pairs": positive_pairs,
            "macro_user_average": float(frame.retrieval_index_positive_coverage.mean()),
            "positive_pair_weighted": float(weighted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True,
                        help="Saved retrieval run containing metrics.json and embeddings.npz.")
    parser.add_argument("--panel", type=Path, required=True,
                        help="Saved validation_users.csv for the already-used development cohort.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; a non-empty directory is never overwritten.")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[100, 200, 500])
    args = parser.parse_args()

    if not args.cutoffs or any(k <= 0 for k in args.cutoffs):
        parser.error("Candidate cutoffs must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")

    run = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    panel_frame = pd.read_csv(args.panel, usecols=["userId", "has_prior_history"])
    panel_users = sorted(set(map(int, panel_frame.userId)))
    if len(panel_users) != len(panel_frame):
        raise ValueError("The saved panel contains duplicate users.")
    prior_history = {int(row.userId): bool(row.has_prior_history)
                     for row in panel_frame.itertuples(index=False)}

    arrays = np.load(args.run / "embeddings.npz", allow_pickle=False)
    user_ids = arrays["user_ids"].astype(np.int64, copy=False)
    item_ids = arrays["item_ids"].astype(np.int64, copy=False)
    user_vectors = arrays["user_vectors"].astype(np.float32, copy=False)
    item_vectors = arrays["item_vectors"].astype(np.float32, copy=False)
    popularity = arrays["popularity"].astype(np.int64, copy=False)
    user_lookup = {int(user): i for i, user in enumerate(user_ids)}

    histories, relevant_by_user = load_panel_data(
        args.ratings, panel_users,
        int(baseline["split"]["train_before_timestamp"]),
        int(baseline["split"]["test_from_timestamp"]))
    if len(relevant_by_user) != len(panel_users):
        raise ValueError(
            "Reconstructed eligible-user count differs from the saved panel; "
            "refusing to score a different cohort."
        )

    cutoffs = sorted(set(args.cutoffs))
    max_k = max(cutoffs)
    retrieval_recs = {}
    popularity_recs = {}
    rows = []
    for user in panel_users:
        seen = histories[int(user)]
        position = user_lookup.get(int(user))
        if position is None:
            retrieval = top_unseen(popularity, seen, max_k)
            has_embedding = False
        else:
            scores = item_vectors @ user_vectors[position]
            order = np.argsort(-scores, kind="stable")
            ranked_items = item_ids[order]
            retrieval = top_unseen(ranked_items, seen, max_k)
            has_embedding = True

        popularity_list = top_unseen(popularity, seen, max_k)
        retrieval_recs[int(user)] = retrieval
        popularity_recs[int(user)] = popularity_list
        relevant = relevant_by_user[int(user)]
        row = {"userId": int(user), "has_retrieval_embedding": has_embedding,
               "has_prior_history": prior_history[int(user)],
               "unseen_positive_count": len(relevant),
               "retrieval_index_positive_coverage": len(relevant & set(map(int, item_ids))) / len(relevant)}
        for cutoff in cutoffs:
            row[f"retrieval_recall@{cutoff}"] = (
                len(set(retrieval[:cutoff]) & relevant) / len(relevant))
            row[f"popularity_recall@{cutoff}"] = (
                len(set(popularity_list[:cutoff]) & relevant) / len(relevant))
        rows.append(row)

    trained_users = [u for u in panel_users if u in user_lookup]
    fallback_users = [u for u in panel_users if u not in user_lookup]
    warm_users = [u for u in panel_users if prior_history[u]]
    true_cold_users = [u for u in panel_users if not prior_history[u]]
    warm_without_embedding = [u for u in warm_users if u not in user_lookup]
    if any(not prior_history[u] for u in trained_users):
        raise ValueError("A user without prior history unexpectedly has a retrieval embedding.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_user_metrics.csv", index=False)
    report = {
        "evaluation": "inference-only candidate recall diagnostic",
        "panel": str(args.panel),
        "panel_users": len(panel_users),
        "eligible_users": len(relevant_by_user),
        "panel_reused_during_prior_development": True,
        "official_test_used": False,
        "training_or_downloads": False,
        "target_definition": "unseen validation-period ratings >= 4; global temporal cutoffs from baseline report",
        "candidate_cutoffs": cutoffs,
        "retrieval_run": str(args.run),
        "retrieval_stage": run.get("stage"),
        "training_interactions": run.get("training_interactions", run.get("training_positive_rows")),
        "indexed_items": int(len(item_ids)),
        "training_catalog_items": int(run["full_training_catalog_items"]),
        "indexed_catalog_coverage": float(len(item_ids) / run["full_training_catalog_items"]),
        "users_with_embedding": len(trained_users),
        "users_without_embedding_use_popularity": len(panel_users) - len(trained_users),
        "users_with_prior_history": len(warm_users),
        "true_cold_users": len(true_cold_users),
        "warm_users_without_embedding": len(warm_without_embedding),
        "retrieval_all_users": summarize(retrieval_recs, relevant_by_user, cutoffs),
        "popularity_all_users": summarize(popularity_recs, relevant_by_user, cutoffs),
        "retrieval_trained_users": summarize(
            {u: retrieval_recs[u] for u in trained_users},
            {u: relevant_by_user[u] for u in trained_users}, cutoffs),
        "popularity_trained_users": summarize(
            {u: popularity_recs[u] for u in trained_users},
            {u: relevant_by_user[u] for u in trained_users}, cutoffs),
        "retrieval_fallback_users": summarize(
            {u: retrieval_recs[u] for u in fallback_users},
            {u: relevant_by_user[u] for u in fallback_users}, cutoffs),
        "popularity_fallback_users": summarize(
            {u: popularity_recs[u] for u in fallback_users},
            {u: relevant_by_user[u] for u in fallback_users}, cutoffs),
        "retrieval_warm_users": summarize(
            {u: retrieval_recs[u] for u in warm_users},
            {u: relevant_by_user[u] for u in warm_users}, cutoffs),
        "popularity_warm_users": summarize(
            {u: popularity_recs[u] for u in warm_users},
            {u: relevant_by_user[u] for u in warm_users}, cutoffs),
        "retrieval_true_cold_users": summarize(
            {u: retrieval_recs[u] for u in true_cold_users},
            {u: relevant_by_user[u] for u in true_cold_users}, cutoffs),
        "popularity_true_cold_users": summarize(
            {u: popularity_recs[u] for u in true_cold_users},
            {u: relevant_by_user[u] for u in true_cold_users}, cutoffs),
        "retrieval_index_positive_coverage_all_users":
            summarize_index_target_coverage(rows),
        "retrieval_index_positive_coverage_trained_users":
            summarize_index_target_coverage(
            [r for r in rows if r["has_retrieval_embedding"]]),
        "retrieval_item_vectors_sha256": sha256(args.run / "embeddings.npz"),
        "saved_faiss_index_sha256_not_loaded": sha256(args.run / "items.faiss"),
        "panel_file_sha256": sha256(args.panel),
        "ratings_sha256": baseline.get("ratings_sha256"),
        "scoring_note": (
            "Scores were recomputed with NumPy float32 dot products from the saved "
            "user/item vectors. The saved FAISS binary was hashed but not loaded; "
            "binary-index equivalence is therefore not verified."
        ),
        "reused_validation_warning": (
            "Development diagnostic only. The same user cohort was used by earlier "
            "candidate comparisons; these results are not an independent test."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
