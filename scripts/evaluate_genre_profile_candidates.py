"""Evaluate a fixed genre-profile candidate baseline without training."""

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


def movie_genre_vectors(movies_path, catalog_ids):
    movies = pd.read_csv(movies_path, usecols=["movieId", "genres"])
    catalog_set = set(map(int, catalog_ids))
    genre_lists = {}
    genre_names = set()
    for row in movies.itertuples(index=False):
        movie_id = int(row.movieId)
        if movie_id not in catalog_set:
            continue
        genres = [g for g in str(row.genres).split("|")
                  if g and g != "(no genres listed)"]
        genre_lists[movie_id] = genres
        genre_names.update(genres)

    names = sorted(genre_names)
    indices = {name: i for i, name in enumerate(names)}
    vectors = np.zeros((len(catalog_ids), len(names)), dtype=np.float32)
    for row, movie_id in enumerate(catalog_ids):
        genres = genre_lists.get(int(movie_id), [])
        if genres:
            vectors[row, [indices[g] for g in genres]] = 1.0 / len(genres)
    return names, vectors


def load_panel(ratings_path, users, train_cutoff, validation_cutoff):
    histories = defaultdict(set)
    positive_genre_counts = defaultdict(lambda: defaultdict(float))
    targets = defaultdict(set)
    columns = ["userId", "movieId", "rating", "timestamp"]
    for chunk in pd.read_csv(ratings_path, usecols=columns, chunksize=500_000):
        selected = chunk[chunk.userId.isin(users)]
        history = selected[selected.timestamp < train_cutoff]
        for user, items in history.groupby("userId").movieId.agg(set).items():
            histories[int(user)].update(map(int, items))
        positives = history[history.rating >= 4.0]
        for user, items in positives.groupby("userId").movieId.agg(set).items():
            for movie_id in items:
                positive_genre_counts[int(user)][int(movie_id)] += 1.0
        validation = selected[(selected.timestamp >= train_cutoff) &
                              (selected.timestamp < validation_cutoff) &
                              (selected.rating >= 4.0)]
        for user, items in validation.groupby("userId").movieId.agg(set).items():
            targets[int(user)].update(map(int, items))

    unseen_targets = {int(user): targets[int(user)] - histories[int(user)]
                      for user in users}
    return histories, positive_genre_counts, unseen_targets


def top_unseen(items, seen, limit):
    selected = []
    for item in items:
        item = int(item)
        if item not in seen:
            selected.append(item)
            if len(selected) == limit:
                break
    return selected


def recall_summary(recommendations, targets, cutoffs):
    eligible = [u for u, relevant in targets.items() if relevant]
    result = {"users": len(eligible),
              "relevant_user_item_pairs": int(sum(len(targets[u]) for u in eligible))}
    for k in cutoffs:
        values = [len(set(recommendations[u][:k]) & targets[u]) / len(targets[u])
                  for u in eligible]
        result[f"recall@{k}"] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--popularity", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--retrieval-diagnostic", type=Path, required=True,
                        help="Per-user metrics from evaluate_candidate_recall.py.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[100, 200, 500])
    args = parser.parse_args()

    if not args.cutoffs or any(k <= 0 for k in args.cutoffs):
        parser.error("Candidate cutoffs must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel, usecols=["userId", "has_prior_history"])
    users = sorted(set(map(int, panel.userId)))
    if len(users) != len(panel):
        raise ValueError("The saved validation panel contains duplicate users.")
    prior_history = {int(row.userId): bool(row.has_prior_history)
                     for row in panel.itertuples(index=False)}

    popularity_frame = pd.read_csv(args.popularity)
    popularity_frame = popularity_frame.sort_values(
        ["positive_count", "movieId"], ascending=[False, True], kind="stable")
    catalog_ids = popularity_frame.movieId.to_numpy(dtype=np.int64)
    pop_order = catalog_ids
    pop_counts = popularity_frame.positive_count.to_numpy(dtype=np.int64)
    genre_names, genre_vectors = movie_genre_vectors(args.movies, catalog_ids)
    item_index = {int(movie_id): i for i, movie_id in enumerate(catalog_ids)}

    histories, positive_genre_counts, targets = load_panel(
        args.ratings, users,
        int(baseline["split"]["train_before_timestamp"]),
        int(baseline["split"]["test_from_timestamp"]))
    if any(not targets[u] for u in users):
        raise ValueError("At least one user in the saved panel has no unseen validation positives.")

    cutoffs = sorted(set(args.cutoffs))
    max_k = max(cutoffs)
    popularity_recs = {}
    genre_recs = {}
    rows = []
    for user in users:
        seen = histories[int(user)]
        positive_items = positive_genre_counts[int(user)]
        preference = np.zeros(len(genre_names), dtype=np.float32)
        for movie_id, count in positive_items.items():
            position = item_index.get(int(movie_id))
            if position is not None:
                preference += count * genre_vectors[position]
        has_profile = bool(preference.sum() > 0)

        if has_profile:
            preference /= preference.sum()
            genre_scores = genre_vectors @ preference
            # Genre affinity defines the content score; popularity and movie ID
            # only break ties. No validation-tuned blend or threshold is used.
            order = np.lexsort((catalog_ids, -pop_counts, -genre_scores))
            genre_list = top_unseen(catalog_ids[order], seen, max_k)
        else:
            genre_list = top_unseen(pop_order, seen, max_k)
        popularity_list = top_unseen(pop_order, seen, max_k)
        genre_recs[int(user)] = genre_list
        popularity_recs[int(user)] = popularity_list

        relevant = targets[int(user)]
        row = {"userId": int(user), "has_prior_history": prior_history[int(user)],
               "has_positive_genre_profile": has_profile,
               "training_history_items": len(seen),
               "unseen_positive_count": len(relevant)}
        for k in cutoffs:
            row[f"genre_recall@{k}"] = len(set(genre_list[:k]) & relevant) / len(relevant)
            row[f"popularity_recall@{k}"] = len(set(popularity_list[:k]) & relevant) / len(relevant)
        rows.append(row)

    retrieval_rows = pd.read_csv(args.retrieval_diagnostic, usecols=["userId"] +
                                 [f"retrieval_recall@{k}" for k in cutoffs])
    per_user = pd.DataFrame(rows).merge(retrieval_rows, on="userId", how="left",
                                        validate="one_to_one")
    if per_user.filter(like="retrieval_recall@").isna().any().any():
        raise ValueError("Could not align retrieval metrics to the saved validation panel.")

    warm_users = [u for u in users if prior_history[u]]
    cold_users = [u for u in users if not prior_history[u]]
    profile_users = [u for u in users
                     if per_user.loc[per_user.userId == u, "has_positive_genre_profile"].iloc[0]]
    # The saved retrieval diagnostic stores per-user recall values, so aggregate
    # those values directly rather than rerunning the tower or index.
    def prior_recall(group, prefix, k):
        return float(group[f"{prefix}_recall@{k}"].mean()) if len(group) else None

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(output / "per_user_metrics.csv", index=False)
    report = {
        "evaluation": "fixed genre-profile candidate baseline; no training",
        "hypothesis": (
            "A per-user genre profile computed only from pre-cutoff ratings >= 4 "
            "can improve warm-user candidate recall over global popularity."
        ),
        "profile_definition": (
            "Normalize the user's counts of positive training interactions across "
            "the genres of those movies; score catalog items by the sum of the "
            "profile weights for their genres. Break score ties by training "
            "positive popularity, then movieId. Users without a positive genre "
            "profile use popularity."
        ),
        "panel": str(args.panel),
        "panel_users": len(users),
        "panel_reused_during_prior_development": True,
        "official_test_used": False,
        "training_or_downloads": False,
        "cutoffs": cutoffs,
        "genres": genre_names,
        "users_with_prior_history": len(warm_users),
        "true_cold_users": len(cold_users),
        "users_with_positive_genre_profile": len(profile_users),
        "targets": "unseen validation-period ratings >= 4; temporal cutoffs from baseline report",
        "genre_profile_all_users": recall_summary(genre_recs, targets, cutoffs),
        "popularity_all_users": recall_summary(popularity_recs, targets, cutoffs),
        "genre_profile_warm_users": recall_summary(
            {u: genre_recs[u] for u in warm_users}, {u: targets[u] for u in warm_users}, cutoffs),
        "popularity_warm_users": recall_summary(
            {u: popularity_recs[u] for u in warm_users}, {u: targets[u] for u in warm_users}, cutoffs),
        "genre_profile_true_cold_users": recall_summary(
            {u: genre_recs[u] for u in cold_users}, {u: targets[u] for u in cold_users}, cutoffs),
        "popularity_true_cold_users": recall_summary(
            {u: popularity_recs[u] for u in cold_users}, {u: targets[u] for u in cold_users}, cutoffs),
        "retrieval_diagnostic_all_users": {
            f"recall@{k}": float(per_user[f"retrieval_recall@{k}"].mean()) for k in cutoffs},
        "retrieval_diagnostic_warm_users": {
            f"recall@{k}": prior_recall(per_user[per_user.has_prior_history], "retrieval", k)
            for k in cutoffs},
        "ratings_sha256": baseline.get("ratings_sha256"),
        "genre_metadata_sha256": sha256(args.movies),
        "prior_two_tower_warning": (
            "Development comparison only. The same validation cohort was reused; "
            "these values are not independent test results."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
