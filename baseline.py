"""A small, reproducible popularity baseline for MovieLens ratings."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def load_ratings(path):
    ratings = pd.read_csv(path, usecols=['userId', 'movieId', 'rating', 'timestamp'])
    if ratings.empty or ratings.isna().any().any():
        raise ValueError('Ratings must be nonempty and have no missing values.')
    for col in ['userId', 'movieId', 'timestamp']:
        values = pd.to_numeric(ratings[col], errors='raise')
        if not ((values >= 0) & (values % 1 == 0)).all():
            raise ValueError(f'{col} must contain nonnegative integers.')
        ratings[col] = values.astype('int64')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='raise')
    if not ratings.rating.between(0.5, 5).all():
        raise ValueError('Expected MovieLens ratings between 0.5 and 5.')
    if ratings.duplicated(['userId', 'movieId']).any():
        raise ValueError('Repeated user/movie pairs need an explicit update policy.')
    return ratings.sort_values('timestamp', kind='stable').reset_index(drop=True)


def split_by_time(ratings):
    """Keep equal timestamps together, including at either boundary."""
    times = ratings.timestamp.sort_values().reset_index(drop=True)
    train_end = int(times.iloc[int(len(times) * 0.8)])
    validation_end = int(times.iloc[int(len(times) * 0.9)])
    train = ratings[ratings.timestamp < train_end].copy()
    validation = ratings[(ratings.timestamp >= train_end) &
                         (ratings.timestamp < validation_end)].copy()
    test = ratings[ratings.timestamp >= validation_end].copy()
    if any(part.empty for part in [train, validation, test]):
        raise ValueError('Not enough distinct timestamps for three nonempty splits.')
    return train, validation, test, train_end, validation_end


def popularity_order(train, positive_rating=4.0):
    # Include all training items, even those with no positive ratings.
    counts = train[train.rating >= positive_rating].groupby('movieId').size()
    items = pd.DataFrame({'movieId': sorted(train.movieId.unique())})
    items['positive_count'] = items.movieId.map(counts).fillna(0).astype(int)
    return items.sort_values(['positive_count', 'movieId'], ascending=[False, True])


def ranking_scores(recommendations, relevant, k):
    if k <= 0 or not relevant:
        raise ValueError('Use positive K and at least one relevant item.')
    hits = [int(item in relevant) for item in recommendations[:k]]
    recall = sum(hits) / len(relevant)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return recall, dcg / ideal


def evaluate(ranked_items, history, heldout, k=10, positive_rating=4.0):
    seen = history.groupby('userId').movieId.agg(set).to_dict()
    relevant = heldout[heldout.rating >= positive_rating].groupby('userId').movieId.agg(set)
    catalog = set(ranked_items)
    rows = []
    recommended_items = set()
    for user, positives in relevant.items():
        prior = seen.get(user, set())
        positives = positives - prior
        if not positives:
            continue
        recommendations = []
        for item in ranked_items:
            if item not in prior:
                recommendations.append(item)
                if len(recommendations) == k:
                    break
        recall, ndcg = ranking_scores(recommendations, positives, k)
        recommended_items.update(recommendations)
        rows.append({'userId': int(user), 'cold_start': user not in seen,
                     'relevant_items': len(positives),
                     'unseen_catalog_positives': len(positives - catalog),
                     'recall': recall, 'ndcg': ndcg,
                     'recommendations': ' '.join(map(str, recommendations))})
    if not rows:
        raise ValueError('No users with positive held-out ratings to evaluate.')
    per_user = pd.DataFrame(rows)
    def summarize(frame):
        return {'users': len(frame), f'recall@{k}': float(frame.recall.mean()) if len(frame) else None,
                f'ndcg@{k}': float(frame.ndcg.mean()) if len(frame) else None}
    metrics = summarize(per_user)
    metrics['warm_users'] = summarize(per_user[~per_user.cold_start])
    metrics['cold_users'] = summarize(per_user[per_user.cold_start])
    metrics['users_without_positive_targets'] = int(heldout.userId.nunique() - len(per_user))
    metrics['unseen_catalog_positive_fraction'] = float(
        per_user.unseen_catalog_positives.sum() / per_user.relevant_items.sum())
    metrics['recommendation_catalog_coverage'] = len(recommended_items) / len(catalog)
    return metrics, per_user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/baseline'))
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--positive-rating', type=float, default=4.0)
    parser.add_argument('--dataset-label', required=True)
    args = parser.parse_args()
    if args.k <= 0 or not 0.5 <= args.positive_rating <= 5:
        parser.error('K must be positive; positive-rating must be between 0.5 and 5.')
    ratings = load_ratings(args.ratings)
    train, validation, test, cutoff1, cutoff2 = split_by_time(ratings)
    ranked = popularity_order(train, args.positive_rating)
    # Freeze the model and catalog at the training boundary. Test users' histories
    # include validation interactions, but these do not change popularity scores.
    validation_metrics, validation_users = evaluate(
        ranked.movieId.tolist(), train, validation, args.k, args.positive_rating)
    test_metrics, test_users = evaluate(
        ranked.movieId.tolist(), pd.concat([train, validation]), test,
        args.k, args.positive_rating)
    checksum = hashlib.sha256()
    with args.ratings.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    report = {'dataset': args.dataset_label, 'ratings_sha256': checksum.hexdigest(),
              'positive_rating': args.positive_rating, 'k': args.k,
              'split': {'train': len(train), 'validation': len(validation), 'test': len(test),
                        'train_before_timestamp': cutoff1, 'test_from_timestamp': cutoff2},
              'model': 'Training positive-rating count; movieId breaks ties',
              'evaluation': 'Full training catalog; observed items excluded; binary relevance; macro user average',
              'validation': validation_metrics, 'test': test_metrics}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    ranked.to_csv(args.output / 'popularity.csv', index=False)
    validation_users.to_csv(args.output / 'validation_users.csv', index=False)
    test_users.to_csv(args.output / 'test_users.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
