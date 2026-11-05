"""Train an ID-based two-tower model and export an exact FAISS retrieval index."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from baseline import load_ratings, popularity_order, ranking_scores, split_by_time
from streaming_data import load_streaming_sample


class TwoTower(tfrs.Model):
    def __init__(self, user_count, item_count, dimensions):
        super().__init__()
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(user_count, dimensions),
            tf.keras.layers.Dense(dimensions, activation='relu'),
            tf.keras.layers.Dense(dimensions),
        ])
        self.item_tower = tf.keras.Sequential([
            tf.keras.layers.Embedding(item_count, dimensions),
            tf.keras.layers.Dense(dimensions, activation='relu'),
            tf.keras.layers.Dense(dimensions),
        ])

    def compute_loss(self, features, training=False):
        users = features['user']
        items = features['item']
        query_vectors = self.user_tower(users)
        candidate_vectors = self.item_tower(items)
        scores = tf.linalg.matmul(query_vectors, candidate_vectors, transpose_b=True)

        # Mask other known positive movies from the same user. Also mask the
        # same movie from another row, which is an accidental in-batch hit.
        same_user = tf.equal(users[:, tf.newaxis], users[tf.newaxis, :])
        same_item = tf.equal(items[:, tf.newaxis], items[tf.newaxis, :])
        diagonal = tf.eye(tf.shape(scores)[0], dtype=tf.bool)
        ignore_as_negative = tf.logical_and(tf.logical_or(same_user, same_item),
                                             tf.logical_not(diagonal))
        scores = tf.where(ignore_as_negative,
                          tf.cast(-1e9, scores.dtype), scores)

        labels = tf.range(tf.shape(scores)[0])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=scores)
        return tf.reduce_mean(losses)


def retrieve(index, item_ids, vector, seen, fallback, k=10):
    """Expand the search when filtering would leave too few recommendations."""
    if vector is None:
        return [int(item) for item in fallback if item not in seen][:k]
    search_size = min(index.ntotal, max(100, k))
    selected = []
    while search_size:
        _, positions = index.search(np.asarray(vector, dtype='float32').reshape(1, -1), search_size)
        selected = [int(item_ids[pos]) for pos in positions[0]
                    if pos >= 0 and item_ids[pos] not in seen][:k]
        if len(selected) >= k or search_size == index.ntotal:
            break
        search_size = min(index.ntotal, search_size * 2)
    for item in fallback:
        if len(selected) >= k:
            break
        if item not in seen and item not in selected:
            selected.append(int(item))
    return selected


def score_validation(index, item_ids, user_ids, user_vectors, train, validation, fallback, k, limit, seed,
                     history_by_user=None):
    histories = (history_by_user if history_by_user is not None else
                 train.groupby('userId').movieId.agg(set).to_dict())
    targets = validation[validation.rating >= 4].groupby('userId').movieId.agg(set)
    eligible_count = len(targets)
    if limit and len(targets) > limit:
        chosen = np.random.default_rng(seed).choice(targets.index.to_numpy(), size=limit, replace=False)
        targets = targets.loc[sorted(chosen)]
    user_lookup = {int(user): position for position, user in enumerate(user_ids)}
    rows = []
    for user, relevant in targets.items():
        seen = histories.get(user, set())
        relevant = relevant - seen
        if not relevant:
            continue
        position = user_lookup.get(int(user))
        vector = None if position is None else user_vectors[position]
        recommended = retrieve(index, item_ids, vector, seen, fallback, k)
        baseline = retrieve(index, item_ids, None, seen, fallback, k)
        recall, ndcg = ranking_scores(recommended, relevant, k)
        base_recall, base_ndcg = ranking_scores(baseline, relevant, k)
        rows.append({'userId': int(user), 'has_training_embedding': position is not None,
                     'has_prior_history': user in histories,
                     'recall': recall, 'ndcg': ndcg,
                     'baseline_recall': base_recall, 'baseline_ndcg': base_ndcg,
                     'recommendations': ' '.join(map(str, recommended))})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('Validation contains no positive targets.')
    def summary(part):
        return {'users': len(part), **{
            name: float(part[column].mean()) if len(part) else None
            for name, column in [(f'recall@{k}', 'recall'), (f'ndcg@{k}', 'ndcg'),
                                 (f'popularity_recall@{k}', 'baseline_recall'),
                                 (f'popularity_ndcg@{k}', 'baseline_ndcg')]}}
    return {'all': summary(frame),
            'trained_users': summary(frame[frame.has_training_embedding]),
            'fallback_users': summary(frame[~frame.has_training_embedding]),
            'eligible_positive_users': eligible_count,
            'evaluation_user_limit': limit}, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-label', required=True)
    parser.add_argument('--interaction-mode', choices=['all-ratings', 'high-ratings'], default='all-ratings',
                        help='All ratings model observed interactions; high ratings models explicit preference only.')
    parser.add_argument('--sampling-strategy', choices=['recent-per-user', 'random'],
                        default='recent-per-user',
                        help='When capped, prefer a few recent interactions from every training user.')
    parser.add_argument('--streaming', action='store_true',
                        help='Read the ratings file in bounded chunks using cutoffs from --split-report.')
    parser.add_argument('--split-report', type=Path,
                        help='Baseline metrics JSON containing the established chronological cutoffs.')
    parser.add_argument('--interactions-per-user', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--dimensions', type=int, default=32)
    parser.add_argument('--max-train-positives', type=int, default=500000,
                        help='0 uses every positive training interaction.')
    parser.add_argument('--eval-users', type=int, default=2000,
                        help='0 evaluates every eligible validation user.')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.dimensions, args.k) <= 0:
        parser.error('Epochs, batch size, dimensions and K must be positive.')
    if args.max_train_positives < 0 or args.eval_users < 0 or args.interactions_per_user <= 0:
        parser.error('Data limits cannot be negative.')
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    faiss.omp_set_num_threads(4)
    print('Loading and splitting ratings...', flush=True)
    history_by_user = None
    stream_counts = None
    ratings_sha256 = None
    if args.streaming:
        if args.split_report is None:
            parser.error('--streaming requires --split-report with the baseline split cutoffs.')
        split_report = json.loads(args.split_report.read_text(encoding='utf-8'))
        split_info = split_report['split']
        cutoff1 = int(split_info['train_before_timestamp'])
        cutoff2 = int(split_info['test_from_timestamp'])
        (train, history_frame, validation, fallback_frame, stream_counts,
         ratings_sha256) = load_streaming_sample(
            args.ratings, cutoff1, cutoff2, args.seed, 4.0, args.interaction_mode,
            args.max_train_positives, args.eval_users, args.interactions_per_user)
        history_by_user = history_frame.groupby('userId').movieId.agg(set).to_dict()
        counts = split_info
        available = (stream_counts['positive_training_rows']
                     if args.interaction_mode == 'high-ratings' else counts['train'])
        interactions = train
    else:
        ratings = load_ratings(args.ratings)
        train, validation, test, cutoff1, cutoff2 = split_by_time(ratings)
        counts = {'train': len(train), 'validation': len(validation), 'test': len(test)}
        del ratings, test  # Test remains untouched during model development.
        history_frame = None
        fallback_frame = None
        interactions = train[['userId', 'movieId', 'rating', 'timestamp']]
        if args.interaction_mode == 'high-ratings':
            interactions = interactions[interactions.rating >= 4]
        available = len(interactions)
        if args.max_train_positives and len(interactions) > args.max_train_positives:
            if args.sampling_strategy == 'recent-per-user':
                user_count = interactions.userId.nunique()
                per_user = min(args.interactions_per_user,
                               max(1, args.max_train_positives // user_count))
                interactions = (interactions.sort_values(['userId', 'timestamp'], kind='stable')
                                .groupby('userId', sort=False, group_keys=False)
                                .tail(per_user))
            else:
                interactions = interactions.sample(args.max_train_positives, random_state=args.seed)
        train = train[['userId', 'movieId', 'rating']]
    user_ids = np.sort(interactions.userId.unique())
    item_ids = np.sort(interactions.movieId.unique())
    if len(item_ids) < 2 or len(user_ids) < 2:
        raise ValueError('Training needs at least two users and two items.')
    user_indices = np.searchsorted(user_ids, interactions.userId.to_numpy()).astype('int32')
    item_indices = np.searchsorted(item_ids, interactions.movieId.to_numpy()).astype('int32')
    dataset = tf.data.Dataset.from_tensor_slices({'user': user_indices, 'item': item_indices})
    dataset = dataset.shuffle(min(len(interactions), 100000), seed=args.seed).batch(args.batch_size).prefetch(1)
    model = TwoTower(len(user_ids), len(item_ids), args.dimensions)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
    print(f'Training on {len(interactions):,} of {available:,} {args.interaction_mode} interactions.', flush=True)
    history = model.fit(dataset, epochs=args.epochs, verbose=2)
    user_vectors = model.user_tower(np.arange(len(user_ids), dtype='int32')).numpy().astype('float32')
    item_vectors = model.item_tower(np.arange(len(item_ids), dtype='int32')).numpy().astype('float32')
    if not np.isfinite(user_vectors).all() or not np.isfinite(item_vectors).all():
        raise ValueError('Training produced non-finite embeddings.')
    index = faiss.IndexFlatIP(args.dimensions)
    index.add(item_vectors)
    fallback = (popularity_order(train).movieId.to_numpy() if fallback_frame is None
                else fallback_frame.movieId.to_numpy())
    print('Evaluating validation users against popularity...', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output / 'items.faiss'))
    np.savez_compressed(args.output / 'embeddings.npz', user_ids=user_ids, item_ids=item_ids,
                        user_vectors=user_vectors, item_vectors=item_vectors, popularity=fallback)
    model.user_tower.save(args.output / 'user_tower.keras')
    model.item_tower.save(args.output / 'item_tower.keras')
    print('Saved model and index. Evaluating validation users...', flush=True)
    metrics, users = score_validation(index, item_ids, user_ids, user_vectors,
                                     train, validation, fallback, args.k, args.eval_users, args.seed,
                                     history_by_user)
    users.to_csv(args.output / 'validation_users.csv', index=False)
    report = {'dataset': args.dataset_label,
              'stage': 'pilot' if len(interactions) < available else 'full-training',
              'seed': args.seed, 'epochs': args.epochs, 'dimensions': args.dimensions,
              'training_interactions': len(interactions), 'available_training_interactions': available,
              'interaction_mode': args.interaction_mode,
              'sampling_strategy': ('recent-per-user-streaming' if args.streaming else
                                    args.sampling_strategy if len(interactions) < available
                                    else 'all-training-rows'),
              'trained_users': len(user_ids), 'indexed_items': len(item_ids),
              'full_training_catalog_items': len(fallback), 'split': counts,
              'ratings_sha256': ratings_sha256,
              'streaming': args.streaming,
              'streaming_sample': stream_counts,
              'train_before_timestamp': cutoff1, 'test_from_timestamp': cutoff2,
              'candidate_policy': 'Exact inner-product search over sampled interaction items; fill short lists with training popularity',
              'training_objective': 'One diagonal positive per row; mask same-user and same-item in-batch pairs as negatives.',
              'cold_start_policy': 'Popularity for users without a learned embedding',
              'test_evaluated': False, 'history': history.history, 'validation': metrics}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
