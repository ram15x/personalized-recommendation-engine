"""Build a bounded, user-balanced MovieLens sample without loading all ratings."""

import hashlib

import numpy as np
import pandas as pd


def load_streaming_sample(path, train_cutoff, validation_cutoff, seed=42,
                          positive_rating=4.0, interaction_mode='all-ratings',
                          max_interactions=500000, eval_users=2000,
                          interactions_per_user=3, chunksize=500000):
    """Read MovieLens twice, retaining bounded training data and evaluation histories."""
    if max_interactions <= 0 or interactions_per_user <= 0:
        raise ValueError('Streaming mode needs a positive interaction cap and per-user limit.')
    columns = ['userId', 'movieId', 'rating', 'timestamp']
    training_users, validation_users = set(), set()
    positive_training_rows = 0

    # Pass one finds the time-split population and gives every training user a
    # place in the sample before selecting any examples.
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        training = chunk[chunk.timestamp < train_cutoff]
        training_users.update(training.userId.unique().tolist())
        positive_training_rows += int((training.rating >= positive_rating).sum())
        validation = chunk[(chunk.timestamp >= train_cutoff) &
                           (chunk.timestamp < validation_cutoff) &
                           (chunk.rating >= positive_rating)]
        validation_users.update(validation.userId.unique().tolist())
    if not training_users or not validation_users:
        raise ValueError('The supplied split cutoffs produced empty training or validation users.')

    ordered_validation_users = sorted(validation_users)
    if eval_users and len(ordered_validation_users) > eval_users:
        selected = np.random.default_rng(seed).choice(
            ordered_validation_users, size=eval_users, replace=False)
        selected_users = set(map(int, selected))
    else:
        selected_users = set(ordered_validation_users)
    ordered_training_users = sorted(training_users)
    if len(ordered_training_users) > max_interactions:
        sampled_training_users = set(map(int, np.random.default_rng(seed).choice(
            ordered_training_users, size=max_interactions, replace=False)))
    else:
        sampled_training_users = training_users
    per_user = min(interactions_per_user,
                   max_interactions // len(sampled_training_users))

    history = {user: set() for user in selected_users}
    targets = {user: set() for user in selected_users}
    sampled = pd.DataFrame(columns=columns)
    popularity_counts = {}
    catalog = set()
    retained_rows = 0
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)

    # Pass two retains a few recent interactions for each user, all prior items
    # for the fixed validation cohort, and the full training popularity counts.
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        training = chunk[chunk.timestamp < train_cutoff]
        catalog.update(training.movieId.unique().tolist())
        counts = training[training.rating >= positive_rating].groupby('movieId').size()
        for item, count in counts.items():
            popularity_counts[int(item)] = popularity_counts.get(int(item), 0) + int(count)

        candidate_rows = training[training.userId.isin(sampled_training_users)]
        if interaction_mode == 'high-ratings':
            candidate_rows = candidate_rows[candidate_rows.rating >= positive_rating]
        if not candidate_rows.empty:
            candidate_rows = (candidate_rows.sort_values(['userId', 'timestamp'], kind='stable')
                              .groupby('userId', sort=False, group_keys=False)
                              .tail(per_user))
            sampled = (candidate_rows.copy().reset_index(drop=True) if sampled.empty
                       else pd.concat([sampled, candidate_rows], ignore_index=True))
            sampled = (sampled.sort_values(['userId', 'timestamp'], kind='stable')
                       .groupby('userId', sort=False, group_keys=False)
                       .tail(per_user).reset_index(drop=True))
            retained_rows = len(sampled)

        selected = chunk[chunk.userId.isin(selected_users)]
        prior = selected[selected.timestamp < train_cutoff]
        for user, items in prior.groupby('userId').movieId.agg(set).items():
            history[int(user)].update(items)
        heldout = selected[(selected.timestamp >= train_cutoff) &
                           (selected.timestamp < validation_cutoff) &
                           (selected.rating >= positive_rating)]
        for user, items in heldout.groupby('userId').movieId.agg(set).items():
            targets[int(user)].update(items)

    fallback = pd.DataFrame({'movieId': sorted(catalog)})
    fallback['positive_count'] = fallback.movieId.map(popularity_counts).fillna(0).astype(int)
    fallback = fallback.sort_values(['positive_count', 'movieId'],
                                    ascending=[False, True]).reset_index(drop=True)
    history_rows = [{'userId': user, 'movieId': item}
                    for user, items in history.items() for item in items]
    target_rows = [{'userId': user, 'movieId': item, 'rating': positive_rating}
                   for user, items in targets.items() for item in items]
    history_frame = pd.DataFrame(history_rows, columns=['userId', 'movieId'])
    target_frame = pd.DataFrame(target_rows, columns=['userId', 'movieId', 'rating'])
    if retained_rows > max_interactions:
        raise RuntimeError('Streaming training sample exceeded its interaction cap.')
    counts = {'training_users': len(training_users),
              'training_users_selected_for_sampling': len(sampled_training_users),
              'validation_users_with_positive_targets': len(validation_users),
              'evaluated_users_selected': len(selected_users),
              'positive_training_rows': positive_training_rows,
              'retained_training_rows': retained_rows,
              'training_catalog_items': len(catalog)}
    return (sampled[['userId', 'movieId', 'rating']], history_frame,
            target_frame, fallback, counts, digest.hexdigest())
