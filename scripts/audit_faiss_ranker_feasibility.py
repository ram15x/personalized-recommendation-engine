"""Count time-bounded rating support for a fixed sample of retriever users."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_END = 1466837400
DEV_END = 1515033908
MIDPOINT = (TRAIN_END + DEV_END) // 2
EXPECTED_RATINGS_SHA256 = '9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', required=True, type=Path)
    parser.add_argument('--embeddings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    actual = sha256(args.ratings)
    if actual != EXPECTED_RATINGS_SHA256:
        raise ValueError(f'Unexpected ratings SHA-256: {actual}')
    with np.load(args.embeddings, allow_pickle=False) as data:
        users = data['user_ids']
        if users.ndim != 1 or len(users) < 10000 or np.any(np.diff(users) <= 0):
            raise ValueError('Expected at least 10,000 sorted unique retriever user IDs')
        chosen = np.sort(np.random.default_rng(42).choice(users, size=10000, replace=False))
    counts = np.zeros((len(chosen), 4), dtype=np.int32)
    for chunk in pd.read_csv(args.ratings, usecols=['userId', 'rating', 'timestamp'], chunksize=500000):
        # Drop all out-of-window rows before storing or counting outcomes.
        part = chunk.loc[(chunk.timestamp >= TRAIN_END) & (chunk.timestamp < DEV_END)]
        part = part.loc[part.userId.isin(chosen)]
        if part.empty:
            continue
        positions = np.searchsorted(chosen, part.userId.to_numpy(dtype=np.int64))
        earlier = part.timestamp.to_numpy() < MIDPOINT
        positive = part.rating.to_numpy() >= 4
        np.add.at(counts[:,0], positions[earlier], 1)
        np.add.at(counts[:,1], positions[earlier & positive], 1)
        np.add.at(counts[:,2], positions[~earlier], 1)
        np.add.at(counts[:,3], positions[~earlier & positive], 1)
    result = {
        'protocol': 'docs/faiss-ranker-feasibility-protocol.md',
        'ratings_sha256': actual,
        'embeddings_sha256': sha256(args.embeddings),
        'sample_size': len(chosen), 'seed': 42,
        'train_end': TRAIN_END, 'midpoint': MIDPOINT, 'development_end': DEV_END,
        'early_rating_rows': int(counts[:,0].sum()), 'early_positive_rows': int(counts[:,1].sum()),
        'late_rating_rows': int(counts[:,2].sum()), 'late_positive_rows': int(counts[:,3].sum()),
        'early_users_with_rating': int((counts[:,0] > 0).sum()),
        'late_users_with_rating': int((counts[:,2] > 0).sum()),
        'users_with_rating_in_both_halves': int(((counts[:,0] > 0) & (counts[:,2] > 0)).sum()),
        'users_with_positive_in_both_halves': int(((counts[:,1] > 0) & (counts[:,3] > 0)).sum()),
        'limitations': ['Reused development period; not independent validation.',
            'Ratings counts are upper bounds on retrievable candidate labels.',
            'No final-period outcomes retained; no model training or scoring.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
