"""Inspect saved recommendations for an anonymized MovieLens user."""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--movies', type=Path, required=True)
    parser.add_argument('--user-id', type=int, required=True)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error('K must be positive.')
    metadata = json.loads((args.model / 'metrics.json').read_text(encoding='utf-8'))
    arrays = np.load(args.model / 'embeddings.npz', allow_pickle=False)
    index = faiss.read_index(str(args.model / 'items.faiss'))
    seen = set()
    for chunk in pd.read_csv(args.ratings, usecols=['userId', 'movieId', 'timestamp'], chunksize=500000):
        history = chunk[(chunk.userId == args.user_id) &
                        (chunk.timestamp < metadata['train_before_timestamp'])]
        seen.update(history.movieId.tolist())
    position = np.flatnonzero(arrays['user_ids'] == args.user_id)
    vector = arrays['user_vectors'][position[0]] if len(position) else None
    result = retrieve(index, arrays['item_ids'], vector, seen, arrays['popularity'], args.k)
    movies = pd.read_csv(args.movies).set_index('movieId')
    print('Source:', 'two-tower retrieval' if vector is not None else 'popularity fallback')
    print('History: ratings before the training cutoff; not a live user profile.')
    for rank, movie in enumerate(result, 1):
        title = movies.loc[movie, 'title'] if movie in movies.index else f'Movie {movie}'
        print(f'{rank:2}. {title} (ID {movie})')


if __name__ == '__main__':
    main()
