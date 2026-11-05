"""Measure saved model vocabulary compatibility; does not read interactions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import faiss


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retriever', type=Path, required=True)
    parser.add_argument('--ranker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    embeddings = args.retriever / 'embeddings.npz'
    index_file = args.retriever / 'items.faiss'
    mapping = args.ranker / 'feature_mapping.npz'
    with np.load(embeddings, allow_pickle=False) as a, np.load(mapping, allow_pickle=False) as b:
        users, items = a['user_ids'], a['item_ids']
        rank_users, rank_items = b['user_ids'], b['item_ids']
        for ids in (users, items, rank_users, rank_items):
            if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
                raise ValueError('ID vocabulary must be nonempty, sorted and unique')
        index = faiss.read_index(str(index_file))
        if not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(items):
            raise ValueError('Expected matching exact inner-product index')
        if not np.array_equal(index.reconstruct_n(0, index.ntotal), a['item_vectors']):
            raise ValueError('FAISS vectors differ from saved embeddings')
        shared_users = np.intersect1d(users, rank_users, assume_unique=True)
        shared_items = np.intersect1d(items, rank_items, assume_unique=True)
        if not len(shared_users) or len(items) < 100:
            raise ValueError('Insufficient shared users or indexed items')
        faiss.omp_set_num_threads(2)
        pos = np.searchsorted(users, shared_users)
        _, hit = index.search(np.asarray(a['user_vectors'][pos], dtype=np.float32), 100)
        if np.any(hit < 0):
            raise ValueError('FAISS returned invalid candidate indices')
        known_count = np.isin(items[hit], rank_items).sum(axis=1)
        result = {
            'artifact_sha256': {str(p): digest(p) for p in (embeddings,index_file,mapping)},
            'retriever_users': int(len(users)), 'ranker_users': int(len(rank_users)),
            'shared_users': int(len(shared_users)),
            'retriever_items': int(len(items)), 'ranker_items': int(len(rank_items)),
            'shared_items': int(len(shared_items)), 'candidates_per_user': 100,
            'all_candidates_known_users': int(np.sum(known_count == 100)),
            'known_candidate_fraction': float(known_count.mean() / 100),
            'known_per_user_min': int(known_count.min()),
            'known_per_user_median': float(np.median(known_count)),
            'known_per_user_max': int(known_count.max()),
            'limitations': ['Raw FAISS top 100; no user-history filter was applied.',
                'Vocabulary overlap is not recommendation quality or a model evaluation.',
                'No ratings, labels, or held-out interactions were read.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
