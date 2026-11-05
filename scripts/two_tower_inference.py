"""Exact FAISS retrieval from a saved, experimental two-tower pilot."""
from pathlib import Path

import numpy as np


class TwoTowerRetriever:
    def __init__(self, index, user_ids, user_vectors, item_ids, item_vectors):
        self.index = index
        self.user_ids = np.asarray(user_ids, dtype=np.int64)
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.user_vectors = np.asarray(user_vectors, dtype=np.float32)
        item_vectors = np.asarray(item_vectors, dtype=np.float32)
        for ids in (self.user_ids, self.item_ids):
            if ids.ndim != 1 or not len(ids) or np.any(ids <= 0) or np.any(np.diff(ids) <= 0):
                raise ValueError("Saved IDs must be sorted, unique, positive integers.")
        if (self.user_vectors.shape != (len(self.user_ids), index.d)
                or item_vectors.shape != (len(self.item_ids), index.d)
                or index.ntotal != len(self.item_ids)
                or not np.isfinite(self.user_vectors).all()
                or not np.isfinite(item_vectors).all()):
            raise ValueError("Saved vector dimensions or values are invalid.")
        if not np.array_equal(index.reconstruct_n(0, index.ntotal), item_vectors):
            raise ValueError("FAISS index does not match the saved item vectors.")

    @classmethod
    def load(cls, directory: Path):
        import faiss
        directory = Path(directory)
        index = faiss.read_index(str(directory / "items.faiss"))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Expected an exact inner-product FAISS index.")
        with np.load(directory / "embeddings.npz", allow_pickle=False) as data:
            return cls(index, data["user_ids"], data["user_vectors"],
                       data["item_ids"], data["item_vectors"])

    def retrieve(self, user_id: int, seen, limit: int = 100) -> dict:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500.")
        position = int(np.searchsorted(self.user_ids, user_id))
        if position == len(self.user_ids) or self.user_ids[position] != user_id:
            return {"item_ids": [], "scores": [], "reason": "unknown_user"}
        seen_ids = {int(item) for item in seen}
        size = min(self.index.ntotal, max(100, limit))
        while size:
            values, indices = self.index.search(self.user_vectors[position:position + 1], size)
            selected = [(int(self.item_ids[i]), float(score))
                        for i, score in zip(indices[0], values[0])
                        if i >= 0 and int(self.item_ids[i]) not in seen_ids][:limit]
            if len(selected) >= limit or size == self.index.ntotal:
                break
            size = min(self.index.ntotal, size * 2)
        return {"item_ids": [item for item, _ in selected],
                "scores": [score for _, score in selected], "reason": None}
