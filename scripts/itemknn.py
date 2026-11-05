"""Validated loader and scorer for sparse ItemKNN retrieval artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sample_evaluation_user_ids(
    training_user_ids: np.ndarray,
    observed_user_ids: set[int],
    max_users: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Sample warm users with any held-out observation, without using rating values."""
    train_ids = np.unique(np.asarray(training_user_ids, dtype=np.int64))
    observed_ids = np.asarray(sorted(map(int, observed_user_ids)), dtype=np.int64)
    if np.any(observed_ids <= 0):
        raise ValueError("Observed user IDs must be positive integers.")
    ids = np.intersect1d(train_ids, observed_ids, assume_unique=True)
    if np.any(ids <= 0):
        raise ValueError("Training user IDs must be positive integers.")
    if max_users < 0:
        raise ValueError("max_users must be zero or positive.")
    if max_users == 0 or max_users >= len(ids):
        return ids
    selected = np.random.default_rng(seed).choice(ids, size=max_users, replace=False)
    return np.sort(selected)


def build_top_neighbors(
    user_item, k: int = 100, block_size: int = 128
):
    """Build row-pruned cosine item neighbors using bounded co-occurrence blocks.

    The exact full item-item matrix can be dense and exceed laptop memory. This
    computes the same row-wise cosine top-k by multiplying only ``block_size``
    item rows at a time. Ties are resolved by ascending item column index.
    """
    # SciPy is a research/build-time dependency only. Keep it out of the
    # recommendation API's import path so the serving image stays smaller.
    from scipy import sparse

    matrix = user_item.tocsr().astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("user_item must have at least two item columns.")
    if k < 1 or block_size < 1:
        raise ValueError("k and block_size must be positive integers.")
    n_items = matrix.shape[1]
    k = min(int(k), n_items - 1)
    degree = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    inverse_sqrt_degree = np.zeros_like(degree)
    seen = degree > 0
    inverse_sqrt_degree[seen] = 1.0 / np.sqrt(degree[seen])

    item_user = matrix.T.tocsr()
    capacity = n_items * k
    rows = np.empty(capacity, dtype=np.int32)
    cols = np.empty(capacity, dtype=np.int32)
    values = np.empty(capacity, dtype=np.float32)
    cursor = 0

    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        co_block = (item_user[start:stop] @ matrix).tocsr()
        for local_row in range(stop - start):
            row = start + local_row
            begin, end = co_block.indptr[local_row:local_row + 2]
            row_cols = co_block.indices[begin:end]
            row_values = co_block.data[begin:end]
            keep = row_cols != row
            row_cols = row_cols[keep]
            if row_cols.size == 0:
                continue
            similarities = (row_values[keep] * inverse_sqrt_degree[row]
                            * inverse_sqrt_degree[row_cols])
            order = np.lexsort((row_cols, -similarities))[:k]
            count = len(order)
            rows[cursor:cursor + count] = row
            cols[cursor:cursor + count] = row_cols[order]
            values[cursor:cursor + count] = similarities[order]
            cursor += count

    result = sparse.csr_matrix(
        (values[:cursor], (rows[:cursor], cols[:cursor])),
        shape=(n_items, n_items), dtype=np.float32,
    )
    result.sort_indices()
    return result


class ItemKNNArtifact:
    """Top-neighbor cosine matrix whose row order is defined by ``item_ids``."""

    def __init__(self, similarity, item_ids: np.ndarray) -> None:
        matrix = similarity.tocsr().astype(np.float32)
        self._set_arrays(matrix.data, matrix.indices, matrix.indptr, matrix.shape, item_ids)

    @classmethod
    def _from_arrays(cls, data, indices, indptr, shape, item_ids):
        instance = cls.__new__(cls)
        instance._set_arrays(data, indices, indptr, shape, item_ids)
        return instance

    def _set_arrays(self, data, indices, indptr, shape, item_ids) -> None:
        data = np.asarray(data, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)
        indptr = np.asarray(indptr, dtype=np.int64)
        shape = tuple(map(int, shape))
        ids = np.asarray(item_ids)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("ItemKNN similarity matrix must be square.")
        if ids.ndim != 1 or len(ids) != shape[0]:
            raise ValueError("Item ID mapping length must equal the matrix dimension.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Item ID mapping must contain integers.")
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("Item IDs must be unique positive integers.")
        if len(ids) > 1 and np.any(np.diff(ids) <= 0):
            raise ValueError("Item IDs must be in strictly increasing row order.")
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("ItemKNN similarities must be finite and non-negative.")
        if indptr.shape != (shape[0] + 1,) or indptr[0] != 0 or indptr[-1] != len(data):
            raise ValueError("ItemKNN CSR row pointers are invalid.")
        if len(indices) != len(data) or np.any(indices < 0) or np.any(indices >= shape[1]):
            raise ValueError("ItemKNN CSR column indices are invalid.")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("ItemKNN CSR row pointers must be nondecreasing.")
        diagonal_rows = np.repeat(np.arange(shape[0]), np.diff(indptr))
        if np.any(diagonal_rows == indices):
            raise ValueError("ItemKNN similarity matrix must not include self-neighbors.")
        nonzero = data != 0
        if not np.all(nonzero):
            # Compact explicit zeros while preserving CSR order.
            kept_rows = diagonal_rows[nonzero]
            indices = indices[nonzero]
            data = data[nonzero]
            counts = np.bincount(kept_rows, minlength=shape[0])
            indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        self.similarity_data = data
        self.similarity_indices = indices
        self.similarity_indptr = indptr
        self.similarity_shape = shape
        self.similarity_nnz = len(data)
        self.item_ids = ids
        self.item_to_row = {int(item_id): row for row, item_id in enumerate(ids)}

    @classmethod
    def load(
        cls, similarity_path: Path, item_ids_path: Path, metadata_path: Path
    ) -> "ItemKNNArtifact":
        return cls.load_verified(similarity_path, item_ids_path, metadata_path)

    @classmethod
    def load_verified(
        cls, similarity_path: Path, item_ids_path: Path, metadata_path: Path
    ) -> "ItemKNNArtifact":
        if not similarity_path.is_file():
            raise FileNotFoundError(f"ItemKNN similarity artifact not found: {similarity_path}")
        if not item_ids_path.is_file():
            raise FileNotFoundError(f"ItemKNN item-ID mapping not found: {item_ids_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"ItemKNN artifact manifest not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("ItemKNN artifact manifest must be a JSON object.")
        actual_similarity_hash = _sha256(similarity_path)
        actual_ids_hash = _sha256(item_ids_path)
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported ItemKNN artifact manifest version.")
        if metadata.get("similarity_sha256") != actual_similarity_hash:
            raise ValueError("ItemKNN similarity file does not match its manifest hash.")
        if metadata.get("item_ids_sha256") != actual_ids_hash:
            raise ValueError("ItemKNN item-ID mapping does not match its manifest hash.")
        if metadata.get("positive_rating_threshold") != 4.0:
            raise ValueError("ItemKNN manifest has an unsupported positive-rating threshold.")
        try:
            neighbors_per_item = int(metadata.get("neighbors_per_item", 0))
            train_cutoff_timestamp = int(metadata.get("train_cutoff_timestamp", 0))
            item_ids_count = int(metadata.get("item_ids_count", -1))
            similarity_nnz = int(metadata.get("similarity_nnz", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("ItemKNN artifact manifest contains invalid counts.") from exc
        if neighbors_per_item != 100:
            raise ValueError("ItemKNN manifest has an unsupported neighbor count.")
        if not isinstance(metadata.get("ratings_sha256"), str) or len(metadata["ratings_sha256"]) != 64:
            raise ValueError("ItemKNN manifest is missing the ratings dataset hash.")
        if train_cutoff_timestamp <= 0:
            raise ValueError("ItemKNN manifest is missing the training cutoff.")
        # scipy.sparse.save_npz's CSR format is a NumPy archive. Read its
        # arrays directly so serving does not import SciPy.
        with np.load(similarity_path, allow_pickle=False) as stored:
            if bytes(stored["format"].tolist()) != b"csr":
                raise ValueError("ItemKNN artifact must use CSR storage.")
            shape = stored["shape"].astype(np.int64, copy=False)
            matrix_data = stored["data"]
            matrix_indices = stored["indices"]
            matrix_indptr = stored["indptr"]
        ids = np.load(item_ids_path, allow_pickle=False)
        artifact = cls._from_arrays(matrix_data, matrix_indices, matrix_indptr, shape, ids)
        if metadata.get("similarity_shape") != list(artifact.similarity_shape):
            raise ValueError("ItemKNN similarity dimensions do not match its manifest.")
        if item_ids_count != len(artifact.item_ids):
            raise ValueError("ItemKNN item-ID count does not match its manifest.")
        if similarity_nnz != artifact.similarity_nnz:
            raise ValueError("ItemKNN nonzero count does not match its manifest.")
        artifact.metadata = metadata
        return artifact

    def score(self, history_item_ids: set[int]) -> dict[int, float]:
        """Sum neighbor similarities for known history IDs, matching eval scoring."""
        rows = [self.item_to_row[item] for item in history_item_ids if item in self.item_to_row]
        if not rows:
            return {}
        scores = np.zeros(len(self.item_ids), dtype=np.float64)
        for row in rows:
            start, end = self.similarity_indptr[row:row + 2]
            np.add.at(scores, self.similarity_indices[start:end], self.similarity_data[start:end])
        return {
            int(self.item_ids[col]): float(score)
            for col, score in enumerate(scores)
            if score > 0
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
