import numpy as np
import pytest
from scipy import sparse

from scripts.itemknn import build_top_neighbors, sample_evaluation_user_ids


def _full_reference(user_item, k):
    matrix = user_item.tocsr().astype(np.float32)
    degree = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    co = (matrix.T @ matrix).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    inv = np.zeros_like(degree)
    nonzero = degree > 0
    inv[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    co = co.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    co.eliminate_zeros()

    rows, cols, values = [], [], []
    for row in range(co.shape[0]):
        start, end = co.indptr[row], co.indptr[row + 1]
        row_cols = co.indices[start:end]
        row_values = co.data[start:end]
        order = np.lexsort((row_cols, -row_values))[:k]
        rows.extend([row] * len(order))
        cols.extend(row_cols[order])
        values.extend(row_values[order])
    return sparse.csr_matrix((values, (rows, cols)), shape=co.shape, dtype=np.float32)


@pytest.mark.parametrize("block_size", [1, 2, 4, 20])
def test_blockwise_neighbor_builder_matches_full_reference(block_size):
    user_item = sparse.csr_matrix(
        np.array([
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
        ], dtype=np.float32)
    )
    expected = _full_reference(user_item, k=2)
    actual = build_top_neighbors(user_item, k=2, block_size=block_size)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-6, atol=1e-7)


def test_neighbor_builder_rejects_invalid_block_or_neighbor_count():
    matrix = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="positive integers"):
        build_top_neighbors(matrix, k=0)
    with pytest.raises(ValueError, match="at least two item columns"):
        build_top_neighbors(sparse.csr_matrix((2, 1)), k=1)


def test_user_sample_is_reproducible_and_does_not_use_rating_values():
    training_ids = np.arange(1, 101, dtype=np.int64)
    observed_ids = set(range(1, 51))
    first = sample_evaluation_user_ids(training_ids, observed_ids, max_users=12, seed=42)
    second = sample_evaluation_user_ids(training_ids, observed_ids, max_users=12, seed=42)
    assert len(first) == 12
    np.testing.assert_array_equal(first, second)
    assert set(first).issubset(observed_ids)


def test_user_sample_rejects_negative_cap():
    with pytest.raises(ValueError, match="zero or positive"):
        sample_evaluation_user_ids(np.array([1, 2]), {1, 2}, max_users=-1)
