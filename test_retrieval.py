import faiss
import numpy as np

from retrieval import retrieve


def test_faiss_ranking_matches_inner_product_and_filters_history():
    vectors = np.array([[1, 0], [0, 1], [0.8, 0.2]], dtype='float32')
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    assert retrieve(index, np.array([10, 20, 30]), np.array([1, 0]), {10}, [10, 20, 30], 2) == [30, 20]


def test_unknown_user_uses_popularity_without_seen_items():
    index = faiss.IndexFlatIP(2)
    assert retrieve(index, np.array([]), None, {10}, [10, 20, 30], 2) == [20, 30]


def test_short_neural_results_are_filled_without_duplicates():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0]], dtype='float32'))
    assert retrieve(index, np.array([10]), np.array([1, 0]), set(), [10, 20, 30], 3) == [10, 20, 30]
