import faiss
import numpy as np
import pytest

from scripts.two_tower_inference import TwoTowerRetriever
from scripts.recommendation_api import TwoTowerRequest, create_app


def test_exact_retrieval_excludes_seen_and_unknown_users():
    vectors = np.array([[1, 0], [0, 1], [.8, .2]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    retriever = TwoTowerRetriever(index, [1], [[1, 0]], [10, 20, 30], vectors)
    result = retriever.retrieve(1, {10}, 2)
    assert result["item_ids"] == [30, 20]
    assert np.allclose(result["scores"], [.8, 0])
    assert retriever.retrieve(2, set())["reason"] == "unknown_user"
    with pytest.raises(ValueError, match="between 1 and 500"):
        retriever.retrieve(1, set(), 501)


def test_rejects_index_vector_mismatch():
    vectors = np.array([[1, 0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    with pytest.raises(ValueError, match="does not match"):
        TwoTowerRetriever(index, [1], [[1, 0]], [10], [[0, 1]])


def test_api_two_tower_and_cold_user_fallback(tmp_path):
    popularity = tmp_path / "pop.csv"
    movies = tmp_path / "movies.csv"
    popularity.write_text("movieId,positive_count\n10,5\n20,3\n", encoding="utf-8")
    movies.write_text("movieId,title,genres\n10,A,Drama\n20,B,Comedy\n", encoding="utf-8")
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    retriever = TwoTowerRetriever(index, [1], [[1, 0]], [10, 20], vectors)
    app = create_app(popularity, movies, experimental_retriever=retriever)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/experimental/two-tower")
    known = route.endpoint(TwoTowerRequest(user_id=1, history_item_ids=[10]))
    assert known["strategy"] == "experimental_two_tower" and known["item_ids"] == [20]
    cold = route.endpoint(TwoTowerRequest(user_id=2, history_item_ids=[10]))
    assert cold == {"strategy": "training_period_popularity", "reason": "unknown_user",
                    "item_ids": [20], "scores": None}
