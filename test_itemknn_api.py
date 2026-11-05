import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.itemknn import ItemKNNArtifact
from scripts.recommendation_api import (RecommendationRequest, TwoTowerRequest,
                                        ExperimentRequest, ExperimentClick, create_app)


def _app(tmp_path, matrix, redis_client=None, experimental_ranker=None, experimental_retriever=None,
         experiment_store=None):
    popularity_path = tmp_path / "popularity.csv"
    movies_path = tmp_path / "movies.csv"
    similarity_path = tmp_path / "similarity.npz"
    item_ids_path = tmp_path / "item_ids.npy"
    metadata_path = tmp_path / "artifact_manifest.json"
    pd.DataFrame(
        {"movieId": [10, 20, 30], "positive_count": [9, 7, 7]}
    ).to_csv(popularity_path, index=False)
    pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["Alpha (2000)", "Beta (2001)", "Gamma (2002)"],
            "genres": ["Drama", "Comedy|Romance", "Action"],
        }
    ).to_csv(movies_path, index=False)
    similarity = sparse.csr_matrix(matrix, dtype=np.float32)
    item_ids = np.array([10, 20, 30], dtype=np.int64)
    sparse.save_npz(similarity_path, similarity)
    np.save(item_ids_path, item_ids, allow_pickle=False)
    metadata = {
        "schema_version": 1,
        "ratings_sha256": "a" * 64,
        "train_cutoff_timestamp": 1_288_571_552,
        "positive_rating_threshold": 4.0,
        "neighbors_per_item": 100,
        "similarity_shape": list(similarity.shape),
        "similarity_nnz": int(similarity.nnz),
        "similarity_sha256": hashlib.sha256(similarity_path.read_bytes()).hexdigest(),
        "item_ids_count": len(item_ids),
        "item_ids_sha256": hashlib.sha256(item_ids_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return create_app(
        popularity_path, movies_path, similarity_path, item_ids_path, metadata_path,
        redis_client=redis_client, experimental_ranker=experimental_ranker,
        ranker_version="fixture-v1" if experimental_ranker else None,
        experimental_retriever=experimental_retriever,
        experiment_store=experiment_store,
    )


def _recommend(app, history, limit=3):
    route = next(route for route in app.routes if getattr(route, "path", None) == "/recommend")
    return route.endpoint(RecommendationRequest(history_item_ids=history, limit=limit))


def _health(app):
    route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    return route.endpoint()


def test_itemknn_ranks_neighbor_and_excludes_history(tmp_path):
    app = _app(
        tmp_path,
        [[0.0, 0.8, 0.1], [0.8, 0.0, 0.2], [0.1, 0.2, 0.0]],
    )
    response = _recommend(app, [10], limit=2)
    assert response.strategy == "itemknn_cosine"
    assert response.personalized is True
    assert [row.movie_id for row in response.recommendations] == [20, 30]
    assert response.history_items_excluded == 1
    assert _health(app)["itemknn_loaded"] is True
    assert _health(app)["personalization_available"] is True


def test_itemknn_uses_popularity_for_empty_or_zero_neighbor_history(tmp_path):
    app = _app(
        tmp_path,
        [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    for history in ([], [30]):
        response = _recommend(app, history, limit=2)
        assert response.strategy == "training_period_popularity"
        assert response.personalized is False
        assert [row.movie_id for row in response.recommendations] == [10, 20]


def test_redis_hit_skips_itemknn_scoring(tmp_path):
    class Cache:
        def __init__(self):
            self.values = {}
            self.get_count = 0
            self.set_count = 0

        def get(self, key):
            self.get_count += 1
            return self.values.get(key)

        def setex(self, key, ttl, value):
            self.set_count += 1
            self.values[key] = value

    cache = Cache()
    app = _app(
        tmp_path,
        [[0.0, 0.8, 0.1], [0.8, 0.0, 0.2], [0.1, 0.2, 0.0]],
        redis_client=cache,
    )
    itemknn = app.state.itemknn
    original_score = itemknn.score
    score_calls = 0

    def counted_score(history):
        nonlocal score_calls
        score_calls += 1
        return original_score(history)

    itemknn.score = counted_score
    first = _recommend(app, [10], limit=2)
    second = _recommend(app, [10], limit=2)
    assert first.model_dump() == second.model_dump()
    assert first.strategy == "itemknn_cosine"
    assert score_calls == 1
    assert cache.get_count == 2
    assert cache.set_count == 1


def test_itemknn_rejects_bad_id_mapping_and_self_edges():
    with pytest.raises(ValueError, match="mapping length"):
        ItemKNNArtifact(sparse.eye(2, format="csr"), np.array([10], dtype=np.int64))
    with pytest.raises(ValueError, match="self-neighbors"):
        ItemKNNArtifact(sparse.eye(2, format="csr"), np.array([10, 20], dtype=np.int64))


def test_itemknn_loader_rejects_mapping_from_a_different_artifact(tmp_path):
    _app(tmp_path, [[0.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]])
    item_ids_path = tmp_path / "item_ids.npy"
    np.save(item_ids_path, np.array([11, 20, 30], dtype=np.int64), allow_pickle=False)
    with pytest.raises(ValueError, match="mapping does not match"):
        ItemKNNArtifact.load(
            tmp_path / "similarity.npz",
            item_ids_path,
            tmp_path / "artifact_manifest.json",
        )


def test_experimental_ranker_is_opt_in_and_cache_separates_users(tmp_path):
    class Cache:
        def __init__(self): self.values = {}
        def get(self, key): return self.values.get(key)
        def setex(self, key, ttl, value): self.values[key] = value

    class Ranker:
        def rank_or_fallback(self, user_id, item_ids, retrieval_scores):
            if user_id == 2:
                return {"item_ids": item_ids, "reason": "unknown_user"}
            return {"item_ids": list(reversed(item_ids)), "reason": None}

    app = _app(tmp_path, [[0, .8, .1], [.8, 0, .2], [.1, .2, 0]],
               redis_client=Cache(), experimental_ranker=Ranker())
    route = next(route for route in app.routes if getattr(route, "path", None) == "/recommend")
    def request(user):
        return route.endpoint(RecommendationRequest(history_item_ids=[10], limit=2,
                                                    experimental_ranker_user_id=user))
    default = request(None)
    ranked = request(1)
    fallback = request(2)
    assert [x.movie_id for x in default.recommendations] == [20, 30]
    assert ranked.strategy == "experimental_dcn"
    assert [x.movie_id for x in ranked.recommendations] == [30, 20]
    assert fallback.strategy == "itemknn_cosine" and fallback.ranking_status == "unknown_user"
    assert [x.movie_id for x in fallback.recommendations] == [20, 30]
    assert len(app.state.redis_client.values) == 3


def test_two_tower_candidates_use_itemknn_features_for_dcn(tmp_path):
    class Retriever:
        def retrieve(self, user_id, seen, limit):
            assert limit == 100 and seen == {10}
            return {"item_ids": [20, 30], "scores": [8.0, 7.0], "reason": None}

    class Ranker:
        def rank_or_fallback(self, user_id, item_ids, retrieval_scores):
            assert user_id == 1 and item_ids == [20, 30]
            assert np.allclose(retrieval_scores, [.8, .1])
            return {"item_ids": [30, 20], "reason": None}

    app = _app(tmp_path, [[0, .8, .1], [.8, 0, .2], [.1, .2, 0]],
               experimental_ranker=Ranker(), experimental_retriever=Retriever())
    route = next(r for r in app.routes if getattr(r, "path", None) == "/experimental/two-tower")
    response = route.endpoint(TwoTowerRequest(user_id=1, history_item_ids=[10],
                                               limit=2, experimental_dcn=True))
    assert response == {"strategy": "experimental_two_tower_dcn", "reason": None,
                        "item_ids": [30, 20], "scores": [7.0, 8.0]}


def test_experiment_tracks_assigned_variant_actual_result_and_click(tmp_path):
    from scripts.experiment_store import ExperimentStore

    class Ranker:
        def rank_or_fallback(self, user_id, item_ids, retrieval_scores):
            return {"item_ids": list(reversed(item_ids)), "reason": None}

    store = ExperimentStore(tmp_path / "events.sqlite", "test-secret-123456789")
    app = _app(tmp_path, [[0, .8, .1], [.8, 0, .2], [.1, .2, 0]],
               experimental_ranker=Ranker(), experiment_store=store)
    recommend = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/experimental/ab/recommend")
    click = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/experimental/ab/click")
    users = {v: next(str(i) for i in range(100) if store.assignment(str(i))[1] == v) for v in ("A", "B")}
    for variant in ("A", "B"):
        response = recommend(ExperimentRequest(participant_id=users[variant], user_id=1,
                                              history_item_ids=[10], limit=2))
        assert response["variant"] == variant
        assert response["recommendation"]["strategy"] == ("itemknn_cosine" if variant == "A" else "experimental_dcn")
        item_id = response["recommendation"]["recommendations"][0]["movie_id"]
        assert click(ExperimentClick(impression_id=response["impression_id"], item_id=item_id))["recorded"]
    assert store.summary()["A"]["item_ctr"] == store.summary()["B"]["item_ctr"] == .5
