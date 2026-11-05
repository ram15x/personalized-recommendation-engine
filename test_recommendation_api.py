from pathlib import Path
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

from scripts.recommendation_api import (
    RecommendationRequest,
    create_app,
    load_catalog,
)


class InProcessClient:
    """Call registered route functions directly without opening local sockets."""

    def __init__(self, app):
        self.app = app

    def _request(self, method, path, **kwargs):
        route = next(
            route for route in self.app.routes
            if getattr(route, "path", None) == path
            and method in getattr(route, "methods", set())
        )
        try:
            if method == "GET":
                result = route.endpoint()
                if isinstance(result, FileResponse):
                    return _Result(
                        200,
                        Path(result.path).read_text(encoding="utf-8"),
                        {"content-type": "text/html; charset=utf-8"},
                    )
                return _Result(200, result)
            payload = RecommendationRequest(**kwargs["json"])
            result = route.endpoint(payload)
            return _Result(200, result.model_dump(mode="json"))
        except HTTPException as exc:
            return _Result(exc.status_code, {"detail": exc.detail})
        except ValidationError as exc:
            return _Result(422, {"detail": exc.errors()})

    def get(self, path, **kwargs):
        return self._request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self._request("POST", path, **kwargs)


class _Result:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {"content-type": "application/json"}
        self.text = body if isinstance(body, str) else ""

    def json(self):
        return self.body


class _MemoryRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}
        self.get_count = 0
        self.set_count = 0

    def get(self, key):
        self.get_count += 1
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.set_count += 1
        self.ttls[key] = ttl
        self.values[key] = value


class _UnavailableRedis:
    def get(self, key):
        raise ConnectionError("Redis unavailable")

    def setex(self, key, ttl, value):
        raise ConnectionError("Redis unavailable")


@pytest.fixture
def client(tmp_path):
    popularity = pd.DataFrame(
        {"movieId": [10, 20, 30], "positive_count": [9, 7, 7]}
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["Alpha (2000)", "Beta (2001)", "Gamma (2002)"],
            "genres": ["Drama", "Comedy|Romance", "Action"],
        }
    )
    popularity_path = tmp_path / "popularity.csv"
    movies_path = tmp_path / "movies.csv"
    popularity.to_csv(popularity_path, index=False)
    movies.to_csv(movies_path, index=False)
    return InProcessClient(create_app(popularity_path, movies_path))


def test_health_reports_nonpersonalized_strategy_and_catalog_size(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "strategy": "training_period_popularity",
        "personalization_available": False,
        "content_fallback_available": True,
        "supported_genres": ["Action", "Comedy", "Drama", "Romance"],
        "redis_cache_enabled": False,
        "redis_last_error": None,
        "catalog_items": 3,
        "itemknn_loaded": False,
        "experimental_ranker_loaded": False,
        "experimental_two_tower_loaded": False,
        "experimental_feature_ranker_loaded": False,
        "experiment_enabled": False,
    }


def test_root_serves_the_local_demo_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Preferred genres" in response.text
    assert "content fallback" in response.text


def test_recommendations_are_ranked_by_training_popularity(client):
    response = client.post("/recommend", json={"limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["personalized"] is False
    assert [row["movie_id"] for row in body["recommendations"]] == [10, 20, 30]
    assert body["recommendations"][1]["genres"] == ["Comedy", "Romance"]


def test_seen_items_are_excluded_without_claiming_personalization(client):
    response = client.post(
        "/recommend", json={"history_item_ids": [10, 10, 999], "limit": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert [row["movie_id"] for row in body["recommendations"]] == [20, 30]
    assert body["history_items_excluded"] == 1
    assert body["personalized"] is False


def test_explicit_genre_preferences_use_content_fallback(client):
    response = client.post(
        "/recommend",
        json={"preferred_genres": [" romance ", "COMEDY"], "limit": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "user_genre_content"
    assert body["personalized"] is True
    assert [item["movie_id"] for item in body["recommendations"]] == [20]
    assert body["recommendations"][0]["matched_genres"] == ["Comedy", "Romance"]


def test_content_fallback_excludes_seen_items_and_rejects_unsupported_genres(client):
    excluded = client.post(
        "/recommend",
        json={"preferred_genres": ["Drama"], "history_item_ids": [10]},
    )
    assert excluded.status_code == 422
    unsupported = client.post(
        "/recommend", json={"preferred_genres": ["Unlisted Genre"]}
    )
    assert unsupported.status_code == 422
    assert "Unsupported genre" in unsupported.json()["detail"]


def test_blank_genre_preference_is_rejected(client):
    response = client.post("/recommend", json={"preferred_genres": ["   "]})
    assert response.status_code == 422


def test_optional_redis_cache_uses_request_key_and_ttl(tmp_path):
    popularity_path = tmp_path / "popularity.csv"
    movies_path = tmp_path / "movies.csv"
    pd.DataFrame({"movieId": [10, 20], "positive_count": [9, 7]}).to_csv(
        popularity_path, index=False
    )
    pd.DataFrame({
        "movieId": [10, 20], "title": ["Alpha", "Beta"],
        "genres": ["Drama", "Comedy|Romance"],
    }).to_csv(movies_path, index=False)
    cache = _MemoryRedis()
    app = create_app(popularity_path, movies_path, redis_client=cache)
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/recommend"
        and "POST" in getattr(route, "methods", set())
    )
    payload = RecommendationRequest(preferred_genres=["Drama"], limit=1)
    first = route.endpoint(payload)
    second = route.endpoint(payload)
    different_preferences = route.endpoint(
        RecommendationRequest(preferred_genres=["Comedy"], limit=1)
    )
    assert first.model_dump() == second.model_dump()
    assert first.strategy == "user_genre_content"
    # There is one cache lookup per request; the second request hits the first key.
    assert cache.get_count == 3
    assert different_preferences.recommendations[0].movie_id == 20
    assert cache.set_count == 2
    assert set(cache.ttls.values()) == {300}
    assert app.state.redis_last_error is None


def test_redis_runtime_error_falls_back_and_is_reported(tmp_path):
    popularity_path = tmp_path / "popularity.csv"
    movies_path = tmp_path / "movies.csv"
    pd.DataFrame({"movieId": [10], "positive_count": [9]}).to_csv(
        popularity_path, index=False
    )
    pd.DataFrame({"movieId": [10], "title": ["Alpha"], "genres": ["Drama"]}).to_csv(
        movies_path, index=False
    )
    app = create_app(popularity_path, movies_path, redis_client=_UnavailableRedis())
    route = next(
        route for route in app.routes
        if getattr(route, "path", None) == "/recommend"
        and "POST" in getattr(route, "methods", set())
    )
    response = route.endpoint(RecommendationRequest(limit=1))
    health = next(
        route.endpoint() for route in app.routes
        if getattr(route, "path", None) == "/health"
    )
    assert response.strategy == "training_period_popularity"
    assert len(response.recommendations) == 1
    assert health["redis_cache_enabled"] is True
    assert health["redis_last_error"] == "ConnectionError"


def test_empty_remaining_catalog_and_invalid_limit_are_rejected(client):
    empty = client.post("/recommend", json={"history_item_ids": [10, 20, 30]})
    assert empty.status_code == 422
    invalid = client.post("/recommend", json={"limit": 51})
    assert invalid.status_code == 422
    bad_id = client.post("/recommend", json={"history_item_ids": [0]})
    assert bad_id.status_code == 422


def test_catalog_loader_rejects_missing_movie_metadata(tmp_path):
    popularity_path = tmp_path / "popularity.csv"
    movies_path = tmp_path / "movies.csv"
    pd.DataFrame({"movieId": [99], "positive_count": [2]}).to_csv(
        popularity_path, index=False
    )
    pd.DataFrame({"movieId": [1], "title": ["Other"], "genres": ["Drama"]}).to_csv(
        movies_path, index=False
    )
    with pytest.raises(ValueError, match="metadata is missing"):
        load_catalog(popularity_path, movies_path)


