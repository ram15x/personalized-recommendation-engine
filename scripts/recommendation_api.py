"""Offline API for popularity, explicit genre preference and optional ItemKNN.

This is a demo-serving layer, not a learned personalized ranker. It filters
items the caller supplies as already seen and returns the fixed training-
period popularity order. New users with no history receive that same fallback.
"""

from __future__ import annotations

import os
import hashlib
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POPULARITY_PATH = PROJECT_ROOT / "reports" / "ml-25m" / "popularity.csv"
DEFAULT_MOVIES_PATH = PROJECT_ROOT / "data" / "ml-25m" / "movies.csv"
DEMO_PATH = PROJECT_ROOT / "demo.html"
DEFAULT_REDIS_TTL_SECONDS = 300


class RecommendationRequest(BaseModel):
    history_item_ids: list[int] = Field(default_factory=list, max_length=2_000)
    preferred_genres: list[str] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=10, ge=1, le=50)
    experimental_ranker_user_id: int | None = Field(default=None, gt=0)

    @field_validator("history_item_ids")
    @classmethod
    def require_positive_movie_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("History movie IDs must be positive integers.")
        return values

    @field_validator("preferred_genres")
    @classmethod
    def clean_genre_preferences(cls, values: list[str]) -> list[str]:
        cleaned = []
        seen = set()
        for value in values:
            genre = value.strip()
            key = genre.casefold()
            if not key:
                raise ValueError("Preferred genres cannot be blank.")
            if key not in seen:
                cleaned.append(genre)
                seen.add(key)
        return cleaned


class TwoTowerRequest(BaseModel):
    user_id: int = Field(gt=0)
    history_item_ids: list[int] = Field(default_factory=list, max_length=2000)
    limit: int = Field(default=10, ge=1, le=100)
    experimental_dcn: bool = False
    experimental_feature_dcn: bool = False

    @field_validator("history_item_ids")
    @classmethod
    def require_positive_history(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("History movie IDs must be positive integers.")
        return values


class ExperimentRequest(BaseModel):
    participant_id: str = Field(min_length=1, max_length=128)
    user_id: int = Field(gt=0)
    history_item_ids: list[int] = Field(default_factory=list, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)


class ExperimentClick(BaseModel):
    impression_id: str = Field(min_length=32, max_length=32)
    item_id: int = Field(gt=0)


class Recommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: list[str]
    positive_training_ratings: int
    matched_genres: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CatalogMovie:
    movieId: int
    title: str
    genres: str
    positive_count: int


class RecommendationResponse(BaseModel):
    strategy: Literal["training_period_popularity", "itemknn_cosine", "user_genre_content", "experimental_dcn"]
    personalized: bool = False
    history_items_excluded: int
    recommendations: list[Recommendation]
    ranking_status: str | None = None


def _read_csv_by_movie_id(path: Path, required: set[str]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name} must contain columns: {', '.join(sorted(required))}.")
        for line_number, row in enumerate(reader, start=2):
            try:
                movie_id = int(row["movieId"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name} has an invalid movieId on row {line_number}.") from exc
            if movie_id <= 0 or movie_id in rows:
                raise ValueError(f"{path.name} movie IDs must be unique positive integers.")
            rows[movie_id] = row
    return rows


def load_catalog(popularity_path: Path, movies_path: Path) -> list[CatalogMovie]:
    if not popularity_path.is_file():
        raise FileNotFoundError(f"Popularity artifact not found: {popularity_path}")
    if not movies_path.is_file():
        raise FileNotFoundError(f"Movie metadata not found: {movies_path}")

    popularity = _read_csv_by_movie_id(popularity_path, {"movieId", "positive_count"})
    movies = _read_csv_by_movie_id(movies_path, {"movieId", "title", "genres"})
    if not popularity:
        raise ValueError("Popularity artifact must be nonempty.")
    catalog = []
    for movie_id, popularity_row in popularity.items():
        try:
            positive_count = int(popularity_row["positive_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Popularity artifact has an invalid count for movie {movie_id}.") from exc
        if positive_count < 0:
            raise ValueError("Positive-rating counts cannot be negative.")
        movie = movies.get(movie_id)
        if movie is None or not movie.get("title"):
            raise ValueError(f"Movie metadata is missing for popular catalog ID {movie_id}.")
        genres = movie.get("genres") or "(no genres listed)"
        catalog.append(CatalogMovie(movie_id, movie["title"], genres, positive_count))
    return sorted(catalog, key=lambda movie: (-movie.positive_count, movie.movieId))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recommendation_cache_key(
    *, known_seen: set[int], preferences: list[str], limit: int,
    strategy: str, catalog_version: str, itemknn_version: str,
    ranker_user_id: int | None = None, ranker_version: str = "none",
) -> str:
    inputs = {
        "api": "recommend-v3",
        "catalog": catalog_version,
        "strategy": strategy,
        "itemknn": itemknn_version,
        "history": sorted(known_seen),
        "genres": sorted(preferences),
        "limit": limit,
        "ranker_user_id": ranker_user_id,
        "ranker_version": ranker_version,
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return "nexvia:recommend:v3:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_app(
    popularity_path: Path | None = None,
    movies_path: Path | None = None,
    item_similarity_path: Path | None = None,
    item_ids_path: Path | None = None,
    item_metadata_path: Path | None = None,
    redis_client=None,
    redis_ttl_seconds: int = DEFAULT_REDIS_TTL_SECONDS,
    experimental_ranker=None,
    ranker_version: str | None = None,
    experimental_retriever=None,
    experimental_feature_ranker=None,
    experiment_store=None,
) -> FastAPI:
    if redis_ttl_seconds < 1:
        raise ValueError("Redis cache TTL must be positive.")
    popularity_path = popularity_path or Path(
        os.environ.get("NEXVIA_POPULARITY_PATH", DEFAULT_POPULARITY_PATH)
    )
    movies_path = movies_path or Path(
        os.environ.get("NEXVIA_MOVIES_PATH", DEFAULT_MOVIES_PATH)
    )
    similarity_setting = os.environ.get("NEXVIA_ITEM_SIMILARITY_PATH")
    item_ids_setting = os.environ.get("NEXVIA_ITEM_IDS_PATH")
    item_metadata_setting = os.environ.get("NEXVIA_ITEMKNN_METADATA_PATH")
    if item_similarity_path is None and similarity_setting:
        item_similarity_path = Path(similarity_setting)
    if item_ids_path is None and item_ids_setting:
        item_ids_path = Path(item_ids_setting)
    if item_metadata_path is None and item_metadata_setting:
        item_metadata_path = Path(item_metadata_setting)
    configured_artifacts = [
        item_similarity_path is not None,
        item_ids_path is not None,
        item_metadata_path is not None,
    ]
    if any(configured_artifacts) and not all(configured_artifacts):
        raise RuntimeError(
            "Configure NEXVIA_ITEM_SIMILARITY_PATH, NEXVIA_ITEM_IDS_PATH, and "
            "NEXVIA_ITEMKNN_METADATA_PATH together, or leave all unset to use popularity."
        )
    try:
        catalog = load_catalog(popularity_path, movies_path)
        itemknn = None
        if all(configured_artifacts):
            from scripts.itemknn import ItemKNNArtifact

            itemknn = ItemKNNArtifact.load(
                item_similarity_path, item_ids_path, item_metadata_path
            )
        ranker_dir = os.environ.get("NEXVIA_EXPERIMENTAL_RANKER_DIR")
        if experimental_ranker is None and ranker_dir:
            from scripts.candidate_ranker_inference import CandidateRanker
            experimental_ranker = CandidateRanker.load(Path(ranker_dir))
            ranker_version = _file_sha256(Path(ranker_dir) / "artifact_manifest.json")
        retriever_dir = os.environ.get("NEXVIA_EXPERIMENTAL_TWO_TOWER_DIR")
        if experimental_retriever is None and retriever_dir:
            from scripts.two_tower_inference import TwoTowerRetriever
            experimental_retriever = TwoTowerRetriever.load(Path(retriever_dir))
        feature_dir = os.environ.get("NEXVIA_EXPERIMENTAL_FEATURE_RANKER_DIR")
        if experimental_feature_ranker is None and feature_dir:
            if not retriever_dir:
                raise ValueError("Feature ranker requires NEXVIA_EXPERIMENTAL_TWO_TOWER_DIR.")
            from scripts.feature_ranker_inference import FeatureRanker
            experimental_feature_ranker = FeatureRanker.load(
                Path(feature_dir), Path(retriever_dir) / "embeddings.npz"
            )
        experiment_db = os.environ.get("NEXVIA_EXPERIMENT_DB")
        experiment_secret = os.environ.get("NEXVIA_EXPERIMENT_SECRET")
        if experiment_store is None and (experiment_db or experiment_secret):
            if not experiment_db or not experiment_secret:
                raise ValueError("Configure both NEXVIA_EXPERIMENT_DB and NEXVIA_EXPERIMENT_SECRET.")
            from scripts.experiment_store import ExperimentStore
            experiment_store = ExperimentStore(Path(experiment_db), experiment_secret)
    except (FileNotFoundError, ValueError, ImportError, csv.Error) as exc:
        raise RuntimeError(f"Could not load recommendation artifacts: {exc}") from exc

    redis_url = os.environ.get("NEXVIA_REDIS_URL")
    if redis_client is None and redis_url:
        try:
            import redis

            redis_client = redis.Redis.from_url(
                redis_url, decode_responses=True,
                socket_connect_timeout=1.0, socket_timeout=1.0,
            )
            redis_client.ping()
        except Exception as exc:
            raise RuntimeError(f"Configured Redis is unavailable: {type(exc).__name__}") from exc
    catalog_version = hashlib.sha256(
        (_file_sha256(popularity_path) + _file_sha256(movies_path)).encode("ascii")
    ).hexdigest()
    itemknn_version = (
        str(itemknn.metadata.get("similarity_sha256", "unknown"))
        if itemknn is not None else "none"
    )
    if experimental_ranker is not None and not ranker_version:
        raise RuntimeError("A version identifier is required for the experimental ranker cache.")

    app = FastAPI(
        title="MovieLens Recommendation Demo",
        version="0.1.0",
        description="Offline popularity-baseline API; not a learned personalized model.",
    )
    app.state.catalog = catalog
    app.state.catalog_item_ids = {movie.movieId for movie in catalog}
    app.state.genre_lookup = {
        genre.casefold(): genre
        for movie in catalog
        for genre in movie.genres.split("|")
        if genre and genre != "(no genres listed)"
    }
    app.state.itemknn = itemknn
    app.state.redis_client = redis_client
    app.state.redis_ttl_seconds = redis_ttl_seconds
    app.state.catalog_version = catalog_version
    app.state.itemknn_version = itemknn_version
    app.state.redis_last_error = None

    @app.get("/", include_in_schema=False)
    def demo_page() -> FileResponse:
        if not DEMO_PATH.is_file():
            raise HTTPException(status_code=503, detail="The local demo page is missing.")
        return FileResponse(DEMO_PATH, media_type="text/html")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "strategy": "itemknn_cosine_with_popularity_fallback"
            if itemknn is not None
            else "training_period_popularity",
            "personalization_available": itemknn is not None,
            "content_fallback_available": True,
            "supported_genres": sorted(app.state.genre_lookup.values()),
            "catalog_items": len(catalog),
            "itemknn_loaded": itemknn is not None,
            "experimental_ranker_loaded": experimental_ranker is not None,
            "experimental_two_tower_loaded": experimental_retriever is not None,
            "experimental_feature_ranker_loaded": experimental_feature_ranker is not None,
            "experiment_enabled": experiment_store is not None,
            "redis_cache_enabled": redis_client is not None,
            "redis_last_error": app.state.redis_last_error,
        }

    @app.post("/experimental/two-tower")
    def experimental_two_tower(payload: TwoTowerRequest) -> dict:
        if experimental_retriever is None:
            raise HTTPException(status_code=503, detail="Experimental two-tower retriever is not configured.")
        if payload.experimental_dcn and (experimental_ranker is None or itemknn is None):
            raise HTTPException(status_code=503, detail="Experimental DCN and ItemKNN must both be configured.")
        if payload.experimental_dcn and payload.experimental_feature_dcn:
            raise HTTPException(status_code=422, detail="Choose one experimental ranker.")
        if payload.experimental_feature_dcn and experimental_feature_ranker is None:
            raise HTTPException(status_code=503, detail="Experimental feature ranker is not configured.")
        seen = set(payload.history_item_ids)
        result = experimental_retriever.retrieve(
            payload.user_id, seen, 100 if payload.experimental_dcn or payload.experimental_feature_dcn else payload.limit
        )
        if result["reason"] == "unknown_user":
            fallback = [movie.movieId for movie in catalog if movie.movieId not in seen][:payload.limit]
            return {"strategy": "training_period_popularity", "reason": "unknown_user",
                    "item_ids": fallback, "scores": None}
        strategy = "experimental_two_tower"
        reason = None
        if payload.experimental_feature_dcn:
            try:
                ordered = experimental_feature_ranker.rank(result["item_ids"], result["scores"])
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            score_by_id = dict(zip(result["item_ids"], result["scores"]))
            result["item_ids"] = ordered
            result["scores"] = [score_by_id[item_id] for item_id in ordered]
            strategy = "experimental_two_tower_feature_dcn"
        if payload.experimental_dcn:
            itemknn_scores = itemknn.score(seen)
            if not itemknn_scores:
                reason = "no_itemknn_scores"
            else:
                decision = experimental_ranker.rank_or_fallback(
                    payload.user_id, result["item_ids"],
                    [itemknn_scores.get(item_id, 0.0) for item_id in result["item_ids"]],
                )
                score_by_id = dict(zip(result["item_ids"], result["scores"]))
                result["item_ids"] = decision["item_ids"]
                result["scores"] = [score_by_id[item_id] for item_id in result["item_ids"]]
                reason = decision["reason"]
                if reason is None:
                    strategy = "experimental_two_tower_dcn"
        return {"strategy": strategy, "reason": reason,
                "item_ids": result["item_ids"][:payload.limit],
                "scores": result["scores"][:payload.limit]}

    @app.post("/experimental/two-tower/display")
    def display_two_tower(payload: TwoTowerRequest) -> dict:
        result = experimental_two_tower(payload)
        catalog_by_id = {movie.movieId: movie for movie in catalog}
        result["movies"] = [
            {"movie_id": movie_id,
             "title": catalog_by_id[movie_id].title if movie_id in catalog_by_id else None}
            for movie_id in result["item_ids"]
        ]
        return result

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(payload: RecommendationRequest) -> RecommendationResponse:
        if payload.experimental_ranker_user_id is not None:
            if experimental_ranker is None:
                raise HTTPException(status_code=503, detail="Experimental ranker is not configured.")
            if payload.preferred_genres:
                raise HTTPException(status_code=422, detail="Experimental ranking requires ItemKNN, without genre preferences.")
        seen = set(payload.history_item_ids)
        known_seen = seen & app.state.catalog_item_ids
        preferences = []
        for genre in payload.preferred_genres:
            canonical = app.state.genre_lookup.get(genre.casefold())
            if canonical is None:
                supported = ", ".join(sorted(app.state.genre_lookup.values()))
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported genre '{genre}'. Choose from: {supported}.",
                )
            preferences.append(canonical)

        if preferences:
            cache_strategy = "user_genre_content"
        elif itemknn is not None and known_seen:
            # Cache before sparse scoring. Use a deterministic request-path key;
            # the response records whether this history had any known neighbors.
            cache_strategy = "itemknn_or_popularity"
        else:
            cache_strategy = "training_period_popularity"
        cache_key = _recommendation_cache_key(
            known_seen=known_seen,
            preferences=preferences,
            limit=payload.limit,
            strategy=cache_strategy,
            catalog_version=app.state.catalog_version,
            itemknn_version=app.state.itemknn_version,
            ranker_user_id=payload.experimental_ranker_user_id,
            ranker_version=ranker_version or "none",
        )
        if redis_client is not None:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    app.state.redis_last_error = None
                    return RecommendationResponse.model_validate_json(cached)
                app.state.redis_last_error = None
            except Exception as exc:
                app.state.redis_last_error = type(exc).__name__

        scores = itemknn.score(known_seen) if itemknn is not None and not preferences else {}
        matched_by_movie: dict[int, list[str]] = {}
        ranking_status = None
        if preferences:
            preference_set = set(preferences)
            ranked = []
            for movie in catalog:
                if movie.movieId in known_seen:
                    continue
                matched = [genre for genre in movie.genres.split("|") if genre in preference_set]
                if matched:
                    ranked.append((movie, matched))
            ranked.sort(key=lambda pair: (-len(pair[1]), -pair[0].positive_count, pair[0].movieId))
            ranked = ranked[:payload.limit]
            if not ranked:
                raise HTTPException(
                    status_code=422,
                    detail="No unseen catalog movies match the supplied genre preferences.",
                )
            for movie, genres in ranked:
                matched_by_movie[movie.movieId] = genres
            available = [movie for movie, _matched_genres in ranked]
            strategy = "user_genre_content"
            use_personalization = True
        else:
            use_itemknn = bool(scores)
            if use_itemknn:
                ranked = sorted(
                    catalog,
                    key=lambda movie: (-scores.get(movie.movieId, 0.0),
                                       -movie.positive_count, movie.movieId),
                )
                candidates = [movie for movie in ranked if movie.movieId not in known_seen][:100]
                if payload.experimental_ranker_user_id is not None:
                    decision = experimental_ranker.rank_or_fallback(
                        payload.experimental_ranker_user_id,
                        [movie.movieId for movie in candidates],
                        [scores.get(movie.movieId, 0.0) for movie in candidates],
                    )
                    by_id = {movie.movieId: movie for movie in candidates}
                    candidates = [by_id[movie_id] for movie_id in decision["item_ids"]]
                    ranking_status = decision["reason"] or "experimental_dcn"
                available = candidates[:payload.limit]
                strategy = "itemknn_cosine"
                if ranking_status == "experimental_dcn":
                    strategy = "experimental_dcn"
                use_personalization = True
            else:
                available = [movie for movie in catalog if movie.movieId not in known_seen][:payload.limit]
                strategy = "training_period_popularity"
                use_personalization = False
                if payload.experimental_ranker_user_id is not None:
                    ranking_status = "no_itemknn_candidates"
        if not available:
            raise HTTPException(
                status_code=422,
                detail="No unseen catalog items remain for the supplied history.",
            )
        recommendations = [
            Recommendation(
                rank=rank,
                movie_id=row.movieId,
                title=row.title,
                genres=[genre for genre in row.genres.split("|") if genre],
                positive_training_ratings=row.positive_count,
                matched_genres=matched_by_movie.get(row.movieId, []),
            )
            for rank, row in enumerate(available, start=1)
        ]
        response = RecommendationResponse(
            strategy=strategy,
            personalized=use_personalization,
            history_items_excluded=len(known_seen),
            recommendations=recommendations,
            ranking_status=ranking_status,
        )
        if redis_client is not None:
            try:
                redis_client.setex(
                    cache_key, redis_ttl_seconds, response.model_dump_json()
                )
                app.state.redis_last_error = None
            except Exception as exc:
                app.state.redis_last_error = type(exc).__name__
        return response

    @app.post("/experimental/ab/recommend")
    def experiment_recommend(payload: ExperimentRequest) -> dict:
        if experiment_store is None or itemknn is None or experimental_ranker is None:
            raise HTTPException(status_code=503, detail="Experiment, ItemKNN, and ranker must be configured.")
        if any(item <= 0 for item in payload.history_item_ids):
            raise HTTPException(status_code=422, detail="History movie IDs must be positive integers.")
        _, variant = experiment_store.assignment(payload.participant_id)
        response = recommend(RecommendationRequest(
            history_item_ids=payload.history_item_ids, limit=payload.limit,
            experimental_ranker_user_id=payload.user_id if variant == "B" else None,
        ))
        impression = experiment_store.record_impression(
            payload.participant_id, variant, response.strategy,
            [item.movie_id for item in response.recommendations],
        )
        return {"variant": variant, "impression_id": impression, "recommendation": response.model_dump()}

    @app.post("/experimental/ab/click")
    def experiment_click(payload: ExperimentClick) -> dict:
        if experiment_store is None:
            raise HTTPException(status_code=503, detail="Experiment is not configured.")
        try:
            recorded = experiment_store.record_click(payload.impression_id, payload.item_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"recorded": recorded}

    @app.get("/experimental/ab/summary")
    def experiment_summary() -> dict:
        if experiment_store is None:
            raise HTTPException(status_code=503, detail="Experiment is not configured.")
        return experiment_store.summary()

    return app


app = create_app()
