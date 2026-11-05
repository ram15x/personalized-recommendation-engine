# Recommendation demo API

## Purpose

The local FastAPI service exposes three transparent offline strategies:

1. **User genre content fallback:** when `preferred_genres` are supplied, rank available movies by the number of matching genres, then break ties by positive training-period rating count and movie ID. This supports a new user who supplies preferences; it is a metadata matching rule, not a trained model.
2. **ItemKNN:** when a hash-verified artifact is configured and the user supplies a usable history, score candidates by the sum of item-item cosine similarities. Docker Compose enables the saved 60/70 development artifact; its live HTTP and browser paths were checked on 5 November 2026.
3. **Popularity fallback:** when no genre preferences or usable ItemKNN scores exist, return the training-period popularity order.

Seen IDs are excluded from each strategy. A genre preference takes precedence over ItemKNN when the request includes both. Genres are matched case-insensitively; unsupported names and a preference with no unseen catalog matches return HTTP 422. The standard recommendation route does not record behavior, train online, or predict clicks. The separate opt-in local A/B prototype can record served responses and submitted clicks; see `docs/experiment.md`.

The page at `/` calls this API and accepts movie IDs and explicit genre preferences. It shows which genres matched for content-based results.

## Start locally

From the project root, with the local MovieLens catalog and popularity artifact present:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn scripts.recommendation_api:app --host 127.0.0.1 --port 8000
```

For native ItemKNN serving, set the three artifact paths below. NumPy is included in the API requirements; TensorFlow, FAISS, pandas and SciPy are not required for this saved sparse inference path.

Default files:

- `reports/ml-25m/popularity.csv`
- `data/ml-25m/movies.csv`

Override with `NEXVIA_POPULARITY_PATH` and `NEXVIA_MOVIES_PATH`.

### Optional Redis response cache

Install the pinned client from `requirements-api.txt`, start a Redis server separately, and set `NEXVIA_REDIS_URL` (for example, `redis://127.0.0.1:6379/0`) before starting the API. The default TTL is 300 seconds. Cache keys are versioned and include the catalog file hashes, ItemKNN artifact hash (or `none`), selected strategy, known history IDs, canonical genre preferences, and requested limit. Requests with different ranking inputs do not share entries.

When Redis is configured, the API pings it at startup and refuses to start if it is unreachable. The API checks the versioned cache before computing ItemKNN scores, so a hit skips sparse scoring. If Redis fails during a request, recommendations are still computed without caching and `/health` reports the last Redis error class. With no `NEXVIA_REDIS_URL`, the API runs without Redis. A regression test confirms repeated ItemKNN requests hit the cache without rescoring; a Redis-server check also passed through Docker Compose.

To enable ItemKNN, configure all three settings together:

- `NEXVIA_ITEM_SIMILARITY_PATH`
- `NEXVIA_ITEM_IDS_PATH`
- `NEXVIA_ITEMKNN_METADATA_PATH`

The loader checks the artifact manifest and file hashes. The corrected development bundle at `reports/development/itemknn-60-70-corrected-cohort-2k-v2/` contains the required similarity matrix, movie-ID mapping, and manifest. For a consistent catalog, pair it with the sibling `training_popularity.csv` and its manifest, generated from the same ratings hash, positive threshold, and exclusive training cutoff. Docker Compose copies these files and sets the matching paths. On 5 November 2026, live HTTP and the browser loaded the artifact, returned strategy `itemknn_cosine` for history ID `318`, returned five results, and excluded `318`. This is a demo-path check, not independent evaluation.

### Experimental DCN ranking

For research only, install `requirements-retrieval.txt`, configure the ItemKNN paths above, and set `NEXVIA_EXPERIMENTAL_RANKER_DIR` to `reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1`. Restart the native API. Supplying a known MovieLens `experimental_ranker_user_id` together with that user's liked-movie history requests reranking of up to 100 ItemKNN candidates. The caller is responsible for matching the user ID to the supplied history; there is no identity verification. The feature is not exposed in the browser UI or enabled in Docker Compose.

The response names `experimental_dcn` when all candidates are in the ranker vocabulary. An unknown or training-label-unsupported user, or any unknown or training-label-unsupported candidate, retains the ItemKNN order, with `ranking_status` explaining the fallback. The support file records IDs that appeared in labeled ranker-training examples; vocabulary membership alone does not prove an embedding was trained. A request without this field retains the normal behavior; requesting the ranker when it is not configured returns 503. Genre preferences cannot be combined with this experimental path. The saved DCN ranked below ItemKNN in its reused development comparison, so enabling it does not establish an improvement or online CTR. Cache keys include the user and model version.

### Experimental two-tower retrieval

For a separate native retrieval check, install the retrieval requirements and set `NEXVIA_EXPERIMENTAL_TWO_TOWER_DIR` to `reports/retrieval-pilot` before starting the API. `POST /experimental/two-tower` accepts `user_id`, `history_item_ids`, and `limit` (1–100). It uses saved user vectors and the exact FAISS index, excludes supplied history, and falls back to popularity for unknown users. `/health` reports whether the retriever loaded. The caller must supply the correct history for the selected MovieLens user; this is not an authenticated user profile system. The pilot performed below popularity on its recorded development comparison, and the endpoint is not enabled in Docker Compose or the browser UI.

To try both stages, configure the ItemKNN and DCN artifacts as described above and include `"experimental_dcn": true`. The endpoint retrieves up to 100 FAISS candidates, calculates the ItemKNN score for each from the supplied history, and passes those ItemKNN scores to the DCN. Returned `scores` remain FAISS inner products, reordered alongside the item IDs. If the user or any candidate is unknown to the DCN, or the history has no ItemKNN scores, the route retains FAISS order and gives a reason. With a missing component it returns 503.

The FAISS candidates come from a different model and later training window than the ItemKNN candidates used to train the DCN. This is an unevaluated distribution change: a working call is not evidence that the combined ranking is useful. In the saved-artifact HTTP check for user 847 and synthetic history 318, unknown candidates caused the route to retain FAISS order. The model-quality comparisons remain those in the saved development reports.

### FAISS-aligned feature DCN (experimental)

For a native research check, set `NEXVIA_EXPERIMENTAL_TWO_TOWER_DIR=reports/retrieval-pilot` and `NEXVIA_EXPERIMENTAL_FEATURE_RANKER_DIR=reports/development/faiss-feature-dcn-mean-init-v2`, install the retrieval requirements, and start the API from the project root. Set `experimental_feature_dcn` to `true` on `POST /experimental/two-tower`. The service retrieves up to 100 candidates, excludes supplied history, and reranks them using FAISS score and frozen popularity rank. The `scores` returned are the original FAISS scores in the new order, not predicted ratings or probabilities. Unknown retriever users fall back to popularity. Do not combine this flag with `experimental_dcn`.

The ranker checks its model and normalization hashes and requires the exact embeddings used for training. A live native HTTP check returned five results and verified the unknown-user fallback. This is an experimental route, not enabled in the default Docker demo. When both artifacts are configured in the native API, the browser page shows an experimental two-stage form; the caller must supply previously rated IDs to exclude them. On the reused development window, the ranker performed better than FAISS ordering but did not clearly outperform popularity ordering on the same candidates; absolute scores were low. See `reports/development/faiss-aligned-ranker-result.md`. No CTR is claimed.

## Endpoints

### `GET /`

Returns the small HTML demo page.

### `GET /health`

Returns server status, default strategy, whether the ItemKNN artifact loaded, catalog size, whether the explicit genre fallback is available, supported genre names, whether Redis caching is enabled, and the last Redis error class if a runtime cache operation failed.

### `POST /recommend`

Request fields:

```json
{
  "history_item_ids": [318, 296],
  "preferred_genres": ["Drama", "Comedy"],
  "limit": 10
}
```

- `history_item_ids`: optional positive MovieLens IDs, at most 2,000. These movies are excluded. If ItemKNN is enabled, they are also treated as liked/highly-rated history; the API cannot infer sentiment from a movie being seen.
- `preferred_genres`: optional list of at most 10 genre names from `/health`. When nonempty, the content fallback is used even if ItemKNN is available.
- `limit`: integer from 1 to 50.

Example cold-start request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend `
  -ContentType 'application/json' `
  -Body '{"preferred_genres":["Drama","Comedy"],"limit":10}'
```

Response fields include `strategy`, `personalized`, the number of known history items excluded, and recommendation rows with rank, movie ID, title, genres, training-period positive-rating count, and `matched_genres` (nonempty for content strategy). Strategy values are `user_genre_content`, `itemknn_cosine`, and `training_period_popularity`. `personalized: true` means the request's explicit genre preferences or a usable ItemKNN history affected ranking; it does not mean a learned ranker was served.

The service returns HTTP 422 for invalid input, unsupported genres, no unseen genre matches, or when no unseen catalog items remain. Rating counts are historical counts, not predicted ratings or business outcomes.

## Verification

The API tests cover popularity order, history exclusion, explicit genre matching, case normalization, unsupported and blank genres, no unseen genre matches, health metadata, request limits, missing catalog metadata, and the HTML route. On 5 November 2026, Docker Compose `/health` returned `ok`, with ItemKNN loaded and Redis enabled without a reported error. Live HTTP returned `itemknn_cosine` for history ID `318`, five recommendations, and one excluded history item. The browser form showed that same ItemKNN path. A second browser request with history ID `318` and Drama/Comedy preferences used the explicit content route and excluded `318`.
