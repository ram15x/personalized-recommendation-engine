# Demo and test verification — 5 November 2026

## Verified in this pass

- The local Docker Compose API answered `GET http://127.0.0.1:8810/health` with `status: ok`, `redis_cache_enabled: true`, and `redis_last_error: null`.
- The browser page at `http://127.0.0.1:8810/` displayed the Movie Recommendations form. A request with history ID `318` and preferences `Drama, Comedy` returned 10 recommendations. The visible cards reported matching genres and did not include movie ID `318`.
- The UI describes this as an offline MovieLens demo with popularity, explicit genre matching, and an optional ItemKNN path. This is evidence for those demo flows only; it does not verify the two-tower model or a trained ranker is being served.
- `python -m pytest -q --ignore=test_retrieval.py`: **41 passed, 1 skipped** after the cache regression test was added.
- The focused run initially found a stale test expectation: the Redis cache is looked up once for each of three requests, while only two cache entries are written because the second request hits the first. The test now expects three lookups. This was a test assertion correction; API behavior was not changed.

## Not verified / failed

- `python -m pytest -q` stops during collection at `test_retrieval.py` because `faiss` is not installed (`ModuleNotFoundError: No module named 'faiss'`). FAISS was not installed for this check, to avoid adding a large package just to make collection proceed.
- The first demo check preceded ItemKNN container integration; the follow-up below verified that route over HTTP and in the browser.
- This local check is not a public deployment. No online A/B test or CTR measurement was performed.

## Evidence and commands

- Browser page title and URL: `Movie Recommendations | Nexvia AIML-03`, `http://127.0.0.1:8810/`.
- API health command: `Invoke-RestMethod -Uri http://127.0.0.1:8810/health -Method Get`.
- Test commands: `python -m pytest -q`; `python -m pytest -q --ignore=test_retrieval.py`; `python -m pytest -q test_itemknn_api.py test_recommendation_api.py`.
- Browser request values: history `318`; preferences `Drama, Comedy`; visible status: “Showing 10 movies. Strategy: content matches for your selected genres.”

## Follow-up: serve the saved ItemKNN candidate

- Added the already-saved ItemKNN matrix, item-ID mapping, and hash manifest to the Compose image; pinned `scipy==1.14.1` in the API image. The Docker build context was **5.25 MB** and did not contain the 678 MB ratings CSV. The build downloaded a **40.8 MB SciPy wheel** plus the other pinned API packages; no MovieLens data was downloaded and no model was trained.
- `docker compose build --progress plain` completed successfully (Docker printed a warning that `--progress` is a global option). `docker compose up -d` started the API and Redis; Redis passed its health check.
- Live `GET /health`: `status=ok`, `itemknn_loaded=true`, `personalization_available=true`, `redis_cache_enabled=true`, `redis_last_error=null`.
- Live `POST /recommend` with `history_item_ids=[318]`, `limit=5`: strategy `itemknn_cosine`; five IDs were `296, 593, 50, 527, 356`; one history item was excluded.
- Live `POST /recommend` with no history: strategy `training_period_popularity`.
- Browser: the same liked-history request displayed “cosine item-neighbor recommendations” and did not display movie `318`. A second request with history `318` and genres Drama/Comedy switched to the content strategy and displayed genre matches, still excluding `318`.
- After these checks, `docker compose down` stopped and removed only this project's API container, Redis container, and Compose network. The local API endpoint then failed to connect, confirming the containers had stopped. The built image and source dataset remain on disk.

This verifies the configured demo route, not independent model performance. The 60/70 ItemKNN development panel was reused during development; the live result is not evidence of generalization or an online outcome.

## Cache-path performance correction

- Code review found that the API computed ItemKNN scores before checking Redis, so a cache hit still paid the sparse-scoring cost. The lookup now happens first; its deterministic request-path cache key and API namespace were bumped to v2 so old responses are not reused under the revised key logic.
- Added a regression test that spies on the ItemKNN scorer and verifies that two identical requests make one score calculation and one cache write.
- `python -m pytest -q test_itemknn_api.py test_recommendation_api.py`: **16 passed** before the new regression test. `python -m pytest -q --ignore=test_retrieval.py`: **41 passed, 1 skipped** after it.
- Docker must be rebuilt after this cache-path change before it runs the updated code. The next rebuild can reuse the cached dependency layer; no new dependency package is needed.

## Serving dependency follow-up

- Replaced pandas catalog handling with Python CSV handling and loaded the saved CSR arrays with NumPy; SciPy remains needed to build the research artifact, but is no longer an API runtime dependency.
- The first regression run found nine failures caused by an obsolete DataFrame empty check. After that fix, two genre-path tests found a tuple handling error. Both were corrected. The subsequent `python -m pytest -q --ignore=test_retrieval.py` reported **42 passed, 1 skipped**. The full suite remains unverified because FAISS is absent.
- A direct check loaded the saved artifact while explicitly blocking pandas and SciPy imports; the ItemKNN and popularity paths returned results. The rebuilt Compose image measured **314,306,932 bytes** with `docker image inspect`. Its build downloaded a 16.5 MB NumPy wheel and smaller API packages; no dataset or model was downloaded or trained.
- The rebuilt container's live `/health` reported ItemKNN loaded and Redis enabled. HTTP requests returned ItemKNN, genre, and popularity strategies; the first two excluded the supplied history ID. `docker compose down` then stopped the API and Redis containers.
- This verifies serving behavior for these checks, not model quality or the remaining required integrations.

## Full suite with retrieval dependencies

- Created a separate Python 3.10 environment with access to existing system packages and installed `faiss-cpu==1.10.0` and `tensorflow-recommenders==0.7.3` with `--no-deps`. The FAISS wheel was 13.7 MB and the TFRS wheel 96 kB; no dataset was downloaded.
- From the project root, ran the isolated environment's `python -m pytest -q -p no:cacheprovider`: **45 passed, 1 skipped in 49.00 seconds**. The skip is the Spark test module because PySpark is absent. The earlier FAISS collection failure is resolved in this isolated environment; the system Python remains unchanged.
- These tests exercise the existing code and fixtures. They do not verify a live Spark job, Triton, Airflow, online A/B testing, or the quality of the deployed recommendation pipeline.

## Saved pilot FAISS index check

- Loaded `reports/retrieval-pilot/items.faiss` and `embeddings.npz` with FAISS 1.10.0. The index contains 12,224 vectors of dimension 32. Reconstructing all index vectors and comparing them to the saved `item_vectors` gave a maximum absolute difference of 0.0.
- For the first 32 saved user vectors, the FAISS top-10 sets matched direct NumPy inner-product top-10 sets for all 32. Maximum absolute returned-score difference was 0.0000019073486328125.
- This checks index/vector consistency for this pilot and sample only. It does not establish recommendation quality, independent evaluation, or equivalence for other checkpoints.

## Saved pilot embedding diagnostic

- On `reports/retrieval-pilot/embeddings.npz`, all saved user and item vectors were finite. A seed-42 random sample of 512 user vectors had L2 norms of 0.773859 / 1.546251 / 2.930737 (minimum / median / maximum); all item vectors had norms 0.393769 / 1.357240 / 6.155774.
- Before excluding seen items, direct inner-product top-1 results for these 512 users contained 209 distinct items; the most frequent, movie 20, was top for 19 users. This rules out a single identical top item for this sample, but does not establish good personalization or explain the poor validation results. It checks only the saved pilot, not the later BPR run.

## Candidate ranker replay and serving interface

- Loaded `dcn_candidate_ranker.keras` and its `feature_mapping.npz` with the existing TensorFlow 2.15 environment. For saved evaluation user 378, the replayed Recall@10 and NDCG@10 were both 0.0, matching the saved per-user record. For the first saved user with nonzero DCN NDCG, user 847, 500 candidates gave Recall@10 0.1111111111111111 and NDCG@10 0.6464018466408364, matching the saved record; 232 candidate IDs mapped to the unknown-item index. Outputs were finite for both checks.
- These are two selected replay checks, not an aggregate reproduction or a quality improvement. The saved ranker expects a training-vocabulary user ID, a training-vocabulary item ID or unknown index, and the standardized `log1p` ItemKNN score. The current demo accepts liked movie IDs rather than a trained user ID, so using this model as its default would use the untrained unknown-user embedding. The ranker also scored below the ItemKNN ordering on its recorded development comparison and remains unpromoted.

- Added `scripts/candidate_ranker_inference.py` as a reusable experimental adapter. It verifies the model and mapping hashes, applies the recorded normalization, rejects unknown users, and returns no score for unknown items so the caller must choose a fallback. It is not connected to the demo API.
- For user 847's saved 500 candidates, the adapter scored 268 known items and left 232 unknown items unscored. Scores for the known items matched a direct model call exactly in this check; an unknown user was rejected. `python -m pytest -q -p no:cacheprovider test_candidate_ranker_inference.py` reported 2 passed. No model was retrained, and this is not evidence of improved ranking quality.
- Added a candidate-to-ranker decision: a completely known candidate list can be reranked; unknown users or any unknown item retain the full retrieval order with a stated reason. Invalid inputs and model errors are surfaced. `python -m pytest -q -p no:cacheprovider test_candidate_ranker_inference.py test_itemknn_api.py` reported 8 passed. This decision is standalone and is not connected to the API or a live experiment.

## Opt-in experimental native API path

- Connected the decision to `/recommend` when the API is explicitly started with `NEXVIA_EXPERIMENTAL_RANKER_DIR` and the request supplies `experimental_ranker_user_id`. It reranks at most 100 ItemKNN candidates, or preserves retrieval order when user or candidate IDs are unknown. The cache key includes the requested user and ranker manifest hash. This feature is not enabled in Docker Compose or the browser UI.
- The first focused test run had one failure because the health response acquired a new field; the expected response was updated. The focused run then reported 20 passed. The full isolated-environment `python -m pytest -q -p no:cacheprovider` reported **49 passed, 1 skipped**; the skip requires PySpark.
- A temporary native uvicorn server loaded the saved ItemKNN and DCN artifacts. Live HTTP `/health` reported both loaded. A default history-318 request returned ItemKNN IDs 296, 593, 50, 527, 356. An opt-in user-847 request returned `experimental_dcn` and IDs 293, 4973, 2324, 527, 2329; an unknown-user request returned `itemknn_cosine`, `ranking_status=unknown_user`, and the default IDs. The server was stopped after the checks. The supplied history and user ID were synthetic request inputs, not proof that they belong together.
- This is a functional check, not a model-quality improvement or an online A/B test. The DCN remains weaker on its recorded reused development comparison, and neither the two-tower nor this experimental ranker is served by Docker Compose.

## Opt-in two-tower FAISS endpoint

- Added `scripts/two_tower_inference.py` and an optional native `/experimental/two-tower` route. The loader checks vector shapes, sorted unique ID mappings, finite vectors, and exact equality between FAISS reconstructed item vectors and the saved vectors. Unknown users fall back to popularity and supplied history is excluded.
- `python -m pytest -q -p no:cacheprovider test_two_tower_inference.py test_recommendation_api.py test_itemknn_api.py`: 20 passed. A temporary native server loaded `reports/retrieval-pilot`; HTTP for user 1 with movie 3949 excluded returned IDs 62644, 64969, 102481, 48780, 93510. An unknown user returned the popularity fallback. The server was stopped afterward.
- This checks the saved pilot and route, not recommendation quality. The two-tower pilot performed below popularity in its saved development comparison. The endpoint is not enabled in Compose or the browser. Its inner-product scores are incompatible with the ItemKNN scores used by the saved DCN ranker; the endpoints do not form a two-tower-to-DCN pipeline.

## Combined experimental route

- Extended `/experimental/two-tower` with `experimental_dcn=true`. It retrieves up to 100 FAISS candidates and computes the saved ranker's expected ItemKNN score feature from the supplied history. Returned scores remain FAISS scores aligned to the final item IDs. A fixture verifies that the DCN receives ItemKNN scores rather than FAISS scores and that FAISS scores follow reordered IDs. The relevant focused suite reported **24 passed**.
- Live HTTP loaded the three saved artifacts. For synthetic request user 847, history 318, the combined request returned the same first five IDs as FAISS alone with reason `unknown_items`; the DCN did not reorder this real request. A fixture exercised successful reordering. The server was stopped after the check.
- The pilot retriever and ranker were trained in different time windows and the DCN was trained on ItemKNN-generated candidates. Combined ranking quality is NOT VERIFIED; the saved individual models were below their recorded baselines. The prior paragraph describes the state before this route was added.

## Ranker vocabulary coverage on saved candidates

- Read the existing 2,000-user 60/70 ItemKNN development candidate export and the saved 50/60 DCN feature mapping. Of 2,000 sampled users, 386 were in the ranker's user vocabulary. Among those 386, only 95 had all of their first 100 candidate items in its item vocabulary; 291 had at least one unknown item. Across those known users' 38,600 candidate positions, 1,354 were unknown items.
- Under the current strict policy, at most 95 of these 2,000 sampled ItemKNN lists (4.75%) could be reranked; the rest would retain candidate order because the user or at least one item is unknown. This is a previously reused ItemKNN development sample, not a measurement of FAISS candidates or live traffic. It establishes a practical coverage limitation and does not evaluate an alternative fallback policy.

## Fixed fallback comparison

- After recording `docs/fallback-evaluation-protocol.md`, ran `python -m scripts.evaluate_ranker_fallback --candidates reports/development/itemknn-60-70-corrected-cohort-2k-v2/candidate_pairs.jsonl.gz --ranker reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1 --output reports/development/fallback-policy-evaluation-v1.json` in the existing isolated environment. No training or data download was performed.
- On 386 known users, including 34 without positive targets, ItemKNN Recall@10/NDCG@10 was 0.05865/0.11075; strict fallback was 0.05568/0.10675; keeping unknown positions and reranking known positions was 0.05958/0.11396. The latter reranked 386 users versus 95 for strict fallback.
- Paired user bootstrap (2,000 resamples) for known-slots minus ItemKNN had 95% intervals [-0.01100, 0.01281] for Recall@10 and [-0.00844, 0.01416] for NDCG@10. Both include zero. The alternative is not promoted: this reused, selected development sample does not establish an improvement or performance on FAISS candidates.

## Manual Airflow retraining skeleton

- Added a manual-only Airflow 3 DAG and separate workflow helper. Preflight verifies the ratings hash and ordered split cutoffs, creates a new per-run directory, then the existing streaming training script runs only if manually triggered. The final step checks result metadata and requests review; it does not promote a model. The streaming script's 500,000-interaction target is not guaranteed to be a strict cap.
- `python -m pytest -q -p no:cacheprovider test_retraining_workflow.py` reported 2 passed using small files and a stubbed training subprocess. The DAG source parsed with Python's `ast.parse`. Airflow DAG import, scheduler behavior, actual training and evaluation are NOT VERIFIED. No training or Airflow download was performed.

### Training-sample cap correction

- The sampler previously allowed the retained sample to exceed its requested limit when the number of training users exceeded that limit. It now deterministically samples users when necessary and verifies the retained interaction count does not exceed the cap. Popularity and validation history still use the full time-bounded training window; the cap does not limit CSV scanning or total memory.
- `python -m pytest -q -p no:cacheprovider test_streaming_data.py test_retraining_workflow.py` reported 3 passed on small fixtures. An initial run emitted a pandas concatenation warning; the empty-frame case was corrected and the repeated run was warning-free. No full-data training or Airflow run was performed. This supersedes the cap limitation noted immediately above.

## Local experiment prototype

- Added opt-in participant assignment, served-response records, click validation/deduplication, and per-arm descriptive counts. The two assigned variants request ItemKNN order and the experimental DCN respectively; actual strategy is recorded separately, including fallbacks. The server records a response as served, not a confirmed browser rendering. It stores a keyed participant hash, not the raw participant ID.
- `python -m pytest -q -p no:cacheprovider test_experiment_store.py test_itemknn_api.py test_recommendation_api.py` reported **21 passed**. Live local HTTP with a temporary database returned A/ItemKNN and B/DCN, accepted one synthetic click in each arm, and returned the expected synthetic counts. These checks are not real-user metrics.
- The first temporary-directory cleanup failed because the server log was still in use (Windows error 32). A subsequent port check showed no listener, and the exact temporary directory was removed. The test server is not running. There is no authenticated experiment, real traffic, live CTR, or measured online A/B outcome.

## Subsequent full suite and Spark notebook inspection

- `python -m pytest -q -p no:cacheprovider` in the isolated environment reported **60 passed, 1 skipped**. The skipped module requires PySpark.
- The existing Colab notebook visibly contains a prior synthetic Spark smoke pass and an older two-test pass using a 6,545-byte source copy. The current local `scripts/spark_prepare_features.py` is 7,162 bytes, and its current test file contains the later staging cleanup test. The notebook is disconnected. Those earlier outputs do not verify the current local code, and no new Colab run was started during this inspection.

## Triton interface contract

- Added a Python backend model and configuration at `triton/model_repository/dcn_ranker/`, reusing the saved ranker adapter. The interface returns a supported-item mask so an unknown-item placeholder is not mistaken for a rank score.
- `python -m pytest -q -p no:cacheprovider test_triton_ranker_contract.py test_candidate_ranker_inference.py`: **4 passed**. These checks use a stub of Triton's Python backend and do not start Triton or load a model in its runtime. Triton installation, actual model loading and live inference remain NOT VERIFIED. No Triton image was downloaded.



## Subsequent verification: portal, tests, and documentation

- The signed-in Nexvia Projects panel was read again. AIML-03 explicitly requests Spark features, TensorFlow Recommenders two-tower embeddings with FAISS, a deep ranker, popularity and content fallbacks with Redis, Triton serving, Airflow retraining, and an online A/B test comparing two ranking models. The checklist requires a demo, GitHub repository, documentation, results, presentation, and individual reflection. No submission was made.
- Command: `python -m pytest -q` from the project root in the isolated Python 3.10 verification environment. Result: `60 passed, 1 skipped, 2 warnings in 26.14s`; Spark was skipped because PySpark was absent locally. Both warnings reported that pytest could not write its cache (`Permission denied`).
- Live HTTP recheck: a Python subprocess started `python -m uvicorn scripts.recommendation_api:app --host 127.0.0.1 --port 18892`, polled `/health` for 40 seconds, then terminated the child. The health check timed out; the captured server log was empty. A direct uvicorn command likewise emitted no output before interruption. This attempt did not establish why startup failed and did not verify current HTTP inference. Earlier successful checks are separately documented above.
- A subsequent in-process `TestClient` attempt did not finish before it was interrupted; it was not a passing check.
- The existing Colab CPU runtime reported `PYSPARK_PRESENT True` and Java 21. The current source and current three tests were not executed in that runtime during this check; only an older two-test result exists in the notebook. Current Spark runtime verification remains outstanding.
- Documentation changes corrected stale descriptions of experimental native model routes, Triton, Airflow, and the local A/B prototype. The README now starts with a lightweight demo instead of instructing a baseline rerun. These text corrections do not validate model quality or runtime integrations.


## Follow-up: native API startup

The earlier 40-second check above was blocked by local socket permissions: `Invoke-RestMethod http://127.0.0.1:18892/health` returned `An attempt was made to access a socket in a way forbidden by its access permissions`. After local network access was enabled for the verification environment, the same venv launched `python -u -m uvicorn scripts.recommendation_api:app --host 127.0.0.1 --port 18892`; `/health` returned HTTP 200 after 2.45 seconds, and `POST /recommend` with history `[318]` and limit 5 returned HTTP 200 with the popularity strategy and five results. The server was terminated after the check. This verifies the default native HTTP path only; it does not measure recommendation quality.


## Rebuilt Docker demo and browser check

- Initial `docker compose up -d --no-build --pull never` used a stale image: `/app/scripts/recommendation_api.py` SHA-256 `d0f53a2056e7987e43b562b36b0ce09c8c228547ae1a254d4df08afd1428ec16` differed from the local file, `093a6175c4eea177cea9b54f6332ae2acfe6b97238e86e6ede37f1c76b068220`. Earlier requests against that image are not evidence for the current API.
- `docker build --network none --pull=false -t project-2-api:latest .` failed because the pip dependency layer was not cached under the no-network build mode; pip could not resolve `fastapi==0.115.2`. `docker compose build --pull=false api` then passed, with the dependency installation layer reported as `CACHED`; no package download was observed.
- After `docker compose up -d --no-build --pull never`, `docker compose exec -T api sha256sum /app/scripts/recommendation_api.py` returned `093a6175c4eea177cea9b54f6332ae2acfe6b97238e86e6ede37f1c76b068220`, matching local source. HTTP `/health`, `/`, and three `/recommend` requests all returned 200. Health reported ItemKNN and Redis enabled, experimental DCN disabled. History `[318]` selected ItemKNN and returned five movies; Drama selected genre matching; empty input selected popularity.
- Chrome at `http://127.0.0.1:8810/`: entering liked movie 318 and limit 5 showed five cards and the cosine item-neighbor strategy. This is a functionality check, not an evaluation of relevance. `docker compose down` stopped and removed the temporary containers and network. No model was trained.


## Local Spark runtime attempt

Installed `pyspark==4.0.5` and `py4j==0.10.9.9` in an isolated Python 3.10 verification environment. Java 19 was available. The first `pytest` run encountered a `PermissionError` in the default pytest temporary directory. A second run set `JAVA_HOME` to the local JDK, `SPARK_LOCAL_IP=127.0.0.1`, `PYSPARK_PYTHON` to the isolated Python, and used a fresh temporary directory; command: `python -m pytest -q --tb=short -p no:cacheprovider --basetemp <temporary-directory> test_spark_prepare_features.py`. Result: **1 passed, 2 failed**. Both failing tests encountered a Hadoop local filesystem error on Windows: `java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset`, including the Parquet write. Spark execution and the current feature job are therefore not fully verified on this machine. No Hadoop binary was downloaded. The package remains in the isolated environment for further Spark work.

## Follow-up full-suite run in an isolated temporary environment — 5 November 2026

- The current system Python 3.10 environment does not have FAISS or TensorFlow Recommenders, so its full-suite attempt stops during collection at the FAISS imports. No packages were added to that environment.
- A temporary Python 3.10 venv with `--system-site-packages` was created and removed after the run. Pinned `faiss-cpu==1.10.0` (13.7 MB wheel) and `tensorflow-recommenders==0.7.3` (96 kB wheel) were installed with `--no-deps`; pinned `numpy==1.26.4` was already available. Pip caching was disabled. No dataset, model, or checkpoint was downloaded.
- The first full run used pytest's default temp location and produced **35 passed, 1 skipped, 28 setup errors** because the sandbox process could not access `C:\Users\ramna\AppData\Local\Temp\pytest-of-ramna` (`PermissionError: [WinError 5] Access is denied`). This was an environment temp-path failure, not a test assertion failure.
- The rerun confined pytest temporary files to the temporary venv directory: `python -m pytest -q -p no:cacheprovider --basetemp <TEMP>\nexvia-p2-faiss-verify\pytest-temp`. Result: **63 passed, 1 skipped in 13.56 seconds**. The skipped module requires PySpark, which is absent from the local system environment.
- This verifies the collected non-Spark suite with the pinned FAISS/TFRS imports in an isolated environment. The Spark module and online A/B outcomes remain outside this run.