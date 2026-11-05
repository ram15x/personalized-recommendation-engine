# Project 2 — Local end-to-end verification
Date: 5 November 2026

## What passed

- Docker Desktop engine: Docker 29.8.0 / Docker Desktop 4.92.0, Linux engine.
- Command: docker compose build --pull=false api
  Result: passed. The Python base image and pinned API dependency layer were already cached. The build context was 96.91 kB. No package or dataset was downloaded during this build.
- Command: docker compose up -d --no-build
  Result: API and Redis started; Redis passed its health check.
- Live HTTP checks against http://127.0.0.1:8810:
  - GET /health: HTTP 200; status ok; 13,173 catalog entries; ItemKNN loaded; Redis enabled; no Redis error. Neural retrieval/ranker models and the experiment store were disabled.
  - GET /: HTTP 200.
  - POST /recommend with history_item_ids [318], limit 5: HTTP 200; strategy itemknn_cosine; returned IDs 296, 593, 50, 527, 356; the history item 318 was excluded.
  - POST /recommend with preferred_genres [Drama], limit 5: HTTP 200; strategy user_genre_content; all five returned entries reported Drama as a matched genre.
  - POST /recommend with an empty request, limit 5: HTTP 200; strategy training_period_popularity.
- Command: docker compose down
  Result: API, Redis, and the Compose network were stopped and removed. A subsequent compose ps showed no running services.

## Test suite

The full test suite ran in the existing isolated Python 3.10 verification environment with PySpark 4.0.5 installed.

- Unprivileged run: 64 passed, 2 Spark setup errors. The local Java gateway could not open a loopback socket (WinError 10013), which is a sandbox permission failure.
- Local-network-enabled run: 64 passed, 2 failed. Both failures are test_spark_prepare_features.py Parquet-write tests. Hadoop raised: “HADOOP_HOME and hadoop.home.dir are unset.” Spark's Windows shell also reported that winutils.exe was not found.
- Therefore, the non-Spark suite passed in this run. The Spark Parquet job and its two write-path tests did not pass on this Windows setup. The output does not establish that Spark works end to end.

Follow-up command, excluding the Spark test module entirely:

```powershell
python -m pytest -q -p no:cacheprovider --ignore=test_spark_prepare_features.py --basetemp <temporary-directory>
```

Result: **63 passed in 19.79 seconds**. This confirms the non-Spark tests in the current checkout; it does not resolve the two Spark write-test failures or validate the Spark job.

Local-network-enabled full-suite command (PowerShell):
$env:JAVA_HOME='<path-to-local-jdk>'
$env:SPARK_LOCAL_IP='127.0.0.1'
$env:PYSPARK_PYTHON='<path-to-isolated-python>'
python -m pytest -q -p no:cacheprovider --basetemp <temporary-directory>

## Scope and limitations

- The Docker demo currently serves ItemKNN with popularity fallback and explicit genre matching. It does not serve the learned TFRS/FAISS retriever or the candidate DCN rankers.
- Learned-model comparisons and ItemKNN metrics use development panels reused during model work. They are not independent test results and do not establish online lift.
- At the time of this initial check, Triton and Airflow runtimes were not started. The Airflow follow-up below supersedes the Airflow portion; Triton remains unverified.
- A/B assignment and click-recording code has synthetic tests only. No real users, impressions, clicks, or CTR measurements are available.
- No MovieLens Spark feature-table run was completed. No training or data download occurred in this verification pass.
- At the time of this verification run, before the public repository was created, the Project-2 folder had local history but no configured GitHub remote. The public release was published later; see the current README and requirements status.
- At inspection, the folder contained 267 files totaling 891.92 MiB; data/ml-25m/ratings.csv accounted for 646.84 MiB. The ratings file is the source needed for the full offline experiments; the Docker demo excludes it from the build context.

## Status

The local Docker demo and its three user-facing recommendation strategies are verified over HTTP. The entire assignment is not complete: Spark runtime validation, Triton inference, execution of an Airflow retraining task, genuine online A/B measurement, presentation, reflection, and portal submission remain outstanding. Do not describe this as production-ready or as a proven improvement over an independent test set.

## Additional current-state checks

The Compose check was repeated against the current folder on 5 November. `docker compose build --pull=false` succeeded from cached layers with a 2.60 kB build context. Live HTTP returned `/health` status `ok`, `catalog_items=13173`, `itemknn_loaded=true`, `redis_cache_enabled=true`, and `redis_last_error=null`. `POST /recommend` with history item 318 returned ItemKNN recommendations 296, 593, 50, 527, and 356; 318 was excluded. A Drama request returned five items with Drama matches. A request with no history used `training_period_popularity`. The API, Redis service, and Compose network were stopped afterward.

The native API was also started with the saved retrieval, ItemKNN, and DCN artifacts. `/health` reported the two-tower model, ranker, and experiment store loaded. A live `/experimental/two-tower` request for MovieLens development user 139 returned five retrieved items with `strategy=experimental_two_tower` and `reason=unknown_items`; the DCN did not rerank that set because its training support did not cover every retrieved item. The live `/experimental/ab/recommend` endpoint assigned one synthetic participant to each arm: A used `itemknn_cosine`, B used `experimental_dcn`. One synthetic click was sent for the first item in each response, and `/experimental/ab/summary` returned five item positions and one click per arm (item CTR 0.2 each). This verifies endpoint plumbing only. It is not real user behavior, an online result, or a valid CTR estimate. The temporary SQLite event database was removed after the smoke check.

Fresh test reruns on the current folder: the non-Spark suite returned **63 passed in 17.96 seconds**. The Spark module returned **1 passed, 2 failed**; the two Parquet-writing tests fail with `HADOOP_HOME and hadoop.home.dir are unset` and the Hadoop Windows `winutils.exe` requirement. No full MovieLens Spark job was run. At the time of this test run, no GitHub remote had been configured. The current public repository and later synchronization are recorded in `docs/requirements-status.md`.

## Two-stage candidate support audit

`reports/development/two-stage-compatibility-2026-11-05.json` records a read-only comparison between the saved `reports/retrieval-pilot` FAISS index and the trained item support in the pointwise candidate DCN. The audit found 1,951 shared users, 5,272 shared items, and a mean supported fraction of 0.6016 across the raw top 100. Per-user supported counts ranged from 16 to 99, with a median of 62; no shared user had all 100 supported. It did not apply seen-item filtering, read ratings, or measure ranking quality. The API's all-candidate support guard therefore avoids mixing scored and unscored candidates, but the audit does not prove a candidate-filtering change would improve recommendations.

## Same-day Docker recheck

Using the already-cached `project-2-api` and `redis:7-alpine` images with `docker compose up -d --no-build`, live HTTP returned `/health` status `ok`, catalog size 13,173, ItemKNN loaded, and Redis cache enabled. `GET /` returned HTTP 200. `POST /recommend` for history item 318 returned five recommendations with strategy `itemknn_cosine` and excluded 318. A Drama-preference request returned the `user_genre_content` strategy and all five results matched Drama. An empty request returned five results using `training_period_popularity`. The API, Redis container, and Compose network were removed with `docker compose down --remove-orphans`. No images or data were downloaded.

## Airflow and Triton runtime follow-up

On 5 November, the official `apache/airflow:slim-3.3.1-python3.12` image was pulled (about 266 MB compressed according to Docker Hub) and run with the project's `airflow/dags` folder mounted read-only. `airflow standalone` initialized a temporary SQLite metadata database. `airflow dags list` showed `nexvia_manual_retraining`; `airflow dags list-import-errors` returned `No data found`. The DAG was paused, and no DAG task, preflight, or training was triggered. The temporary container and downloaded image were removed after the check.

Triton was not run. The NVIDIA registry manifest for `nvcr.io/nvidia/tritonserver:25.10-py3-min` totals about 4.5 GB of compressed image layers; the `26.10-py3` manifest totals about 8.8 GB. No Triton image was downloaded. The Python backend adapter therefore remains unverified in a Triton server. Its contract tests do not establish runtime inference.

The full suite was rerun on 5 November from the existing Python 3.10 verification environment: **64 passed, 2 failed**. Both failures are Spark Parquet-write tests on Windows; the errors report missing `HADOOP_HOME`/`hadoop.home.dir` and `winutils.exe`. No new training or dataset download occurred.

