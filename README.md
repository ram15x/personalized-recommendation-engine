# Large-Scale Personalized Recommendation Engine

A two-stage recommendation prototype: retrieve a short candidate list, then rank it. The current code is an offline research prototype, not a production recommender.

The target system and what is implemented versus pending are described in [the architecture note](docs/architecture.md).

## Current state

The local Docker demo uses ItemKNN for liked movie history, popularity when there is no usable history, explicit genre matching when genres are supplied, and Redis response caching. A native experimental API loads saved two-tower and DCN artifacts. Its tested two-tower-to-DCN request fell back because the ranker did not support every retrieved candidate. A read-only vocabulary audit found a median of 62 supported items among the raw top 100 and no complete 100-item match for the 1,951 shared users; that audit did not apply user-history filtering or measure recommendation quality. The DCN candidate model reranked ItemKNN candidates in the opt-in A/B route, but its recorded offline comparison was below ItemKNN. Neither learned path is promoted as the default. The current full suite has two Spark Parquet-write failures on Windows because Hadoop `winutils.exe` is unavailable; the other 63 tests pass. Airflow 3 started in its slim container and discovered the manual DAG without import errors; its retraining task was not run. Triton serving and genuine online A/B outcomes are not verified. See the status note and dated verification report for exact evidence.

The public GitHub repository is [ram15x/personalized-recommendation-engine](https://github.com/ram15x/personalized-recommendation-engine); local `main` tracks `origin/main`. Raw MovieLens data, model checkpoints, indexes, and per-user exports are excluded from Git. A clean clone must obtain its own MovieLens files and regenerate the local popularity and ItemKNN artifacts before it can build the full Docker demo. GroupLens says it typically does not permit public redistribution of MovieLens datasets; do not commit the ratings file or derived user-level exports. See the [GroupLens dataset terms](https://grouplens.org/datasets/movielens/) and the included usage notes.

## Run it

For the CPU demo, install Docker Desktop and start it, then run from this folder:

```powershell
docker compose up --build --wait
```

Compose waits for the API health check and Redis before returning. Open `http://127.0.0.1:8810/`. Stop the services with `docker compose down`. The demo uses the included movie catalog and precomputed ItemKNN and popularity artifacts; it does not train a model or read the full ratings file. See **Docker demo (CPU)** below for the artifact paths and limitations.

For native API use, install Python 3.10, create a virtual environment, and install only the serving requirements:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-api.txt
.\.venv\Scripts\python -m uvicorn scripts.recommendation_api:app --host 127.0.0.1 --port 8000
```

The native API defaults to `reports/ml-25m/popularity.csv` and `data/ml-25m/movies.csv`. The optional ItemKNN settings are described below. Research and testing dependencies are separate; installing TensorFlow or running the offline baseline is not necessary to use the demo. Do not overwrite the saved evaluation reports when running new experiments.

### Spark feature tables

The Spark preparation job is `scripts/spark_prepare_features.py`. It writes separate Parquet tables for training interactions, the following validation interval, training-only user aggregates, genres for items in the training catalog, and training-only item popularity. Its default interval is the previously explored MovieLens 60/70 development window (cutoffs recorded in `docs/requirements-status.md`); this is a development pipeline, not an independent evaluation. Validation outcomes do not contribute to user, item, or popularity aggregates. Spark scans the source CSV to apply the time filter, but rows at or after the validation end are excluded from output tables and aggregates.

Windows note: on this machine the Spark Parquet-write tests fail because Hadoop `winutils.exe` and `HADOOP_HOME` are missing. The temporal-bound validation passes; the Parquet writer and full MovieLens Spark job remain unverified here. Use a configured Hadoop environment or Linux/Colab for runtime validation.

Install Spark separately and run from the project root:

```powershell
python -m pip install -r requirements-spark.txt
python scripts/spark_prepare_features.py `
  --ratings data/ml-25m/ratings.csv `
  --movies data/ml-25m/movies.csv `
  --output reports/spark-features-60-70
```

The output path must be new; the job refuses to overwrite it. The job writes to a temporary staging folder and publishes the completed folder only after table writes and counts succeed; a failed run removes its staging folder. This produces feature tables only. It does not train or evaluate a recommender, serve predictions, or establish online CTR. A prior synthetic Spark smoke was reported during development, but no durable run log is saved in this package; the current staging-folder cleanup change has not been run in Spark. A full 25M run was attempted in Colab but stopped because the ratings file was absent from the expected path; no data was transferred.

## Local recommendation API

The API defaults to the measured MovieLens 25M training-period popularity baseline. A request can instead supply explicit genre preferences for a transparent content-based cold-start list; a configured ItemKNN artifact can use a liked/highly-rated history. An optional Redis response cache is enabled with `NEXVIA_REDIS_URL`; it is off by default. Opt-in native API paths for the experimental two-tower pilot and DCN, including a combined experimental call, are described in [the API documentation](docs/api.md); they are not enabled in Docker Compose or the browser UI. Their saved development results were below their respective baselines, and combined ranking quality is unmeasured. The same server serves the HTML interface in `demo.html` at `/`.

With the MovieLens 25M files and the baseline artifact in place, start the API from this folder:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn scripts.recommendation_api:app --host 127.0.0.1 --port 8000
```

The default files are `reports/ml-25m/popularity.csv` and `data/ml-25m/movies.csv`. Override them with `NEXVIA_POPULARITY_PATH` and `NEXVIA_MOVIES_PATH` if needed. Check `http://127.0.0.1:8000/health`, then send a request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommend -ContentType 'application/json' -Body '{"history_item_ids":[318],"limit":10}'
```

The service rejects missing or inconsistent artifacts at startup and returns an explicit error when the supplied history excludes the entire catalog or no unseen movies match the requested genres. The content fallback uses existing movie metadata and needs no model training or extra dataset. The optional experimental ranker is not an independently validated production ranking model.

### Docker demo (CPU)

Docker Compose starts the API and Redis and loads the saved sparse ItemKNN matrix plus popularity counts built at the same 60% training cutoff. No GPU or training is needed to run it. The build context includes only the movie catalog, small precomputed tables, and ItemKNN files; it excludes the 678 MB `ratings.csv`. Serving uses NumPy and does not require pandas or SciPy.

```powershell
docker compose up --build --wait
```

Open `http://127.0.0.1:8810/`. Movie IDs entered with no genre preferences are treated as liked/highly rated history; seen items are excluded. Preferred genres select the explicit metadata fallback. With no usable history or preferences, the API uses training-period popularity. The Docker ItemKNN artifact is a 60/70 development candidate selected after examining earlier results, so it is not independent validation and does not establish production performance. The Compose demo does not serve the TFRS/FAISS two-tower or DCN ranker.

Stop the local containers with:

```powershell
docker compose down
```

### Optional ItemKNN serving path

The native API has an optional sparse ItemKNN path behind three settings: `NEXVIA_ITEM_SIMILARITY_PATH`, `NEXVIA_ITEM_IDS_PATH`, and `NEXVIA_ITEMKNN_METADATA_PATH`. The API requirements include NumPy for this path; TensorFlow and FAISS are not needed for serving it. When configuring this artifact, also set `NEXVIA_POPULARITY_PATH` to its sibling `training_popularity.csv`; this keeps popularity fallback and tie-breaking on the same time cutoff. Docker Compose does this automatically and loads the artifact through its hash-checked manifest. The live HTTP route and browser form returned five cosine-neighbor recommendations for history ID 318 and excluded 318. Requests with explicit genres still use the content fallback; requests without usable history use cutoff-aligned popularity. The ItemKNN evaluation uses a reused development window and is not independent validation. Two candidate-conditioned DCN checks on saved earlier/later exports ranked below ItemKNN ordering on the 386 shared users, so neither was promoted or served. See the [pointwise report](reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1.md) and [pairwise report](reports/development/dcn-pairwise-candidates-50-60-to-60-70-v1.md). Since ItemKNN was fit on ratings `>= 4`, supplied history should represent liked or highly rated movies; the API cannot infer preference from a movie being merely seen. `/health` reports whether ItemKNN loaded; each recommendation response names the strategy.

Redis is checked before the sparse ItemKNN calculation, so a repeated identical request can return without repeating that scoring work. The cache is versioned and uses a 300-second TTL; a regression test confirms an ItemKNN cache hit skips rescoring.

The evaluator saves `positive_item_ids.npy` and the hash-checked manifest alongside the similarity matrix. The 60/70 artifact hashes and dimensions were verified. The fixture tests still cover mapping mismatch and fallback; these tests are separate from the MovieLens evaluation results.

Both ItemKNN evaluation scripts accept `--export-candidate-lists`. The corrected evaluator selects users with any observed rating in the held-out window, then reports both all-observed warm-user metrics and the positive-target subgroup. By default it samples at most 2,000 eligible warm users (seed 42), without using rating values to select them. Set `--max-users 0` to request all eligible users. Item-item similarities are computed in 64-item blocks. The corrected 50/60 and 60/70 candidate exports each contain 2,000 user lists with 500 candidates per user; unrated candidates remain unknown, not negative labels. Two DCN candidate-ranker checks used the 50/60 export for training and evaluated the 386 shared users in 60/70. Both scored below ItemKNN order on Recall@10 and NDCG@10 and are not promoted. The earlier 86-user pilot is retained and labeled low-support.

For request/response fields and error behavior, see [the API documentation](docs/api.md).

### FAISS-aligned experimental ranker

An opt-in native API route retrieves candidates from the saved two-tower FAISS index and reranks them with a compact feature-based deep-and-cross model. On a reused development window it ranked better than FAISS order but did not clearly beat popularity order on the same candidates; absolute metrics were low. The route is not enabled in the default Docker demo. When configured in the native API, it is also available through the experimental browser form. Setup, response fields, and limitations: [API documentation](docs/api.md); results: [development report](reports/development/faiss-aligned-ranker-result.md).

### Ranking-model work in progress

The pointwise ranker previously treated IDs in its candidate vocabulary as trained even when they had no labeled training example. The native inference guard now requires labeled-training support and falls back otherwise. Earlier ranker results were produced before this correction and do not measure the updated behavior; see [the support audit](reports/development/ranker-training-support-audit.md).

`scripts/dcn_ranker.py` contains a DCN-v1 scorer. A bounded 70/80 rating diagnostic performed worse than a simple user/item-mean baseline. On retrieved candidates, the pointwise DCN scored NDCG@10 0.10127 vs 0.11075 for ItemKNN order (paired difference -0.00903, 95% interval -0.02198 to +0.00219). A pairwise objective was also worse: NDCG@10 0.05441 vs 0.11075 (difference -0.05621, interval -0.07410 to -0.03913). Neither is promoted or evidence of CTR. Both checks use reused development data. The pointwise model is available only through an opt-in native API path; Docker Compose and the browser UI do not enable it. See the [pointwise report](reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1.md), [pairwise report](reports/development/dcn-pairwise-candidates-50-60-to-60-70-v1.md), [earlier diagnostic](reports/development/dcn-rating-diagnostic-70-80-2026-11-05-corrected.md), and [ranker protocol](docs/ranking-model-protocol.md).

## Data

Official source: https://grouplens.org/datasets/movielens/25m/

Download: https://files.grouplens.org/datasets/movielens/ml-25m.zip

Keep the supplied README with the data and review its usage terms before sharing datasets. Raw data is excluded from Git. Share download instructions rather than copying the dataset into the final public folder until its terms have been checked.

`ml-latest-small` is only a development fixture, downloaded separately from GroupLens. Its scores are not results on MovieLens 25M.

## What the baseline measures

- Split all ratings by global time: about 80% training, 10% validation and 10% test. Equal timestamps stay together.
- Treat a rating of 4 or 5 as a positive interaction.
- Rank movies by the number of positive training ratings. Break ties by movie ID.
- Recommend from the complete training catalog, excluding anything the user rated before that evaluation window.
- Keep the model fixed after training. Validation interactions are added to the history used to filter test recommendations, not to popularity counts.
- Compute binary Recall@10 and NDCG@10 per user, then average users equally.
- Count positive movies absent from the training catalog as misses. Report that fraction instead of hiding it.
- Report warm and cold users separately. Here cold means no observed history before the evaluation window; later embedding models must separately track users absent from model training.

This is an offline rating-based experiment. A high rating is a proxy for preference, not an observed click. It cannot establish click-through rate or an online A/B-test improvement. Users without positive targets are excluded from ranking averages and counted separately. The model and candidate catalog are intentionally frozen; new-item availability is not modeled.

## Current development status

The source, documentation, tests, and presentation are published in the public [GitHub repository](https://github.com/ram15x/personalized-recommendation-engine). Raw MovieLens data and local model artifacts remain outside Git.

The folder also contains ID-only TensorFlow Recommenders two-tower retrieval pilots with exact FAISS inner-product search. The saved 500,000-interaction pilot scored below popularity on its reused 2,000-user validation cohort, especially for users with learned embeddings. A fixed genre-profile candidate diagnostic was also below popularity on that same cohort. Neither result is an independent test.

A saved Colab run using all 9.98 million training-positive pairs also scored below popularity on the same reused validation window. Its warm-user Recall@10 was 0.00026 versus 0.01902 for popularity; the saved output and its limitations are recorded in [the Colab run audit](reports/existing-colab-retrieval-audit.md). This finding means the larger sample did not fix the observed ranking problem; the audit does not identify its cause.

An additional time-window audit split the existing first 80% period into an earlier 70% training window and a following 10% validation window. Of the 19,403 users with a positive target, 81.65% had no prior rating history; 25.93% of distinct positive-target items were absent from the earlier training catalog. Historical retrieval checkpoints trained through this validation period, so they cannot be compared on it as held-out candidates. See [requirements and status](docs/requirements-status.md), [experiment protocol](docs/experiment-protocol.md), and the detailed counts in `reports/development/temporal-split-audit.json`.

The first three-epoch Colab evaluation on the 70/80 development window used only the 18,405 positively rated training items as its catalog denominator; its reported 100% coverage was invalid. The corrected run used all 22,931 movies rated in the training window. The two-tower indexed 18,405 (80.26%). Among 3,550 users with learned embeddings, retrieval Recall@100 was 0.00438 on all eligible targets; on the matched subset whose target items were indexed, retrieval Recall@100 was 0.00494 versus 0.14821 for popularity. Warm top-10 Recall@10 was 0.00045 with retrieval versus 0.03362 with popularity. The 15,843 users without history and 10 users with history but no embedding used popularity fallback, so overall scores are not evidence of personalized retrieval. This result does not support shipping the learned retriever. The split is a development window, not an independent test. See the [corrected run report](reports/development/two_tower_70_80_fullcatalog_2026-11-05.md) and [metrics](reports/development/two_tower_70_80_fullcatalog_2026-11-05.json); the earlier, flawed output remains preserved in the [initial diagnostic](reports/development/two_tower_70_80_colab_diagnostic.md).

Reports contain the dataset checksum, cutoffs, counts, metrics, popularity scores and per-user recommendations. Keep test results for reporting; use validation when choosing models and parameters.

A fixed item-item cosine retrieval candidate was compared with popularity on temporal development windows. Earlier saved 50/60 and 60/70 reports used users selected for at least one future rating >=4. The corrected 60/70 run sampled 2,000 training-positive users with any observed rating in the held-out window (seed 42), including 30 users with no >=4 target. On all 2,000 users, ItemKNN vs popularity matched Recall@100 was 0.23527 vs 0.16095 (paired bootstrap difference +0.07439; 95% CI +0.06753 to +0.08139); matched NDCG@10 was 0.14733 vs 0.12411 (+0.02324; 95% CI +0.01596 to +0.03015). Matched Recall@10 difference was +0.00188 (95% CI -0.00317 to +0.00672), which does not establish a difference. The positive-target subgroup contains 1,970 users; catalog coverage for positive target pairs was 73.1%. This is still a reused development window: ItemKNN was selected after earlier results, so it is not independent confirmation. It measures retrieval only, not ranking, CTR, or online lift. The native API defaults to popularity when ItemKNN artifacts are not configured; Docker Compose configures the ItemKNN candidate. Corrected run details: [metrics](reports/development/itemknn-60-70-corrected-cohort-2k-v2/metrics.json); historical reports: [50/60](reports/development/itemknn-50-60-results-2026-11-05.md) and [60/70](reports/development/itemknn-60-70-replication-results-2026-11-05.md).

## Next steps

1. Keep the native API fallback as popularity when no ItemKNN artifact is configured. Docker Compose uses the evaluated ItemKNN artifact for liked-history recommendations, with popularity fallback. The development window was reused, so its measurements are not independent confirmation.
2. Two candidate-conditioned DCN objectives were evaluated; neither beat the unchanged ItemKNN order, so neither is promoted. Keep the API on its current default until an alternative passes the same-candidate, paired evaluation gate. The official final 10% remains untouched.
3. The Docker Compose `/health`, ItemKNN history path, explicit-genre path, and browser form were checked on 5 November 2026. The ItemKNN route works for this configured artifact; its validation panel was reused during development, so do not claim independent model validation.
4. Redis was enabled in the checked Compose run. The current Spark Parquet-write tests fail on this Windows setup because Hadoop winutils.exe is missing. The Airflow 3 standalone runtime discovered the manual DAG with no import errors, but no retraining task was run. Triton serving remains unverified. A/B code has synthetic checks only; real-user outcomes and CTR are unmeasured because MovieLens has no click/impression traffic.
5. Complete the demo, architecture/data/API documentation, results, presentation and reflection after the measured pipeline is stable.

The portal lists these integrations as required; the items above are not yet complete. A genuine online A/B test needs traffic and a defined measurement window. Offline MovieLens metrics must not be described as CTR.

## Dataset usage note (checked against the downloaded README)

The MovieLens 25M README requires acknowledgement, prohibits redistribution without separate permission, and requires permission for commercial or revenue-bearing use. Keep raw data out of the public submission folder; include the official download URL instead. Clarify any intended company deployment with the manager before moving beyond an educational research prototype.

Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5(4). https://doi.org/10.1145/2827872






