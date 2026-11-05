# Two-tower retrieval

This is the first learned candidate generator. User IDs and movie IDs pass through separate embedding and dense layers. Their dot product is the retrieval score. The model uses TensorFlow Recommenders model infrastructure and a custom in-batch softmax objective. It masks other known-positive items from the same user and repeated movie IDs. TFRS 0.7.3 here does not accept the newer `score_mask` argument, so the code applies this mask directly.

Architecture reference: https://www.tensorflow.org/recommenders/examples/basic_retrieval

## Environment

The project-local `.venv-retrieval` was removed during the November 4, 2026 cleanup because virtual environments are generated files and are not portable. Recreate it from this project folder with Python 3.10:

```powershell
python -m venv .venv-retrieval
.\.venv-retrieval\Scripts\python -m pip install -r requirements-retrieval.txt
```

The pinned stack uses TensorFlow 2.15 / Keras 2, TensorFlow Recommenders 0.7.3 and FAISS CPU. Installation may download packages; this cleanup did not reinstall them. Do not upload the recreated environment.
## Run the pilot

From the project folder in PowerShell:

```powershell
.\.venv-retrieval\Scripts\python retrieval.py --ratings data/ml-25m/ratings.csv --split-report reports/ml-25m/metrics.json --streaming --output reports/retrieval-streaming-pilot --dataset-label ml-25m --interaction-mode all-ratings --sampling-strategy recent-per-user --interactions-per-user 3 --epochs 3 --max-train-positives 500000 --eval-users 2000
```

The default pilot samples a maximum of 500,000 interactions after the global time split by taking up to about three recent interactions from each user. It is not a full-data training run. Vocabulary comes only from those sampled training interactions. Use `--sampling-strategy random` to compare a random interaction sample, and `--interaction-mode high-ratings` to train only on ratings >=4. The default `all-ratings` mode treats every observed rating as an implicit positive; evaluation still uses rating >=4, so interpret that comparison carefully. A zero training limit uses all interactions; a zero evaluation limit uses every eligible validation user.

## Evaluation choices

- Uses the same time boundaries and rating >= 4 relevance rule as the baseline.
- Keeps held-out positives outside the learned catalog in the recall denominator.
- Searches every indexed item with exact FAISS inner product. This is not an approximate index or a large-scale latency claim.
- Removes previously rated movies. Expands the search when filtering exhausts the shortlist.
- Falls back to training popularity for users without a trained embedding. Fills short neural lists from the full training catalog.
- Reports the baseline on precisely the same validation users, plus trained-user and fallback-user groups.
- Does not evaluate the test set during training or model selection.

This first model uses IDs only. Content features, popularity-bias correction, ranking, serving, orchestration and online experiments remain to be built. Early 500,000-interaction pilots are below popularity on validation; do not treat them as an improvement. The full-data local CPU attempt and recent-per-user preparation attempt did not produce complete runs.

## Saved files

Each run saves `metrics.json`, `validation_users.csv`, `items.faiss`, `embeddings.npz`, `user_tower.keras` and `item_tower.keras`. The NPZ includes ID mappings, embeddings and the popularity fallback. Only load artifacts you trust.

To inspect a saved recommendation list:

```powershell
.\.venv-retrieval\Scripts\python recommend.py --model reports/retrieval-pilot --ratings data/ml-25m/ratings.csv --movies data/ml-25m/movies.csv --user-id 1
```

This command filters history at the training cutoff. It is an offline inspection tool for anonymized MovieLens IDs, not recommendations for the intern.

The first pilot is a useful end-to-end implementation but scores below popularity. A full CPU run was attempted; its training process disappeared before producing saved outputs. Use the saved pilot for reference, and continue with stratified sampling or Colab before spending time tuning model dimensions.






