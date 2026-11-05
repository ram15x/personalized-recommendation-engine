# DCN rating diagnostic — development result

## Decision

**Do not promote the DCN checkpoint.** On this fixed development diagnostic it scored worse than the simple user/item-mean baseline on rating error and on the observed-item ranking metrics below. This is not an end-to-end recommender evaluation, and the 70/80 window was reused earlier in Project 2 development.

## Run setup

- Dataset: local MovieLens 25M `ratings.csv`; SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`.
- Time boundaries: train rows before timestamp `1421944985` (70% boundary); validation rows from that boundary up to but excluding `1466837400` (80% boundary). The official final 10% was not used for fitting or scoring.
- Full period sizes: 17,500,066 train rows and 2,500,010 validation rows.
- Training: deterministic 500,000-row reservoir sample from the 70% training period; 2,000 validation users sampled with at least five validation ratings each; 252,294 validation ratings; seed 42; DCN-v1 with 32-d user/item embeddings, three cross layers, MLP (128, 64), dropout 0.1; Adam 1e-3; MSE; batch 4,096; four epochs.
- Baseline: clipped `user_mean + item_mean - global_mean`, with each missing user/item side independently falling back to the training global mean.
- Label: explicit MovieLens rating regression. Binary relevance for the diagnostic ranking table is observed rating >= 4.

## Results

| Measure | Mean/item baseline | DCN | DCN minus baseline |
|---|---:|---:|---:|
| Rating RMSE (252,294 observed rows) | 1.03752 | 1.16772 | +0.13020, worse |
| Rating MAE (252,294 observed rows) | 0.78367 | 0.87150 | +0.08783, worse |
| Observed-item macro Recall@10 (1,995 users with a positive) | 0.25236 | 0.24227 | -0.01008 |
| Observed-item macro NDCG@10 (same users) | 0.74256 | 0.70903 | -0.03353 |

Warm/cold breakdown for the observed-item ranking diagnostic:

| Group | Users | Baseline Recall@10 | DCN Recall@10 | Baseline NDCG@10 | DCN NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Warm (training history exists) | 357 | 0.41806 | 0.39373 | 0.70070 | 0.64199 |
| Cold (no training history) | 1,638 | 0.21624 | 0.20926 | 0.75168 | 0.72364 |

DCN raw (unclipped) sensitivity metrics were RMSE 1.16864 and MAE 0.87244; the primary comparison above clips its predictions to the valid 0.5–5.0 rating range, matching the baseline's clipping.

The ranking candidates here are the movies those users actually rated in the validation window, not unseen items retrieved by ItemKNN. These figures therefore do **not** measure top-N recommendation quality. No confidence interval was computed, and these metrics should not be used to claim an improvement.

Training and validation MSE both decreased over four epochs (train `2.0439, 0.7866, 0.7088, 0.6458`; validation `1.5859, 1.4881, 1.4613, 1.3657`). Despite the falling loss, the final DCN was worse than the mean/item baseline in this comparison. Lower loss alone did not establish usefulness.

## Reproducible files

- Training code: `scripts/train_dcn_rating_diagnostic.py`
- Model: `scripts/dcn_ranker.py`
- Run report: `reports/development/dcn-rating-diagnostic-70-80-2026-11-05-clipped/metrics.json`
- Per-user metrics: `reports/development/dcn-rating-diagnostic-70-80-2026-11-05-clipped/per_user_metrics.csv`
- Saved DCN: removed during project cleanup because this diagnostic checkpoint was rejected; the metrics and per-user evaluation artifacts remain available in this run directory.
- Checkpoint SHA-256: `24b7c5c0ba8fbc20524a884e704d45e98c8a84109e7ab325dad431b8759378b8`
- Supporting user/item ID mappings and numeric normalization arrays are in the same run directory; their hashes are recorded in `metrics.json`.

Exact command:

```powershell
python -m scripts.train_dcn_rating_diagnostic --ratings data\ml-25m\ratings.csv --movies data\ml-25m\movies.csv --output reports\development\dcn-rating-diagnostic-70-80-2026-11-05-clipped --max-train-rows 500000 --validation-users 2000 --min-validation-ratings 5 --epochs 4 --batch-size 4096 --seed 42
```

## Corrections and failed attempts

1. Running the file as `python scripts\train_dcn_rating_diagnostic.py` failed before training with `ModuleNotFoundError: No module named 'scripts.dcn_ranker'`; the package-module command above is the working invocation.
2. The first training attempt found a mismatch between the maximum metadata movie ID and the train-period item array, then stopped before creating run files. The genre matrix was resized to the maximum metadata ID; IDs unseen in training receive zero genre features.
3. The first completed run used a faulty baseline fallback; its `metrics.json` is marked **SUPERSEDED**. A second run corrected the baseline but compared raw DCN predictions with a clipped baseline; its `metrics.json` is also marked **SUPERSEDED**. The clipped-comparison run above uses independent user/item fallbacks and the same valid rating range for both models; it is the only result to quote.

## Next development gate

Build and freeze time-safe candidate-pair examples from the ItemKNN retriever, then compare a simple rating scorer and DCN on the exact same retrieved candidates. Report retrieval ceiling separately from reranking, preserve warm/cold support, and keep the final 10% untouched. If the DCN does not beat the simple ranker under that protocol, retain the simpler scorer. No clicks or online A/B data are available, so CTR remains unmeasured.
