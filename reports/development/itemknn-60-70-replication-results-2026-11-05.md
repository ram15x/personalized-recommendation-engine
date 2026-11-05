# ItemKNN 60/70 temporal replication results

## Run identity and data

- Evaluator: `scripts/evaluate_itemknn_60_70_replication.py`, SHA-256 `db1450aa7f598eadfcbf079b1e545143dc042aa9c7bd28fb73d72544ced6fbd2` (13,787 bytes).
- MovieLens `ratings.csv` SHA-256: `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`.
- Colab T4 run returned exit code 0. The metrics file SHA-256 shown by the runtime was `25a0372b0c632d2caa1ebdd95d57a09eaa8130d9c1d19e9264f720a127e80e94`; the JSON reconstructed in this folder matches that hash exactly. The raw per-user CSV and sparse similarity artifact were not transferred to this folder.
- Time split: 15,000,057 training rows before timestamp `1,288,571,552`; 2,500,009 validation rows through timestamp `1,421,944,985`. Equal timestamps stay together. No official final 10% or historical 80–90% window was used.
- Positive target is rating >= 4. This is an implicit-preference proxy, not a click. Item similarity is cosine-normalized binary co-rating, top 100 neighbors per item; user scores sum similarities from positive history. All previously rated items are excluded.
- Support: 4,820 warm users, 223,407 eligible positive target pairs; 163,419 target pairs (73.15%) were in the train catalog. Popularity was compared on the same users and catalog.

## Results

| Metric | ItemKNN | Popularity | Support/notes |
|---|---:|---:|---|
| Recall@10, all eligible targets | 0.03873 | 0.03502 | 4,820 users |
| NDCG@10, all eligible targets | 0.15279 | 0.12393 | 4,820 users |
| Recall@10, targets in shared catalog | 0.04770 | 0.04142 | 163,419 target pairs across the same 4,820 users |
| NDCG@10, targets in shared catalog | 0.15410 | 0.12495 | Same matched support |
| Recall@100, all eligible targets | 0.19215 | 0.13593 | 4,820 users |
| Recall@100, targets in shared catalog | 0.23987 | 0.16303 | Same matched support |
| ItemKNN Recall@500, all eligible targets | 0.45285 | Not recorded | Do not infer a popularity comparison at 500 |

Paired user bootstrap on the matched catalog-support metrics (2,000 resamples, seed 42) gave ItemKNN-minus-popularity median differences: Recall@100 `+0.07689` (95% interval `[+0.07209, +0.08152]`), Recall@10 `+0.00637` (`[+0.00333, +0.00934]`), and NDCG@10 `+0.02928` (`[+0.02462, +0.03392]`).

## Interpretation and limits

This is a positive temporal development replication against popularity on this 60/70 window. It is not independent model selection: ItemKNN was chosen after viewing 50/60 results. Both windows are development data. Ratings are preference proxies, unrated movies are unknown, and no click or online A/B evidence exists. This candidate does not include a second-stage ranker. The current API remains the popularity-only baseline.

### Evidence files

- Frozen replication protocol: `reports/development/itemknn-60-70-replication-preregistration-2026-11-05.md`.
- Reconstructed metrics JSON (content SHA-256 verified against the Colab runtime output): `reports/development/itemknn-60-70-metrics.json`.
- Evaluator source: `scripts/evaluate_itemknn_60_70_replication.py`.
