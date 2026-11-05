# ItemKNN 50/60 temporal development results

## Run identity and data

- Evaluator: `scripts/evaluate_itemknn_50_60.py`, SHA-256 `ce77bd8c8d293bafcac4b2b7914e994814ff94fae7cb67b581a23dcfdf248f4b` (13,823 bytes).
- MovieLens `ratings.csv` SHA-256: `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`.
- Colab T4 run returned exit code 0. The metrics file SHA-256 shown by the runtime was `16471cd70d1cc35e9bf359fcc112c01bc7eccaee92106f05d27a4097c45fe997`; the JSON reconstructed in this folder matches that hash exactly. The raw per-user CSV and sparse similarity artifact were not transferred to this folder.
- Time split: 12,500,047 training rows before timestamp `1,198,868,375`; 2,500,010 validation rows through timestamp `1,288,571,552`. Equal timestamps stay together. No official final 10% or historical 80–90% window was used.
- Positive target is rating >= 4. This is an implicit-preference proxy, not a click. Item similarity is cosine-normalized binary co-rating, top 100 neighbors per item; user scores sum similarities from positive history. All previously rated items are excluded.
- Support: 4,274 warm users, 185,309 eligible positive target pairs; 130,011 target pairs (70.16%) were in the train catalog. Popularity was compared on the same users and catalog.

## Results

| Metric | ItemKNN | Popularity | Support/notes |
|---|---:|---:|---|
| Recall@10, all eligible targets | 0.04019 | 0.03262 | 4,274 users |
| NDCG@10, all eligible targets | 0.15449 | 0.12230 | 4,274 users |
| Recall@10, targets in shared catalog | 0.05483 | 0.04339 | 130,011 target pairs across the same 4,274 users |
| NDCG@10, targets in shared catalog | 0.15751 | 0.12424 | Same matched support |
| Recall@100, all eligible targets | 0.18739 | 0.13542 | 4,274 users |
| Recall@100, targets in shared catalog | 0.25153 | 0.17710 | Same matched support |
| ItemKNN Recall@500, all eligible targets | 0.41117 | Not recorded | Do not infer a popularity comparison at 500 |

Paired user bootstrap on the matched catalog-support metrics (2,000 resamples, seed 42) gave ItemKNN-minus-popularity median differences: Recall@100 `+0.07433` (95% interval `[+0.07006, +0.07901]`), Recall@10 `+0.01139` (`[+0.00863, +0.01410]`), and NDCG@10 `+0.03327` (`[+0.02854, +0.03798]`).

## Interpretation and limits

This is a positive offline development comparison against popularity on this particular 50/60 temporal window. ItemKNN was selected after inspecting earlier development results, so the window is not an independent final test. Ratings are preference proxies, unrated movies are unknown rather than confirmed negatives. This evaluates recommendation ordering only: it does not establish online CTR, A/B lift, production behavior, or the required second-stage ranker. The current API remains the popularity-only baseline.

### Evidence files

- Frozen protocol: `reports/development/itemknn-50-60-preregistration-2026-11-05.md`.
- Reconstructed metrics JSON (content SHA-256 verified against the Colab runtime output): `reports/development/itemknn-50-60-metrics.json`.
- Evaluator source: `scripts/evaluate_itemknn_50_60.py`.
