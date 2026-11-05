# BPR two-tower comparison: measured result

## Run identity and integrity

- Runtime: Google Colab T4, Python 3.13.16, TensorFlow 2.21.0, TFRS 0.7.3, FAISS 1.15.1.
- Source script: `scripts/train_bpr_objective_comparator.py`; SHA-256 `6cd990e04701ef24292e902ca06793ce3caae07e1b073e0a1da0c26724f6e003` (18,398 bytes). The uploaded copy was verified to the same hash before launch and compiled in Colab.
- Ratings: existing `/content/ml-25m/ratings.csv`, 678,260,987 bytes; SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`. No data download occurred.
- New output directory: `/content/nexvia_aiml03_bpr_70_80_seed42`; it did not exist at preflight. The run completed with exit code 0 in 425.59 seconds total; model training reported 358.59 seconds.
- Saved Colab `metrics.json` SHA-256: `7fbd7e594ada1531586bf6e91eeb44e8f95f9e161f8a249be216a3967daf3fa7`. The Colab output directory contains `metrics.json`, `per_user_metrics.csv`, `embeddings.npz`, `items.faiss`, `user_tower.keras`, and `item_tower.keras`.

## Fixed setup

The comparator keeps the corrected two-tower architecture, 32-dimensional embeddings, 17,500,066-row training interval, 2,500,010-row following validation interval, 8,750,986 positive training pairs (`rating >=4`), 512 batch size, Adagrad at 0.05, 3 epochs, and exact FAISS inner-product candidate index. It changes the training objective to pairwise BPR and samples one unrated-in-training catalog item per positive per epoch. Negative seeds were 42, 43, and 44. The official final 10% and historical 80–90% panel were not used.

## Results

The validation cohort contains 3,550 users with learned embeddings and 124,777 unseen positive target pairs. The candidate index represents 110,095 of those targets (88.23%).

| Metric | BPR tower (18,405 indexed items) | Popularity (22,931-item training catalog) |
|---|---:|---:|
| Candidate Recall@100, all eligible targets | 0.13203 | — |
| Candidate Recall@500, all eligible targets | 0.29110 | — |
| Recall@100, targets restricted to the same indexed items | 0.14519 | 0.14821 |
| Recall@10 on each system's available catalog | 0.03187 | 0.03362 |
| NDCG@10 on each system's available catalog | 0.09818 | 0.10279 |

Paired user bootstrap (2,000 resamples, seed 42), BPR minus popularity on the same indexed candidate catalog:

| Metric | Median paired difference | 95% interval |
|---|---:|---:|
| Recall@100 | -0.00303 | [-0.00463, -0.00137] |
| Recall@10 | -0.00190 | [-0.00279, -0.00107] |
| NDCG@10 | -0.00484 | [-0.00634, -0.00329] |

The BPR objective produced a much stronger candidate ranking than the earlier softmax two-tower on this same reused window: matched-catalog Recall@100 moved from 0.00494 to 0.14519. However, its paired comparison against popularity remains negative with the interval below zero, and its top-10 metrics are also lower. This is evidence of a substantial development-panel gain over that weak prior tower, not evidence that BPR is better than popularity or that it generalizes. The next model-choice step must use a newly frozen time-disjoint evaluation window. Do not promote this checkpoint based on these reused-panel results.

## Training behavior and limits

Recorded training loss by epoch was 0.06760, 0.09940, and 0.11510. The rise means the training trace does not support a convergence claim. The run uses an explicit-feedback rating threshold as a preference proxy; sampled unrated items can be unknown rather than disliked. The window has already been used for development decisions and is not an independent test. No click-through rate or online A/B outcome was measured.

## Exact Colab command

```python
subprocess.run([
    sys.executable,
    "/content/train_bpr_objective_comparator (1).py",
    "--ratings", "/content/ml-25m/ratings.csv",
    "--output", "/content/nexvia_aiml03_bpr_70_80_seed42",
], check=True)
```

The run was started only after confirming the uploaded script hash, dataset hash, successful `py_compile`, and absent output path. The model files remain in the Colab runtime; the local repository records the metrics and run provenance, not a promoted model artifact.
