# Controlled two-tower development run

## Result and scope

One T4 Colab run completed on the frozen 70/80 MovieLens 25M time window. The run trained an ID-only two-tower model for three epochs, then wrote metrics, per-user rows, top-10 examples, embeddings, a FAISS index, and tower models under `/content/nexvia_aiml03_dev_70_80`. The Colab output was read and its metrics copied to `two_tower_70_80_colab_metrics.json`. **The model artifacts remain in Colab temporary storage; they were not copied into this project.**

This is a development diagnostic, not a result to present as a valid full-catalog comparison. Inspection found that the run script built its catalog from positive-only training rows. The audited split has 22,931 distinct items in all training ratings but only 18,405 items in ratings >=4. Thus the reported 100% catalog coverage is an artifact of using the 18,405-item positive subset as both numerator and denominator. The controlled run indexed only that subset and also compared popularity only within it. The project's established popularity baseline includes every item seen in training, including items with zero positive ratings (`baseline.py`, `popularity_order`). The run therefore does not satisfy the full-training-catalog comparison in `docs/experiment-protocol.md` and must not be cited as a valid full-catalog model comparison.

## What the run actually reported

| Cohort / measure | Two-tower candidate list | Popularity | Support |
|---|---:|---:|---:|
| Warm-user Recall@10 | 0.000412 | 0.033624 | 3,550 users; 124,777 eligible positive targets |
| Warm-user NDCG@10 | 0.001836 | 0.102786 | same |
| Warm-user candidate Recall@100, same indexed catalog | 0.005030 | 0.148211 | 3,550 users; 110,095 in-index target pairs |
| All-user Recall@10 | 0.069557 | 0.075633 | 19,403 users; 1,230,107 eligible positive targets |
| All-user NDCG@10 | 0.313277 | 0.331747 | same |

The warm/all Recall@10 and NDCG@10 values above are from the script's hybrid list: retrieval results first, then popularity fill. Candidate Recall@100 is from raw retrieval. These values are faithfully copied from the run, but are restricted to the positive-only catalog and are not a defensible final comparison. The low warm-user values show that this particular run did not establish useful personalized retrieval.

## Run configuration and integrity checks

- Dataset: official MovieLens 25M `ratings.csv`; SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385` (verified by the script before parsing).
- Split: train `timestamp < 1421944985`; validation `1421944985 <= timestamp < 1466837400`; each split row count matched the audit. The script states it did not use the old 80–90% validation or final 10% periods.
- Training: 8,750,986 rating>=4 user-item pairs, batch size 512, 32 dimensions, Adagrad 0.05, 3 epochs. Reported fit time 246.20 seconds; loss values 5.9745, 5.9743, 5.9746.
- Validation support: 19,403 users with positive targets; 3,550 have positive-item embeddings. The split audit's 3,560 “seen in train” count uses any prior rating; the run's 3,550 “warm” count requires at least one prior rating>=4, so those counts describe different cohorts.
- Runtime: Google Colab GPU runtime; TensorFlow reported a physical GPU. The session UI identified a T4. The run used about 1.1 GB of temporary Colab disk at completion. The UI showed approximately 10 minutes elapsed for the cell; that display is not an exact wall-clock measurement.
- It downloaded/extracted MovieLens only in Colab `/content`; it did not create another local dataset copy.

## Corrective action before any further model comparison

The local `scripts/train_dev_retrieval.py` now tracks all training item IDs, ranks popularity across the full catalog, reports validation target availability, separates candidate-generation metrics from the hybrid retrieval-plus-popularity-fill ranking metrics, and bounds the popularity scans. `python -m py_compile scripts/train_dev_retrieval.py` completed on Python 3.10.1. The corrected Colab run was **not completed**: the notebook patch cell failed with `SyntaxError: unterminated string literal (detected at line 6)` before launching training. Thus the original run's metrics remain diagnostic only. Re-run the corrected script on the same frozen split before selecting hyperparameters or claiming an improvement.

Process note: I also proceeded to the full three-epoch run before completing the protocol's proposed short sanity checks for gradients, ID mapping, and score distributions. Those checks remain undone. That was a sequencing mistake; no claim of model quality or improvement follows from the run.

## Evidence files

- Run configuration, logic, and saved-output paths: `scripts/train_dev_retrieval.py`.
- Exact copied metrics: `reports/development/two_tower_70_80_colab_metrics.json`.
- Split counts and catalog sizes: `reports/development/temporal-split-audit.json`.
- Full-catalog popularity definition: `baseline.py` (`popularity_order`).
- Predeclared evaluation rules: `docs/experiment-protocol.md`.
