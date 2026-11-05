# BPR temporal replication: preregistered protocol

Status: frozen before the 60–70% window is scored or used for training.

## Question

Does the previously fixed BPR two-tower recipe retain its performance relative to the training-period popularity baseline on an earlier temporal window? This is a robustness diagnostic. The BPR objective and architecture were chosen after seeing results on the later 70–80% development window, so this is **not** an independent model-selection test.

## Data and split

- Dataset: existing local MovieLens 25M `data/ml-25m/ratings.csv`; expected SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`.
- Positive preference proxy: rating >= 4.0; not clicks or observed exposure.
- Train: timestamp < `1288571552` (approximately the first 60% of rows in the previously defined first 80% period).
- Validation: `1288571552` <= timestamp < `1421944985` (approximately the following 10 percentage points, ending at the previous 70/80 split cutoff).
- Timestamp ties stay wholly on one side of each cutoff. Realized row counts will be recorded from the actual timestamps.
- No labels/metrics from the official final 10% are used. No external data is downloaded.

## Frozen model and comparator

- Keep the BPR comparator implementation and all settings from `reports/development/bpr-objective-comparison-2026-11-05.md`: ID towers, dimension 32, batch 512, 3 epochs, Adagrad 0.05, one uniformly sampled item not rated by that user in the training period for each positive, seeds 42/43/44, and the same L2 term.
- Build the item index from positive-rated training items. Compare the learned tower against popularity computed only from positive training ratings.
- Do not change architecture, features, threshold, optimizer, epochs, negative sampler, metrics, or split after observing this window.

## Outcomes to report

Primary metric: warm-user Recall@100 restricted to positive targets represented in the shared training-positive item index. Also report candidate Recall@100 and @500, Recall@10, NDCG@10, catalog coverage, warm-user support, and 2,000 paired user-bootstrap intervals for BPR minus popularity on identical users/targets/catalog. Record per-user predictions and exact run hashes.

## Decision rule

No promotion from this one window. If BPR is still below popularity, keep popularity as the served fallback and investigate the rating/negative-sampling mismatch before adding a deep ranker. If it exceeds popularity, call it a promising replication only; a later untouched evaluation would still be needed before claiming generalization.

## Provenance caveat

Because this model family was selected after observing the later 70–80% development results, the earlier period is a temporal replication with selection-history caveat, not an independent test set. This run must not be used to claim CTR or online A/B performance.
