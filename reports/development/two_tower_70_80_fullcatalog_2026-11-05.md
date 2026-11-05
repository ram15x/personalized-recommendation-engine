# Corrected 70/80 two-tower diagnostic

## Run record

- Run date: 5 November 2026; Colab UI showed 2:02 AM, timezone not independently recorded.
- Runtime: Google Colab T4; TensorFlow 2.21.0, TensorFlow Recommenders 0.7.3, FAISS 1.15.1.
- Data: existing Colab `/content/ml-25m/ratings.csv`, SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`. No data was downloaded for this run.
- Split: train timestamps `< 1421944985` (17,500,066 ratings); validation `1421944985 <= timestamp < 1466837400` (2,500,010 ratings). Neither the historical 80–90% panel nor final 10% was used.
- Target: ratings `>=4` treated as positive. Training used 8,750,986 user-item pairs and an ID-only, 32-dimensional two-tower model with batch 512, 3 epochs, Adagrad 0.05, and custom in-batch softmax loss.
- Loss by epoch: 5.97453, 5.97425, 5.97465. Training time: 260.47 seconds.
- The corrected run used a 22,931-item catalog of all movies rated in training; only 18,405 items (80.26%) had positive-training embeddings and could enter the FAISS index.
- Model weights and per-user predictions remain in Colab. The local project holds the metrics; subgroup metrics below were read from the saved `per_user_predictions.csv` without training or model inference.

## Results

On 3,550 users with a learned user embedding, top-100/500 retrieval Recall was 0.00438 / 0.02164. Of their 124,777 eligible positive targets, 110,095 (88.23%) had an item vector in the index. Restricting the candidate-recall denominator to those indexed targets, retrieval Recall@100 was 0.00494 versus 0.14821 for popularity on the same users and indexed targets. The final top-10 retrieval list reached Recall@10 0.00045 and NDCG@10 0.00189, versus popularity at 0.03362 and 0.10279. The indexed target set is mostly representable; the retriever nevertheless fails to rank those items near the top.

The read-only cohort aggregation clarifies the all-user scores:

| Earlier history before cutoff | Learned user embedding | Users | Eligible target pairs | Target pairs represented in index | Index coverage | Retrieval plus popularity fill: R@10 / NDCG@10 | Popularity: R@10 / NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Yes | Yes | 3,550 | 124,777 | 110,095 | 88.23% | 0.00045 / 0.00189 | 0.03362 / 0.10279 |
| Yes | No | 10 | 446 | 436 | 97.76% | 0.10838 / 0.30374 | 0.10838 / 0.30374 |
| No | No | 15,843 | 1,104,884 | 1,060,073 | 95.94% | 0.08502 / 0.38307 | 0.08502 / 0.38307 |

Thus 15,843 validation users had no prior history and 10 had earlier ratings but no learned embedding. Both groups used popularity; their metrics are not evidence of learned personalization. Across all users, 1,170,604 of 1,230,107 eligible target pairs (95.16%) corresponded to indexed items. This is item representability, not retrieval success.

Across all 19,403 users, the combined list had Recall@10 0.06956 and NDCG@10 0.31329, below popularity alone at 0.07563 and 0.33175. Those overall scores are driven by the fallback and hide the poor warm-user retriever.

## Decision

The positive-only item-catalog denominator error in the initial evaluation was corrected. The full-catalog run completed, but the learned retrieval model remains materially worse than popularity on users who have learned embeddings. Do not ship or claim improvement for this retriever. Do not add a ranker yet: a ranker cannot recover relevant items that candidate generation misses. Next diagnose the custom loss and training signal, ID mappings, vector norms and score distribution, then compare a simple time-safe collaborative baseline with popularity on the same frozen development window. Keep the final 10% out of model selection.

## Limits

This is one development run on a window used to guide the next work, not an independent test. MovieLens ratings are explicit feedback; rating `>=4` is a preference proxy, not clicks or CTR. The model has ID-only features and lacks a content path for unseen IDs. Three epochs with nearly flat loss do not establish convergence. The 70/80 panel has already been used for development decisions. See the accompanying JSON for exact counts and metrics.
