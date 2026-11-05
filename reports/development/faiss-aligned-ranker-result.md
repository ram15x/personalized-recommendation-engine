# FAISS-aligned feature-ranker development result

The fixed protocol is `docs/faiss-aligned-ranker-protocol.md`. The existing two-tower pilot and FAISS index generated up to 100 unseen candidates per user in each half of the reused 80/90 development period. The first half supplied 8,313 observed candidate ratings for training; the second supplied 3,181 users, 3,047 with unseen positive ratings. No final-period outcomes were retained or used. The export `faiss-aligned-80-90-candidates-v1/candidates.jsonl.gz` is 6,387,956 bytes and has SHA-256 `11af9ed7179d550c9fdc21d6f08b574d6586a157e4bd39f76482f6e4c8d0bcc6`.

A fixed four-epoch CPU run used a feature-only deep-and-cross rating scorer (FAISS score and training-period popularity rank, without user/item ID embeddings). Training MSE by epoch was 102.14, 84.32, 69.06 and 56.49. These losses are high for 0.5–5 rating targets; no training improvement or usable rating prediction is claimed.

| Same candidates, 3,047 users with positive targets | Macro Recall@10 | Macro NDCG@10 |
|---|---:|---:|
| FAISS order | 0.00634 | 0.01343 |
| Feature DCN | 0.00763 | 0.01513 |
| Popularity order within the same candidates | 0.01156 | 0.02482 |

Paired user-bootstrap DCN-minus-FAISS NDCG@10 median = +0.00172, 95% interval approximately [-0.000004, +0.00352]; this includes zero. DCN-minus-popularity NDCG@10 median = -0.00967, interval [-0.01192, -0.00733]. The overall positive-target candidate recall@100, weighted by target count, was 0.03724; 0.72407 of positive targets were in the indexed catalog. Candidate-conditioned ranking metrics and per-user result files were not saved and are NOT VERIFIED. See `faiss-feature-dcn-80-90-v1/metrics.json` for complete figures and intervals.

The saved model was reloaded and produced finite, differing scores for two synthetic feature vectors; this checks serialization and input dependence, not correctness or model quality. One synthetic output was outside the rating range; scores are unbounded. The ranker is not wired into the API or promoted. The development interval and decisions were reused, ratings are exposure-biased and not clicks, and these findings do not establish generalization.

Hashes: `ranker.keras` SHA-256 `16c503e12450286001fb8d0af404e9dcf0d7205fa054ab94abbdaa43459d7856`; `normalization.npz` `61f8d322459b73d98986096366a9e4c5cc9ebeb9c59d8f79e1e6e27855658e7b`; `metrics.json` `1f402cf92b9ea56ca29674e42853036765b8cc5298d8d193f2dd5415bd450100`. TensorFlow and Requests emitted warnings during execution. No dataset was downloaded.

## Mean-initialized v2 and native integration

The targeted v2 run changed the output initialization to the training-label mean; the protocol records this after inspecting v1 and before fitting v2. Its four training MSE values were 0.9721, 0.9575, 0.9407 and 0.9328. On the same 3,047 users and FAISS candidates, Recall@10/NDCG@10 were 0.01062/0.02319 for v2, 0.00634/0.01343 for FAISS order, and 0.01156/0.02482 for popularity order. Paired NDCG@10 DCN-minus-FAISS median +0.00979, 95% interval [+0.00757, +0.01197]; DCN-minus-popularity median -0.00159, interval [-0.00336, +0.0000074]. This reused-window result does not establish an independent gain and does not clearly beat popularity. The full numbers are in `faiss-feature-dcn-mean-init-v2/metrics.json`.

The native API has an opt-in feature-DCN route. A live local HTTP check loaded both saved artifacts, returned five two-stage recommendations, and returned popularity fallback for an unknown user. This checks execution, not quality. The feature ranker is experimental and is not the Docker/browser default.

A native browser check with both artifacts configured submitted user 847 and displayed ten movie titles through `/experimental/two-tower/display` (HTTP 200). The browser check does not establish recommendation quality; supplied history is required for seen-item exclusion. The temporary server was stopped.
