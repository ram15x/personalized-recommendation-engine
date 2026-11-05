# Pre-registered BPR objective comparison

## Question and hypothesis

The saved ID-only two-tower learned almost no separation on one fixed training batch, and its warm-user candidate recall was far below popularity. The next controlled question is whether replacing its in-batch softmax with a pairwise ranking objective and explicit unobserved-item negatives improves candidate ordering. This is a testable hypothesis, not an expected result.

The comparison follows Bayesian Personalized Ranking (BPR), which optimizes pairwise ordering of an observed positive above a sampled unobserved item. It is not assumed to outperform the existing run. The reference paper defines the objective for personalized ranking and discusses matrix-factorization and adaptive-kNN applications: https://proceedings.mlr.press/r7/rendle09a.html. TFRS also describes retrieval as learning query/candidate representations and contrasting paired examples against other candidates: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval.

## Frozen run settings

| Setting | Fixed value |
|---|---|
| Data | Existing MovieLens 25M `ratings.csv`; checksum must match the corrected run |
| Split | Train timestamp `<1421944985`; validation `[1421944985, 1466837400)` |
| Training positives | Rating `>=4`; all 8,750,986 positive pairs |
| User/item ID sets | Same positive-trained users and items as the corrected two-tower run |
| Candidate negatives | One item per positive pair, sampled uniformly from the positive-trained item catalog; reject every item the user rated in the training period (any rating); resample each epoch |
| Seed | 42 for initialization; negative and row-shuffle seeds 42, 43, 44 for epochs 1–3 |
| Architecture | Same user and item tower layers and dimension 32 as `scripts/train_dev_retrieval.py` |
| Objective change | Pairwise `softplus(score_negative - score_positive)` BPR loss with `1e-6` L2 on the input ID embeddings; no extra item bias |
| Optimizer / batch / epochs | Adagrad 0.05 / 512 / 3, held constant for the first comparison |
| Candidate index | Exact FAISS inner product over the same 18,405 positive-trained item IDs |
| Primary comparison | Warm-user paired Recall@100 on the same indexed candidate catalog; also report Recall@500, matched-catalog Recall@10/NDCG@10, and full-target Recall@10/NDCG@10 |
| Baseline | Popularity trained on the same time cutoff and scored on the exact same users and available catalog |
| Test data | No final-10% ratings or historical 80–90% panel |
| Uncertainty summary | 2,000 paired user bootstrap resamples with fixed seed 42; intervals are conditional on this reused development cohort |
| Output | New, uniquely named run directory; existing corrected run remains untouched |

The held-out `rating>=4` targets remain a proxy for preference, not clicks. An unrated item is an uncertain negative in MovieLens, so a positive result would still need cautious interpretation. The run is a development comparison on a window already used to guide this project; it is not an independent test.

## Reproduction command

After uploading the current script to Colab and confirming its SHA-256, the ratings checksum, and that the output path does not exist, run:

```bash
python /content/train_bpr_objective_comparator.py \
  --ratings /content/ml-25m/ratings.csv \
  --output /content/nexvia_aiml03_bpr_70_80_seed42
```

The script refuses an existing output directory or a ratings checksum mismatch. It makes no download attempt.

## Decision rule

Report every fixed run metric, including per-user paired differences against popularity on the same warm-user cohort and the bootstrap intervals. Do not call the model improved from a single reused window or ship it based only on this comparison. If the pairwise model still loses to popularity, stop model-architecture changes and diagnose data/target and evaluation assumptions before another fit. Do not add a ranking stage until candidate generation has a credible result.

## Execution status

Protocol was written before the comparator run. The exact local script was uploaded to Colab and its SHA-256 (`6cd990e04701ef24292e902ca06793ce3caae07e1b073e0a1da0c26724f6e003`) and the existing ratings checksum were verified before training. The new run completed in 358.59 seconds in the T4 runtime and wrote a new output directory. Its warm-user matched-catalog Recall@100 was 0.14519 versus popularity at 0.14821; the paired-user bootstrap interval for the difference was [-0.00463, -0.00137]. Top-10 Recall/NDCG also remained lower. These results are from a development window reused during model work, so they do not establish independent generalization. See [the result report](../reports/development/bpr-objective-comparison-2026-11-05.md) and [the earlier readiness record](../reports/development/bpr-comparator-readiness-2026-11-05.md), which preserves the initial failed synchronization attempt.
