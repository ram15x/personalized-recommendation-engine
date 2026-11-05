# Pairwise DCN candidate-ranker check

## Decision

Do **not** promote this pairwise DCN. It ranked below the unchanged ItemKNN candidate order on the same 386 shared users. Unlike the pointwise DCN run, its paired bootstrap intervals for Recall@10 and NDCG@10 were entirely below zero. The API is unchanged.

## Why this objective was tested

The preceding DCN was trained with squared error on star values, but was judged by candidate ordering. Pairwise ranking objectives directly train a score difference between a preferred and a less-preferred item; Rendle et al. describe this approach for personalized ranking in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://proceedings.mlr.press/r7/rendle09a.html). That paper concerns implicit-feedback BPR. This experiment is a limited adaptation to MovieLens: it forms pairs only from **observed** ratings >=4 versus observed ratings <4. Unrated movies remain unknown, not negative examples. The research motivation does not guarantee improvement, and the measured result here was worse.

## Fixed protocol

- Training input: saved top-500 ItemKNN candidates and observed ratings from the 50/60 window. Evaluation input: saved candidates from 60/70, evaluated only for the 386 user IDs shared with training.
- The architecture and input features stayed the same as the pointwise candidate DCN: 32-dimensional user/item embeddings, three cross layers, MLP widths 128 and 64, dropout 0.1, and training-standardized `log1p(ItemKNN score)`.
- The one intended change was the objective: binary cross-entropy from the difference between a >=4-star candidate score and a <4-star candidate score. Training used Adam 0.001, four epochs, batch size 2,048, seed 42.
- Users with both observed classes generated all possible within-user pairs, then at most 64 pairs per user were sampled deterministically. 1,575 of 2,000 training users had both classes. Of 1,262,923 possible observed pairs, 72,732 were sampled. The 33,929 positive and 31,420 lower-rated observed candidate labels were used; unrated items were not.
- The same evaluation cohort and ItemKNN candidate order as the pointwise DCN run were used. This makes it a direct development comparison, not independent confirmation.

## Results

Evaluation included 386 users, 10,099 observed candidate ratings (5,610 >=4), and 193,000 candidate positions. 18,732 (9.71%) candidate item IDs were outside the training item vocabulary and used the reserved unknown-item embedding. Unrated items in the top 10 count as non-hits, so the ranking metrics remain affected by rating exposure/selection bias.

| Ordering/scorer | Macro Recall@10 | Macro NDCG@10 |
|---|---:|---:|
| ItemKNN candidate order | 0.05865 | 0.11075 |
| Pairwise DCN | 0.03197 | 0.05441 |
| User/item rating-mean baseline | 0.02122 | 0.03528 |

| Paired difference, 2,000 user-bootstrap resamples | Median difference | 95% interval |
|---|---:|---:|
| Pairwise DCN minus ItemKNN, Recall@10 | -0.02657 | [-0.04099, -0.01233] |
| Pairwise DCN minus ItemKNN, NDCG@10 | -0.05621 | [-0.07410, -0.03913] |
| Pairwise DCN minus mean baseline, Recall@10 | +0.01060 | [+0.00313, +0.01937] |
| Pairwise DCN minus mean baseline, NDCG@10 | +0.01919 | [+0.00982, +0.02894] |

The training pairwise loss fell from 0.6544 to 0.4480; training pair accuracy rose from 0.6180 to 0.7804. Those are training diagnostics only. The later-window rank metrics show that the fit did not transfer well enough to outrank ItemKNN on this cohort.

## Limits and reproduction

Both windows were already used in Project 2 development, and the 386-user overlap is a small, selected cohort. MovieLens contains explicit ratings, not clicks or impressions. These results do not measure CTR, online A/B lift, production value, or independent generalization. No raw ratings CSV was opened, no downloads or package installs were made, and the official final 10% was not used. The API and its default were not changed.

From the Project-2 folder:

```powershell
python -m scripts.train_pairwise_candidate_dcn `
  --train-candidates reports\development\itemknn-50-60-candidate-train-2k\candidate_pairs.jsonl.gz `
  --eval-candidates reports\development\itemknn-60-70-corrected-cohort-2k-v2\candidate_pairs.jsonl.gz `
  --output reports\development\dcn-pairwise-candidates-50-60-to-60-70-v1 `
  --epochs 4 --batch-size 2048 --max-pairs-per-user 64 --seed 42
```

The run took 18.31 seconds in the checked Windows Python 3.10 / TensorFlow 2.15 CPU process. Files and hashes are in `reports/development/dcn-pairwise-candidates-50-60-to-60-70-v1/`. This is an unpromoted experiment.
