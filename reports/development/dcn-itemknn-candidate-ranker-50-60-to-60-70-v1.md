# Candidate-level DCN ranker check

## Decision

Do **not** promote the DCN over the ItemKNN candidate order. On the 386 users shared by the two sampled development exports, the DCN's point estimates were lower than ItemKNN for both macro Recall@10 and NDCG@10. The paired user-bootstrap intervals for those differences include zero. The DCN did beat a simple user/item rating-mean scorer, but that does not show it improves the retrieval pipeline. The API default and serving path were not changed.

## Fixed protocol

- Training input: the saved top-500 ItemKNN candidate export from the MovieLens 50/60 temporal window; evaluation input: the saved export from the later 60/70 window.
- Training used only observed explicit ratings attached to candidates in the earlier export. Unrated candidates were omitted from the loss; they were not negative examples.
- The ranker was a DCN-v1 rating regressor with 32-dimensional user and item embeddings, three cross layers, an MLP of 128 and 64 units, dropout 0.1, Adam at 0.001, MSE loss, four epochs, batch size 2,048, seed 42.
- Its only numeric input was the training-standardized `log1p(ItemKNN retrieval score)`. Normalization values were fit on observed training candidate ratings only. The comparison scorer was the clipped `user_mean + item_mean - global_mean` computed from those same earlier observed candidate ratings. ItemKNN order was the unchanged candidate order in the later export.
- Evaluation was restricted to user IDs appearing in both exports. No raw ratings CSV was read, no package was installed, no data was downloaded, and the official final 10% was not used.

## Results

There were 386 overlapping users, 10,099 observed later-window candidate ratings, and 5,610 positive candidate ratings (rating >= 4). Every method ranked the same 500 candidates per user. The evaluation catalog had 193,000 candidate positions; 18,732 item IDs (9.71%) were outside the training candidate vocabulary and used the model's reserved unknown-item embedding. Unrated items in the top 10 count as non-hits for these observed-label metrics; that convention is biased by which movies received ratings and is not evidence that unrated movies were disliked.

| Ordering/scorer | Macro Recall@10 | Macro NDCG@10 |
|---|---:|---:|
| ItemKNN candidate order | 0.05865 | 0.11075 |
| DCN rating predictions | 0.04916 | 0.10127 |
| User/item rating-mean baseline | 0.02122 | 0.03528 |

| Paired difference, 2,000 user-bootstrap resamples | Median difference | 95% interval |
|---|---:|---:|
| DCN minus ItemKNN, Recall@10 | -0.00936 | [-0.02085, 0.00202] |
| DCN minus ItemKNN, NDCG@10 | -0.00903 | [-0.02198, 0.00219] |
| DCN minus mean baseline, Recall@10 | +0.02788 | [+0.01613, +0.04009] |
| DCN minus mean baseline, NDCG@10 | +0.06593 | [+0.05051, +0.08239] |

On the 10,099 observed candidate ratings, DCN RMSE/MAE were 0.9054/0.7007; the mean baseline was 0.7791/0.5745. Thus the DCN also had worse rating error than the simple baseline. The ranking comparison is the relevant reranking check: it did not improve the ItemKNN order.

## Limits

This is a small, sampled, reused development cohort, not an independent test. The 386-user overlap is selected by presence in both exports. MovieLens ratings are explicit preferences, not impressions or clicks. The observed-label metric treats unobserved candidates as non-hits and is subject to exposure bias. The result cannot support CTR, online A/B, production, or generalization claims. It also does not establish that all DCN recipes fail; it rejects this fixed candidate-ranker recipe for promotion on the available evidence.

## Reproduction and artifacts

Command, from the Project-2 folder:

```powershell
python -m scripts.train_candidate_dcn `
  --train-candidates reports\development\itemknn-50-60-candidate-train-2k\candidate_pairs.jsonl.gz `
  --eval-candidates reports\development\itemknn-60-70-corrected-cohort-2k-v2\candidate_pairs.jsonl.gz `
  --output reports\development\dcn-itemknn-candidate-ranker-50-60-to-60-70-v1 `
  --epochs 4 --batch-size 2048 --seed 42
```

The run took 15.02 seconds in the observed Windows Python 3.10 / TensorFlow 2.15 CPU process. It used only the two candidate exports (about 22 MB compressed combined). The saved model, feature mapping, per-user metrics, aggregate metrics, and file hashes are in this directory. This artifact is an unpromoted experiment; the API remains unchanged.
