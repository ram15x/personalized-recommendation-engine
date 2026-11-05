# Ranker training support audit

The current candidate trainer builds its vocabularies from all exported users and candidate IDs (`scripts/train_candidate_dcn.py:91`, `:93`), but builds training rows from `observed_ratings` only (`:104`–`:109`). The model uses separate user and item embedding tables (`scripts/dcn_ranker.py:70`–`:71`). IDs without labeled examples therefore have no direct supervised training example under this code. The exact source revision that trained the saved checkpoint is NOT VERIFIED; the saved metrics do not record its source hash.

The training export hash matches the value recorded in the saved metrics: `c2521cbd2598574b16cb1ab97ed3eeacf4eb778d07bc488600a9fbbf8579d6c7`. Scanning that export found 1,874 of 2,000 vocabulary users and 2,787 of 6,100 vocabulary items in at least one labeled example. This does not prove every supported embedding learned a useful representation.

A new `trained_support.json` records those labeled IDs and the training-export hash. It has SHA-256 `07c16266464e8ac9ce4dd05d8bd0342dd53baf7bd5b93acc7174853341a068cc`; the updated artifact manifest has SHA-256 `9baa589662d7c55585ccf67433c482957c790eb814fc57534660e76384eb0d49`. The checkpoint and feature mapping were not changed. The loader verifies the files and export reference and falls back for unsupported IDs (`scripts/candidate_ranker_inference.py`). New trainer outputs also record support (`scripts/train_candidate_dcn.py:271`).

## Checks

- Focused suite: 13 passed (`test_candidate_ranker_inference.py`, `test_itemknn_api.py`, `test_triton_ranker_contract.py`).
- Full local suite: `python -m pytest -q -p no:cacheprovider`: 61 passed, 1 skipped in 18.49 seconds. Spark was skipped.
- Saved model load and inference: 1,874 supported users, 2,787 supported items; a supported item produced a finite score; a vocabulary item without labeled support returned no score; unsupported user and unsupported item triggered fallback. TensorFlow emitted deprecation warnings and Requests emitted a dependency warning.
- Live native HTTP: with ItemKNN and the saved experimental ranker configured, `/health` reported the ranker loaded and a request using a vocabulary user without labeled support returned HTTP 200, `itemknn_cosine` and `ranking_status: unknown_user` with five items. The temporary server was stopped.
- Corrected fallback evaluation on the existing reused 60/70 development cohort (`fallback-policy-trained-support-v2.json`): 386 users, 83 strict reranks. ItemKNN NDCG@10 = 0.11075; strict policy = 0.10641. Paired user-bootstrap median difference = -0.00391, 95% interval [-0.01131, +0.00166]. The alternative known-slots policy is exploratory and is not used by the API. These results do not establish an improvement. The original fallback result used vocabulary-only support and is historical, not a measurement of this corrected guard. Candidate-conditioned ranker reports likewise used the earlier scoring behavior; they remain development diagnostics, not evidence about the corrected serving policy.

No model was retrained. No new dataset was downloaded. The final test period was not accessed for these checks. The model checkpoint and numeric mappings were not modified.
