> Historical policy: the original result used vocabulary membership, not labeled-training support. It does not evaluate the corrected inference guard; see `reports/development/ranker-training-support-audit.md`.

# Ranker fallback evaluation protocol

Before running: use the existing 60/70 ItemKNN candidate export and the saved 50/60 pointwise DCN, without retraining. Evaluate the 386 users in both the export and saved ranker user vocabulary. Use the first 100 ItemKNN candidates per user. Positives are the observed ratings of at least 4 among the exported candidates; unobserved ratings remain unknown. Report macro Recall@10 and NDCG@10 on the same users for: (1) the original ItemKNN order, (2) the current strict policy, which reranks only when every item is known, and (3) a proposed policy that keeps unknown-item positions unchanged while sorting known items by DCN score in the remaining positions. Break score ties by movie ID. Use 2,000 paired user bootstrap resamples with seed 42 to compare each policy against ItemKNN.

This is a reused development panel. It cannot establish independent generalization, click-through rate, or performance on FAISS candidates. No policy will be promoted solely from this comparison. The results will include all selected users, including those without positive targets (their metrics are zero).
