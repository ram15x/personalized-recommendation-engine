# Item-to-item collaborative retrieval: frozen 50/60 development protocol

## Reason for this candidate

The existing ID-only neural retriever and BPR variant both remain below the time-matched popularity reference. Before adding another neural architecture or a ranking layer, this experiment checks a classical, inspectable collaborative method: item-item cosine similarity over positive co-ratings. Item-based collaborative filtering is a long-established recommendation method (Sarwar et al., WWW 2001). A later sequential-recommendation comparison found nearest-neighbor methods competitive with transformer methods on some datasets; this motivates testing the baseline, not assuming it will win.

## Frozen method

- Dataset: existing local MovieLens 25M ratings.csv only; require the existing expected SHA-256 before reading.
- Temporal windows: training is the earliest 50% of rows by global timestamp; validation is the following 50–60% interval. Compute cutoffs from the first 80% baseline using whole timestamp ties. Check that the computed 60% cutoff equals the previously audited timestamp 1,288,571,552. No official final 10% or historical 80–90% window is used.
- Positive training evidence: ratings >= 4, treated as a preference proxy. One binary user-item entry per positive pair.
- Similarity: cosine-normalized item co-rating counts; zero the diagonal; keep the top 100 neighbors per item, with stable deterministic tie-breaking.
- User score: sum item similarities from that user's positive training history. Exclude every movie the user rated before the validation cutoff. Tie-break equal scores by training-period positive popularity descending, then movie ID ascending; this also yields a popularity fill for zero-score items.
- Comparison: popularity is counted only from positive training ratings, on the exact same full training-rated catalog, warm-user support, and validation targets.
- Metrics: candidate Recall@100 and @500; Recall@10 and NDCG@10; per-user paired bootstrap with 2,000 resamples, seed 42. Report warm support, target pairs, catalog coverage, and per-user metrics.
- Decision gate: do not call the candidate useful or promote it unless the paired 95% interval for Recall@100 is above zero and Recall@10/NDCG@10 are not below popularity. No hyperparameter sweep or score-based choice will be made on this panel.

## Selection caveat

The 50–60% interval has not been used in the saved experiment reports. However, item-item collaborative filtering is being selected after reviewing earlier development runs, so this is a fresh temporal development comparison, not an independent final test. It cannot establish production performance, CTR, or online A/B results.

## Primary references

- B. Sarwar et al., Item-Based Collaborative Filtering Recommendation Algorithms, WWW 2001: https://doi.org/10.1145/371920.372071
- Sequential recommendation: A study on transformers, nearest neighbors and sampled metrics, Information Sciences, 2022: https://doi.org/10.1016/j.ins.2022.07.079
