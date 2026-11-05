# ItemKNN temporal replication: fixed 60/70 window

This replication uses the exact item-neighbor method frozen in itemknn-50-60-preregistration-2026-11-05.md: binary rating >= 4 training evidence, cosine-normalized co-rating, top 100 neighbors per item, summed similarities from positive user history, exclusion of all prior-rated movies, popularity tie-break and fill, same training-rated catalog, and identical warm-user targets.

The training interval is timestamp < 1,288,571,552 (the audited 60% boundary). Validation is 1,288,571,552 <= timestamp < 1,421,944,985 (the audited 60–70% interval). The script verifies the computed upper boundary. The ratings file SHA must match the frozen MovieLens 25M checksum. No official final 10% or historical 80–90% panel is used.

Report candidate Recall@100 and @500; Recall@10/NDCG@10; popularity on identical warm users and catalog; paired user bootstrap with 2,000 resamples and seed 42; supports and catalog coverage. Do not tune or alter the method after looking at this result. This is a temporal development replication after inspecting the 50/60 result, not an independent final test. Do not promote a model or claim generalization from this result alone.
