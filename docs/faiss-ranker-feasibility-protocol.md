# FAISS-aligned ranker feasibility check

Hypothesis: the saved two-tower pilot can provide a bounded, time-safe candidate source for training and checking a candidate ranker on the already-used 80/90 development period. This check measures available support; it does not train, tune, score recommendations, or demonstrate quality.

Before examining the result: select 10,000 user IDs uniformly without replacement from the saved retriever vocabulary using NumPy random seed 42. This selection depends only on the saved retriever, not future labels. Split the existing development period `[1466837400, 1515033908)` at timestamp `1490935654` (its arithmetic midpoint). Count selected users with any rating and with a rating >=4 in each half and with observations in both halves. Read and retain only rows before the 90% boundary and only aggregate counts. Verify the ratings CSV hash against the existing recorded hash. Do not use the final period for model selection.

This is a feasibility check, not a representative real-world sample. Counts of observed ratings are an upper bound on usable FAISS candidate labels, since a later step must require that the rated item was actually retrieved and not previously seen. MovieLens ratings are not impressions or clicks. The chosen development period has already been reused, and any later metrics there cannot be called independent. Do not promote a model on the basis of this support check.


## Candidate-label support follow-up

Fixed before running: use the same 10,000 seed-42 retriever users and time boundaries. Find sampled users with any observed rating in either half. For those users, reconstruct history before the first half; compute up to 100 exact FAISS candidates excluding movies with prior ratings. The first-half labels are observed ratings on those candidates. For the second half, use the same fixed retriever but exclude both prior and first-half rated movies; labels are second-half observed ratings on those candidates. Report numbers of candidate labels and users with labels in each half and both halves. Do not train or calculate ranking metrics. Restrict outcome retention to timestamps before the development end. This is a candidate-support diagnostic on a reused development period, not an independent evaluation, and observed ratings remain exposure-biased.
