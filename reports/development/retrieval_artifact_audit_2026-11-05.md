# Saved retrieval artifact audit

## Scope and evidence

Read-only audit of the corrected 70/80 two-tower run. No weights were updated, no model was trained, and the official final 10% split was not used. The results below were produced in Colab notebook `Untitled2.ipynb` (Drive ID `1M3w8YciRhz2tDm5EGLiqIIF1EGD_K1a4`), audit cell 3. The locally saved `scripts/audit_saved_retrieval_run.py` was syntax-checked locally but was not executed locally because this environment does not have TensorFlow Recommenders or FAISS installed.

The main run details and its validation results are in `two_tower_70_80_fullcatalog_2026-11-05.md` and its JSON companion. The tested fixed batch was 512 rating>=4 training rows, seed 42, selected from the first 500,000-row ratings chunk. This is one batch, not a representative sample of all training data.

## Results

| Check | Result |
|---|---:|
| User tower vs saved user vectors, max absolute difference | 0.0 |
| Item tower vs saved item vectors, max absolute difference | 0.0 |
| Reconstructed FAISS vectors vs saved item vectors, max absolute difference | 0.0 |
| FAISS item vectors | 18,405 |
| Saved user and item vectors finite | Yes |
| Sampled rows / unique users / unique items | 512 / 407 / 421 |
| User and item ID mapping round-trip exact | Yes / Yes |
| Sampled-batch loss | 6.23630285 |
| Finite / nonzero gradient tensors | 10/10 / 10/10 |
| Gradient global norm | 0.00411310 |
| Mean positive logit minus hardest-negative logit | -0.00379546 |
| Masked top-1 accuracy on this batch | 0.001953125 |
| Weights changed during audit | No |

Saved embedding norm ranges were small: user min/median/max `0.02719 / 0.08391 / 0.20990`; item min/median/max `0.03317 / 0.11827 / 0.22919`.

## Interpretation and limit

The exported towers, saved vector arrays, and FAISS index agree exactly for the checked vectors. The sampled batch has finite gradients, but its near-zero negative margin and near-chance top-1 show little separation on that batch. This does **not** prove a bug or explain the full run’s poor validation quality. It does rule out a mismatch among the sampled ID mappings, saved vectors, and FAISS vectors as the cause of this particular audit discrepancy.

The validation evidence remains poor for the learned warm-user retriever: Recall@100 was `0.00438` across eligible targets and `0.00494` on the indexed-target subset, while popularity reached `0.14821` on that same indexed-target subset. Those are development results from the reused 70/80 window, not independent test results. See the corrected run report for full context.

## NOT VERIFIED

- The one-batch gradient diagnostic does not establish full-training gradient behavior, convergence, or generalization.
- Root cause of poor retrieval quality is not established.
- The Colab runtime artifacts are not stored in this local project folder; the metrics and audit values above are transcribed from the notebook outputs.
- The local audit script has only been syntax-checked, not run against local model artifacts.

## Reproduction note

The notebook output can be regenerated only while its `/content` ratings and corrected run artifacts are present. The local script accepts explicit artifact, ratings, and training-script paths; it does not download data or modify weights.
