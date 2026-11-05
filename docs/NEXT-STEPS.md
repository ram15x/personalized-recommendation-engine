# What to do next — checked October 24, 2026

> Historical planning memo. Experiments and results changed after this date. For the current AIML-03 status, use `docs/requirements-status.md`, `docs/experiment-protocol.md`, and the ItemKNN result reports. The fixed ItemKNN candidate later exceeded popularity on matched-catalog metrics in two development windows, but was selected during development and has not been integrated into the API. The current demo remains popularity-only.

## The honest workload

The portal lists these study estimates:

| Module | Portal estimate |
|---|---:|
| Foundations & data preparation | 2,130 min = 35.5 h |
| Classical machine learning | 2,130 min = 35.5 h |
| Advanced ML & deep learning | 2,100 min = 35 h |
| Applied workflows & MLOps | 1,980 min = 33 h |
| Capstone & final assessment | 660 min = 11 h |
| **Total** | **9,000 min = 150 h** |

From October 24 to November 20, there are about 27 calendar days. At 4–5 hours per day, that gives about 108–135 hours total. The module estimates alone exceed that by 15–42 hours, before either project. These portal estimates may not equal clock time for an experienced AI/ML graduate, but we do not yet have a written exemption or confirmed scope. Emails call modules supplementary while the portal says all modules are needed for certification.

Both assigned project briefs list an estimated effort of 14 days and the same November 20 deadline. Their detailed stacks and deliverables are broad. It would be unsafe to promise a complete, production-grade implementation of every listed service and integration inside the remaining period without a manager-approved scope reduction.

## What the evidence says about AIML-03

- The popularity baseline on MovieLens 25M test data: Recall@10 0.05184, NDCG@10 0.27137. This test set must stay final; do not tune against it.
- The two-tower pilot used 500,000 of 9,981,093 training interactions rated >= 4. Its validation report used one seeded sample of 2,000 users, with popularity scored on the same users: retrieval Recall@10 0.05405 / NDCG@10 0.27892; popularity 0.05652 / 0.29081.
- Among the 528 validation users with a trained user embedding, retrieval was worse than popularity: Recall@10 0.00940 vs 0.01878; NDCG@10 0.02432 vs 0.06937. The remaining 1,472 used popularity fallback, so their equal scores do not show a learned-model gain.
- The pilot indexed 12,224 movies out of 34,461 in the full training catalog (35.5%). Its top ten may be filled from popularity. The candidate-coverage limitation is substantial.
- These are early validation findings, not final test results. A falling loss does not show good held-out retrieval.
- TFRS `remove_accidental_hits` masks an in-batch candidate when it is the same item as the target. It does not by itself mask a different known-positive item for the same user. Since this code draws multiple rows per user into shuffled batches, some rated-positive items may be presented as negatives. This is a plausible issue to check, not a proven cause of the score.
- Full 9.98-million-interaction training did not produce saved outputs in the current Windows CPU-only TensorFlow setup. The 500,000-row pilot remains the working artifact.

## Recommended next moves

1. Get written clarification from Rishi/HR on whether all 150 estimated module hours are truly required, the minimum acceptable scope for BOTH capstones, how a late start affects evaluation, and whether a screen-recorded prototype is acceptable in place of all production integrations. A draft message is in the chat, not sent.
2. Do not start full training on the Windows CPU or build the ranking stage yet. The baseline is still ahead.
3. If the expected full-data retrieval work remains in scope, move training to a GPU runtime only after confirming which GPU and TensorFlow/TFRS versions are actually available. Colab's official FAQ says resources, GPU type and usage limits vary and are not guaranteed; save checkpoints and outputs frequently.
4. Repair the retrieval experiment before scaling: compare on the same fixed validation cohort; report trained-user coverage and candidate-catalog coverage; design a loss or batch strategy that does not treat other known positives from the same user as negatives; save checkpoints; use validation for choices and keep the test data untouched.
5. Work AIML-02 in parallel at prototype scale, starting with a legally accessible public dataset and a small 2D classification demo. Do not claim clinical use or deploy a diagnostic system. Confirm whether the brief really requires both 2D classification AND 3D segmentation, Orthanc, React, Docker and all other listed integrations.
6. Keep a daily cap of 4–5 hours, with a short saved checkpoint at the end of each session and no unattended overnight jobs. Replan from actual elapsed work rather than trying to "catch up" by skipping sleep.

## Research sources

- TensorFlow says native-Windows TensorFlow GPU support ended after 2.10; newer versions need WSL2 for supported NVIDIA GPU use: https://www.tensorflow.org/install/pip
- Colab resources and GPU type are not guaranteed; availability and usage limits vary: https://research.google.com/colaboratory/faq.html
- TensorFlow Recommenders describes two-stage retrieval/ranking, in-batch retrieval training, and its accidental-hit mask: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval

No external message was sent and no model was trained during this analysis.

## Later retrieval diagnostics — October 24, 2026

These are separate development runs, each evaluated against popularity on the same seeded 2,000-user MovieLens 25M validation cohort. They are not final test scores.

- A custom in-batch mask excluded other same-user positive examples and repeated movie IDs from the negative set. It did **not** improve results: all-user Recall@10 0.05171 vs popularity 0.05652; NDCG@10 0.27289 vs 0.29081. On 528 users with trained embeddings, Recall@10 was 0.00053 vs 0.01878 and NDCG@10 was 0.00150 vs 0.06937. The metric summary remains in `reports/retrieval-masked-pilot`; its generated model and embedding binaries were removed because the variant was rejected.
- A separate 500,000-row pilot trained on all observed rating interactions rather than only ratings >=4. It also did **not** improve results: Recall@10 0.05193 vs 0.05652; NDCG@10 0.27371 vs 0.29081. Only 516 of the sampled 2,000 users had learned embeddings; those users were far below popularity. It indexed 14,226 of 34,461 training-catalog items (41.3%). The metric summary remains in `reports/retrieval-all-ratings-pilot`; its generated model and embedding binaries were removed because the variant was rejected.
- The all-ratings objective treats every rating as an observed positive interaction, while evaluation still measures ratings >=4. That mismatch is deliberate for this diagnostic and may be part of why results stay low; the experiment does not show that all-ratings training is suitable for the final model.
- The first high-rating pilot remains the best of these pilot neural models, but still trails popularity. The high-rating same-user-mask pilot is substantially worse. None justifies replacing the popularity fallback.
- A recent-per-user preparation run did not finish and produced no report. The child Python process disappeared without printing an error; a memory/resource limit is plausible but unconfirmed. Do not claim a proven OOM diagnosis.

### Updated recommendation

Stop trying small changes on the same local pipeline for now. First confirm project scope with the manager. If retrieval remains a priority, use one carefully bounded GPU session or a memory-efficient streaming data pipeline; record the exact environment; verify validation-user coverage before training; define whether the target is a watch/rating or a high rating; and keep popularity as the measured benchmark and cold-start fallback. Run only one controlled change at a time. Do not spend time on the ranking model until retrieval is competitive on validation.

## Latest run: bounded user-balanced pilot

The streaming command completed without materializing the full ratings table. The default per-user sample trained on 413,616 recent interactions from 137,883 training users. Three CPU epochs finished and model artifacts were saved. On the fixed 2,000-user validation cohort, two-tower Recall@10 0.05117 / NDCG@10 0.27061 remained below popularity 0.05652 / 0.29081. Only 570 users had embeddings, and those results were much worse than their popularity baseline. The test set stayed untouched. The output is in `reports/retrieval-streaming-pilot`.
