# Experiment notes

## October 24, 2026 — First baseline

Started with popularity because it is quick to inspect and makes a useful cold-start fallback. No model training or deployment claims yet.

The small development dataset contains 100,836 ratings. The temporal split produced 80,668 training rows, 10,084 validation rows and 10,084 test rows. Its test Recall@10 was 0.06790 and NDCG@10 was 0.27864, over 72 users with positive targets. These are development-only numbers, not MovieLens 25M results.

Four tests passed: hand-calculated metrics, timestamp boundary isolation, seen-item filtering with unseen-catalog misses, and an explicit error for missing positive targets.

The available Python installation is 3.10.1 with pandas 2.2.3. PyTorch is 2.5.0+cpu, so it cannot currently use the laptop GPU. The popularity baseline does not need CUDA. Set up an isolated GPU environment before the neural stage.

Questions still open with the manager: whether modules are mandatory for certification, whether every listed integration is required, acceptable dataset subsets, and expectations for the online A/B test. No message has been sent.

## Full MovieLens 25M baseline — completed October 24

The full run completed successfully over 25,000,095 ratings: 20,000,076 training, 2,500,009 validation and 2,500,010 test rows.

Validation: Recall@10 = 0.05825, NDCG@10 = 0.29435 (17,847 evaluated users).
Test: Recall@10 = 0.05184, NDCG@10 = 0.27137 (18,276 evaluated users).

The test average mixes 6,330 warm and 11,946 cold users. Warm-user NDCG@10 is 0.07479, so the overall score alone is not a good measure of personalization. 11.48% of positive test targets are absent from the frozen training catalog. These misses remain in the denominator.

Artifacts: reports/ml-25m/metrics.json, popularity.csv, validation_users.csv, test_users.csv. All four tests also passed from this Desktop copy.

Next: prepare reusable features and a two-tower retrieval baseline, using validation for model decisions. No online CTR, A/B-test or neural-model results exist yet.

Hardware correction: nvidia-smi reports 4096 MiB VRAM (4 GB) on the RTX 3050 Laptop GPU.

## Follow-up decision — October 24, 2026

A full one-epoch pass over all 9,981,093 positive training rows did not finish cleanly in this Windows CPU-only TensorFlow environment. It produced no saved model artifacts, so there are no full-data run results to report. The working reference remains the saved, reproducible 500,000-row pilot above. Continue full training in a Colab T4 session or start with a smaller all-users stratified sample. The local full-data run was stopped after its training process disappeared without output.

## October 24 — Negative diagnostic results

The same-user masked loss pilot did worse than popularity on the same validation cohort. A 500,000-example all-ratings pilot also did worse and changed the target from high ratings to all observed ratings, so it is not directly the preferred final objective. Both outputs and metrics are saved in separate report folders. Neither is evidence of progress in quality.

A recent-per-user sample preparation attempt ended after the child Python process disappeared before any log/report; resource exhaustion is plausible, not confirmed. Stop repeating the same large local preparation. Next priority: get project scope in writing, then pick either a bounded GPU session or a genuinely streaming sampler. Keep validation fixed and test sealed.

## Streaming user-balanced run — October 24, 2026

The bounded loader completed on the full MovieLens 25M file. It kept the previously established cutoffs from `reports/ml-25m/metrics.json` and scanned the ratings CSV in chunks. For training it retained up to three most recent interactions per training user: 413,616 interactions across 137,883 users, out of 20,000,076 total training interactions. The fixed validation cohort was 2,000 users. This resolved the earlier data-preparation problem, though it did not improve recommendation quality.

After three CPU epochs, overall Recall@10 was 0.05117 and NDCG@10 was 0.27061, below popularity on the exact same users (0.05652 and 0.29081). Only 570 of those users had learned embeddings; for them, Recall@10 was 0.00020 vs 0.01897 for popularity and NDCG@10 0.00147 vs 0.07235. The other 1,430 used the popularity fallback. The index contains 12,859 of 34,461 training-catalog items (37.3%). Training loss stayed near 6.068 across all epochs. No test users were evaluated.

Saved in `reports/retrieval-streaming-pilot`. This is a negative validation result. Keep popularity as the current choice; don't advance to ranking or claim a personalized-model gain.
