# Audit of the saved Project 2 Colab retrieval run

Read-only review on 5 November 2026 of the saved Google Colab notebook [Nexvia_AIML03_Streaming_Retrieval.ipynb](https://colab.research.google.com/drive/11oP8Zl8vrVt55X6BJd6hQbArFEYmgIuK). The notebook header showed it was last edited on 24 October. No notebook cell was run or edited during this review. Its saved outputs are historical evidence, not a fresh run.

## Full-positive two-tower run (saved notebook cell 10)

- Dataset checksum shown in the notebook matched the local MovieLens 25M `ratings.csv` checksum: `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`.
- Training used **9,981,093** positive user-item pairs (`rating >= 4`), **137,682** users, and indexed **24,740** items out of a **34,461**-item training catalog (**71.8%** coverage).
- The saved output reports TensorFlow Recommenders 0.7.3, FAISS 1.15.1, a visible GPU, batch size 512, embedding size 32, and 3 epochs. The displayed training duration is **277.6 seconds**. Epoch losses were **5.104267, 5.104278, 5.104559**; they did not materially fall across these epochs. This describes the logged values; it does not establish the cause.
- The selected 2,000-user validation cohort came from the old 80–90% time window. The notebook reports `test_period_used: false`; the final 10% was not scored in this run.
- Full-cohort hybrid Recall@10/NDCG@10 was **0.05120 / 0.27065**, below popularity at **0.05652 / 0.29081**.
- On 567 users with learned embeddings, two-tower top-10 Recall@10/NDCG@10 was **0.000262 / 0.001484**, below popularity at **0.01902 / 0.07259**.
- For a more like-for-like candidate-stage check, the notebook restricted the comparison to the same indexed catalog and the 555 warm users with at least one eligible target there (12 were skipped; 17,085 target pairs). Two-tower Recall@100/NDCG@100 was **0.00446 / 0.00242**; popularity on that same catalog was **0.13285 / 0.09420**.

## What this does and does not establish

This is evidence that the full 9.98-million-positive run did not outperform the popularity fallback on the saved, reused validation cohort. It means that simply increasing the sampled pilot to all eligible positive pairs did not solve the observed warm-user ranking problem in this run. It does **not** identify the root cause, prove the architecture can never work, or provide an independent test result.

The same notebook contains an earlier 413,616-row pilot and extra epochs selected on validation NDCG. Those are separate historical experiments and are not a clean independent comparison. The complete notebook output should be reviewed before making any claim that the selection procedure was fixed in advance.

## Record limitations

- The notebook is in the user's Google Drive, not copied into this project folder. A download attempt did not complete, so no local notebook archive or checkpoint was created from it.
- Whether the model artifacts written under Colab `/content/nexvia_aiml03_full_positive` were later persisted elsewhere is **NOT VERIFIED**.
- This audit did not train, score, download data, or use the final 10% test window.

