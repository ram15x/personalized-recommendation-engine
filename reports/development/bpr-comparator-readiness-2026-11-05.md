# BPR comparator readiness check

## Scope

This check prepared, but did not train, the pre-registered BPR two-tower comparison. No dataset download occurred, the official final 10% period was not read, and no existing checkpoint or baseline artifact was modified.

## Verified

- `scripts/train_bpr_objective_comparator.py` passes local `python -m py_compile`. SHA-256: `6cd990e04701ef24292e902ca06793ce3caae07e1b073e0a1da0c26724f6e003` (18,398 bytes).
- Local command `python -m pytest -q test_baseline.py`: **4 passed in 2.34s**.
- A helper smoke check with dependency stubs sampled six negatives against four known user-item training pairs and found zero collisions; the small Recall/NDCG example returned `1.0 / 0.63092975`. This checks only the sampler/metric helpers, not model training or retrieval.
- Colab preflight in the open `Untitled2.ipynb` confirmed `/content/ml-25m/ratings.csv` exists, is 678,260,987 bytes, and has SHA-256 `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385`, matching the comparator's expected checksum.
- That preflight found `/content/train_bpr_objective_comparator.py` present at 17,787 bytes and `/content/nexvia_aiml03_bpr_70_80_seed42` absent. This was before the attempted synchronization of the newer local script.
- Local Python was 3.10.1. `tensorflow`, `pandas`, `numpy`, and `pytest` were importable; `tensorflow_recommenders` and `faiss` were not.

## Failure and what did not happen

- The local script was uploaded to Colab, but it did not match the edited local copy. A guarded patch cell stopped with the exact error `AssertionError: expected original thread setup exactly once`. This was a safety check in the sync cell; it stopped before writing the remote script.
- No BPR training or evaluation command was launched. There is no BPR model, metric, or improvement result from this attempt.
- The full local command `python -m pytest -q` stopped during collection with `ModuleNotFoundError: No module named 'faiss'` from `test_retrieval.py`. No dependency installation was attempted.
- The remote script hash after the attempted patch was not rechecked; local/remote source equality is therefore **NOT VERIFIED**.

## Next safe action

The readiness note above records the initial failed synchronization only. The follow-up run is complete: a fresh 18,398-byte script upload matched the local SHA-256, compiled in Colab, and ran against the checksum-verified ratings file to a new output directory with exit code 0. See [the BPR comparison report](bpr-objective-comparison-2026-11-05.md) for the measured results. Do not treat the reused 70/80 panel as an independent test or promote the model from this comparison alone.
