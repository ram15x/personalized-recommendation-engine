"""TensorFlow-free NumPy inference for the saved DCN ranker in Triton."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def export_weights(artifact_dir: Path) -> Path:
    """Export the checked Keras DCN weights and add their hash to the manifest."""
    from scripts.candidate_ranker_inference import CandidateRanker

    artifact_dir = Path(artifact_dir)
    checked = CandidateRanker.load(artifact_dir)
    model = checked.model
    arrays: dict[str, np.ndarray] = {
        "user_embedding": model.user_embedding.get_weights()[0],
        "item_embedding": model.item_embedding.get_weights()[0],
    }
    for i, layer in enumerate(model.cross):
        kernel, bias = layer.get_weights()
        arrays[f"cross_{i}_kernel"] = kernel
        arrays[f"cross_{i}_bias"] = bias
    for i, layer in enumerate(model.deep):
        kernel, bias = layer.get_weights()
        arrays[f"deep_{i}_kernel"] = kernel
        arrays[f"deep_{i}_bias"] = bias
    arrays["output_kernel"], arrays["output_bias"] = model.output_layer.get_weights()

    target = artifact_dir / "triton_numpy_weights.npz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    np.savez(target, **arrays)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{digest}  {target.name}\n", encoding="ascii"
    )
    return target


if __name__ == "__main__":
    output = export_weights(Path(sys.argv[1]))
    print(output)


class NumpyCandidateRanker:
    """Same frozen ID mapping/support as CandidateRanker, with NumPy DCN math."""

    def __init__(self, directory: Path):
        from scripts.candidate_ranker_inference import _sha256

        directory = Path(directory)
        manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
        for name in ("dcn_candidate_ranker.keras", "feature_mapping.npz", "trained_support.json", "metrics.json"):
            expected = manifest.get("files_sha256", {}).get(name)
            if expected != _sha256(directory / name):
                raise ValueError(f"Ranker artifact hash mismatch: {name}")
        weights_path = directory / "triton_numpy_weights.npz"
        checksum_path = directory / "triton_numpy_weights.npz.sha256"
        expected_weights_hash = checksum_path.read_text(encoding="ascii").split()[0]
        if expected_weights_hash != _sha256(weights_path):
            raise ValueError("Ranker NumPy weight hash mismatch.")
        with np.load(directory / "feature_mapping.npz", allow_pickle=False) as values:
            self.users = values["user_ids"].astype(np.int64)
            self.items = values["item_ids"].astype(np.int64)
            self.mean = values["numeric_mean"].astype(np.float32)
            self.std = values["numeric_std"].astype(np.float32)
        support = json.loads((directory / "trained_support.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        source_hash = support.get("training_export_sha256")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or source_hash != metrics.get("training_export_sha256")):
            raise ValueError("Trained support does not match the recorded training export.")
        self.trained_users = np.asarray(support["user_ids"], dtype=np.int64)
        self.trained_items = np.asarray(support["item_ids"], dtype=np.int64)
        with np.load(directory / "triton_numpy_weights.npz", allow_pickle=False) as weights:
            self.w = {name: weights[name].astype(np.float32) for name in weights.files}

    def score(self, user_id: int, item_ids, retrieval_scores) -> list[float | None]:
        ids = np.asarray(item_ids, dtype=np.int64)
        retrieval = np.asarray(retrieval_scores, dtype=np.float32)
        if (ids.ndim != 1 or retrieval.shape != ids.shape or np.any(ids <= 0)
                or len(np.unique(ids)) != len(ids) or not np.isfinite(retrieval).all()
                or np.any(retrieval < 0)):
            raise ValueError("Candidate IDs and nonnegative finite scores must be aligned and unique.")
        user_pos = int(np.searchsorted(self.users, user_id))
        if (user_pos == len(self.users) or self.users[user_pos] != user_id
                or not np.isin(user_id, self.trained_users)):
            raise ValueError("Unknown ranker user; use the candidate-order fallback.")
        item_pos = np.searchsorted(self.items, ids)
        known = item_pos < len(self.items)
        known[known] = self.items[item_pos[known]] == ids[known]
        known &= np.isin(ids, self.trained_items)
        result: list[float | None] = [None] * len(ids)
        if not known.any():
            return result
        ui = np.full(int(known.sum()), user_pos + 1, dtype=np.int64)
        ii = item_pos[known] + 1
        numeric = (np.log1p(retrieval[known, None]) - self.mean) / self.std
        x0 = np.concatenate([self.w["user_embedding"][ui], self.w["item_embedding"][ii], numeric], axis=1)
        crossed = x0
        cross_count = sum(key.startswith("cross_") and key.endswith("_kernel") for key in self.w)
        for i in range(cross_count):
            projection = np.sum(crossed * self.w[f"cross_{i}_kernel"], axis=1, keepdims=True)
            crossed = x0 * projection + self.w[f"cross_{i}_bias"] + crossed
        deep = x0
        deep_count = sum(key.startswith("deep_") and key.endswith("_kernel") for key in self.w)
        for i in range(deep_count):
            deep = np.maximum(deep @ self.w[f"deep_{i}_kernel"] + self.w[f"deep_{i}_bias"], 0)
        predictions = (np.concatenate([crossed, deep], axis=1) @ self.w["output_kernel"]
                       + self.w["output_bias"]).reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError("Ranker produced non-finite predictions.")
        for ix, prediction in zip(np.flatnonzero(known), predictions):
            result[int(ix)] = float(prediction)
        return result
