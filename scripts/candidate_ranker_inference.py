"""Load and score the saved experimental DCN candidate ranker.

Unknown users are rejected and unknown items are returned with no score. The
caller must choose a fallback ordering; this module does not promote a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateRanker:
    def __init__(self, model, users, items, mean, std, trained_users=None, trained_items=None):
        self.model = model
        self.users = np.asarray(users, dtype=np.int64)
        self.items = np.asarray(items, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if (self.users.ndim != 1 or self.items.ndim != 1 or not len(self.users)
                or not len(self.items) or np.any(np.diff(self.users) <= 0)
                or np.any(np.diff(self.items) <= 0)):
            raise ValueError("Ranker ID vocabularies must be sorted and unique.")
        if (self.mean.shape != (1,) or self.std.shape != (1,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("Ranker feature normalization is invalid.")
        self.trained_users = np.asarray(self.users if trained_users is None else trained_users, dtype=np.int64)
        self.trained_items = np.asarray(self.items if trained_items is None else trained_items, dtype=np.int64)
        for supported, vocabulary in ((self.trained_users, self.users), (self.trained_items, self.items)):
            if (supported.ndim != 1 or not len(supported) or np.any(np.diff(supported) <= 0)
                    or not np.isin(supported, vocabulary).all()):
                raise ValueError("Trained ID support must be sorted, unique, and within the vocabulary.")

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
        for name in ("dcn_candidate_ranker.keras", "feature_mapping.npz", "trained_support.json", "metrics.json"):
            if manifest.get("files_sha256", {}).get(name) != _sha256(directory / name):
                raise ValueError(f"Ranker artifact hash mismatch: {name}")
        with np.load(directory / "feature_mapping.npz", allow_pickle=False) as m:
            users, items = m["user_ids"], m["item_ids"]
            mean, std = m["numeric_mean"], m["numeric_std"]
        support = json.loads((directory / "trained_support.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        source_hash = support.get("training_export_sha256")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or source_hash != metrics.get("training_export_sha256")):
            raise ValueError("Trained support does not match the recorded training export.")
        import tensorflow as tf
        import scripts.dcn_ranker  # Register the saved model's custom layers.
        model = tf.keras.models.load_model(directory / "dcn_candidate_ranker.keras", compile=False)
        return cls(model, users, items, mean, std, support["user_ids"], support["item_ids"])

    def score(self, user_id: int, item_ids, retrieval_scores) -> list[float | None]:
        ids = np.asarray(item_ids, dtype=np.int64)
        scores = np.asarray(retrieval_scores, dtype=np.float32)
        if (ids.ndim != 1 or scores.shape != ids.shape or np.any(ids <= 0)
                or len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all()
                or np.any(scores < 0)):
            raise ValueError("Candidate IDs and nonnegative finite scores must be aligned and unique.")
        user_pos = int(np.searchsorted(self.users, user_id))
        if user_pos == len(self.users) or self.users[user_pos] != user_id or user_id not in self.trained_users:
            raise ValueError("Unknown ranker user; use the candidate-order fallback.")
        positions = np.searchsorted(self.items, ids)
        known = positions < len(self.items)
        known[known] = self.items[positions[known]] == ids[known]
        known &= np.isin(ids, self.trained_items)
        output: list[float | None] = [None] * len(ids)
        if not np.any(known):
            return output
        inputs = {
            "user_id": np.full(int(known.sum()), user_pos + 1, dtype=np.int32),
            "item_id": (positions[known] + 1).astype(np.int32),
            "numeric": (np.log1p(scores[known, None]) - self.mean) / self.std,
        }
        predictions = np.asarray(self.model(inputs, training=False)).reshape(-1)
        if len(predictions) != int(known.sum()) or not np.isfinite(predictions).all():
            raise ValueError("Ranker produced invalid predictions.")
        for index, prediction in zip(np.flatnonzero(known), predictions):
            output[int(index)] = float(prediction)
        return output

    def rank_or_fallback(self, user_id: int, item_ids, retrieval_scores) -> dict:
        """Rerank a complete known candidate list, or retain its retrieval order.

        Unknown items cannot be compared fairly with scored items, so a single
        unknown candidate causes the complete list to retain retrieval order.
        Invalid inputs and model errors still raise; they are not hidden.
        """
        ids = list(item_ids)
        user_pos = int(np.searchsorted(self.users, user_id))
        if user_pos == len(self.users) or self.users[user_pos] != user_id or user_id not in self.trained_users:
            # Validate the candidate inputs as well, even on a fallback.
            values = np.asarray(retrieval_scores, dtype=np.float32)
            id_array = np.asarray(ids, dtype=np.int64)
            if (id_array.ndim != 1 or values.shape != id_array.shape or np.any(id_array <= 0)
                    or len(np.unique(id_array)) != len(id_array)
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError("Candidate IDs and nonnegative finite scores must be aligned and unique.")
            return {"item_ids": ids, "strategy": "retrieval_fallback", "reason": "unknown_user"}
        scores = self.score(user_id, ids, retrieval_scores)
        if any(value is None for value in scores):
            return {"item_ids": ids, "strategy": "retrieval_fallback", "reason": "unknown_items"}
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
        return {"item_ids": [ids[index] for index in order], "strategy": "experimental_dcn", "reason": None}
