"""Export the ItemKNN training-window popularity table without loading all ratings."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = ROOT / "data" / "ml-25m" / "ratings.csv"
DEFAULT_ARTIFACT_DIR = (
    ROOT / "reports" / "development" / "itemknn-60-70-corrected-cohort-2k-v2"
)
CHUNK_ROWS = 500_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(ratings_path: Path, artifact_dir: Path) -> tuple[Path, Path]:
    artifact_path = artifact_dir / "artifact_manifest.json"
    metrics_path = artifact_dir / "metrics.json"
    ids_path = artifact_dir / "positive_item_ids.npy"
    output_path = artifact_dir / "training_popularity.csv"
    output_manifest_path = artifact_dir / "training_popularity_manifest.json"

    if not ratings_path.is_file():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    for path in (output_path, output_manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    actual_hash = sha256(ratings_path)
    if actual_hash != artifact.get("ratings_sha256"):
        raise ValueError("Ratings file hash differs from the ItemKNN artifact manifest.")
    cutoff = int(artifact["train_cutoff_timestamp"])
    threshold = float(artifact["positive_rating_threshold"])

    counts: Counter[int] = Counter()
    train_rows = 0
    for frame in pd.read_csv(
        ratings_path,
        usecols=["movieId", "rating", "timestamp"],
        chunksize=CHUNK_ROWS,
    ):
        train = frame.loc[frame["timestamp"] < cutoff]
        train_rows += len(train)
        positive = train.loc[train["rating"] >= threshold, "movieId"]
        counts.update(map(int, positive))

    if train_rows != int(metrics["train_rows"]):
        raise ValueError(
            f"Training rows differ from the saved metrics: {train_rows} vs {metrics['train_rows']}."
        )
    if sum(counts.values()) != int(metrics["positive_training_pairs"]):
        raise ValueError("Positive interaction count differs from the saved metrics.")
    item_ids = np.load(ids_path, allow_pickle=False)
    counted_ids = np.asarray(sorted(counts), dtype=np.int64)
    if not np.array_equal(counted_ids, item_ids.astype(np.int64, copy=False)):
        raise ValueError("Popularity item IDs differ from the ItemKNN item-ID mapping.")

    table = pd.DataFrame(
        [(movie_id, counts[movie_id]) for movie_id in counted_ids],
        columns=["movieId", "positive_count"],
    ).sort_values(
        ["positive_count", "movieId"],
        ascending=[False, True],
        kind="stable",
    )
    temporary_csv = output_path.with_suffix(".csv.tmp")
    temporary_manifest = output_manifest_path.with_suffix(".json.tmp")
    manifest = {
        "schema_version": 1,
        "source": str(ratings_path.name),
        "ratings_sha256": actual_hash,
        "train_cutoff_timestamp_exclusive": cutoff,
        "positive_rating_threshold": threshold,
        "training_rows": train_rows,
        "positive_training_pairs": sum(counts.values()),
        "item_count": len(counted_ids),
    }
    try:
        table.to_csv(temporary_csv, index=False)
        manifest["popularity_sha256"] = sha256(temporary_csv)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_csv, output_path)
        os.replace(temporary_manifest, output_manifest_path)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        output_manifest_path.unlink(missing_ok=True)
        raise
    return output_path, output_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    csv_path, manifest_path = export(args.ratings, args.artifact_dir)
    print(json.dumps({
        "popularity_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "popularity_sha256": sha256(csv_path),
        "manifest_sha256": sha256(manifest_path),
        "bytes": csv_path.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
