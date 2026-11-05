import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_itemknn_popularity import export, sha256


def test_export_is_time_aligned_manifest_checked_and_refuses_overwrite(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    ratings = tmp_path / "ratings.csv"
    pd.DataFrame({
        "userId": [1, 1, 2, 2],
        "movieId": [10, 10, 20, 30],
        "rating": [5.0, 4.0, 4.0, 3.0],
        "timestamp": [8, 10, 7, 9],
    }).to_csv(ratings, index=False)
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps({
        "ratings_sha256": sha256(ratings),
        "train_cutoff_timestamp": 10,
        "positive_rating_threshold": 4.0,
    }), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps({
        "train_rows": 3,
        "positive_training_pairs": 2,
    }), encoding="utf-8")
    np.save(artifact_dir / "positive_item_ids.npy", np.array([10, 20], dtype=np.int64))

    csv_path, manifest_path = export(ratings, artifact_dir)
    result = pd.read_csv(csv_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.to_dict("records") == [
        {"movieId": 10, "positive_count": 1},
        {"movieId": 20, "positive_count": 1},
    ]
    assert manifest["training_rows"] == 3
    assert manifest["positive_training_pairs"] == 2
    assert manifest["popularity_sha256"] == sha256(csv_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export(ratings, artifact_dir)
