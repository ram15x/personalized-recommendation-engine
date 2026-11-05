import sqlite3

import pytest

from scripts.experiment_store import ExperimentStore


def test_assignment_impression_click_and_item_ctr(tmp_path):
    store = ExperimentStore(tmp_path / "experiment.sqlite", "a-secure-test-secret")
    user_hash, variant = store.assignment("local-participant")
    assert store.assignment("local-participant") == (user_hash, variant)
    impression = store.record_impression("local-participant", variant, "itemknn_cosine", [10, 20])
    assert store.record_click(impression, 10) is True
    assert store.record_click(impression, 10) is False
    with pytest.raises(ValueError, match="displayed item"):
        store.record_click(impression, 99)
    result = store.summary()[variant]
    assert result["participants"] == 1 and result["impressions"] == 1
    assert result["item_ctr"] == .5 and result["clicked_items"] == 1
    with sqlite3.connect(store.path) as conn:
        assert "local-participant" not in str(conn.execute("SELECT * FROM impressions").fetchall())


def test_rejects_invalid_assignment_and_impressions(tmp_path):
    store = ExperimentStore(tmp_path / "events.sqlite", "a-secure-test-secret")
    _, variant = store.assignment("user")
    with pytest.raises(ValueError, match="assignment"):
        store.record_impression("user", "B" if variant == "A" else "A", "itemknn", [10])
    with pytest.raises(ValueError, match="unique"):
        store.record_impression("user", variant, "itemknn", [10, 10])
