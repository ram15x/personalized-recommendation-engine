import hashlib
import json

import pytest

from scripts import retraining_workflow as workflow


def test_preflight_refuses_duplicates_and_review_checks_split(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "retrieval.py").write_text("", encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    ratings.write_bytes(b"test data\n")
    checksum = hashlib.sha256(ratings.read_bytes()).hexdigest()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"ratings_sha256": checksum, "split": {
        "train_before_timestamp": 100, "test_from_timestamp": 200}}), encoding="utf-8")
    run = workflow.prepare_run(project, ratings, split, tmp_path / "runs", "manual/1")
    with pytest.raises(FileExistsError):
        workflow.prepare_run(project, ratings, split, tmp_path / "runs", "manual/1")

    calls = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    workflow.train(run, "python")
    assert "--streaming" in calls[0][0] and calls[0][1]["check"] is True
    model = run / "model"
    model.mkdir()
    metrics = {"test_evaluated": False, "ratings_sha256": checksum,
               "train_before_timestamp": 100, "test_from_timestamp": 200}
    (model / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert workflow.inspect_result(run)["status"] == "review_required"
    metrics["test_evaluated"] = True
    (model / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        workflow.inspect_result(run)


def test_preflight_rejects_wrong_data_hash(tmp_path):
    (tmp_path / "retrieval.py").write_text("", encoding="utf-8")
    ratings = tmp_path / "ratings.csv"
    ratings.write_bytes(b"data")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"ratings_sha256": "0" * 64, "split": {
        "train_before_timestamp": 1, "test_from_timestamp": 2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        workflow.prepare_run(tmp_path, ratings, split, tmp_path / "runs", "run")
