"""Guardrails for a manually triggered two-tower retraining job."""
import hashlib
import json
import subprocess
from pathlib import Path


def prepare_run(project: Path, ratings: Path, split_report: Path, output_root: Path, run_id: str) -> Path:
    project, ratings, split_report, output_root = map(Path, (project, ratings, split_report, output_root))
    if not run_id or not ratings.is_file() or not split_report.is_file() or not (project / "retrieval.py").is_file():
        raise ValueError("Existing project, ratings, split report, and run ID are required.")
    report = json.loads(split_report.read_text(encoding="utf-8"))
    split = report["split"]
    train_end, validation_end = int(split["train_before_timestamp"]), int(split["test_from_timestamp"])
    if train_end <= 0 or validation_end <= train_end:
        raise ValueError("Split timestamps must be positive and ordered.")
    expected_hash = report.get("ratings_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("The split report must contain the ratings SHA-256.")
    digest = hashlib.sha256()
    with ratings.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Ratings hash does not match the split report.")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
    run_dir.mkdir(exist_ok=False)
    config = {"run_id": run_id, "project": str(project.resolve()), "ratings": str(ratings.resolve()),
              "split_report": str(split_report.resolve()), "train_before_timestamp": train_end,
              "validation_end_timestamp": validation_end, "ratings_sha256": expected_hash}
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return run_dir


def train(run_dir: Path, python_executable: str) -> None:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    output = run_dir / "model"
    if output.exists():
        raise FileExistsError(f"Run output exists; refusing overwrite: {output}")
    command = [python_executable, str(Path(config["project"]) / "retrieval.py"),
               "--ratings", config["ratings"], "--split-report", config["split_report"],
               "--streaming", "--output", str(output), "--dataset-label", "ml-25m",
               "--interaction-mode", "high-ratings", "--max-train-positives", "500000",
               "--interactions-per-user", "3", "--epochs", "3", "--eval-users", "2000"]
    subprocess.run(command, cwd=config["project"], check=True)


def inspect_result(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "model" / "metrics.json").read_text(encoding="utf-8"))
    if (metrics.get("test_evaluated") is not False
            or metrics.get("ratings_sha256") != config["ratings_sha256"]
            or metrics.get("train_before_timestamp") != config["train_before_timestamp"]
            or metrics.get("test_from_timestamp") != config["validation_end_timestamp"]):
        raise ValueError("Run metrics do not match the recorded data and split contract.")
    # Evaluation and human review are separate: no automatic promotion.
    return {"status": "review_required", "metrics_path": str(run_dir / "model" / "metrics.json")}
