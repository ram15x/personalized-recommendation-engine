"""Manual-only Airflow 3 DAG for a bounded, review-required pilot run."""
from datetime import datetime, timezone
import os
from pathlib import Path

from airflow.sdk import dag, task, get_current_context


@dag(schedule=None, start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1, tags=["nexvia", "experimental"])
def nexvia_manual_retraining():
    @task(retries=0)
    def preflight():
        from scripts.retraining_workflow import prepare_run
        project = Path(os.environ["NEXVIA_PROJECT_ROOT"])
        return str(prepare_run(project, Path(os.environ["NEXVIA_RATINGS_PATH"]),
                               Path(os.environ["NEXVIA_SPLIT_REPORT"]),
                               Path(os.environ["NEXVIA_RETRAIN_OUTPUT_ROOT"]),
                               get_current_context()["run_id"]))

    @task(retries=0)
    def train(run_dir):
        from scripts.retraining_workflow import train as run_training
        run_training(Path(run_dir), os.environ["NEXVIA_RETRAIN_PYTHON"])
        return run_dir

    @task(retries=0)
    def require_review(run_dir):
        from scripts.retraining_workflow import inspect_result
        return inspect_result(Path(run_dir))

    require_review(train(preflight()))


nexvia_manual_retraining()
