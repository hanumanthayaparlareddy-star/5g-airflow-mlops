from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


PIPELINE_ROOT = Path("/opt/5g-mlops")
RAW_DATA = PIPELINE_ROOT / "data/raw/5g_metrics.csv"
DATA_DIR = PIPELINE_ROOT / "data/processed/airflow-run"
MODEL_DIR = PIPELINE_ROOT / "models/airflow-run"
F1_OUTPUT = DATA_DIR / "f1_score.txt"
F1_THRESHOLD = "0.90"

MLFLOW_URI = (
    "http://mlflow-mlflow.mlflow.svc.cluster.local:5000"
)


def run_command(args, env):
    """Run one pipeline stage and fail immediately on errors."""
    result = subprocess.run(
        args,
        cwd=str(PIPELINE_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline command failed with exit code "
            f"{result.returncode}: {' '.join(args)}"
        )

    return result.stdout


def run_5g_pipeline():
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_URI

    if not RAW_DATA.is_file():
        raise FileNotFoundError(
            f"Raw dataset is missing: {RAW_DATA}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    featured_file = DATA_DIR / "featured.csv"
    train_file = DATA_DIR / "train.csv"
    test_file = DATA_DIR / "test.csv"
    model_file = MODEL_DIR / "model.pkl"

    # 1. Validate the raw dataset.
    run_command(
        [sys.executable, "src/data_validation.py", str(RAW_DATA)],
        env,
    )

    # 2. Engineer features.
    run_command(
        [
            sys.executable,
            "src/feature_engineering.py",
            str(RAW_DATA),
            str(featured_file),
        ],
        env,
    )

    # 3. Split and preprocess the engineered dataset.
    run_command(
        [
            sys.executable,
            "src/preprocess.py",
            str(featured_file),
            str(DATA_DIR),
        ],
        env,
    )

    # 4. Train and capture the MLflow run ID.
    training_output = run_command(
        [
            sys.executable,
            "src/train.py",
            str(train_file),
            str(MODEL_DIR),
        ],
        env,
    )

    match = re.search(
        r"Run ID\s*:\s*([0-9a-fA-F]+)",
        training_output,
    )
    if not match:
        raise RuntimeError(
            "Training completed but its MLflow run ID "
            "could not be extracted from the output."
        )

    run_id = match.group(1)
    print(f"Captured MLflow run ID: {run_id}")

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Trained model file is missing: {model_file}"
        )

    # 5. Evaluate. A failed F1 gate raises an error and stops here.
    run_command(
        [
            sys.executable,
            "src/evaluate.py",
            str(model_file),
            str(test_file),
            str(F1_OUTPUT),
            F1_THRESHOLD,
        ],
        env,
    )

    # 6. Register only after evaluation succeeds.
    run_command(
        [
            sys.executable,
            "src/register_model.py",
            run_id,
        ],
        env,
    )

    print(
        "PIPELINE SUCCEEDED: quality gate passed "
        "and the model was registered."
    )
    return run_id


with DAG(
    dag_id="5g_ml_pipeline",
    description="Validate, train, evaluate, and register the 5G congestion model",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["5g-mlops", "mlflow", "model-training"],
) as dag:
    run_pipeline = PythonOperator(
        task_id="validate_train_evaluate_register",
        python_callable=run_5g_pipeline,
    )
