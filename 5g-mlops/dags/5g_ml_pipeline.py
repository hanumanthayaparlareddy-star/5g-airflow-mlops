from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from kubernetes.client import models as k8s


PIPELINE_ROOT = Path("/opt/5g-mlops")
RAW_DATA = PIPELINE_ROOT / "data/raw/5g_metrics.csv"

# These directories live on the shared PVC.
SHARED_DIR = PIPELINE_ROOT / "shared"
DATA_DIR = SHARED_DIR / "data/processed/airflow-run"
MODEL_DIR = SHARED_DIR / "models/airflow-run"

FEATURED_FILE = DATA_DIR / "featured.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "model.pkl"
F1_OUTPUT = DATA_DIR / "f1_score.txt"

F1_THRESHOLD = "0.90"
MLFLOW_URI = "http://mlflow-mlflow.mlflow.svc.cluster.local:5000"


# Add the PVC to each KubernetesExecutor task pod.
POD_OVERRIDE = {
    "pod_override": k8s.V1Pod(
        spec=k8s.V1PodSpec(
            containers=[
                k8s.V1Container(
                    name="base",
                    volume_mounts=[
                        k8s.V1VolumeMount(
                            name="5g-mlops-shared",
                            mount_path=str(SHARED_DIR),
                        )
                    ],
                )
            ],
            volumes=[
                k8s.V1Volume(
                    name="5g-mlops-shared",
                    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                        claim_name="5g-mlops-shared-pvc"
                    ),
                )
            ],
        )
    )
}


def get_env():
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    return env


def run_command(args):
    result = subprocess.run(
        args,
        cwd=str(PIPELINE_ROOT),
        env=get_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(args)}"
        )

    return result.stdout


def validate_data():
    if not RAW_DATA.is_file():
        raise FileNotFoundError(f"Dataset missing: {RAW_DATA}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    run_command(
        [sys.executable, "src/data_validation.py", str(RAW_DATA)]
    )


def feature_engineering():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            "src/feature_engineering.py",
            str(RAW_DATA),
            str(FEATURED_FILE),
        ]
    )


def preprocess_data():
    run_command(
        [
            sys.executable,
            "src/preprocess.py",
            str(FEATURED_FILE),
            str(DATA_DIR),
        ]
    )


def train_model():
    output = run_command(
        [
            sys.executable,
            "src/train.py",
            str(TRAIN_FILE),
            str(MODEL_DIR),
        ]
    )

    match = re.search(r"Run ID\s*:\s*([0-9a-fA-F]+)", output)
    if not match:
        raise RuntimeError(
            "Training finished, but the MLflow run ID was not found."
        )

    run_id = match.group(1)

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model file missing: {MODEL_FILE}")

    print(f"Captured MLflow run ID: {run_id}", flush=True)

    # PythonOperator returns this value to XCom for registration.
    return run_id


def evaluate_model():
    run_command(
        [
            sys.executable,
            "src/evaluate.py",
            str(MODEL_FILE),
            str(TEST_FILE),
            str(F1_OUTPUT),
            F1_THRESHOLD,
        ]
    )


def register_model(ti):
    run_id = ti.xcom_pull(task_ids="train_model")

    if not run_id:
        raise RuntimeError("No MLflow run ID was received from train_model.")

    run_command(
        [
            sys.executable,
            "src/register_model.py",
            run_id,
        ]
    )


with DAG(
    dag_id="5g_ml_pipeline",
    description="5G congestion prediction: validate, engineer, preprocess, train, evaluate and register",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["5g-mlops", "mlflow", "model-training"],
) as dag:

    validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
        executor_config=POD_OVERRIDE,
    )

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
        executor_config=POD_OVERRIDE,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_data,
        executor_config=POD_OVERRIDE,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        executor_config=POD_OVERRIDE,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        executor_config=POD_OVERRIDE,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        executor_config=POD_OVERRIDE,
    )

    (
        validation_task
        >> feature_task
        >> preprocess_task
        >> train_task
        >> evaluate_task
        >> register_task
    )
