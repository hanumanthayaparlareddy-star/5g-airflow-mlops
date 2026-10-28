from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/5g-mlops"
PYTHON = "python"

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


def get_run_id_from_training_output(ti):
    """
    In production, training should return its run_id through
    a structured artifact or XCom. This function is a placeholder
    for that explicit handoff.
    """
    run_id = ti.xcom_pull(
        task_ids="train_model",
        key="mlflow_run_id",
    )

    if not run_id:
        raise ValueError(
            "Training task did not publish its MLflow run ID"
        )

    return run_id


def evaluate_model(ti):
    import subprocess

    run_id = get_run_id_from_training_output(ti)

    subprocess.run(
        [
            PYTHON,
            f"{PROJECT_ROOT}/src/evaluate_model.py",
            "--run-id",
            run_id,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def register_model(ti):
    import subprocess

    run_id = get_run_id_from_training_output(ti)

    subprocess.run(
        [
            PYTHON,
            f"{PROJECT_ROOT}/src/register_model.py",
            "--run-id",
            run_id,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


with DAG(
    dag_id="5g_ml_pipeline",
    description="5G congestion prediction MLOps pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["5g", "mlops", "feast", "mlflow"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} src/ingest_data.py"
        ),
    )

    validate = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} src/validate_data.py"
        ),
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} src/feature_engineering.py"
        ),
    )

    feast_apply = BashOperator(
        task_id="feast_apply",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} scripts/feast_apply.py"
        ),
    )

    feast_materialize = BashOperator(
        task_id="feast_materialize",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} scripts/feast_materialize.py"
        ),
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} src/train_model.py"
        ),
        do_xcom_push=True,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    (
        ingest
        >> validate
        >> feature_engineering
        >> feast_apply
        >> feast_materialize
        >> train
        >> evaluate
        >> register
    )
