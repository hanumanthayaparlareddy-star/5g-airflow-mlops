from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def verify_task_pod():
    print("KubernetesExecutor task pod ran successfully.")
    return "success"


with DAG(
    dag_id="airflow_kubernetes_smoke_test",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["5g-mlops", "kubernetes-executor"],
) as dag:
    smoke_test = PythonOperator(
        task_id="verify_task_pod",
        python_callable=verify_task_pod,
    )
