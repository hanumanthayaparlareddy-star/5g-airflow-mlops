import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "5G-Congestion-Model",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    run = client.get_run(args.run_id)
    gate_status = run.data.tags.get("quality_gate")

    if gate_status != "PASSED":
        raise SystemExit(
            f"Registration blocked: quality_gate={gate_status!r}"
        )

    model_uri = f"runs:/{args.run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        registered.version,
    )

    print(
        f"Registered model: {MODEL_NAME}, "
        f"version: {registered.version}, "
        "alias: champion"
    )


if __name__ == "__main__":
    main()
