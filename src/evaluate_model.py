import argparse
import os
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from feast import FeatureStore
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", "/root/5g-mlops")
)
TRAINING_DATA = Path(
    os.getenv(
        "TRAINING_DATA",
        PROJECT_ROOT / "data/processed/5g_training_data.parquet"
    )
)
FEAST_REPO = Path(
    os.getenv(
        "FEAST_REPO",
        PROJECT_ROOT
        / "feature_store/5g_features/feature_repo"
    )
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MIN_F1 = float(os.getenv("MIN_F1", "0.90"))
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.90"))

FEATURES = [
    "users", "prb_utilization", "latency", "packet_loss",
    "user_prb_interaction", "latency_packet_loss",
    "high_prb_flag", "high_latency_flag",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(TRAINING_DATA)

    # Reproduce the same split as train_model.py for this lab.
    _, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["congestion"].astype(int),
    )

    # For production, retrieve the test features from Feast using
    # event-time historical retrieval, just as in training.
    store = FeatureStore(repo_path=str(FEAST_REPO))
    entity_df = test_df[
        ["cell_id", "event_timestamp"]
    ].copy()

    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["event_timestamp"],
        utc=True,
    )

    historical = store.get_historical_features(
        entity_df=entity_df,
        features=[
            f"five_g_cell_features:{feature}"
            for feature in FEATURES
        ],
    ).to_df()

    labels = test_df[
        ["cell_id", "event_timestamp", "congestion"]
    ].copy()
    labels["event_timestamp"] = pd.to_datetime(
        labels["event_timestamp"],
        utc=True,
    )
    historical["event_timestamp"] = pd.to_datetime(
        historical["event_timestamp"],
        utc=True,
    )

    test_features = historical.merge(
        labels,
        on=["cell_id", "event_timestamp"],
        how="inner",
        validate="one_to_one",
    )

    if test_features.empty:
        raise ValueError("No matching test features returned by Feast")

    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.sklearn.load_model(
        f"runs:/{args.run_id}/model"
    )

    predictions = model.predict(test_features[FEATURES])
    y_true = test_features["congestion"].astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }

    passed = (
        metrics["f1"] >= MIN_F1
        and metrics["accuracy"] >= MIN_ACCURACY
    )

    client = mlflow.tracking.MlflowClient()
    client.log_metric(args.run_id, "gate_f1", metrics["f1"])
    client.log_metric(
        args.run_id,
        "gate_accuracy",
        metrics["accuracy"],
    )
    client.set_tag(
        args.run_id,
        "quality_gate",
        "PASSED" if passed else "FAILED",
    )

    print(json.dumps({
        "run_id": args.run_id,
        "metrics": metrics,
        "minimum_f1": MIN_F1,
        "minimum_accuracy": MIN_ACCURACY,
        "passed": passed,
    }, indent=2))

    if not passed:
        raise SystemExit(
            "Quality gate FAILED. Do not register/promote this model."
        )


if __name__ == "__main__":
    main()
