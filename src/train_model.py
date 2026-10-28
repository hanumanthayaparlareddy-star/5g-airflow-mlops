import os
import json
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "5G-Congestion-Prediction",
)

FEATURES = [
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "user_prb_interaction",
    "latency_packet_loss",
    "high_prb_flag",
    "high_latency_flag",
]

FEATURE_SERVICE = "five_g_cell_features"
TARGET = "congestion"


def get_historical_features(df, store):
    entity_df = df[
        ["cell_id", "event_timestamp"]
    ].copy()

    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["event_timestamp"],
        utc=True,
    )

    feature_refs = [
        f"{FEATURE_SERVICE}:{feature}"
        for feature in FEATURES
    ]

    training_features = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    # Preserve the label using its original row timestamp and cell.
    training_features["event_timestamp"] = pd.to_datetime(
        training_features["event_timestamp"],
        utc=True,
    )

    labels = df[
        ["cell_id", "event_timestamp", TARGET]
    ].copy()

    labels["event_timestamp"] = pd.to_datetime(
        labels["event_timestamp"],
        utc=True,
    )

    training_features = training_features.merge(
        labels,
        on=["cell_id", "event_timestamp"],
        how="inner",
        validate="one_to_one",
    )

    if training_features.empty:
        raise ValueError(
            "Feast historical retrieval returned no matched labels"
        )

    return training_features


def main():
    if not TRAINING_DATA.exists():
        raise FileNotFoundError(TRAINING_DATA)

    df = pd.read_parquet(TRAINING_DATA)

    store = FeatureStore(
        repo_path=str(FEAST_REPO)
    )

    training_df = get_historical_features(
        df,
        store,
    )

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in training_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training columns: {missing}"
        )

    X = training_df[FEATURES]
    y = training_df[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both congestion classes"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
        }

        mlflow.log_params({
            "algorithm": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": 42,
            "feature_count": len(FEATURES),
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tag(
            "project",
            "5g-congestion-prediction",
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        mlflow.log_dict(
            {
                "features": FEATURES,
                "target": TARGET,
            },
            "feature_contract.json",
        )

        print(json.dumps({
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_uri": f"runs:/{run.info.run_id}/model",
        }, indent=2))


if __name__ == "__main__":
    main()
