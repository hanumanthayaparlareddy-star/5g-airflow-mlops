import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", "/root/5g-mlops")
)

INPUT_DATA = Path(
    os.getenv(
        "STAGING_DATA",
        PROJECT_ROOT / "data/processed/5g_metrics_staging.csv"
    )
)

FEAST_SOURCE = Path(
    os.getenv(
        "FEAST_SOURCE",
        PROJECT_ROOT / "data/feast/5g_metrics_feast.parquet"
    )
)

TRAINING_DATA = Path(
    os.getenv(
        "TRAINING_DATA",
        PROJECT_ROOT / "data/processed/5g_training_data.parquet"
    )
)

FEATURE_COLUMNS = [
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "user_prb_interaction",
    "latency_packet_loss",
    "high_prb_flag",
    "high_latency_flag",
]


def build_features(df):
    df = df.copy()

    df["users"] = pd.to_numeric(
        df["users"], errors="raise"
    )
    df["prb_utilization"] = pd.to_numeric(
        df["prb_utilization"], errors="raise"
    )
    df["latency"] = pd.to_numeric(
        df["latency"], errors="raise"
    )
    df["packet_loss"] = pd.to_numeric(
        df["packet_loss"], errors="raise"
    )

    # Example engineered features.
    df["user_prb_interaction"] = (
        df["users"] * df["prb_utilization"] / 100.0
    )

    df["latency_packet_loss"] = (
        df["latency"] * df["packet_loss"]
    )

    df["high_prb_flag"] = (
        df["prb_utilization"] >= 80
    ).astype("int64")

    df["high_latency_flag"] = (
        df["latency"] >= 100
    ).astype("int64")

    return df


def main():
    if not INPUT_DATA.exists():
        raise FileNotFoundError(INPUT_DATA)

    df = pd.read_csv(INPUT_DATA)

    required = [
        "event_timestamp",
        "cell_id",
        "users",
        "prb_utilization",
        "latency",
        "packet_loss",
        "congestion",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"],
        utc=True,
        errors="raise",
    )

    featured = build_features(df)

    FEAST_SOURCE.parent.mkdir(
        parents=True, exist_ok=True
    )
    TRAINING_DATA.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Includes label for training and offline evaluation.
    featured.to_parquet(
        TRAINING_DATA,
        index=False,
    )

    # Feast source contains features and entity/timestamp,
    # but deliberately excludes the target label.
    feast_columns = [
        "event_timestamp",
        "cell_id",
        *FEATURE_COLUMNS,
    ]

    featured[feast_columns].to_parquet(
        FEAST_SOURCE,
        index=False,
    )

    print(f"Training dataset: {TRAINING_DATA}")
    print(f"Feast source: {FEAST_SOURCE}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
