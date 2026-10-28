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

REQUIRED_COLUMNS = [
    "event_timestamp",
    "cell_id",
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "congestion",
]


def main():
    if not INPUT_DATA.exists():
        raise FileNotFoundError(INPUT_DATA)

    df = pd.read_csv(INPUT_DATA)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Required columns contain null values"
        )

    ranges = {
        "users": (0, None),
        "prb_utilization": (0, 100),
        "latency": (0, None),
        "packet_loss": (0, 100),
    }

    for column, (minimum, maximum) in ranges.items():
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if (df[column] < minimum).any():
            raise ValueError(
                f"{column} contains negative values"
            )

        if maximum is not None:
            if (df[column] > maximum).any():
                raise ValueError(
                    f"{column} exceeds {maximum}"
                )

    labels = set(
        pd.to_numeric(
            df["congestion"],
            errors="raise",
        ).unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels 0/1, found {labels}"
        )

    if len(labels) != 2:
        raise ValueError(
            "Training data must contain both classes"
        )

    print("Data validation PASSED")
    print(f"Rows validated: {len(df)}")


if __name__ == "__main__":
    main()
