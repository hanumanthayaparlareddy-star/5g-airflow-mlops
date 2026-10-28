import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", "/root/5g-mlops")
)

RAW_DATA = Path(
    os.getenv(
        "RAW_DATA",
        PROJECT_ROOT / "data/raw/5g_metrics.csv"
    )
)

OUTPUT_DATA = Path(
    os.getenv(
        "STAGING_DATA",
        PROJECT_ROOT / "data/processed/5g_metrics_staging.csv"
    )
)

REQUIRED_COLUMNS = [
    "users",
    "prb_utilization",
    "latency",
    "packet_loss",
    "congestion",
]


def main():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_DATA}"
        )

    df = pd.read_csv(RAW_DATA)

    # Create cell IDs for demo datasets that do not have them.
    if "cell_id" not in df.columns:
        df["cell_id"] = [
            f"cell_{i:03d}"
            for i in range(1, len(df) + 1)
        ]

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if "event_timestamp" not in df.columns:
        df["event_timestamp"] = pd.date_range(
            end=pd.Timestamp.now(tz="UTC"),
            periods=len(df),
            freq="min",
        )
    else:
        df["event_timestamp"] = pd.to_datetime(
            df["event_timestamp"],
            utc=True,
            errors="raise",
        )

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_DATA, index=False)

    print(f"Rows ingested: {len(df)}")
    print(f"Output: {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
