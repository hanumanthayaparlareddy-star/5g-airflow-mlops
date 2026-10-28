import os
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", "/root/5g-mlops")
)

FEAST_REPO = Path(
    os.getenv(
        "FEAST_REPO",
        PROJECT_ROOT
        / "feature_store/5g_features/feature_repo"
    )
)

FEAST_SOURCE = Path(
    os.getenv(
        "FEAST_SOURCE",
        PROJECT_ROOT / "data/feast/5g_metrics_feast.parquet"
    )
)


def main():
    if not FEAST_SOURCE.exists():
        raise FileNotFoundError(
            f"Feast source not found: {FEAST_SOURCE}"
        )

    df = pd.read_parquet(FEAST_SOURCE)

    if df.empty:
        raise ValueError("Feast source is empty")

    timestamps = pd.to_datetime(
        df["event_timestamp"],
        utc=True,
        errors="raise",
    )

    start_time = timestamps.min().isoformat()
    end_time = timestamps.max().isoformat()

    subprocess.run(
        [
            "feast",
            "-c",
            str(FEAST_REPO),
            "materialize",
            start_time,
            end_time,
        ],
        check=True,
    )

    print("Feast materialization completed")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")


if __name__ == "__main__":
    main()
