import os
import subprocess
from pathlib import Path

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


def main():
    if not FEAST_REPO.exists():
        raise FileNotFoundError(
            f"Feast repository not found: {FEAST_REPO}"
        )

    subprocess.run(
        [
            "feast",
            "-c",
            str(FEAST_REPO),
            "apply",
        ],
        check=True,
    )

    print("Feast definitions applied successfully")


if __name__ == "__main__":
    main()
