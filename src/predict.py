import os

import mlflow
import mlflow.pyfunc
import pandas as pd

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "5G-Congestion-Model",
)

FEATURES = [
    "users", "prb_utilization", "latency", "packet_loss",
    "user_prb_interaction", "latency_packet_loss",
    "high_prb_flag", "high_latency_flag",
]


def make_features(payload):
    users = float(payload["users"])
    prb = float(payload["prb_utilization"])
    latency = float(payload["latency"])
    loss = float(payload["packet_loss"])

    return {
        "users": users,
        "prb_utilization": prb,
        "latency": latency,
        "packet_loss": loss,
        "user_prb_interaction": users * prb / 100.0,
        "latency_packet_loss": latency * loss,
        "high_prb_flag": int(prb >= 80.0),
        "high_latency_flag": int(latency >= 100.0),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    telemetry = {
        "cell_id": "cell_001",
        "users": 125,
        "prb_utilization": 87.5,
        "latency": 115,
        "packet_loss": 2.1,
    }

    X = pd.DataFrame(
        [make_features(telemetry)],
        columns=FEATURES,
    )

    prediction = int(model.predict(X)[0])

    print({
        "cell_id": telemetry["cell_id"],
        "congestion_prediction": prediction,
        "label": (
            "CONGESTED"
            if prediction == 1
            else "NOT_CONGESTED"
        ),
    })


if __name__ == "__main__":
    main()
