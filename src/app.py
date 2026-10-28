import os
from functools import lru_cache

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

app = FastAPI(
    title="5G Congestion Prediction API",
    version="1.0.0",
)


class Telemetry(BaseModel):
    cell_id: str
    users: int = Field(ge=0)
    prb_utilization: float = Field(ge=0, le=100)
    latency: float = Field(ge=0)
    packet_loss: float = Field(ge=0, le=100)


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@champion"
    )


def make_features(payload):
    users = payload.users
    prb = payload.prb_utilization
    latency = payload.latency
    loss = payload.packet_loss

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Telemetry):
    try:
        model = load_model()

        features = pd.DataFrame(
            [make_features(payload)],
            columns=FEATURES,
        )

        prediction = int(model.predict(features)[0])

        return {
            "cell_id": payload.cell_id,
            "congestion_prediction": prediction,
            "label": (
                "CONGESTED"
                if prediction == 1
                else "NOT_CONGESTED"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model prediction is temporarily unavailable",
        ) from exc
