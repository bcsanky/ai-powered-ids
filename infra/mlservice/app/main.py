from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="ML Service Skeleton")

class ScoreRequest(BaseModel):
    event_id: str
    payload: dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(req: ScoreRequest):
    # Később itt lesz a valódi modell-inferencia
    return {
        "event_id": req.event_id,
        "anomaly_score": 0.123,
        "label": "unknown",
        "model_version": "skeleton-v1"
    }
