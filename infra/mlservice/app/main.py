from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import get_config
from app.schemas import HealthResponse, ScoreRequest, ScoreResponse
from app.scoring import build_scorer
from ml.src.scoring_runtime import result_to_dict


app = FastAPI(title="AI IDS ML Scoring Service")
scorer = build_scorer()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = get_config()
    detail = None
    if not scorer.model_loaded:
        try:
            scorer.load()
        except Exception as exc:
            detail = str(exc)

    return HealthResponse(
        status="ok",
        model_loaded=scorer.model_loaded,
        model_version=cfg.model_version,
        model_root=str(cfg.model_root),
        preprocess_path=str(cfg.preprocess_path),
        detail=detail,
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    try:
        result = scorer.score_event(
            event_id=req.event_id,
            features=req.features,
            rule_flag=req.rule_flag,
            rule_level=req.rule_level,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "model_loaded": False,
                "error": str(exc),
            },
        ) from exc

    return ScoreResponse(**result_to_dict(result))
