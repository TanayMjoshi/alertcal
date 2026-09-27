"""HTTP face of alertcal (FastAPI).

Run locally:  uvicorn alertcal.api:app --reload      then open http://127.0.0.1:8000/docs

Endpoints: GET /health, POST /calibrate, POST /evaluate, POST /curve. Bodies are validated by
pydantic; a ValueError from core (alpha out of range, slice too small) becomes HTTP 422.
"""
from __future__ import annotations

from typing import Annotated, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .core import realised_fpr, recalibrate, sample_size_curve

app = FastAPI(
    title="alertcal",
    version=__version__,
    description=(
        "Recalibrate anomaly-alert thresholds after deployment shift. "
        "Send anomaly scores (higher = more anomalous); get thresholds and realised "
        "false-positive rates back. Nothing here sees raw data or a model."
    ),
)

Score = Annotated[float, Field(allow_inf_nan=False)]
Alpha = Annotated[float, Field(gt=0.0, lt=1.0, description="target false-positive rate, in (0, 1)")]


class ScoresIn(BaseModel):
    scores: List[Score] = Field(
        ..., min_length=1, description="Anomaly scores of benign events. Higher = more anomalous."
    )


class CalibrateRequest(ScoresIn):
    alpha: Alpha = 0.05


class CalibrateResponse(BaseModel):
    threshold: float
    alpha: float
    n: int


class EvaluateRequest(ScoresIn):
    threshold: float = Field(..., description="Cutoff to test. Use scores NOT used to set it.")


class EvaluateResponse(BaseModel):
    fpr: float
    threshold: float
    n: int


class CurveRequest(ScoresIn):
    alpha: Alpha = 0.05
    sizes: List[int] = Field(default=[50, 100, 200, 500, 1000, 2000], min_length=1)
    draws: int = Field(20, ge=1, le=1000)
    seed: int = 0

    @field_validator("sizes")
    @classmethod
    def _positive_sizes(cls, sizes: List[int]) -> List[int]:
        if any(n < 1 for n in sizes):
            raise ValueError("every size must be >= 1")
        return sizes


class CurvePoint(BaseModel):
    mean: float
    p5: float
    p95: float


class CurveResponse(BaseModel):
    alpha: float
    draws: int
    seed: int
    n: int
    curve: Dict[int, CurvePoint]


@app.exception_handler(ValueError)
async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/calibrate", response_model=CalibrateResponse, summary="Threshold at the (1 - alpha) quantile")
def calibrate(req: CalibrateRequest) -> CalibrateResponse:
    thr = recalibrate(req.scores, req.alpha)
    return CalibrateResponse(threshold=thr, alpha=req.alpha, n=len(req.scores))


@app.post("/evaluate", response_model=EvaluateResponse, summary="Realised false-positive rate of a threshold")
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    fpr = realised_fpr(req.scores, req.threshold)
    return EvaluateResponse(fpr=fpr, threshold=req.threshold, n=len(req.scores))


@app.post("/curve", response_model=CurveResponse, summary="Realised FPR versus calibration slice size")
def curve(req: CurveRequest) -> CurveResponse:
    result = sample_size_curve(req.scores, req.alpha, sizes=req.sizes, draws=req.draws, seed=req.seed)
    return CurveResponse(
        alpha=req.alpha,
        draws=req.draws,
        seed=req.seed,
        n=len(req.scores),
        curve={n: CurvePoint(mean=m, p5=lo, p95=hi) for n, (m, lo, hi) in result.items()},
    )
