"""API de scoring.  Ejecutar:  uvicorn api.main:app --reload   ->  http://127.0.0.1:8000/docs"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from models.explainability import Explainer
from models.scoring import load, predict_pd, prepare
from utils.scorecard import band_for, pd_to_score, recommended_limit

app = FastAPI(title="Credit Risk Scoring API (demo con datos sintéticos)", version="1.0.0")


class Application(BaseModel):
    sector: Literal["Pesca", "Industria", "Mineria", "Comercio"]
    years_operating: float = Field(ge=0, le=100, examples=[8])
    annual_revenue: float = Field(gt=0, examples=[1_500_000])
    avg_days_late: float = Field(ge=0, le=365, examples=[6])
    pct_invoices_late: float = Field(ge=0, le=1, examples=[0.12])
    credit_utilization: float = Field(ge=0, le=2, examples=[0.45])
    bureau_score: float = Field(ge=300, le=900, examples=[680])
    bureau_delinquencies: int = Field(ge=0, le=50, examples=[0])
    dso_trend_days: float = Field(ge=-120, le=120, examples=[2])
    requested_limit: float = Field(gt=0, examples=[90_000])


class Decision(BaseModel):
    pd_12m: float
    score: int
    banda: str
    decision: str
    limite_recomendado: float
    razones: list[dict]


@lru_cache
def _model():
    m = load()
    return m, Explainer(m)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=Decision)
def score(app_in: Application) -> Decision:
    model, explainer = _model()
    df = pd.DataFrame([app_in.model_dump()])
    pd_ = float(predict_pd(model, df)[0])
    s = pd_to_score(pd_)
    band, decision = band_for(s)
    return Decision(pd_12m=round(pd_, 4), score=s, banda=band, decision=decision,
                    limite_recomendado=recommended_limit(app_in.requested_limit, band),
                    razones=explainer.reasons(prepare(df)))
