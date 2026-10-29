import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from models.scoring import load, predict_pd

client = TestClient(app)

BASE = {"sector": "Industria", "years_operating": 8, "annual_revenue": 1_500_000, "avg_days_late": 6,
        "pct_invoices_late": 0.12, "credit_utilization": 0.45, "bureau_score": 680, "bureau_delinquencies": 0,
        "dso_trend_days": 2, "requested_limit": 90_000}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_endpoint_contract():
    r = client.post("/score", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["pd_12m"] < 1
    assert body["banda"] in "ABCDE"
    assert len(body["razones"]) == 3
    assert body["limite_recomendado"] <= BASE["requested_limit"]


def test_invalid_input_rejected():
    bad = {**BASE, "pct_invoices_late": 1.7}
    assert client.post("/score", json=bad).status_code == 422


def test_monotonic_in_days_late():
    """Más días de atraso nunca debe bajar la PD (restricción monótona)."""
    model = load()
    rows = pd.DataFrame([{**BASE, "avg_days_late": d} for d in range(0, 121, 10)])
    p = predict_pd(model, rows)
    assert all(b >= a - 1e-9 for a, b in zip(p, p[1:]))


def test_better_bureau_score_lowers_pd():
    model = load()
    p = predict_pd(model, pd.DataFrame([{**BASE, "bureau_score": 500}, {**BASE, "bureau_score": 820}]))
    assert p[1] <= p[0]
