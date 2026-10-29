"""Entrenamiento y scoring del modelo de probabilidad de default (PD)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ARTIFACT = Path(__file__).parents[1] / "artifacts" / "pd_model.joblib"
NUMERIC = ["years_operating", "log_revenue", "avg_days_late", "pct_invoices_late", "credit_utilization",
           "bureau_score", "bureau_delinquencies", "dso_trend_days", "requested_pct_revenue"]
SECTOR_DUMMIES = ["sector_Pesca", "sector_Mineria", "sector_Comercio"]
FEATURES = NUMERIC + SECTOR_DUMMIES
# Restricciones monótonas: el riesgo NUNCA baja si empeora el comportamiento de pago (regulatoriamente explicable)
MONOTONE = {"years_operating": -1, "log_revenue": -1, "avg_days_late": 1, "pct_invoices_late": 1,
            "credit_utilization": 1, "bureau_score": -1, "bureau_delinquencies": 1, "dso_trend_days": 1,
            "requested_pct_revenue": 1}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["years_operating"] = df.years_operating
    X["log_revenue"] = np.log(df.annual_revenue)
    for c in ["avg_days_late", "pct_invoices_late", "credit_utilization", "bureau_score",
              "bureau_delinquencies", "dso_trend_days"]:
        X[c] = df[c]
    X["requested_pct_revenue"] = df.requested_limit / df.annual_revenue * 100
    for s in ["Pesca", "Mineria", "Comercio"]:
        X[f"sector_{s}"] = (df.sector == s).astype(int)
    return X[FEATURES].astype(float)


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def evaluate(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"ROC_AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4), "KS": round(ks_stat(y, p), 4),
            "Brier": round(brier_score_loss(y, p), 4), "PD_media_predicha": round(float(np.mean(p)), 4),
            "Default_rate_real": round(float(np.mean(y)), 4)}


def train(df: pd.DataFrame, seed: int = 0) -> dict:
    X, y = prepare(df), df.default_12m
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    Xtr_fit, Xval, ytr_fit, yval = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr, ytr)
    mono = tuple(MONOTONE.get(f, 0) for f in FEATURES)
    xgb = XGBClassifier(n_estimators=1500, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=10, monotone_constraints=mono, eval_metric="logloss",
                        early_stopping_rounds=50, random_state=seed, n_jobs=2)
    xgb.fit(Xtr_fit, ytr_fit, eval_set=[(Xval, yval)], verbose=False)

    results = {"Regresión logística": evaluate(yte, logreg.predict_proba(Xte)[:, 1]),
               "XGBoost monótono": evaluate(yte, xgb.predict_proba(Xte)[:, 1])}
    return {"logreg": logreg, "xgb": xgb, "X_test": Xte, "y_test": yte, "X_train": Xtr, "metrics": results}


def save(model, path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, path)


def load(path: Path = ARTIFACT):
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: python train.py")
    return joblib.load(path)["model"]


def predict_pd(model, applicants: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(applicants))[:, 1]
