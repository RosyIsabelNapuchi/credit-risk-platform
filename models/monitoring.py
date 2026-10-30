"""Monitoreo de modelos en producción: estabilidad (PSI) y back-test de calibración."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE, PSI_ALERT = 0.10, 0.25          # umbrales de uso común en riesgo de crédito


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index con cortes por cuantiles de la muestra de desarrollo.

    PSI = sum((a% - e%) * ln(a% / e%)).  < 0.10 estable · 0.10-0.25 vigilar · > 0.25 cambio significativo.
    """
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # variable casi constante (p. ej. dummies): usar valores únicos
        cats = np.union1d(np.unique(e), np.unique(a))
        e_pct = np.array([(e == c).mean() for c in cats])
        a_pct = np.array([(a == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e_pct = np.histogram(e, edges)[0] / len(e)
        a_pct = np.histogram(a, edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, 1e-4, None), np.clip(a_pct, 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "estable"
    return "vigilar" if value < PSI_ALERT else "alerta"


def calibration_status(pd_pred: float, default_obs: float, n: int) -> str:
    """Compara default observado vs. PD predicha con un intervalo binomial aproximado al 95 %."""
    se = np.sqrt(pd_pred * (1 - pd_pred) / n)
    if abs(default_obs - pd_pred) <= 1.96 * se:
        return "calibrado"
    return "subestima riesgo" if default_obs > pd_pred else "sobreestima riesgo"


def feature_psi(dev: pd.DataFrame, new: pd.DataFrame) -> pd.Series:
    return pd.Series({c: psi(dev[c], new[c]) for c in dev.columns}).sort_values(ascending=False)
