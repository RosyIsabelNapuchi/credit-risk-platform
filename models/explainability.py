"""Razones de la decisión por solicitante (reason codes) con SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

LABELS = {
    "years_operating": "años de operación",
    "log_revenue": "tamaño de ventas anuales",
    "avg_days_late": "días promedio de atraso",
    "pct_invoices_late": "% de facturas pagadas tarde",
    "credit_utilization": "uso de la línea de crédito",
    "bureau_score": "puntaje de buró",
    "bureau_delinquencies": "morosidades reportadas en buró",
    "dso_trend_days": "tendencia del DSO",
    "requested_pct_revenue": "monto solicitado vs. ventas",
    "sector_Pesca": "sector pesca",
    "sector_Mineria": "sector minería",
    "sector_Comercio": "sector comercio",
}


class Explainer:
    def __init__(self, model):
        self._exp = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._exp.shap_values(X))

    def reasons(self, X_row: pd.DataFrame, top: int = 3) -> list[dict]:
        """Top factores que SUBEN y BAJAN el riesgo para una sola fila."""
        sv = self.shap_values(X_row)[0]
        order = np.argsort(-np.abs(sv))[:top]
        return [{"factor": LABELS.get(X_row.columns[i], X_row.columns[i]),
                 "valor": round(float(X_row.iloc[0, i]), 3),
                 "efecto": "aumenta el riesgo" if sv[i] > 0 else "reduce el riesgo",
                 "impacto_log_odds": round(float(sv[i]), 3)} for i in order]
