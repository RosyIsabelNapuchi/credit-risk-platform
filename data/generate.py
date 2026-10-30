"""Solicitudes de crédito B2B SINTÉTICAS.

Se simula un PD "verdadero" con una función logística de variables financieras,
de comportamiento de pago y de un puntaje de buró simulado. No se usa ningún
dato real de clientes ni de burós de crédito.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ("Pesca", "Industria", "Mineria", "Comercio")


def make_applications(n: int = 9000, seed: int = 11, quality_shift: float = 0.0,
                      macro_shock: float = 0.0) -> pd.DataFrame:
    """Genera solicitudes. Para simular monitoreo:
    - quality_shift < 0: llega una cartera de peor calidad (cambian las variables -> el PSI lo detecta).
    - macro_shock > 0: sube el riesgo por un factor NO observado (p. ej. recesión): las variables no
      cambian, pero el default real sí. El PSI no lo ve; solo el back-test de default lo detecta.
    """
    rng = np.random.default_rng(seed)
    sector = rng.choice(SECTORS, n, p=[0.25, 0.35, 0.2, 0.2])
    years = np.clip(rng.gamma(2.2, 4.5, n), 0.3, 45).round(1)
    revenue = np.exp(rng.normal(14.2, 1.1, n)).round(-3)                    # ventas anuales USD
    quality = rng.normal(0, 1, n) + quality_shift               # calidad crediticia latente
    avg_days_late = np.clip(rng.gamma(1.5, 4, n) * np.exp(-0.6 * quality), 0, 120).round(1)
    pct_late = np.clip(0.15 - 0.08 * quality + rng.normal(0, 0.08, n), 0, 1).round(3)
    utilization = np.clip(0.45 - 0.15 * quality + rng.normal(0, 0.18, n), 0, 1.3).round(3)
    bureau_score = np.clip(650 + 70 * quality + rng.normal(0, 45, n), 300, 900).round()
    delinquencies = rng.poisson(np.exp(-0.2 - 0.7 * quality))
    dso_trend = (rng.normal(0, 6, n) - 3 * quality).round(1)
    requested = (revenue * rng.uniform(0.02, 0.12, n)).round(-2)

    sector_eff = pd.Series(sector).map({"Pesca": 0.35, "Industria": 0.0, "Mineria": -0.1, "Comercio": 0.2}).to_numpy()
    logit = (-4.7 + 0.035 * avg_days_late + 1.6 * pct_late + 1.2 * utilization
             - 0.006 * (bureau_score - 650) + 0.25 * delinquencies + 0.03 * dso_trend
             - 0.03 * np.minimum(years, 20) - 0.15 * (np.log(revenue) - 14.2)
             + 0.25 * np.log(requested / revenue * 100 + 1) + sector_eff + macro_shock)
    pd_true = 1 / (1 + np.exp(-logit))
    default = (rng.random(n) < pd_true).astype(int)
    return pd.DataFrame({
        "sector": sector, "years_operating": years, "annual_revenue": revenue,
        "avg_days_late": avg_days_late, "pct_invoices_late": pct_late, "credit_utilization": utilization,
        "bureau_score": bureau_score, "bureau_delinquencies": delinquencies, "dso_trend_days": dso_trend,
        "requested_limit": requested, "default_12m": default,
    })
