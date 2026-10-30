"""Conversión PD -> puntaje y política de decisión (convención 'points to double the odds')."""
from __future__ import annotations

import math

BASE_SCORE, BASE_ODDS, PDO = 600, 30, 20       # 600 pts = odds buenos:malos 30:1; +20 pts duplica las odds

BANDS = [  # (score mínimo, banda, decisión)
    (625, "A", "Aprobar"),
    (600, "B", "Aprobar"),
    (575, "C", "Aprobar con límite reducido"),
    (550, "D", "Revisión de analista"),
    (-math.inf, "E", "Rechazar / pago adelantado"),
]
LIMIT_FACTOR = {"A": 1.0, "B": 1.0, "C": 0.6, "D": 0.3, "E": 0.0}


def pd_to_score(pd_: float) -> int:
    pd_ = min(max(pd_, 1e-6), 1 - 1e-6)
    odds = (1 - pd_) / pd_
    factor = PDO / math.log(2)
    offset = BASE_SCORE - factor * math.log(BASE_ODDS)
    return int(round(offset + factor * math.log(odds)))


def band_for(score: int) -> tuple[str, str]:
    for min_score, band, decision in BANDS:
        if score >= min_score:
            return band, decision
    raise AssertionError("inalcanzable")


def recommended_limit(requested: float, band: str) -> float:
    return round(requested * LIMIT_FACTOR[band], -2)
