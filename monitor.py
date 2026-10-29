"""Simula tres meses de producción y aplica el tablero de monitoreo del modelo.

Uso:  python monitor.py   (requiere haber corrido antes: python train.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.generate import make_applications
from models.monitoring import PSI_ALERT, PSI_STABLE, calibration_status, feature_psi, psi, psi_status
from models.scoring import load, prepare

ROOT = Path(__file__).parent
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

SCENARIOS = {
    "Mes estable": dict(seed=21, quality_shift=0.0, macro_shock=0.0),
    "Cartera más riesgosa": dict(seed=22, quality_shift=-0.5, macro_shock=0.0),
    "Shock macro no observado": dict(seed=23, quality_shift=0.0, macro_shock=0.7),
}


def main() -> None:
    model = load()
    dev = make_applications()                       # muestra de desarrollo (misma semilla que train.py)
    X_dev = prepare(dev)
    p_dev = model.predict_proba(X_dev)[:, 1]

    rows, details = [], {}
    for name, cfg in SCENARIOS.items():
        new = make_applications(n=1500, **cfg)
        X_new = prepare(new)
        p_new = model.predict_proba(X_new)[:, 1]
        fpsi = feature_psi(X_dev, X_new)
        score_psi = psi(p_dev, p_new)
        row = {"escenario": name, "PSI_score": round(score_psi, 3), "estado_PSI": psi_status(score_psi),
               "variable_mas_inestable": fpsi.index[0], "PSI_variable": round(float(fpsi.iloc[0]), 3),
               "PD_predicha": round(float(p_new.mean()), 4), "default_observado": round(float(new.default_12m.mean()), 4),
               "estado_calibracion": calibration_status(float(p_new.mean()), float(new.default_12m.mean()), len(new)),
               "AUC": round(roc_auc_score(new.default_12m, p_new), 3)}
        rows.append(row)
        details[name] = fpsi.round(3).to_dict()
    tab = pd.DataFrame(rows).set_index("escenario")
    tab.to_csv(ROOT / "reports" / "monitoreo.csv")
    (ROOT / "reports" / "monitoreo_psi_variables.json").write_text(json.dumps(details, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(tab.T)

    # Figura: PSI del score y brecha de calibración por escenario
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "font.size": 10})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    names = list(tab.index)
    a1.barh(names[::-1], tab.PSI_score[::-1], color=C1, height=0.55)
    a1.axvline(PSI_STABLE, color=INK2, linestyle="--", linewidth=1)
    a1.axvline(PSI_ALERT, color=INK2, linestyle=":", linewidth=1)
    a1.set_ylim(-0.5, len(names) - 0.1)
    a1.text(PSI_STABLE, len(names) - 0.35, " vigilar", color=INK2, fontsize=8)
    a1.text(PSI_ALERT, len(names) - 0.35, " alerta", color=INK2, fontsize=8)
    a1.set_xlim(0, max(PSI_ALERT * 1.25, tab.PSI_score.max() * 1.2))
    for y, v in enumerate(tab.PSI_score[::-1]):
        a1.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    a1.set(title="Estabilidad: PSI del score", xlabel="PSI")
    a1.grid(axis="y", visible=False)
    y = np.arange(len(names))[::-1]
    a2.scatter(tab.PD_predicha * 100, y, color=C1, s=60, label="PD predicha", zorder=3)
    a2.scatter(tab.default_observado * 100, y, color=C2, s=60, label="Default observado", zorder=3)
    for yi, (p, o) in zip(y, zip(tab.PD_predicha * 100, tab.default_observado * 100)):
        a2.plot([p, o], [yi, yi], color=GRID, linewidth=2, zorder=2)
    a2.set_yticks(y, names)
    a2.set(title="Calibración: predicho vs. observado", xlabel="% de default")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "04_monitoreo.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
