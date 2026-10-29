"""Entrena, evalúa y guarda el modelo de PD.   Uso: python train.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from data.generate import make_applications
from models.explainability import LABELS, Explainer
from models.scoring import save, train
from utils.scorecard import band_for, pd_to_score

ROOT = Path(__file__).parent
FIG = ROOT / "reports" / "figures"
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e6e5e0"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "legend.frameon": False, "lines.linewidth": 2, "font.size": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = make_applications()
    df.to_csv(ROOT / "data" / "applications_synthetic.csv", index=False)
    print(f"Solicitudes sintéticas: {len(df):,} | tasa de default: {df.default_12m.mean():.1%}")

    out = train(df)
    xgb, logreg, Xte, yte = out["xgb"], out["logreg"], out["X_test"], out["y_test"]
    metrics = pd.DataFrame(out["metrics"]).T
    print("\n=== Métricas en test (25 % hold-out) ===\n", metrics)
    save(xgb)

    p_x, p_l = xgb.predict_proba(Xte)[:, 1], logreg.predict_proba(Xte)[:, 1]
    scores = np.array([pd_to_score(p) for p in p_x])
    bands = [band_for(s)[0] for s in scores]
    bt = pd.DataFrame({"banda": bands, "default": yte.to_numpy(), "pd": p_x}).groupby("banda").agg(
        solicitudes=("default", "size"), default_real=("default", "mean"), pd_media=("pd", "mean"))
    bt["pct_solicitudes"] = bt.solicitudes / bt.solicitudes.sum()
    bt = bt.round(4)
    bt.to_csv(ROOT / "reports" / "bandas_test.csv")
    print("\nTasa de default por banda (test):\n", bt)

    expl = Explainer(xgb)
    sv = expl.shap_values(Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=[LABELS[c] for c in Xte.columns]).sort_values()

    # ejemplo de explicación para el caso más riesgoso del test
    i = int(np.argmax(p_x))
    ejemplo = {"pd": round(float(p_x[i]), 3), "score": int(scores[i]), "banda": bands[i],
               "razones": expl.reasons(Xte.iloc[[i]])}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(
        {"metricas_test": out["metrics"], "iteraciones_xgb": int(xgb.best_iteration),
         "ejemplo_explicacion": ejemplo}, indent=2, ensure_ascii=False))
    print("\nEjemplo de reason codes:", json.dumps(ejemplo, indent=1, ensure_ascii=False))

    # --- Figuras
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, name, c in ((p_l, "Regresión logística", C1), (p_x, "XGBoost monótono", C2)):
        fpr, tpr, _ = roc_curve(yte, p)
        a1.plot(fpr, tpr, color=c, label=f"{name} (Gini {out['metrics'][name]['Gini']:.2f})")
    a1.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    a1.set(title="Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    a1.legend(loc="lower right")
    dec = pd.DataFrame({"p": p_x, "y": yte.to_numpy()})
    dec["decil"] = pd.qcut(dec.p, 10, labels=False)
    cal = dec.groupby("decil").agg(pred=("p", "mean"), real=("y", "mean"))
    lim = max(cal.max()) * 1.1
    a2.plot([0, lim], [0, lim], color=GRID, linestyle="--", linewidth=1, label="Calibración perfecta")
    a2.plot(cal.pred, cal.real, color=C2, marker="o", markersize=5, label="XGBoost monótono")
    a2.set(title="Calibración por decil de PD", xlabel="PD media predicha", ylabel="Default observado")
    a2.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "01_roc_calibracion.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bt.index, bt.default_real * 100, color=C1, width=0.6)
    for x, (v, n) in enumerate(zip(bt.default_real * 100, bt.pct_solicitudes * 100)):
        ax.text(x, v + 0.6, f"{v:.1f}%\n({n:.0f}% de solicitudes)", ha="center", fontsize=8.5, color=INK2)
    ax.set(title="Default observado por banda de riesgo (test)", xlabel="Banda", ylabel="Default 12m (%)")
    ax.set_ylim(0, bt.default_real.max() * 100 * 1.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "02_default_por_banda.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(imp.index, imp.values, color=C1, height=0.62)
    ax.set(title="Importancia global (|SHAP| medio)", xlabel="Impacto medio en log-odds")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "03_shap_importancia.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
