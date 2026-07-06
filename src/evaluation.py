"""
#src/evaluation.py

Evaluación de modelos, curvas ROC y PR para el diseño 2×2 del TFG.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcula AUC-ROC, PR-AUC y las curvas para un modelo entrenado.

    Devuelve
    --------
    dict con claves:
      auc, pr_auc, fpr, tpr, precision, recall, y_proba
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc    = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    fpr, tpr, _  = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    # Alertas de diagnóstico
    if auc > 0.95:
        print(f"  ALERTA: AUC={auc:.4f} > 0.95 — posible data leakage, verificar pipeline")
    if y_pred.sum() == 0:
        print("  ALERTA: el modelo predice todo 0 — revisar scale_pos_weight o sampling_strategy")
    if y_pred.sum() == len(y_pred):
        print("  ALERTA: el modelo predice todo 1 — revisar umbral o desbalanceo")

    return {
        "auc":       auc,
        "pr_auc":    pr_auc,
        "fpr":       fpr,
        "tpr":       tpr,
        "precision": prec,
        "recall":    rec,
        "y_proba":   y_proba,
    }


# Colores y estilos por experimento
_STYLES = {
    "XGBoost + Config A":  {"color": "#2196F3", "ls": "-"},
    "XGBoost + Config B":  {"color": "#F44336", "ls": "-"},
    "RUSBoost + Config A": {"color": "#4CAF50", "ls": "--"},
    "RUSBoost + Config B": {"color": "#FF9800", "ls": "--"},
}


def plot_roc_curves(
    results: dict[str, dict],
    save_path: str | Path,
) -> None:
    """
    Genera curvas ROC superpuestas para los 4 experimentos.

    Parámetros
    ----------
    results   : dict cuyas claves son etiquetas ('XGBoost + Config A', ...)
                y los valores son el dict devuelto por evaluate_model()
    save_path : ruta donde guardar la figura
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Azar (AUC=0.50)")

    for label, res in results.items():
        style = _STYLES.get(label, {})
        ax.plot(
            res["fpr"], res["tpr"],
            color=style.get("color", None),
            ls=style.get("ls", "-"),
            lw=2,
            label=f"{label}  (AUC={res['auc']:.4f})",
        )

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax.set_title("Curvas ROC — Diseño 2×2", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figura guardada: {save_path}")


def plot_pr_curves(
    results: dict[str, dict],
    save_path: str | Path,
    prevalence: float = 0.0074,
) -> None:
    """
    Genera curvas Precision-Recall superpuestas con línea de baseline.

    Parámetros
    ----------
    results    : mismo formato que plot_roc_curves
    save_path  : ruta de destino
    prevalence : prevalencia en el test (≈ 261/35166 ≈ 0.0074)
                 se dibuja como línea horizontal de referencia (clasificador aleatorio)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(
        prevalence, color="k", ls=":", lw=1,
        label=f"Azar (PR-AUC≈{prevalence:.4f})",
    )

    for label, res in results.items():
        style = _STYLES.get(label, {})
        ax.plot(
            res["recall"], res["precision"],
            color=style.get("color", None),
            ls=style.get("ls", "-"),
            lw=2,
            label=f"{label}  (PR-AUC={res['pr_auc']:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Curvas Precision-Recall — Diseño 2×2", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figura guardada: {save_path}")
