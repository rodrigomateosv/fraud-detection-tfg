"""
#src/interpretability.py

Funciones reutilizables de TreeSHAP para el análisis de interpretabilidad
del modelo XGBoost + Config B del TFG de detección de fraude contable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


# Lista canónica de features Schilit (posiciones 5 y 6 usan columnas del CSV)
SCHILIT_10: list[str] = [
    "receivables_index",
    "inventory_index",
    "payables_index",
    "accrual_ratio",
    "soft_assets",
    "dpi",
    "gross_margin_index",
    "sales_growth_index",
    "leverage_index",
    "asset_quality_index",
]


def compute_shap_values(
    model,
    X: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Calcula TreeSHAP values para un modelo XGBoost.

    Parámetros
    ----------
    model : XGBClassifier entrenado
    X     : DataFrame de features (test set)

    Devuelve
    --------
    (shap_values, expected_value)
      shap_values    : ndarray de shape (n_obs, n_features), clase positiva
      expected_value : valor base del explainer (log-odds o probabilidad)
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # XGBoost binario: shap_values es un ndarray 2D directamente
    # (no lista de dos arrays como en algunos modelos sklearn)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # clase positiva (fraude)

    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        exp_val = float(exp_val[1] if len(exp_val) > 1 else exp_val[0])
    else:
        exp_val = float(exp_val)

    return shap_vals, exp_val


def get_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calcula importancia global como |SHAP| medio por feature.

    Parámetros
    ----------
    shap_values   : ndarray (n_obs, n_features), clase positiva
    feature_names : lista de nombres de columnas (mismo orden que shap_values)

    Devuelve
    --------
    DataFrame ordenado por mean_abs_shap descendente con columnas:
      [rank, feature, mean_abs_shap, mean_shap, std_shap, n_obs]
    """
    abs_shap  = np.abs(shap_values)
    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": abs_shap.mean(axis=0),
        "mean_shap":     shap_values.mean(axis=0),
        "std_shap":      shap_values.std(axis=0),
        "n_obs":         len(shap_values),
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def classify_schilit_vs_raw(feature_names: list[str]) -> dict[str, str]:
    """
    Clasifica cada feature como 'Schilit' o 'Raw'.

    Parámetros
    ----------
    feature_names : lista de nombres de columnas del dataset

    Devuelve
    --------
    dict {feature_name: 'Schilit' | 'Raw'}
    """
    return {f: ("Schilit" if f in SCHILIT_10 else "Raw") for f in feature_names}
