
"""
#src/features.py

Feature engineering basado en Schilit (2018) — Financial Shenanigans, 4th ed.
Combina fórmulas literales del libro (DSO, DSI, DPO) con formalizaciones de
Beneish (1999), Sloan (1996) y Dechow et al. (2011).

NOTA TÉCNICA — PP&E bruto vs. neto:
  Features 6 (depreciation_index) y 10 (asset_quality_index) usan ppegt
  (PP&E bruto) porque ppent (neto) no está en el dataset. Beneish/Dechow
  usan PP&E neto; esto explica correlaciones <1.0 con dpi y soft_assets del CSV.
"""
import numpy as np
import pandas as pd

# ── Conjuntos de features ────────────────────────────────────────────────────

# 28 variables raw de Compustat (Bao et al. 2020, configuración A)
RAW_28: list[str] = [
    "act", "ap", "at", "ceq", "che", "cogs", "csho", "dlc", "dltis", "dltt",
    "dp", "ib", "invt", "ivao", "ivst", "lct", "lt", "ni", "ppegt", "pstk",
    "re", "rect", "sale", "sstk", "txp", "txt", "xint", "prcc_f",
]

# 10 features Schilit para modelado.
# Posiciones 5-6 ('soft_assets', 'dpi') provienen directamente del CSV;
# las versiones recalculadas (soft_assets_ratio, depreciation_index) se añaden
# como columnas de validación pero NO entran en SCHILIT_10.
SCHILIT_10: list[str] = [
    "receivables_index",
    "inventory_index",
    "payables_index",
    "accrual_ratio",
    "soft_assets",        # columna del CSV (feature 5)
    "dpi",                # columna del CSV (feature 6)
    "gross_margin_index",
    "sales_growth_index",
    "leverage_index",
    "asset_quality_index",
]

# 10 columnas nuevas que añade build_schilit_features()
# (incluye soft_assets_ratio y depreciation_index como columnas de validación)
SCHILIT_COMPUTED: list[str] = [
    "receivables_index",
    "inventory_index",
    "payables_index",
    "accrual_ratio",
    "soft_assets_ratio",    # validación de 'soft_assets' del CSV
    "depreciation_index",   # validación de 'dpi' del CSV
    "gross_margin_index",
    "sales_growth_index",
    "leverage_index",
    "asset_quality_index",
]


def build_schilit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade 10 columnas Schilit al DataFrame de entrada.

    Parámetros
    ----------
    df : pd.DataFrame con columnas Compustat raw + gvkey + fyear

    Devuelve
    --------
    pd.DataFrame con 10 nuevas columnas (SCHILIT_COMPUTED).
    Las primeras observaciones de cada empresa tienen NaN en las features
    que requieren lag — esto es correcto, NO rellenar con cero.

    IMPORTANTE: no se aplica winsorización ni imputación aquí.
    Eso se realiza en el pipeline de modelado (Día 3).
    """
    df = df.sort_values(["gvkey", "fyear"]).copy()

    def lag(col: str) -> pd.Series:
        """Valor del año anterior por empresa. Nunca shift() sin groupby."""
        return df.groupby("gvkey")[col].shift(1)

    def lag_s(s: pd.Series) -> pd.Series:
        """Lag para una serie derivada que comparte el índice de df."""
        return s.groupby(df["gvkey"]).shift(1)

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        """División segura: denominador cero → NaN."""
        return num / den.replace(0, np.nan)

    # ── Proxy de CFO por balance (Sloan 1996) ───────────────────────────────
    # cfo_proxy = ni - (Δact - Δche - Δlct + Δdlc + Δtxp)
    cfo_proxy = df["ni"] - (
        (df["act"] - lag("act"))
        - (df["che"] - lag("che"))
        - (df["lct"] - lag("lct"))
        + (df["dlc"] - lag("dlc"))
        + (df["txp"] - lag("txp"))
    )

    # ── Feature 1 — Receivables Index (DSRI / DSO) ──────────────────────────
    # Beneish 1999 DSRI · Schilit Cap.3 pp.41,49-50 · alto → fraude
    # Ratio creciente de cobros/ventas señala reconocimiento prematuro de ingresos
    df["receivables_index"] = safe_div(
        safe_div(df["rect"], df["sale"]),
        safe_div(lag("rect"), lag("sale")),
    )

    # ── Feature 2 — Inventory Index (DSI) ───────────────────────────────────
    # Schilit Cap.6 p.97 · alto → fraude
    # Inventario acumulándose más rápido que el coste señala inventario ficticio
    df["inventory_index"] = safe_div(
        safe_div(df["invt"], df["cogs"]),
        safe_div(lag("invt"), lag("cogs")),
    )

    # ── Feature 3 — Payables Index (DPO) ────────────────────────────────────
    # Schilit Cap.12 pp.184-185 · alto → fraude
    # Estirar pagos a proveedores infla artificialmente el CFO reportado
    df["payables_index"] = safe_div(
        safe_div(df["ap"], df["cogs"]),
        safe_div(lag("ap"), lag("cogs")),
    )

    # ── Feature 4 — Accrual Ratio ────────────────────────────────────────────
    # Sloan 1996 · Dechow & Dichev 2002 · Schilit Cap.7 pp.108-111 · alto → fraude
    # Beneficio muy superior al flujo de caja predice reversiones futuras
    df["accrual_ratio"] = safe_div(df["ni"] - cfo_proxy, df["at"])

    # ── Feature 5 — Soft Assets Ratio (columna de validación) ───────────────
    # Dechow 2011 · Schilit Cap.6 pp.85-87 (WorldCom) · alto → fraude
    # Activos intangibles/opacos como % del activo total
    # SCHILIT_10 usa 'soft_assets' del CSV; esta columna sirve solo para validar
    df["soft_assets_ratio"] = safe_div(
        df["at"] - df["ppegt"] - df["che"], df["at"]
    )

    # ── Feature 6 — Depreciation Index (columna de validación) ──────────────
    # Beneish 1999 DEPI · Schilit Cap.6 p.91 (Ultimate Software) · alto → fraude
    # Tasa de amortización decreciente señala alargamiento de vidas útiles
    # SCHILIT_10 usa 'dpi' del CSV; esta columna sirve solo para validar
    dep_rate = safe_div(df["dp"], df["ppegt"])
    df["depreciation_index"] = safe_div(lag_s(dep_rate), dep_rate)

    # ── Feature 7 — Gross Margin Index ──────────────────────────────────────
    # Beneish 1999 GMI · Schilit Cap.5 p.71 (IBM) · alto → fraude
    # Margen bruto deteriorándose (ratio t-1/t > 1) presiona a manipular
    gm = safe_div(df["sale"] - df["cogs"], df["sale"])
    df["gross_margin_index"] = safe_div(lag_s(gm), gm)

    # ── Feature 8 — Sales Growth Index ──────────────────────────────────────
    # Beneish 1999 SGI · Schilit pp.49,69 · alto → fraude (proxy de motivación)
    # Crecimiento rápido crea presión insostenible para seguir batiendo expectativas
    df["sales_growth_index"] = safe_div(df["sale"], lag("sale"))

    # ── Feature 9 — Leverage Index ───────────────────────────────────────────
    # Beneish 1999 LVGI · Schilit pp.11-13 (Lehman Repo 105) · dirección AMBIGUA
    # Presión de covenants puede motivar manipulación, pero no es directional
    df["leverage_index"] = safe_div(
        safe_div(df["lt"], df["at"]),
        safe_div(lag("lt"), lag("at")),
    )

    # ── Feature 10 — Asset Quality Index ────────────────────────────────────
    # Beneish 1999 AQI · Schilit Cap.6 pp.85-91 · alto → fraude
    # Activos no corrientes opacos (excl. act y ppegt) creciendo como % del total
    aq = safe_div(df["at"] - df["act"] - df["ppegt"], df["at"])
    df["asset_quality_index"] = safe_div(aq, lag_s(aq))

    # Reemplazar ±inf por NaN (divisiones entre valores extremos o cero)
    df[SCHILIT_COMPUTED] = df[SCHILIT_COMPUTED].replace([np.inf, -np.inf], np.nan)

    return df


def get_feature_sets() -> dict[str, list[str]]:
    """
    Devuelve configuraciones de features para el modelado.

    config_a : RAW_28              (Bao et al. 2020 baseline)
    config_b : RAW_28 + SCHILIT_10 (aumentado con Schilit)
    """
    return {
        "config_a": list(RAW_28),
        "config_b": list(RAW_28) + list(SCHILIT_10),
    }
