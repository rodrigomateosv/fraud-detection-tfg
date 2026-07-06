"""
#src/preprocessing.py

Split temporal con tratamiento de fraude serial (Bao et al. 2022 Erratum)
y pipeline de winsorización + imputación sin fuga de datos del test set.
"""
import numpy as np
import pandas as pd


def split_temporal_with_serial_fraud(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train/test temporal y elimina la contaminación
    por fraude serial inter-split.

    Diseño Bao et al. (2020):
      Train : fyear <= 2001
      Buffer: fyear == 2002  (EXCLUIDO)
      Test  : 2003 <= fyear <= 2008

    Tratamiento fraude serial (Erratum Bao et al. 2022):
      Las empresas que aparecen como fraude en el test set NO pueden
      aparecer como fraude en el train, porque el modelo vería el
      patrón fraudulento antes que el regulador.  Se resetean a 0.

    Números esperados (validados contra EDA del Día 1):
      Train antes reset : 71.748 obs, 514 fraudes
      Observaciones reseteadas : 104
      Train después reset : 71.748 obs, 410 fraudes
      Test : 35.166 obs, 261 fraudes
    """
    train_raw = df[df["fyear"] <= 2001].copy()
    test      = df[(df["fyear"] >= 2003) & (df["fyear"] <= 2008)].copy()

    print(f"Train (antes reset) : {len(train_raw):>7,} obs  |  {int(train_raw['misstate'].sum()):>4} fraudes")
    print(f"Test                : {len(test):>7,} obs  |  {int(test['misstate'].sum()):>4} fraudes")

    # Identificar gvkeys con misstate=1 en el test
    test_fraud_gvkeys = set(test.loc[test["misstate"] == 1, "gvkey"].unique())

    # Resetear esas gvkeys en el train
    mask_reset = train_raw["gvkey"].isin(test_fraud_gvkeys) & (train_raw["misstate"] == 1)
    n_reset = int(mask_reset.sum())
    train = train_raw.copy()
    train.loc[mask_reset, "misstate"] = 0

    print(f"\nObservaciones de fraude reseteadas en train : {n_reset}")
    print(f"Train (después reset): {len(train):>7,} obs  |  {int(train['misstate'].sum()):>4} fraudes")

    # Validar que no hay solapamiento de fraudes entre train y test
    train_fraud_gvkeys = set(train.loc[train["misstate"] == 1, "gvkey"].unique())
    solapamiento = train_fraud_gvkeys & test_fraud_gvkeys
    assert len(solapamiento) == 0, (
        f"ERROR: {len(solapamiento)} gvkeys aparecen como fraude en train Y en test "
        f"tras el reset. Revisar el tratamiento de fraude serial."
    )
    print("Validación anti-leakage: train_fraude ∩ test_fraude = ∅  ✓")

    # Alertas sobre números esperados
    if len(train_raw) != 71748:
        print(f"  AVISO: train obs esperadas 71.748, obtenidas {len(train_raw):,}")
    if int(train_raw["misstate"].sum()) != 514:
        print(f"  AVISO: fraudes train antes reset esperados 514, obtenidos {int(train_raw['misstate'].sum())}")
    if n_reset != 104:
        print(f"  AVISO: reseteados esperados 104, obtenidos {n_reset}")
    if int(train["misstate"].sum()) != 410:
        print(f"  AVISO: fraudes train después reset esperados 410, obtenidos {int(train['misstate'].sum())}")
    if len(test) != 35166:
        print(f"  AVISO: test obs esperadas 35.166, obtenidas {len(test):,}")
    if int(test["misstate"].sum()) != 261:
        print(f"  AVISO: fraudes test esperados 261, obtenidos {int(test['misstate'].sum())}")

    return train, test


def winsorize_and_impute(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winsorización al [P1, P99] e imputación por mediana intra-fyear.

    CRÍTICO — anti-leakage:
      Todos los estadísticos (percentiles, medianas) se calculan
      ÚNICAMENTE con el train set y luego se aplican al test.

    Parámetros
    ----------
    X_train, X_test : DataFrames que deben contener feature_cols.
                      Pueden contener 'fyear' como columna auxiliar para
                      la imputación intra-año (si no está, se usa mediana global).
    feature_cols    : lista de columnas de features a procesar

    Devuelve
    --------
    (X_train_proc, X_test_proc) con exactamente feature_cols como columnas
    """
    # Copiar el DataFrame completo para no modificar el original
    X_tr = X_train.copy()
    X_te = X_test.copy()

    n_out_tr = n_out_te = 0
    n_nan_tr = n_nan_te = 0

    # ── Winsorización [P1, P99] ──────────────────────────────────────────────
    lo_bounds = X_tr[feature_cols].quantile(0.01)
    hi_bounds = X_tr[feature_cols].quantile(0.99)

    for col in feature_cols:
        lo, hi = lo_bounds[col], hi_bounds[col]
        n_out_tr += int((X_tr[col] < lo).sum() + (X_tr[col] > hi).sum())
        n_out_te += int((X_te[col] < lo).sum() + (X_te[col] > hi).sum())
        X_tr[col] = X_tr[col].clip(lo, hi)
        X_te[col] = X_te[col].clip(lo, hi)

    print(f"Winsorización [P1,P99]: {n_out_tr:,} outliers en train  |  {n_out_te:,} en test")

    # ── Imputación por mediana intra-fyear (calculada en train) ─────────────
    # fyear puede venir como columna auxiliar (no en feature_cols).
    if "fyear" in X_tr.columns and "fyear" not in feature_cols:
        fyear_tr = X_tr["fyear"]
        fyear_te = X_te["fyear"]
    else:
        fyear_tr = fyear_te = None

    for col in feature_cols:
        n_nan_tr += int(X_tr[col].isna().sum())
        n_nan_te += int(X_te[col].isna().sum())

        if fyear_tr is not None:
            # Mediana por fyear en train
            med_by_year = X_tr[col].groupby(fyear_tr).median()
            global_med  = X_tr[col].median()

            # Imputar train
            nan_mask_tr = X_tr[col].isna()
            X_tr.loc[nan_mask_tr, col] = fyear_tr[nan_mask_tr].map(med_by_year).fillna(global_med)

            # Imputar test: usar fyears del train; fyears ausentes → mediana global
            nan_mask_te = X_te[col].isna()
            X_te.loc[nan_mask_te, col] = fyear_te[nan_mask_te].map(med_by_year).fillna(global_med)
        else:
            global_med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(global_med)
            X_te[col] = X_te[col].fillna(global_med)

    print(f"Imputación mediana   : {n_nan_tr:,} NaN en train        |  {n_nan_te:,} en test")

    # Verificaciones finales (solo sobre feature_cols)
    X_tr_out = X_tr[feature_cols]
    X_te_out = X_te[feature_cols]
    assert X_tr_out.isna().sum().sum() == 0, "ERROR: quedan NaN en train tras imputación"
    assert X_te_out.isna().sum().sum() == 0, "ERROR: quedan NaN en test tras imputación"
    assert not np.isinf(X_tr_out.values).any(), "ERROR: inf en train tras winsorización"
    assert not np.isinf(X_te_out.values).any(), "ERROR: inf en test tras winsorización"

    return X_tr_out, X_te_out
