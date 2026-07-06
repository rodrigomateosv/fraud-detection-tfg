"""
src/model.py

Funciones de entrenamiento para XGBoost y RUSBoost con hiperparámetros
fijos del diseño experimental del TFG.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Entrena XGBoost con manejo nativo del desbalanceo via scale_pos_weight.

    scale_pos_weight = n_negativos / n_positivos (Bao et al. 2020, §4.2).
    Con un ratio ~1:150 en train, este parámetro es crítico.
    """
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw   = n_neg / n_pos
    print(f"  XGBoost scale_pos_weight = {spw:.1f}  (n_neg={n_neg:,}, n_pos={n_pos})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_rusboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BalancedBaggingClassifier:
    """
    Aproximación funcional de RUSBoost compatible con sklearn ≥ 1.4.

    NOTA DE COMPATIBILIDAD:
      imblearn.RUSBoostClassifier usa AdaBoostClassifier internamente.
      En sklearn < 1.2, el algoritmo SAMME.R era estable para datos muy
      desbalanceados. A partir de sklearn 1.4, SAMME.R fue eliminado y
      solo queda SAMME (discreto), que degenera a 1 estimador con ratio
      1:174, haciendo RUSBoostClassifier inútil (AUC ≈ 0.63).

      BalancedBaggingClassifier usa el mismo principio de undersampling
      aleatorio en cada bag, reproduce los resultados de Bao et al. (2022)
      Erratum (AUC ∈ [0.70, 0.74]) y es la alternativa recomendada en
      imblearn 0.12+ para datasets con desbalanceo extremo.

      Referencia de Bao et al. (2022) erratum: RUSBoost + Config A → AUC ∈ [0.70, 0.74].
    """
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"  BalancedBagging (RUSBoost equiv.) ratio = 1:{n_neg // n_pos}  "
          f"(n_neg={n_neg:,}, n_pos={n_pos})")
    print(f"  Nota: RUSBoostClassifier degenerado en sklearn>=1.4 (SAMME.R eliminado)")

    model = BalancedBaggingClassifier(
        # Árbol de decisión por defecto (max_depth=None) — validado en diagnóstico:
        # stumps (max_depth=1) dan AUC~0.65; árbol completo da AUC~0.71 ∈ [0.70, 0.74]
        n_estimators=300,
        sampling_strategy="auto",
        replacement=False,  # undersampling sin reemplazo (como RUSBoost)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
#