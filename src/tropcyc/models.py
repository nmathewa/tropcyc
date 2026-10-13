"""Intensity-change models with one interface: fit(table) and predict(table).

`table` is the output of features.build_table (plus ERA5 columns or patches
for the models that use them). Predictions are 24 h intensity change in kt.

The baselines matter: a skill score is only meaningful against them, and an
ML model that cannot beat "intensity stays the same" is not useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TRACK_FEATURES


class Persistence:
    """No change: vmax in 24 h equals vmax now."""

    name = "persistence"

    def __init__(self, target: str = "dv_24"):
        self.target = target

    def fit(self, table: pd.DataFrame):
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(table))


class Trend:
    """The last 24 h change continues, damped: a common naive forecast."""

    name = "trend"

    def __init__(self, target: str = "dv_24", damping: float | None = None):
        self.target = target
        self.damping = damping

    def fit(self, table: pd.DataFrame):
        if self.damping is None:  # least-squares damping factor
            x, y = table["dvmax_24"].to_numpy(), table[self.target].to_numpy()
            self.damping = float(np.dot(x, y) / max(np.dot(x, x), 1e-9))
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.damping * table["dvmax_24"].to_numpy()


class Climatology:
    """Mean training change for storms of similar current intensity."""

    name = "climatology"
    bins = [0, 34, 64, 83, 96, 113, 137, 250]  # TD, TS, Cat 1-5 edges in kt

    def __init__(self, target: str = "dv_24"):
        self.target = target

    def fit(self, table: pd.DataFrame):
        cat = pd.cut(table["vmax"], self.bins, right=False)
        self.means_ = table.groupby(cat, observed=False)[self.target].mean()
        self.overall_ = float(table[self.target].mean())
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        cat = pd.cut(table["vmax"], self.bins, right=False)
        return cat.map(self.means_).astype(float).fillna(self.overall_).to_numpy()


class GradientBoosting:
    """Gradient-boosted trees on tabular predictors (SHIPS-like in spirit).

    Uses LightGBM when installed, otherwise scikit-learn's histogram GBM.
    Both handle missing values (e.g. no mslp) natively.
    """

    name = "gbm"

    def __init__(self, target: str = "dv_24", features: list[str] | None = None, **params):
        self.target = target
        self.features = features or list(TRACK_FEATURES)
        self.params = {"n_estimators": 400, "learning_rate": 0.03, **params}

    def _make(self):
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(num_leaves=31, min_child_samples=40, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, verbose=-1, **self.params)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            p = dict(self.params)
            return HistGradientBoostingRegressor(max_iter=p.pop("n_estimators"), **p)

    def fit(self, table: pd.DataFrame):
        self.model_ = self._make().fit(table[self.features], table[self.target])
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(table[self.features])

    def feature_importance(self) -> pd.Series:
        imp = getattr(self.model_, "feature_importances_", None)
        if imp is None:
            return pd.Series(dtype=float)
        return pd.Series(imp, index=self.features).sort_values(ascending=False)


MODELS = {m.name: m for m in (Persistence, Trend, Climatology, GradientBoosting)}
