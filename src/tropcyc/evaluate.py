"""Leak-free splits and forecast verification.

Rows from the same storm are strongly correlated (consecutive 6 h fixes look
almost identical), so a random row split lets a model memorise a storm and
score far too well. Splits here are always by whole season: training years
come strictly before validation years, which come before test years. That is
also how the model would be used, on storms it has never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import RI_THRESHOLD_KT


def split_by_year(table: pd.DataFrame, val_start: int, test_start: int):
    """Return (train, val, test): years < val_start, [val_start, test_start), >= test_start."""
    # Assign each storm to the year it formed, so no storm straddles two sets.
    year = table.groupby("sid")["time"].transform("min").dt.year
    train = table[year < val_start]
    val = table[(year >= val_start) & (year < test_start)]
    test = table[year >= test_start]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def scores(y_true, y_pred, baseline_pred=None) -> dict:
    """MAE, RMSE, bias (kt) and, if given a baseline, MAE skill (%) over it."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    out = {
        "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }
    if baseline_pred is not None:
        base_mae = np.mean(np.abs(np.asarray(baseline_pred, float) - y_true))
        out["skill_pct"] = float(100 * (1 - out["mae"] / base_mae)) if base_mae > 0 else np.nan
    return out


def ri_scores(dv_true, dv_pred, threshold: float = RI_THRESHOLD_KT) -> dict:
    """Rapid-intensification detection: probability of detection, false alarm ratio, CSI."""
    obs = np.asarray(dv_true) >= threshold
    fc = np.asarray(dv_pred) >= threshold
    hits = int(np.sum(obs & fc))
    misses = int(np.sum(obs & ~fc))
    false_alarms = int(np.sum(~obs & fc))
    return {
        "ri_events": int(obs.sum()),
        "pod": hits / (hits + misses) if hits + misses else np.nan,
        "far": false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan,
        "csi": hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan,
    }


def compare(models, train: pd.DataFrame, test: pd.DataFrame, target: str = "dv_24") -> pd.DataFrame:
    """Fit each model on train, score on test; skill is relative to persistence."""
    y = test[target].to_numpy()
    baseline = np.zeros(len(test))  # persistence
    rows = {}
    for model in models:
        pred = model.fit(train).predict(test)
        rows[model.name] = {**scores(y, pred, baseline), **ri_scores(y, pred)}
    return pd.DataFrame(rows).T
