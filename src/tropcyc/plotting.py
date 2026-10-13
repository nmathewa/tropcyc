"""Verification plots. Maps and storm tracks come from tropycal itself."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_storm_track(dataset, sid: str, **kwargs):
    """Draw one storm's track with tropycal (`dataset` is a tropycal TrackDataset)."""
    return dataset.get_storm(sid).plot(**kwargs)


def plot_intensity_forecast(table: pd.DataFrame, pred, sid: str, lead: int = 24, ax=None):
    """Observed vmax for one storm, with each forecast drawn at its valid time."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(8, 4))[1]
    mask = (table["sid"] == sid).to_numpy()
    s = table[mask]
    valid = s["time"] + pd.Timedelta(hours=lead)
    ax.plot(s["time"], s["vmax"], "k-o", ms=3, label="best track")
    ax.plot(valid, s["vmax"] + np.asarray(pred)[mask], "C3-", lw=1.5, label=f"{lead} h forecast")
    ax.set_ylabel("max sustained wind (kt)")
    ax.set_title(f"{s['name'].iloc[0]} ({sid})" if "name" in s and len(s) else sid)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_scatter(y_true, y_pred, ax=None, title: str = ""):
    """Predicted against observed 24 h change, with the RI threshold marked."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "k--", lw=1)
    ax.axvline(30, color="C3", lw=0.8, ls=":")
    ax.axhline(30, color="C3", lw=0.8, ls=":")
    ax.set_xlabel("observed change (kt)")
    ax.set_ylabel("predicted change (kt)")
    ax.set_title(title)
    return ax
