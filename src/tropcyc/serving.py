"""Pick, train and save the model the server uses, and make live forecasts.

Only models that need nothing but the best track are candidates: the
ERA5-based ones cannot run in real time, because ERA5 is published about
five days late.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import compare, split_by_year
from .features import RI_THRESHOLD_KT, latest_features
from .models import Climatology, GradientBoosting, Persistence, Trend
from .tracks import clean, from_storm_dicts

LIVE_CANDIDATES = (Persistence, Trend, Climatology, GradientBoosting)


def select_and_train(table: pd.DataFrame, val_start: int, test_start: int, target: str = "dv_24") -> dict:
    """Score every live-capable model on the validation seasons, refit the best on all data."""
    train, val, _ = split_by_year(table, val_start, test_start)
    scores = compare([m(target) for m in LIVE_CANDIDATES], train, val, target=target)
    best = scores["mae"].astype(float).idxmin()
    model = {m.name: m for m in LIVE_CANDIDATES}[best](target).fit(table)
    return {
        "model": model,
        "name": best,
        "target": target,
        "validation": scores.astype(float).round(3).to_dict(orient="index"),
        "val_years": [val_start, test_start - 1],
        "trained_on": {
            "samples": int(len(table)),
            "storms": int(table["sid"].nunique()),
            "first": str(table["time"].min()),
            "last": str(table["time"].max()),
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_bundle(bundle: dict, path) -> None:
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path) -> dict:
    import joblib

    return joblib.load(path)


def forecast(bundle: dict, tracks: pd.DataFrame) -> pd.DataFrame:
    """24 h intensity forecast from each storm's latest fix."""
    cols = ["sid", "name", "time", "lat", "lon", "vmax", "mslp", "dvmax_24"]
    feats = latest_features(clean(tracks, min_fixes=1)) if len(tracks) else pd.DataFrame()
    if feats.empty:
        return pd.DataFrame(columns=cols + ["dv_24", "vmax_24", "valid_time", "rapid_intensification"])
    out = feats[cols].copy()
    out["dv_24"] = np.round(bundle["model"].predict(feats), 1)
    out["vmax_24"] = np.clip(out["vmax"] + out["dv_24"], 15, None).round(0)
    out["valid_time"] = out["time"] + pd.Timedelta(hours=24)
    out["rapid_intensification"] = out["dv_24"] >= RI_THRESHOLD_KT
    return out


def fetch_active_storms(basin: str = "all") -> dict:
    """Current storms from tropycal's realtime feed, as {sid: storm dict}."""
    from tropycal import realtime

    rt = realtime.Realtime()
    storms = {}
    for sid in rt.list_active_storms(basin=basin):
        storm = rt.get_storm(sid)
        storms[sid] = storm.dict
    return storms


def forecast_active(bundle: dict, storms: dict) -> pd.DataFrame:
    return forecast(bundle, from_storm_dicts(storms))
