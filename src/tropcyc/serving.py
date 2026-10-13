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


def load_bundle(path, source: str = "gfs", future_source: str | None = None) -> dict:
    """A .joblib bundle from `tropcyc train`, or a transformer .pt run live on `source`."""
    if str(path).endswith(".pt"):
        return transformer_bundle(path, source, future_source)
    import joblib

    return joblib.load(path)


class LiveTransformer:
    """Runs a trained transformer on current storms, fetching environment from `source`."""

    def __init__(self, model, source, future_source=None):
        self.model, self.source, self.future_source = model, source, future_source

    def predict_live(self, tracks: pd.DataFrame, rows: pd.DataFrame):
        from .transformer import build_env_store

        self.model.tracks = tracks
        self.model.store = build_env_store(rows, tracks, self.source, extrapolate=True,
                                           progress=False, workers=8, future_source=self.future_source)
        if not self.model.store:
            raise RuntimeError("no environment patches could be fetched for the active storms")
        dv, p = self.model._predict_loader(self.model._loader(rows.reset_index(drop=True), False))
        return dv, p


def transformer_bundle(path, source: str = "gfs", future_source: str | None = None) -> dict:
    from .sources import make_source
    from .transformer import TransformerModel

    model = TransformerModel.load(path, verbose=False)
    val = [h for h in model.history_ if h["finetune"]] or model.history_
    label = f" + {future_source} future" if future_source else ""
    return {
        "model": LiveTransformer(model, make_source(source, live=True),
                                 make_source(future_source, live=True) if future_source else None),
        "name": f"transformer ({model.size}, {source}{label})",
        "kind": "sequence",
        "target": model.target,
        "validation": {"transformer": {"mae": min(h["val_mae"] for h in val)}} if val else {},
        "checkpoint": str(path),
    }


def forecast(bundle: dict, tracks: pd.DataFrame) -> pd.DataFrame:
    """24 h intensity forecast from each storm's latest fix."""
    cols = ["sid", "name", "time", "lat", "lon", "vmax", "mslp", "dvmax_24"]
    feats = latest_features(clean(tracks, min_fixes=1)) if len(tracks) else pd.DataFrame()
    if feats.empty:
        return pd.DataFrame(columns=cols + ["dv_24", "vmax_24", "valid_time", "rapid_intensification"])
    out = feats[cols].copy()
    if bundle.get("kind") == "sequence":
        dv, p_ri = bundle["model"].predict_live(clean(tracks, min_fixes=1), feats)
        out["dv_24"] = np.round(dv, 1)
        out["ri_probability"] = np.round(p_ri, 3)
    else:
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
