"""Track-derived predictors and intensity-change targets.

Everything here comes from the best track alone, so it is cheap and needs no
gridded data. It is the feature set for the persistence-style baselines and
the gradient-boosting model; the CNN adds ERA5 patches on top (see era5.py).

Lags and leads are looked up by *time*, not by row position, so a gap in a
track (for example a storm that is briefly extratropical) never pairs two
fixes that are not really 6, 12 or 24 hours apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
RI_THRESHOLD_KT = 30  # Kaplan & DeMaria (2003): +30 kt in 24 h

TRACK_FEATURES = [
    "vmax",
    "mslp",
    "dvmax_6",
    "dvmax_12",
    "dvmax_24",
    "lat",
    "abs_lat",
    "lon",
    "u_motion",
    "v_motion",
    "speed",
    "age_h",
    "doy_sin",
    "doy_cos",
]


def _value_at(df: pd.DataFrame, col: str, hours: int) -> pd.Series:
    """Value of `col` for the same storm `hours` from each row (NaN if absent)."""
    shifted = df[["sid", "time", col]].copy()
    shifted["time"] = shifted["time"] - pd.Timedelta(hours=hours)
    merged = df[["sid", "time"]].merge(shifted, on=["sid", "time"], how="left")
    return merged[col].to_numpy()


def _motion(lat0, lon0, lat1, lon1, hours):
    """Eastward and northward storm motion in km/h."""
    dlon = ((lon1 - lon0 + 180) % 360) - 180
    mean_lat = np.deg2rad((lat0 + lat1) / 2)
    u = np.deg2rad(dlon) * EARTH_RADIUS_KM * np.cos(mean_lat) / hours
    v = np.deg2rad(lat1 - lat0) * EARTH_RADIUS_KM / hours
    return u, v


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TRACK_FEATURES to a cleaned track table."""
    out = df.sort_values(["sid", "time"]).reset_index(drop=True).copy()
    for h in (6, 12, 24):
        out[f"dvmax_{h}"] = out["vmax"] - _value_at(out, "vmax", -h)

    lat_prev = _value_at(out, "lat", -6)
    lon_prev = _value_at(out, "lon", -6)
    out["u_motion"], out["v_motion"] = _motion(lat_prev, lon_prev, out["lat"], out["lon"], 6)
    out["speed"] = np.hypot(out["u_motion"], out["v_motion"])

    out["abs_lat"] = out["lat"].abs()
    out["age_h"] = (out["time"] - out.groupby("sid")["time"].transform("min")).dt.total_seconds() / 3600
    doy = out["time"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_targets(df: pd.DataFrame, leads=(24,)) -> pd.DataFrame:
    """Add dv_<lead> (future minus current vmax, kt) and ri_24 (0/1)."""
    out = df.copy()
    for lead in leads:
        out[f"dv_{lead}"] = _value_at(out, "vmax", lead) - out["vmax"]
    if 24 in leads:
        out["ri_24"] = np.where(out["dv_24"].isna(), np.nan, (out["dv_24"] >= RI_THRESHOLD_KT).astype(float))
    return out


def build_table(tracks: pd.DataFrame, leads=(24,)) -> pd.DataFrame:
    """Features + targets; rows missing a target or the 24 h history are dropped."""
    table = add_targets(add_track_features(tracks), leads)
    needed = ["dvmax_24", "u_motion"] + [f"dv_{lead}" for lead in leads]
    return table.dropna(subset=needed).reset_index(drop=True)
