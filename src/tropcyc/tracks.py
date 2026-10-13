"""Best-track data as one tidy table.

tropycal does the downloading and parsing (HURDAT2 or IBTrACS). This module
flattens its per-storm dictionaries into a single pandas DataFrame, one row per
storm per 6-hour fix, which is what everything else in tropcyc works on.

Columns: sid, name, time, lat, lon, vmax (kt), mslp (hPa), type
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["sid", "name", "time", "lat", "lon", "vmax", "mslp", "type"]

# HURDAT2 / tropycal storm types counted as tropical. Extratropical (EX),
# lows (LO), waves (WV) and disturbances (DB) are dropped: their intensity
# changes follow different physics and would only add noise to the target.
TROPICAL_TYPES = ("TD", "TS", "HU")
SUBTROPICAL_TYPES = ("SD", "SS")


def load_tropycal(basin: str = "north_atlantic", source: str = "hurdat", **kwargs):
    """Load a tropycal TrackDataset (downloads on first use)."""
    from tropycal import tracks

    return tracks.TrackDataset(basin=basin, source=source, **kwargs)


def from_storm_dicts(storms, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Flatten tropycal storm dictionaries ({sid: {"time": [...], "vmax": [...], ...}}).

    Historical (TrackDataset.data) and realtime (RealtimeStorm.dict) storms share
    this layout, so both go through here.
    """
    frames = []
    for sid, storm in storms.items():
        if not len(storm["time"]):
            continue
        year = int(storm.get("year", pd.Timestamp(storm["time"][0]).year))
        if year_range and not (year_range[0] <= year <= year_range[1]):
            continue
        n = len(storm["time"])
        frames.append(
            pd.DataFrame(
                {
                    "sid": sid,
                    "name": storm.get("name", ""),
                    "time": pd.to_datetime(storm["time"]),
                    "lat": storm["lat"],
                    "lon": storm["lon"],
                    "vmax": storm["vmax"],
                    "mslp": storm.get("mslp", [np.nan] * n),
                    "type": storm.get("type", ["TS"] * n),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def from_tropycal(dataset, year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Flatten a tropycal TrackDataset into the tropcyc track table."""
    return from_storm_dicts(dataset.data, year_range)


def clean(
    df: pd.DataFrame,
    include_subtropical: bool = False,
    synoptic_only: bool = True,
    min_fixes: int = 4,
) -> pd.DataFrame:
    """Keep tropical, synoptic-time fixes with a valid intensity.

    synoptic_only drops the extra landfall / peak records HURDAT2 inserts
    between 00/06/12/18 UTC, so every step is exactly 6 hours.
    """
    keep = TROPICAL_TYPES + (SUBTROPICAL_TYPES if include_subtropical else ())
    out = df[df["type"].isin(keep)].copy()
    out["vmax"] = pd.to_numeric(out["vmax"], errors="coerce")
    out.loc[out["mslp"] <= 0, "mslp"] = np.nan
    out = out.dropna(subset=["vmax", "lat", "lon"])
    if synoptic_only:
        t = out["time"]
        out = out[(t.dt.minute == 0) & (t.dt.hour % 6 == 0)]
    out = out.sort_values(["sid", "time"]).drop_duplicates(["sid", "time"])
    counts = out.groupby("sid")["time"].transform("size")
    return out[counts >= min_fixes].reset_index(drop=True)
