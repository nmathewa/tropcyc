"""Storm tracks from TC vitals, the operational storm messages in every GFS cycle.

NHC and JTWC issue a "TC vitals" record for each active storm every 6 hours
(position, max wind, central pressure); NCEP stores them with each GFS cycle
on AWS (2021 onward). They give

  * a live feed of current storms that needs no NHC access, and
  * a global, 6-hourly track archive for training.

They are *operational* estimates made in real time, not the post-season best
track (HURDAT2 / IBTrACS), which revises some intensities. Use best track
where you can; this is the fallback when it is not reachable.
"""

from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .tracks import COLUMNS

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
MS_TO_KT = 1.943844
BASINS = {"L": "North Atlantic", "E": "East Pacific", "C": "Central Pacific", "W": "West Pacific",
          "A": "Arabian Sea", "B": "Bay of Bengal", "S": "South Indian", "P": "South Pacific"}


def url(cycle: pd.Timestamp) -> str:
    return f"{BUCKET}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.syndata.tcvitals.tm00"


def download(start, end, folder, workers: int = 32) -> int:
    """Save every cycle's TC vitals in [start, end] to folder/YYYYMMDDHH.txt (skips existing)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def get(t):
        path = folder / f"{t:%Y%m%d%H}.txt"
        if path.exists():
            return 1
        try:
            path.write_bytes(urllib.request.urlopen(url(t), timeout=60).read())
            return 1
        except Exception:  # no file for this cycle (quiet period or gap)
            return 0

    cycles = pd.date_range(pd.Timestamp(start).floor("6h"), pd.Timestamp(end).floor("6h"), freq="6h")
    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(get, cycles))


def parse_line(line: str) -> dict | None:
    """One TC vitals record -> dict (None for invests, malformed or test records)."""
    if len(line) < 70:
        return None
    sid = line[5:8].strip()
    try:
        number = int(sid[:2])
    except ValueError:
        return None
    if number >= 80:  # 80-99: invests and test systems, not storms
        return None
    # Fixed-width format: fields can touch (e.g. "1008C0400"), so never split on spaces.
    try:
        t = pd.Timestamp(f"{line[19:27]} {line[28:30]}:{line[30:32]}")
        lat, lon = line[33:37].strip(), line[38:43].strip()
        la = int(lat[:-1]) / 10 * (1 if lat[-1] == "N" else -1)
        lo = int(lon[:-1]) / 10 * (1 if lon[-1] == "E" else -1)
        pcen = float(line[52:56])
        vmax_ms = float(line[67:69])
    except (ValueError, IndexError):
        return None
    if vmax_ms <= 0:
        return None
    return {"center": line[:4].strip(), "id": sid, "name": line[9:18].replace(":", " ").strip(),
            "time": t, "lat": la, "lon": ((lo + 180) % 360) - 180,
            "vmax": round(vmax_ms * MS_TO_KT), "mslp": pcen if pcen > 800 else np.nan}


def parse_text(text: str) -> list[dict]:
    return [r for r in map(parse_line, text.splitlines()) if r]


def _storm_ids(df: pd.DataFrame, max_gap_days: float = 5) -> pd.Series:
    """Unique storm id: basin id + the year the storm first appeared.

    Ids like 03S are reused every season, and Southern Hemisphere storms run
    across 1 January, so consecutive fixes of one id belong to the same storm
    unless more than `max_gap_days` apart.
    """
    sids = pd.Series(index=df.index, dtype=object)
    for _, g in df.sort_values("time").groupby("id"):
        new = g["time"].diff() > pd.Timedelta(days=max_gap_days)
        start_year = g["time"].groupby(new.cumsum()).transform("min").dt.year
        sids[g.index] = g["id"] + start_year.astype(str)
    return sids


def to_tracks(records: list[dict]) -> pd.DataFrame:
    """Records -> the standard tropcyc track table (sid, name, time, lat, lon, vmax, mslp, type)."""
    if not records:
        return pd.DataFrame(columns=COLUMNS + ["basin"])
    df = pd.DataFrame(records).sort_values(["time", "center"]).drop_duplicates(["id", "time"])
    df["sid"] = _storm_ids(df)
    # TC vitals carry no storm type: classify by intensity (all records are tropical/subtropical systems)
    df["type"] = np.select([df["vmax"] < 34, df["vmax"] < 64], ["TD", "TS"], "HU")
    df["basin"] = df["id"].str[-1].map(BASINS).fillna("other")
    # a storm's name changes when it is named (e.g. "TWENTY" -> "SIMON"): keep the last one
    df["name"] = df.groupby("sid")["name"].transform("last")
    return df[COLUMNS + ["basin"]].sort_values(["sid", "time"]).reset_index(drop=True)


def load_archive(folder) -> pd.DataFrame:
    records = []
    for path in sorted(Path(folder).glob("*.txt")):
        records += parse_text(path.read_text(errors="ignore"))
    return to_tracks(records)


def fetch_recent(hours: int = 48, now=None) -> pd.DataFrame:
    """Tracks of the storms in the latest available cycle, with their last `hours` of history."""
    now = pd.Timestamp(now or pd.Timestamp.utcnow().tz_localize(None)).floor("6h")
    cycles = pd.date_range(now - pd.Timedelta(hours=hours + 12), now, freq="6h")

    def get(t):
        try:
            return parse_text(urllib.request.urlopen(url(t), timeout=60).read().decode(errors="ignore"))
        except Exception:
            return None

    with ThreadPoolExecutor(12) as ex:
        results = dict(zip(cycles, ex.map(get, cycles), strict=True))
    available = [t for t, r in results.items() if r is not None]
    if not available:
        raise RuntimeError("no recent TC vitals found on the GFS bucket")
    latest = max(available)
    tracks = to_tracks([r for t in available for r in results[t]])
    active = set(tracks.loc[tracks["time"] == latest, "sid"])
    return tracks[tracks["sid"].isin(active)].reset_index(drop=True)


def active_storms_dict(hours: int = 48) -> dict:
    """Live storms in the {sid: storm dict} layout the server expects (tropycal-style)."""
    tracks = fetch_recent(hours)
    out = {}
    for sid, g in tracks.groupby("sid"):
        out[sid] = {k: list(g[k]) for k in ("time", "lat", "lon", "vmax", "mslp", "type")}
        out[sid]["name"] = g["name"].iloc[-1]
        out[sid]["basin"] = g["basin"].iloc[-1]
    return out
