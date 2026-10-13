"""Active storms as tropycal storms, and tropycal track maps for the live page.

Storms come from tropycal's realtime feed (NHC, via tropycal.realtime) when it
is reachable, and from TC vitals (NHC + JTWC, all basins) otherwise or for the
basins tropycal's default feed does not cover. Either way each storm becomes a
tropycal-style storm dictionary, so tropycal draws every map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# TC vitals basin letter -> ATCF basin prefix and tropycal basin name
ATCF = {"L": ("AL", "north_atlantic"), "E": ("EP", "east_pacific"), "C": ("CP", "east_pacific"),
        "W": ("WP", "west_pacific"), "A": ("IO", "north_indian"), "B": ("IO", "north_indian"),
        "S": ("SH", "south_indian"), "P": ("SH", "south_pacific")}


def ace(times, vmax, types) -> float:
    """Accumulated cyclone energy (10^4 kt^2) over 6-hourly tropical fixes at >= 34 kt."""
    total = 0.0
    for t, v, ty in zip(times, vmax, types, strict=True):
        t = pd.Timestamp(t)
        if t.minute == 0 and t.hour % 6 == 0 and ty in ("TS", "HU") and np.isfinite(v) and v >= 34:
            total += v * v
    return round(total * 1e-4, 1)


def _to_tropycal_dict(g: pd.DataFrame) -> dict:
    """One storm's TC vitals rows -> a tropycal storm dictionary."""
    sid = g["sid"].iloc[0]
    letter = sid[2]
    prefix, basin = ATCF.get(letter, ("XX", "all"))
    atcf_id = f"{prefix}{sid[:2]}{sid[3:]}"
    n = len(g)
    d = {"id": atcf_id, "operational_id": atcf_id, "name": g["name"].iloc[-1],
         "year": int(sid[3:]), "season": int(sid[3:]), "basin": basin,
         "source": "tcvitals", "source_info": "NHC / JTWC TC vitals (GFS bucket)",
         "source_method": "", "source_url": "",
         "time": list(g["time"].dt.to_pydatetime()), "lat": list(g["lat"].astype(float)),
         "lon": list(g["lon"].astype(float)), "vmax": list(g["vmax"].astype(float)),
         "mslp": list(g["mslp"].astype(float)), "type": list(g["type"]),
         "special": [""] * n, "extra_obs": [0] * n, "wmo_basin": [basin] * n}
    d["ace"] = ace(d["time"], d["vmax"], d["type"])
    return d


def from_tcvitals(hours: int = 240) -> dict:
    """Active storms (latest cycle) with up to `hours` of history, keyed by ATCF id."""
    from .tcvitals import fetch_recent

    tracks = fetch_recent(hours)
    return {d["id"]: d for d in (_to_tropycal_dict(g) for _, g in tracks.groupby("sid"))}


def from_tropycal_realtime() -> dict:
    """Active storms from tropycal's realtime feed (NHC best-track files), keyed by ATCF id."""
    from tropycal import realtime

    rt = realtime.Realtime()
    out = {}
    for sid in rt.list_active_storms():
        d = dict(rt.get_storm(sid).dict)
        if not d.get("invest", False):
            d.setdefault("ace", ace(d["time"], d["vmax"], d["type"]))
            out[sid] = d
    return out


def active_storms(hours: int = 240) -> tuple[dict, list[str]]:
    """Merged active storms {ATCF id: tropycal dict} and the sources that answered.

    tropycal's realtime feed wins for the storms it has; TC vitals fill in the rest.
    """
    storms, sources = {}, []
    try:
        storms.update(from_tcvitals(hours))
        sources.append("TC vitals")
    except Exception:  # feed down: carry on with whatever else answers
        pass
    try:
        rt = from_tropycal_realtime()
        storms.update(rt)
        if rt:
            sources.append("tropycal realtime (NHC)")
    except Exception:
        pass
    return storms, sources


def to_tracks(storms: dict) -> pd.DataFrame:
    """tropycal dicts -> the tropcyc track table (with sid = ATCF id) for forecasting."""
    from .tracks import from_storm_dicts

    return from_storm_dicts(storms)


def plot_storm(d: dict, path, forecast_row=None) -> None:
    """tropycal track map of one storm, with the tropcyc 24 h intensity forecast marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tropycal.tracks.storm import Storm

    ax = Storm(d).plot()
    if forecast_row is not None:
        lon, lat = float(d["lon"][-1]), float(d["lat"][-1])
        import cartopy.crs as ccrs

        ax.plot(lon, lat, marker="*", markersize=22, markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.4, transform=ccrs.PlateCarree(), zorder=40)
        label = (f"tropcyc 24 h forecast: {forecast_row.vmax_24:.0f} kt "
                 f"({forecast_row.dv_24:+.0f} kt)").replace("-", "−")
        ax.annotate(label, xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=(14, 14), textcoords="offset points", fontsize=12, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.3", "alpha": 0.9}, zorder=41)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")


def plot_overview(storms: dict, path) -> None:
    """One tropycal map with every active storm's track."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tropycal.tracks.plot import TrackPlot

    lats = np.concatenate([np.asarray(d["lat"], float) for d in storms.values()])
    lons = np.concatenate([np.asarray(d["lon"], float) for d in storms.values()])
    # Storms on both sides of the Pacific: centre the map on the date line, longitudes 0-360
    east, west = (lons < 0).any(), (lons >= 0).any()
    centre = 180.0 if east and west and np.ptp(lons % 360) < np.ptp(lons) else 0.0
    if centre == 180.0:
        lons = lons % 360
    pad = 10
    domain = {"w": float(lons.min() - pad), "e": float(lons.max() + 3 * pad),  # legend sits top right
              "s": float(max(lats.min() - pad, -70)), "n": float(min(lats.max() + pad, 70))}
    tp = TrackPlot()
    tp.create_cartopy(proj="PlateCarree", central_longitude=centre)
    tp.plot_storms(list(storms.values()), domain=domain, title="Active tropical cyclones")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close("all")
