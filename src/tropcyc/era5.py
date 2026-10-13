"""Storm-centred ERA5 patches, read straight from the public ARCO-ERA5 Zarr store.

Nothing is downloaded in bulk: for each best-track fix we read a small box
(by default 10 x 10 degrees, 41 x 41 points at 0.25 deg) around the storm
centre, only for the variables and levels we need. Results are cached to one
.npz file, so the slow cloud reads happen once.

Channels (in this order):
    sst      sea surface temperature (K)
    shear_u  u(200 hPa) - u(850 hPa), deep-layer shear, east component (m/s)
    shear_v  v(200 hPa) - v(850 hPa), north component (m/s)
    rh_700   mid-level relative humidity (%), from specific humidity + temperature
    vo_850   low-level relative vorticity (1e-5 /s), from the 850 hPa winds

ARCO-ERA5 stores neither RH nor vorticity on pressure levels, so both are
derived here from the fields it does have.

These are the ingredients of most intensity-change studies: warm ocean, weak
shear and a moist mid troposphere favour intensification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CHANNELS = ["sst", "shear_u", "shear_v", "rh_700", "vo_850"]
GRID_DEG = 0.25


def open_era5(url: str = ARCO_ERA5):
    """Lazily open ARCO-ERA5 (needs gcsfs + zarr; anonymous access)."""
    import xarray as xr

    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"}, consolidated=True)


def _box_indices(lat: float, lon: float, half_width_deg: float):
    """Row/column indices of a box on the ERA5 grid (lat 90..-90, lon 0..359.75)."""
    n = int(round(half_width_deg / GRID_DEG))
    i0 = int(round((90 - lat) / GRID_DEG))
    j0 = int(round((lon % 360) / GRID_DEG))
    rows = np.clip(np.arange(i0 - n, i0 + n + 1), 0, 720)
    cols = np.arange(j0 - n, j0 + n + 1) % 1440  # wraps across the date line
    return rows, cols


def extract_patch(ds, time, lat: float, lon: float, half_width_deg: float = 5.0) -> np.ndarray:
    """One (channels, H, W) float32 patch centred on (lat, lon) at `time`."""
    rows, cols = _box_indices(lat, lon, half_width_deg)
    snap = ds.sel(time=pd.Timestamp(time), method="nearest").isel(latitude=rows, longitude=cols)

    def lev(var, level):
        return snap[var].sel(level=level).values

    sst = snap["sea_surface_temperature"].values
    shear_u = lev("u_component_of_wind", 200) - lev("u_component_of_wind", 850)
    shear_v = lev("v_component_of_wind", 200) - lev("v_component_of_wind", 850)
    rh = relative_humidity(lev("specific_humidity", 700), lev("temperature", 700), 700.0)
    lats = snap["latitude"].values
    vo = relative_vorticity(lev("u_component_of_wind", 850), lev("v_component_of_wind", 850), lats) * 1e5
    return np.stack([sst, shear_u, shear_v, rh, vo]).astype(np.float32)


def relative_humidity(q, t_k, p_hpa: float):
    """RH (%) from specific humidity (kg/kg) and temperature (K); ARCO-ERA5 has no RH field."""
    e = q * p_hpa / (0.622 + 0.378 * q)                       # vapour pressure, hPa
    tc = t_k - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))            # Bolton (1980), hPa
    return 100.0 * e / es


def relative_vorticity(u, v, lats):
    """dv/dx - du/dy (1/s) on a regular 0.25 deg patch; ARCO-ERA5 has no vorticity field.

    `lats` are the patch's row latitudes (descending, as in ERA5).
    """
    r = 6.371e6
    d = np.deg2rad(GRID_DEG)
    coslat = np.cos(np.deg2rad(lats))[:, None]
    dv_dx = np.gradient(v, axis=1) / (r * coslat * d)
    du_dy = -np.gradient(u, axis=0) / (r * d)  # rows run north to south
    return dv_dx - du_dy


def build_patches(table: pd.DataFrame, cache: str | Path, ds=None, half_width_deg: float = 5.0,
                  progress: bool = True) -> dict:
    """Patches for every row of `table`, cached in `cache` (.npz). Resumable."""
    cache = Path(cache)
    store = load_patches(cache) if cache.exists() else {}
    todo = [r for r in table.itertuples() if (r.sid, pd.Timestamp(r.time)) not in store]
    if todo:
        ds = ds if ds is not None else open_era5()
        for k, r in enumerate(todo, 1):
            store[(r.sid, pd.Timestamp(r.time))] = extract_patch(ds, r.time, r.lat, r.lon, half_width_deg)
            if progress and k % 100 == 0:
                print(f"  {k}/{len(todo)} patches")
                save_patches(store, cache)
        save_patches(store, cache)
    return store


def save_patches(store: dict, path: str | Path) -> None:
    keys = list(store)
    np.savez_compressed(
        path,
        sid=np.array([k[0] for k in keys]),
        time=np.array([k[1].value for k in keys], dtype=np.int64),
        x=np.stack([store[k] for k in keys]),
    )


def load_patches(path: str | Path) -> dict:
    f = np.load(path)
    times = pd.to_datetime(f["time"])
    return {(s, t): x for s, t, x in zip(f["sid"], times, f["x"], strict=True)}


def patch_summary(store: dict, table: pd.DataFrame, radius_km: float = 500.0) -> pd.DataFrame:
    """Area means of each channel within `radius_km` of the centre, as table columns.

    Lets the gradient-boosting model use the environment too, as scalars
    (e.g. mean SST, shear magnitude) instead of images.
    """
    first = next(iter(store.values()))
    h = first.shape[-1]
    yy, xx = np.mgrid[:h, :h] - h // 2
    disc = np.hypot(yy, xx) * GRID_DEG * 111.0 <= radius_km
    rows = []
    for r in table.itertuples():
        x = store.get((r.sid, pd.Timestamp(r.time)))
        if x is None:
            rows.append([np.nan] * (len(CHANNELS) + 1))
            continue
        means = [np.nanmean(c[disc]) for c in x]
        rows.append(means + [float(np.hypot(means[1], means[2]))])
    cols = [f"env_{c}" for c in CHANNELS] + ["env_shear"]
    return pd.DataFrame(rows, columns=cols, index=table.index)
