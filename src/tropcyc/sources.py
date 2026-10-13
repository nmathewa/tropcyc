"""Where the storm environment comes from: ERA5, GFS, or an AI model's output.

Every source returns the same 5 channels (era5.CHANNELS) on the same
storm-centred box, by default 1 deg spacing and +-10 deg (21 x 21 points,
~2000 km across), so a model trained on one source can be fine-tuned or run
on another:

    ERA5Source       WeatherBench2 ERA5 at 1 deg, 6-hourly, 1959-2023  -> pre-training
    GFSSource        NOAA GFS on AWS (1p00 or 0p25), 2021-present, in
                     real time                                        -> fine-tuning, live
    NOAAAISource     NOAA's real-time GraphCast-GFS (2024-05/2026) and AIGFS
                     (2026-) on AWS                                    -> AI-model environment
    XarraySource     any gridded dataset with init time and lead time: GraphCast /
                     GenCast output you ran yourself, WeatherNext exports,
                     WeatherBench2 forecast Zarrs

Why 1 deg: the transformer looks at the large-scale environment (shear,
SST, mid-level moisture), which 1 deg resolves. Reading ERA5 at 0.25 deg
costs ~170 MB per time step (each cloud chunk is a whole global field), so
pre-training would stream ~1.7 TB; at 1 deg it is ~35 MB per *day*.

`patch(init, lat, lon, lead_hours)` returns the environment valid at
init + lead_hours. Lead 0 is the analysis; lead 24 is a forecast, which is how
a model sees the *future* environment the storm will move into (the trick the
operational SHIPS model relies on). For reanalysis there is no forecast, so
lead 24 simply means the reanalysis 24 h later ("perfect prognosis").
"""

from __future__ import annotations

import tempfile
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .era5 import relative_humidity, relative_vorticity

OMEGA = 7.2921e-5
DEFAULT_GRID = 1.0
DEFAULT_HALF_WIDTH = 10.0
WB2_ERA5_1DEG = ("gs://weatherbench2/datasets/era5/"
                 "1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr")


def box(lat: float, lon: float, grid: float, half_width: float):
    """Target latitudes (north -> south) and longitudes (0..360) of a storm-centred box.

    The centre snaps to the grid so every source samples the same points.
    """
    n = int(round(half_width / grid))
    clat = round(lat / grid) * grid
    clon = round((lon % 360) / grid) * grid
    lats = np.clip(clat + grid * np.arange(n, -n - 1, -1), -90, 90)
    lons = (clon + grid * np.arange(-n, n + 1)) % 360
    return lats, lons


class _LRU:
    """Tiny thread-safe cache for decoded global fields."""

    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()
        self.key_locks: dict = {}

    def get(self, key, load):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:  # one download per key, even with many threads asking
            with self.lock:
                if key in self.data:
                    return self.data[key]
            value = load()
            with self.lock:
                self.data[key] = value
                while len(self.data) > self.size:
                    self.data.popitem(last=False)
                self.key_locks.pop(key, None)
            return value


class ERA5Source:
    """ERA5 from WeatherBench2 (1 deg, 6-hourly), read one day-chunk at a time."""

    name = "era5"
    VARS = ["sea_surface_temperature", "u_component_of_wind", "v_component_of_wind",
            "relative_humidity", "vorticity"]

    def __init__(self, url: str = WB2_ERA5_1DEG, grid: float = DEFAULT_GRID,
                 half_width_deg: float = DEFAULT_HALF_WIDTH, ds=None):
        self.url, self.grid, self.half_width = url, grid, half_width_deg
        self._ds, self._open_lock = ds, threading.Lock()
        self._blocks = _LRU(6)

    @property
    def ds(self):
        with self._open_lock:
            if self._ds is None:
                import xarray as xr

                self._ds = xr.open_zarr(self.url, storage_options={"token": "anon"})[self.VARS]
        return self._ds

    def _block(self, valid: pd.Timestamp):
        ds = self.ds
        step = ds.chunks["time"][0] if ds.chunks else 1
        i = int(ds.indexes["time"].get_indexer([valid], method="nearest")[0])
        b = i // step

        def load():
            sub = ds.isel(time=slice(b * step, (b + 1) * step)).sel(level=[200, 700, 850])
            return sub.load()

        return self._blocks.get(b, load)

    def patch(self, init, lat, lon, lead_hours: int = 0) -> np.ndarray:
        valid = pd.Timestamp(init) + pd.Timedelta(hours=lead_hours)
        lats, lons = box(lat, lon, self.grid, self.half_width)
        snap = self._block(valid).sel(time=valid, method="nearest")
        snap = snap.sel(latitude=lats, longitude=lons, method="nearest")
        snap = snap.transpose(..., "latitude", "longitude")

        def lev(var, level):
            return snap[var].sel(level=level).values

        rh = lev("relative_humidity", 700)
        if np.nanmax(rh) <= 1.5:  # stored as a fraction in some WeatherBench2 versions
            rh = rh * 100
        return np.stack([
            snap["sea_surface_temperature"].values,
            lev("u_component_of_wind", 200) - lev("u_component_of_wind", 850),
            lev("v_component_of_wind", 200) - lev("v_component_of_wind", 850),
            rh,
            lev("vorticity", 850) * 1e5,
        ]).astype(np.float32)


class GFSSource:
    """GFS GRIB2 from the NOAA open-data bucket on AWS.

    Only the 8 GRIB messages needed are downloaded (HTTP range requests using
    the .idx files): about 1 MB per cycle and lead at 1 deg, 5 MB at 0.25 deg.
    Decoded global fields are cached, so all storms at one cycle share one
    download.
    """

    name = "gfs"
    BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    FIELDS = {  # key: (GRIB variable, level as written in the .idx)
        "t_sfc": ("TMP", "surface"),
        "land": ("LAND", "surface"),
        "u200": ("UGRD", "200 mb"),
        "v200": ("VGRD", "200 mb"),
        "u850": ("UGRD", "850 mb"),
        "v850": ("VGRD", "850 mb"),
        "rh700": ("RH", "700 mb"),
        "absv850": ("ABSV", "850 mb"),
    }
    PRODUCTS = {1.0: "1p00", 0.5: "0p50", 0.25: "0p25"}

    def __init__(self, grid: float = DEFAULT_GRID, half_width_deg: float = DEFAULT_HALF_WIDTH,
                 cache_dir: str | Path | None = None, live: bool = False):
        if grid not in self.PRODUCTS:
            raise ValueError(f"GFS grids: {sorted(self.PRODUCTS)}")
        self.grid, self.half_width, self.live = grid, half_width_deg, live
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "tropcyc-gfs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fields = _LRU(8)

    def _path_tag(self):
        return f"gfs{self.PRODUCTS[self.grid]}"

    def _url(self, init: pd.Timestamp, lead: int) -> str:
        return (f"{self.BUCKET}/gfs.{init:%Y%m%d}/{init:%H}/atmos/"
                f"gfs.t{init:%H}z.pgrb2.{self.PRODUCTS[self.grid]}.f{lead:03d}")

    def _ranges(self, url: str) -> dict:
        with urllib.request.urlopen(url + ".idx", timeout=60) as r:
            rows = [ln.split(":") for ln in r.read().decode().strip().splitlines()]
        out = {}
        for key, (var, level) in self.FIELDS.items():
            for i, row in enumerate(rows):
                if row[3] == var and row[4] == level:
                    end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ""
                    out[key] = (int(row[1]), end)
                    break
            else:
                raise KeyError(f"{var} {level} not in {url}.idx")
        return out

    def fields(self, init, lead: int = 0) -> dict:
        """Global 2-D arrays for one cycle and lead (lat 90..-90, lon 0..360-grid)."""
        init = pd.Timestamp(init)

        def load():
            import xarray as xr

            url = self._url(init, lead)
            out = {}
            for name, (start, end) in self._ranges(url).items():
                path = self.cache_dir / f"{self._path_tag()}_{init:%Y%m%d%H}_f{lead:03d}_{name}.grib2"
                if not path.exists():
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        path.write_bytes(r.read())
                with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                    out[name] = next(iter(ds.data_vars.values())).values.astype(np.float32)
            return out

        return self._fields.get((init, lead), load)

    def _live_fields(self, init, lead_hours):
        """In real time the newest cycle appears ~4 h late: fall back to older cycles."""
        last_err = None
        for back in (0, 6, 12, 18):
            try:
                return self.fields(pd.Timestamp(init) - pd.Timedelta(hours=back), lead_hours + back)
            except Exception as e:  # 404 / missing idx while the cycle is still uploading
                last_err = e
        raise last_err

    def patch(self, init, lat, lon, lead_hours: int = 0) -> np.ndarray:
        f = self._live_fields(init, lead_hours) if self.live else self.fields(init, lead_hours)
        lats, lons = box(lat, lon, self.grid, self.half_width)
        rows = np.rint((90 - lats) / self.grid).astype(int)
        cols = np.rint(lons / self.grid).astype(int) % f["t_sfc"].shape[1]
        b = {k: v[np.ix_(rows, cols)] for k, v in f.items()}
        sst = np.where(b["land"] > 0.5, np.nan, b["t_sfc"])  # skin temp over sea = SST
        coriolis = 2 * OMEGA * np.sin(np.deg2rad(lats))[:, None]
        vo = (b["absv850"] - coriolis) * 1e5  # GFS gives absolute vorticity
        return np.stack([sst, b["u200"] - b["u850"], b["v200"] - b["v850"], b["rh700"], vo]
                        ).astype(np.float32)


class NOAAAISource(GFSSource):
    """NOAA's real-time AI forecasts on AWS: GraphCast-GFS (Feb 2024 - May 2026) and
    AIGFS (Apr 2026 onward), both GraphCast-family models initialised from GFS.

    Same GRIB2 + .idx layout as GFS, so the same range-request download is used.
    These models output winds, specific humidity and temperature but not SST,
    RH or vorticity: RH and vorticity are derived, SST comes from GFS at init.
    Native 0.25 deg fields are block-averaged to the 1 deg box (like ERA5's
    conservative regridding) so all sources see comparable smoothing.
    """

    BUCKET = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
    FIELDS = {
        "u200": ("UGRD", "200 mb"),
        "v200": ("VGRD", "200 mb"),
        "u850": ("UGRD", "850 mb"),
        "v850": ("VGRD", "850 mb"),
        "q700": ("SPFH", "700 mb"),
        "t700": ("TMP", "700 mb"),
    }
    NATIVE = 0.25

    def __init__(self, model: str = "aigfs", grid: float = DEFAULT_GRID,
                 half_width_deg: float = DEFAULT_HALF_WIDTH, cache_dir=None, live: bool = False,
                 sst_source=None):
        if model not in ("aigfs", "graphcastgfs"):
            raise ValueError("model must be 'aigfs' or 'graphcastgfs'")
        self.model = self.name = model
        self.grid, self.half_width, self.live = grid, half_width_deg, live
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "tropcyc-noaa-ai"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fields = _LRU(8)
        self.sst_source = sst_source or GFSSource(grid=grid, half_width_deg=half_width_deg, live=live)

    def _url(self, init: pd.Timestamp, lead: int) -> str:
        # NOAA renamed the files during the archive; use whichever exists for this cycle.
        d, h, f = f"{init:%Y%m%d}", f"{init:%H}", f"{lead:03d}"
        if self.model == "aigfs":
            candidates = [f"aigfs.{d}/{h}/model/atmos/grib2/aigfs.t{h}z.pres.f{f}.grib2"]
        else:
            folder = f"graphcastgfs.{d}/{h}/forecasts_13_levels"
            candidates = [f"{folder}/graphcastgfs.t{h}z.pgrb2.0p25.f{f}",
                          f"{folder}/aigfs.t{h}z.pres.f{f}.grib2"]
        for c in candidates:
            url = f"{self.BUCKET}/{c}"
            try:
                urllib.request.urlopen(urllib.request.Request(url + ".idx", method="HEAD"), timeout=30)
                return url
            except Exception:
                continue
        raise FileNotFoundError(f"no {self.model} forecast for {init:%Y-%m-%d %H}Z f{f}")

    def _path_tag(self):
        return self.model

    def patch(self, init, lat, lon, lead_hours: int = 0) -> np.ndarray:
        f = self._live_fields(init, lead_hours) if self.live else self.fields(init, lead_hours)
        lats, lons = box(lat, lon, self.grid, self.half_width)
        b = {k: _block_mean(v, lats, lons, self.grid) for k, v in f.items()}
        rh = relative_humidity(b["q700"], b["t700"], 700.0)
        vo = relative_vorticity(b["u850"], b["v850"], lats, self.grid) * 1e5
        sst = self.sst_source.patch(init, lat, lon, 0)[0]  # SST barely changes in 24 h
        return np.stack([sst, b["u200"] - b["u850"], b["v200"] - b["v850"], rh, vo]).astype(np.float32)


def _block_mean(field: np.ndarray, lats, lons, grid: float) -> np.ndarray:
    """Average a global (lat 90..-90, lon 0..360) field over grid-sized boxes at (lats, lons)."""
    nlat, nlon = field.shape
    res = 180.0 / (nlat - 1)
    k = max(1, int(round(grid / res)))
    rows = np.rint((90 - lats) / res).astype(int)[:, None] + np.arange(k) - k // 2
    cols = np.rint(lons / res).astype(int)[:, None] + np.arange(k) - k // 2
    rows = np.clip(rows, 0, nlat - 1)
    cols = cols % nlon
    sub = field[rows[:, None, :, None], cols[None, :, None, :]]  # (lat, lon, k, k)
    return np.nanmean(sub, axis=(2, 3))


class XarraySource:
    """Forecasts from any xarray dataset: GraphCast, GenCast, WeatherNext, WeatherBench2.

    Needs dims for init time, lead time, level, latitude and longitude, and the
    variables named in `variables` (defaults follow GraphCast / WeatherBench2
    naming). RH and vorticity are used if present, otherwise derived from
    specific humidity + temperature and from the winds. GraphCast has no SST
    output, so pass `sst_source` (e.g. GFSSource) to take SST at init time.

        ds = xr.open_zarr("graphcast_2022.zarr")       # your GraphCast run
        src = XarraySource(ds, name="graphcast", sst_source=GFSSource())
    """

    DEFAULT_VARIABLES = {
        "sst": "sea_surface_temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "q": "specific_humidity",
        "t": "temperature",
        "rh": "relative_humidity",
        "vo": "vorticity",
    }

    def __init__(self, ds, name: str = "xarray", variables: dict | None = None,
                 init_dim: str = "time", lead_dim: str = "prediction_timedelta",
                 grid: float = DEFAULT_GRID, half_width_deg: float = DEFAULT_HALF_WIDTH, sst_source=None):
        self.ds, self.name, self.sst_source = ds, name, sst_source
        self.var = {**self.DEFAULT_VARIABLES, **(variables or {})}
        self.init_dim, self.lead_dim = init_dim, lead_dim
        self.grid, self.half_width = grid, half_width_deg

    def patch(self, init, lat, lon, lead_hours: int = 0) -> np.ndarray:
        lats, lons = box(lat, lon, self.grid, self.half_width)
        sel = {self.init_dim: pd.Timestamp(init)}
        if self.lead_dim in self.ds.dims:
            sel[self.lead_dim] = pd.Timedelta(hours=lead_hours)
        snap = self.ds.sel(sel, method="nearest").sel(latitude=lats, longitude=lons, method="nearest")
        snap = snap.transpose(..., "latitude", "longitude")

        def lev(key, level):
            return snap[self.var[key]].sel(level=level).values

        if self.var["sst"] in snap:
            sst = snap[self.var["sst"]].values
        elif self.sst_source is not None:  # SST barely changes in 24 h: take it at init time
            sst = self.sst_source.patch(init, lat, lon, 0)[0]
        else:
            sst = np.full((len(lats), len(lons)), np.nan)
        if self.var["rh"] in snap:
            rh = lev("rh", 700)
        else:
            rh = relative_humidity(lev("q", 700), lev("t", 700), 700.0)
        if self.var["vo"] in snap:
            vo = lev("vo", 850) * 1e5
        else:
            vo = relative_vorticity(lev("u", 850), lev("v", 850), lats, self.grid) * 1e5
        return np.stack([sst, lev("u", 200) - lev("u", 850), lev("v", 200) - lev("v", 850), rh, vo]
                        ).astype(np.float32)


def make_source(spec: str, live: bool = False, grid: float = DEFAULT_GRID):
    """'era5', 'gfs', 'aigfs', 'graphcastgfs', or 'xarray:<path or gs:// zarr>'."""
    if spec == "era5":
        return ERA5Source(grid=grid)
    if spec == "gfs":
        return GFSSource(grid=grid, live=live)
    if spec in ("aigfs", "graphcastgfs"):
        return NOAAAISource(spec, grid=grid, live=live)
    if spec.startswith("xarray:"):
        import xarray as xr

        path = spec.split(":", 1)[1]
        opener = xr.open_zarr if path.rstrip("/").endswith(".zarr") else xr.open_dataset
        return XarraySource(opener(path), name=Path(path).stem, grid=grid,
                            sst_source=GFSSource(grid=grid, live=live))
    raise ValueError(f"unknown source {spec!r}")
