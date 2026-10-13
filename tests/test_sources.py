import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from tropcyc.sources import GFSSource, XarraySource, box, make_source  # noqa: E402


def test_box_snaps_and_wraps():
    lats, lons = box(22.4, -179.6, 1.0, 10.0)
    assert len(lats) == len(lons) == 21
    assert lats[0] == 32 and lats[-1] == 12  # north -> south, centre snapped to 22
    assert lons[10] == 180 and lons[0] == 170 and lons[-1] == 190 % 360  # wraps the date line


class ConstSST:
    def patch(self, init, lat, lon, lead_hours=0):
        return np.full((5, 21, 21), 301.0, np.float32)


def graphcast_like():
    """GraphCast-style output: lat south->north, init 'time' + 'prediction_timedelta', no SST."""
    lat = np.arange(0, 40.25, 0.25)
    lon = np.arange(260, 300.25, 0.25)
    lev = [200, 700, 850]
    shape = (1, 2, 3, len(lat), len(lon))
    coords = {"time": [pd.Timestamp("2022-09-27")], "prediction_timedelta": pd.to_timedelta([0, 24], "h"),
              "level": lev, "latitude": lat, "longitude": lon}
    dims = list(coords)
    u = np.zeros(shape, np.float32)
    u[:, :, 0] = 20.0  # 20 m/s westerlies at 200 hPa only
    return xr.Dataset({
        "u_component_of_wind": (dims, u),
        "v_component_of_wind": (dims, np.zeros(shape, np.float32)),
        "specific_humidity": (dims, np.full(shape, 0.008, np.float32)),
        "temperature": (dims, np.full(shape, 283.15, np.float32)),
    }, coords=coords)


def test_xarray_source_graphcast_layout():
    src = XarraySource(graphcast_like(), name="graphcast", sst_source=ConstSST())
    x = src.patch("2022-09-27", 22.4, -83.6, lead_hours=24)
    assert x.shape == (5, 21, 21)
    assert np.allclose(x[0], 301.0)          # SST borrowed from the other source
    assert np.allclose(x[1], 20.0)           # shear u = u200 - u850
    assert np.allclose(x[4], 0.0, atol=1e-6)  # uniform wind -> no vorticity
    assert 50 < x[3].mean() < 100            # RH derived from q and T


def test_make_source():
    assert make_source("gfs", live=True).live
    with pytest.raises(ValueError):
        make_source("nope")


def test_gfs_live_falls_back_to_older_cycle(monkeypatch):
    src = GFSSource(live=True)
    calls = []

    def fields(init, lead):
        calls.append((pd.Timestamp(init), lead))
        if len(calls) < 2:
            raise OSError("404: cycle not uploaded yet")
        return {k: np.zeros((181, 360), np.float32) for k in GFSSource.FIELDS}

    monkeypatch.setattr(src, "fields", fields)
    src.patch("2026-10-12 12:00", 20, -60, 24)
    assert calls == [(pd.Timestamp("2026-10-12 12:00"), 24), (pd.Timestamp("2026-10-12 06:00"), 30)]


def test_block_mean_averages_to_coarser_grid():
    from tropcyc.sources import _block_mean

    field = np.zeros((721, 1440), np.float32)
    field[:, ::2] = 2.0  # alternating 0/2 columns average to 1 over each 1 deg box
    lats, lons = box(20, -60, 1.0, 3.0)
    assert np.allclose(_block_mean(field, lats, lons, 1.0), 1.0)


def test_noaa_ai_source_derives_missing_fields(monkeypatch):
    from tropcyc.sources import NOAAAISource

    src = NOAAAISource("aigfs", sst_source=ConstSST())
    es = 6.112 * np.exp(17.67 * 10 / (10 + 243.5))
    q_half = 0.5 * 0.622 * es / (700 - 0.378 * es)  # ~50 % RH at 10 C
    fields = {"u200": 15.0, "v200": 0.0, "u850": 5.0, "v850": 0.0, "q700": q_half, "t700": 283.15}
    monkeypatch.setattr(src, "fields", lambda init, lead: {
        k: np.full((721, 1440), v, np.float32) for k, v in fields.items()})
    x = src.patch("2026-10-12", 15, -45, 24)
    assert x.shape == (5, 21, 21)
    assert np.allclose(x[0], 301.0) and np.allclose(x[1], 10.0) and np.allclose(x[2], 0.0)
    assert np.allclose(x[3], 50.0, atol=1.0) and np.allclose(x[4], 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        NOAAAISource("nope")
