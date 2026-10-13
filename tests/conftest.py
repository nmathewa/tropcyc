import numpy as np
import pandas as pd
import pytest


class FakeTrackDataset:
    """Stands in for tropycal.tracks.TrackDataset: only `.data` is used."""

    def __init__(self, data):
        self.data = data


def make_storm(sid, year, n=24, seed=0, start_vmax=25.0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(f"{year}-08-01")
    time = [t0 + pd.Timedelta(hours=6 * i) for i in range(n)]
    vmax = np.clip(start_vmax + np.cumsum(rng.normal(2, 6, n)), 20, 160).round(0)
    return {
        "id": sid, "name": f"STORM{sid[-4:]}", "year": year, "time": time,
        "lat": list(np.linspace(12, 28, n)), "lon": list(np.linspace(-40, -75, n)),
        "vmax": list(vmax), "mslp": list(1010 - 0.8 * vmax),
        "type": ["TD"] + ["TS"] * (n - 2) + ["EX"],
    }


@pytest.fixture
def fake_dataset():
    data = {}
    for k, year in enumerate(range(2000, 2012)):
        for j in range(4):
            sid = f"AL{j:02d}{year}"
            data[sid] = make_storm(sid, year, seed=100 * k + j)
    return FakeTrackDataset(data)
