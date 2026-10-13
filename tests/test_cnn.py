import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tropcyc import build_table, clean, from_tropycal, split_by_year  # noqa: E402
from tropcyc.cnn import CNNModel  # noqa: E402
from tropcyc.era5 import load_patches, patch_summary, save_patches  # noqa: E402


def fake_store(table, size=17, seed=0):
    rng = np.random.default_rng(seed)
    store = {}
    for r in table.itertuples():
        x = rng.normal(size=(5, size, size)).astype(np.float32)
        x[0] += 300 + r.dv_24 / 10  # make SST weakly informative
        x[0, 0, 0] = np.nan  # a land point
        store[(r.sid, pd.Timestamp(r.time))] = x
    return store


def test_cnn_trains_and_predicts(fake_dataset, tmp_path):
    table = build_table(clean(from_tropycal(fake_dataset)))
    store = fake_store(table)
    train, _, test = split_by_year(table, 2008, 2010)
    model = CNNModel(store, epochs=2, batch_size=32, verbose=False).fit(train)
    pred = model.predict(test)
    assert pred.shape == (len(test),) and np.isfinite(pred).all()
    model.save(tmp_path / "cnn.pt")


def test_patch_roundtrip_and_summary(fake_dataset, tmp_path):
    table = build_table(clean(from_tropycal(fake_dataset))).head(20)
    store = fake_store(table)
    save_patches(store, tmp_path / "p.npz")
    back = load_patches(tmp_path / "p.npz")
    assert set(back) == set(store)
    summary = patch_summary(back, table)
    assert summary.shape == (20, 6) and summary["env_sst"].between(290, 320).all()


def test_derived_era5_fields():
    from tropcyc.era5 import relative_humidity, relative_vorticity

    # saturated air at 700 hPa, 10 C: RH ~ 100 %
    t = np.array(283.15)
    es = 6.112 * np.exp(17.67 * 10 / (10 + 243.5))
    q_sat = 0.622 * es / (700 - 0.378 * es)
    assert np.isclose(relative_humidity(q_sat, t, 700.0), 100.0, atol=0.5)

    # solid-body rotation u = -w*y, v = w*x has vorticity 2w everywhere
    lats = np.arange(20, 10 - 0.25, -0.25)            # descending, like ERA5
    lons = np.arange(-90, -80 + 0.25, 0.25)
    lon2, lat2 = np.meshgrid(lons, lats)
    r = 6.371e6
    x = np.deg2rad(lon2 + 85) * r * np.cos(np.deg2rad(lat2))
    y = np.deg2rad(lat2 - 15) * r
    w = 5e-5
    vo = relative_vorticity(-w * y, w * x, lats)
    assert np.allclose(vo[5:-5, 5:-5], 2 * w, rtol=0.05)
