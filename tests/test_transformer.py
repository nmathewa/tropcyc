import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tropcyc import build_table, clean, from_tropycal, split_by_year  # noqa: E402
from tropcyc.evaluate import scores  # noqa: E402
from tropcyc.transformer import (  # noqa: E402
    FUTURE_LEAD,
    TransformerModel,
    build_env_store,
    env_key,
    future_positions,
    load_store,
    save_store,
)


class FakeSource:
    """Warm 'SST' ahead of the storm when it is about to intensify; `bias` mimics another model's climate."""

    def __init__(self, tracks, bias=0.0, size=17):
        self.v = tracks.set_index(["sid", "time"])["vmax"]
        self.bias, self.size, self.calls = bias, size, 0

    def patch(self, init, lat, lon, lead_hours=0):
        self.calls += 1
        rng = np.random.default_rng(abs(hash((round(lat, 2), round(lon, 2), lead_hours))) % 2**32)
        x = rng.normal(size=(5, self.size, self.size)).astype(np.float32)
        x[0] += 300 + self.bias
        x[0, 0, 0] = np.nan
        return x


@pytest.fixture
def data(fake_dataset):
    tracks = clean(from_tropycal(fake_dataset))
    return tracks, build_table(tracks)


def test_store_has_history_and_future(data, tmp_path):
    tracks, table = data
    rows = table.head(10)
    src = FakeSource(tracks)
    store = build_env_store(rows, tracks, src, cache=tmp_path / "s.npz", progress=False)
    r = rows.iloc[-1]
    assert env_key(r.sid, r.time, 0) in store
    assert env_key(r.sid, r.time - pd.Timedelta(hours=24), 0) in store
    assert env_key(r.sid, r.time, FUTURE_LEAD) in store
    # resumable: a second build with the cache downloads nothing
    src2 = FakeSource(tracks)
    build_env_store(rows, tracks, src2, cache=tmp_path / "s.npz", progress=False)
    assert src2.calls == 0
    assert set(load_store(tmp_path / "s.npz")) == set(store)


def test_future_position_extrapolates_live(data):
    tracks, _ = data
    last = tracks.groupby("sid").tail(1)
    known = future_positions(tracks).merge(last[["sid", "time"]])
    assert known["flat"].isna().all()  # nothing after the final fix
    est = future_positions(tracks, extrapolate=True).merge(last[["sid", "time"]])
    assert np.isfinite(est["flat"]).all()
    assert (est["flat"].to_numpy() > last["lat"].to_numpy()).all()  # synthetic storms move north


def test_pretrain_finetune_predict(data, tmp_path):
    tracks, table = data
    train, val, _ = split_by_year(table, 2008, 2010)
    store = build_env_store(table, tracks, FakeSource(tracks), progress=False)
    m = TransformerModel(store, tracks, epochs=3, batch_size=32, verbose=False).fit(train)
    pred = m.predict(val)
    assert pred.shape == (len(val),) and np.isfinite(pred).all()
    assert scores(val["dv_24"], pred)["mae"] < 60
    m.save(tmp_path / "pre.pt")

    # fine-tune on a "GFS-like" source with a different SST climate
    store2 = build_env_store(table, tracks, FakeSource(tracks, bias=1.5), progress=False)
    ft = TransformerModel.load(tmp_path / "pre.pt", store=store2, tracks=tracks, epochs=2, verbose=False)
    before = ft.predict(val)
    ft.fit(train, finetune=True)
    assert len(ft.history_) > len(m.history_)  # pre-training history kept
    p = ft.predict_ri_probability(val)
    assert ((p >= 0) & (p <= 1)).all() and not np.allclose(before, ft.predict(val))


def test_missing_history_is_masked(data):
    tracks, table = data
    store = build_env_store(table.head(5), tracks, FakeSource(tracks), progress=False)
    first = table.groupby("sid").head(1).head(3)  # earliest samples: the oldest fixes are missing
    for k in [k for k in store if k[2] == 0][:3]:
        store.pop(k)
    m = TransformerModel(store, tracks, epochs=1, verbose=False).fit(table.head(5))
    assert np.isfinite(m.predict(first)).all()


def test_store_roundtrip(tmp_path):
    store = {env_key("AL01", "2020-01-01", 0): np.ones((5, 3, 3), np.float32),
             env_key("AL01", "2020-01-01", 24): np.zeros((5, 3, 3), np.float32)}
    save_store(store, tmp_path / "x.npz")
    assert set(load_store(tmp_path / "x.npz")) == set(store)


def test_future_source_used_for_future_token(data):
    tracks, table = data
    hist, fut = FakeSource(tracks), FakeSource(tracks, bias=50)
    store = build_env_store(table.head(4), tracks, hist, future_source=fut, progress=False)
    leads = {k[2]: v for k, v in store.items()}
    assert np.nanmean(leads[FUTURE_LEAD][0]) > 340 and np.nanmean(leads[0][0]) < 310
    assert fut.calls == len([k for k in store if k[2] == FUTURE_LEAD])


def test_checkpoint_is_weights_only(data, tmp_path):
    tracks, table = data
    store = build_env_store(table.head(8), tracks, FakeSource(tracks), progress=False)
    TransformerModel(store, tracks, epochs=1, verbose=False).fit(table.head(8)).save(tmp_path / "m.pt")
    ckpt = torch.load(tmp_path / "m.pt", weights_only=True)  # raises if anything but tensors/plain data
    assert set(ckpt) == {"state", "stats", "size", "target", "history"}
