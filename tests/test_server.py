import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tropcyc import build_table, clean, from_tropycal  # noqa: E402
from tropcyc.server import create_app  # noqa: E402
from tropcyc.serving import load_bundle, save_bundle, select_and_train  # noqa: E402

from .conftest import make_storm  # noqa: E402


@pytest.fixture
def bundle(fake_dataset, tmp_path):
    table = build_table(clean(from_tropycal(fake_dataset)))
    b = select_and_train(table, val_start=2008, test_start=2010)
    save_bundle(b, tmp_path / "live.joblib")
    return load_bundle(tmp_path / "live.joblib")


def live_storms():
    long = make_storm("AL142026", 2026, n=12, seed=7)
    long["type"] = ["TS"] * 12  # still tropical at the latest fix
    short = make_storm("EP032026", 2026, n=3, seed=8)  # under 24 h of history
    return {"AL142026": long, "EP032026": short}


def test_select_and_train(bundle):
    assert bundle["name"] in {"persistence", "trend", "climatology", "gbm"}
    v = bundle["validation"]
    assert bundle["name"] == min(v, key=lambda k: v[k]["mae"])


def test_storms_endpoint(bundle):
    client = TestClient(create_app(bundle, fetch=live_storms))
    body = client.get("/storms").json()
    sids = [s["sid"] for s in body["storms"]]
    assert sids == ["AL142026"]  # short-history storm skipped
    s = body["storms"][0]
    assert s["vmax_24"] == max(15, round(s["vmax"] + s["dv_24"]))
    assert client.get("/storms/al142026").status_code == 200
    assert client.get("/storms/EP032026").status_code == 404
    assert "AL142026" in client.get("/").text
    assert client.get("/model").json()["name"] == bundle["name"]


def test_feed_failure(bundle):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("NHC down")
        return live_storms()

    client = TestClient(create_app(bundle, fetch=flaky, refresh_minutes=0))
    assert client.get("/storms").status_code == 200
    r = client.get("/storms")  # refetch fails -> last good result is served
    assert r.status_code == 200 and r.json()["storms"]
    assert "NHC down" in client.get("/health").json()["feed_error"]

    def down():
        raise ConnectionError("NHC down")

    cold = TestClient(create_app(bundle, fetch=down))
    assert cold.get("/storms").status_code == 503
    assert "Feed problem" in cold.get("/").text


def test_predict(bundle):
    client = TestClient(create_app(bundle, fetch=live_storms))
    s = live_storms()["AL142026"]
    fixes = [{"time": str(t), "lat": la, "lon": lo, "vmax": v}
             for t, la, lo, v in zip(s["time"], s["lat"], s["lon"], s["vmax"], strict=True)]
    r = client.post("/predict", json={"name": "test", "fixes": fixes})
    assert r.status_code == 200 and "dv_24" in r.json()
    assert client.post("/predict", json={"fixes": fixes[:2]}).status_code == 422
