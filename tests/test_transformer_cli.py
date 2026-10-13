import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tropcyc import build_table, clean, from_tropycal  # noqa: E402
from tropcyc.cli import main  # noqa: E402
from tropcyc.server import create_app  # noqa: E402
from tropcyc.serving import LiveTransformer  # noqa: E402
from tropcyc.tracks import from_storm_dicts  # noqa: E402
from tropcyc.transformer import TransformerModel, build_env_store, save_store  # noqa: E402

from .test_server import live_storms  # noqa: E402
from .test_transformer import FakeSource  # noqa: E402


def test_pretrain_finetune_cli_then_serve(fake_dataset, tmp_path, capsys):
    tracks = clean(from_tropycal(fake_dataset))
    table = build_table(tracks)
    table.to_csv(tmp_path / "table.csv", index=False)
    tracks.to_csv(tmp_path / "table_tracks.csv", index=False)
    save_store(build_env_store(table, tracks, FakeSource(tracks), progress=False), tmp_path / "era5.npz")
    gfs_like = FakeSource(tracks, bias=1.0)
    save_store(build_env_store(table, tracks, gfs_like, progress=False), tmp_path / "gfs.npz")

    main(["pretrain", str(tmp_path / "table.csv"), "--store", str(tmp_path / "era5.npz"),
          "--years", "2000", "2008", "--epochs", "2", "-o", str(tmp_path / "pre.pt")])
    out = capsys.readouterr().out
    assert "held-out seasons 2007-2008" in out and "transformer" in out and "persistence" in out

    main(["finetune", str(tmp_path / "table.csv"), "--model", str(tmp_path / "pre.pt"),
          "--store", str(tmp_path / "gfs.npz"), "--years", "2009", "2011", "--epochs", "2",
          "-o", str(tmp_path / "ft.pt")])
    assert (tmp_path / "ft.pt").exists()

    # live: the server fetches environment for the active storms from the (fake) source
    model = TransformerModel.load(tmp_path / "ft.pt", verbose=False)
    storms = live_storms()
    bundle = {"model": LiveTransformer(model, FakeSource(clean(from_storm_dicts(storms), min_fixes=1))),
              "name": "transformer", "kind": "sequence"}
    body = TestClient(create_app(bundle, fetch=live_storms)).get("/storms").json()
    s = body["storms"][0]
    assert s["sid"] == "AL142026" and 0 <= s["ri_probability"] <= 1
