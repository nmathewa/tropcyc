"""Command line: tropcyc build-dataset | era5 | evaluate | train | serve | env-store | pretrain | finetune

    tropcyc build-dataset --basin north_atlantic --years 1982 2024 -o data/table.csv
    tropcyc era5 data/table.csv -o data/patches.npz
    tropcyc evaluate data/table.csv --val-start 2014 --test-start 2019 [--patches data/patches.npz --cnn]
    tropcyc train data/table.csv -o models/live.joblib
    tropcyc serve --model models/live.joblib

Transformer (pre-train on ERA5, fine-tune on GFS, run live on GFS):
    tropcyc env-store data/table.csv --source era5 --years 1980 2020 -o data/env_era5.npz
    tropcyc pretrain data/table.csv --store data/env_era5.npz --years 1980 2020 -o models/tf_era5.pt
    tropcyc env-store data/table.csv --source gfs --years 2021 2025 -o data/env_gfs.npz
    tropcyc finetune data/table.csv --model models/tf_era5.pt --store data/env_gfs.npz \
        --years 2021 2025 -o models/tf_gfs.pt
    tropcyc serve --model models/tf_gfs.pt --source gfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__


def _read_table(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def _tracks_path(table_path) -> Path:
    p = Path(table_path)
    return p.with_name(p.stem + "_tracks.csv")


def _years(df: pd.DataFrame, years) -> pd.DataFrame:
    if not years:
        return df.reset_index(drop=True)
    y = df.groupby("sid")["time"].transform("min").dt.year
    return df[(y >= years[0]) & (y <= years[1])].reset_index(drop=True)


def cmd_build(a):
    from .features import build_table
    from .tracks import clean, from_tropycal, load_tropycal

    ds = load_tropycal(basin=a.basin, source=a.source)
    tracks = clean(from_tropycal(ds, tuple(a.years)), include_subtropical=a.subtropical)
    table = build_table(tracks, leads=tuple(a.leads))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False)
    tracks.to_csv(_tracks_path(a.output), index=False)  # full tracks: the transformer's history tokens
    print(f"{len(table)} samples from {table['sid'].nunique()} storms -> {a.output}")


def cmd_era5(a):
    from .era5 import build_patches

    table = _read_table(a.table)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    store = build_patches(table, a.output, half_width_deg=a.half_width)
    print(f"{len(store)} patches -> {a.output}")


def cmd_evaluate(a):
    from .evaluate import compare, split_by_year
    from .features import TRACK_FEATURES
    from .models import Climatology, GradientBoosting, Persistence, Trend

    table = _read_table(a.table)
    target = f"dv_{a.lead}"
    models = [Persistence(target), Trend(target), Climatology(target), GradientBoosting(target)]

    store = None
    if a.patches:
        from .era5 import load_patches, patch_summary

        store = load_patches(a.patches)
        keys = set(store)
        has_patch = [(s, t) in keys for s, t in zip(table["sid"], table["time"], strict=True)]
        table = table[has_patch].reset_index(drop=True)
        table = pd.concat([table, patch_summary(store, table)], axis=1)
        env = [c for c in table.columns if c.startswith("env_")]
        gbm_env = GradientBoosting(target, features=list(TRACK_FEATURES) + env)
        gbm_env.name = "gbm+env"
        models.append(gbm_env)
        if a.cnn:
            from .cnn import CNNModel

            models.append(CNNModel(store, target, epochs=a.epochs))

    train, val, test = split_by_year(table, a.val_start, a.test_start)
    if a.final:  # report on test, training on train+val
        train, holdout = pd.concat([train, val], ignore_index=True), test
    else:
        holdout = val
    print(f"train {len(train)}  {'test' if a.final else 'val'} {len(holdout)} samples")
    result = compare(models, train, holdout, target=target)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(result)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(a.output)


def cmd_train(a):
    from .serving import save_bundle, select_and_train

    table = _read_table(a.table)
    bundle = select_and_train(table, a.val_start, a.test_start, target=f"dv_{a.lead}")
    save_bundle(bundle, a.output)
    print(pd.DataFrame(bundle["validation"]).T[["mae", "rmse", "skill_pct"]].round(2))
    print(f"best on {a.val_start}-{a.test_start - 1}: {bundle['name']}, refit on all data -> {a.output}")


def cmd_serve(a):
    from .server import run

    run(a.model, host=a.host, port=a.port, refresh_minutes=a.refresh, source=a.source,
        future_source=a.future_source, feed=a.feed)


def cmd_env_store(a):
    from .sources import make_source
    from .transformer import build_env_store

    table = _years(_read_table(a.table), a.years)
    tracks = _read_table(a.tracks or _tracks_path(a.table))
    future = make_source(a.future_source) if a.future_source else None
    print(f"{len(table)} samples: history from {a.source}, future from {a.future_source or a.source} ...")
    store = build_env_store(table, tracks, make_source(a.source), cache=a.output, workers=a.workers,
                            future_source=future)
    print(f"{len(store)} patches -> {a.output}")


def _transformer_run(a, finetune: bool):
    from .evaluate import compare
    from .models import GradientBoosting, Persistence
    from .transformer import TransformerModel, load_store

    table = _years(_read_table(a.table), a.years)
    tracks = _read_table(a.tracks or _tracks_path(a.table))
    store = load_store(a.store)
    table = table[[(str(s), t, 0) in store for s, t in zip(table["sid"], table["time"], strict=True)]]
    table = table.reset_index(drop=True)
    kw = {"epochs": a.epochs, "batch_size": a.batch_size, "device": a.device}
    if finetune:
        model = TransformerModel.load(a.model, store=store, tracks=tracks, **kw)
    else:
        model = TransformerModel(store, tracks, size=a.size, lr=a.lr, **kw)

    # hold out the last season(s) to report skill, then refit nothing: early stopping
    # inside fit already uses the most recent storms of the training part.
    year = table.groupby("sid")["time"].transform("min").dt.year
    test_start = year.max() - a.test_years + 1
    train, test = table[year < test_start], table[year >= test_start]
    model.fit(train.reset_index(drop=True), finetune=finetune)
    model.save(a.output)
    print(f"saved -> {a.output}")
    if len(test):
        print(f"\nheld-out seasons {test_start}-{year.max()} ({len(test)} samples):")
        result = compare([Persistence(), GradientBoosting(), _Fitted(model)], train, test)
        print(result[["n", "mae", "rmse", "skill_pct", "ri_events", "pod", "far"]].round(2))


class _Fitted:
    """Wrap an already-trained model so compare() does not retrain it."""

    def __init__(self, model):
        self.model, self.name = model, model.name

    def fit(self, table):
        return self

    def predict(self, table):
        return self.model.predict(table)


def cmd_site_update(a):
    from .serving import forecast, load_bundle
    from .sitegen import write_site
    from .tcvitals import fetch_recent

    bundle = load_bundle(a.model)
    tracks = fetch_recent(48)
    fc = forecast(bundle, tracks)
    scores = pd.read_csv(a.scores)
    hist = _read_table(a.tracks)
    years = f"{hist['time'].min():%Y}–{hist['time'].max():%Y}"
    issued = tracks["time"].max() if len(tracks) else pd.Timestamp.utcnow().tz_localize(None).floor("6h")
    write_site(bundle, tracks, fc, scores, a.out, issued, hist["sid"].nunique(), years)
    print(f"{len(fc)} storm forecasts, issued {issued:%Y-%m-%d %H} UTC -> {a.out}/forecast.md, results.md")


def cmd_tcvitals(a):
    from .features import build_table
    from .tcvitals import download, load_archive
    from .tracks import clean

    n = download(a.start, a.end or pd.Timestamp.utcnow().tz_localize(None), a.folder)
    tracks = clean(load_archive(a.folder))
    table = build_table(tracks)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False)
    tracks.to_csv(_tracks_path(a.output), index=False)
    print(f"{n} cycles; {tracks['sid'].nunique()} storms; {len(table)} samples -> {a.output}")


def cmd_pretrain(a):
    _transformer_run(a, finetune=False)


def cmd_finetune(a):
    _transformer_run(a, finetune=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tropcyc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"tropcyc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset", help="best tracks (via tropycal) -> feature table")
    b.add_argument("--basin", default="north_atlantic")
    b.add_argument("--source", default="hurdat", choices=["hurdat", "ibtracs"])
    b.add_argument("--years", nargs=2, type=int, default=[1982, 2024])
    b.add_argument("--leads", nargs="+", type=int, default=[24])
    b.add_argument("--subtropical", action="store_true", help="keep SD/SS fixes")
    b.add_argument("-o", "--output", default="data/table.csv")
    b.set_defaults(func=cmd_build)

    e = sub.add_parser("era5", help="storm-centred ERA5 patches for a feature table")
    e.add_argument("table")
    e.add_argument("--half-width", type=float, default=5.0, help="degrees")
    e.add_argument("-o", "--output", default="data/patches.npz")
    e.set_defaults(func=cmd_era5)

    v = sub.add_parser("evaluate", help="train models and compare them")
    v.add_argument("table")
    v.add_argument("--lead", type=int, default=24)
    v.add_argument("--val-start", type=int, default=2014)
    v.add_argument("--test-start", type=int, default=2019)
    v.add_argument("--final", action="store_true", help="score on the test years (do this last)")
    v.add_argument("--patches", help="ERA5 patches .npz; adds gbm+env")
    v.add_argument("--cnn", action="store_true", help="also train the CNN (needs --patches)")
    v.add_argument("--epochs", type=int, default=40)
    v.add_argument("-o", "--output")
    v.set_defaults(func=cmd_evaluate)

    t = sub.add_parser("train", help="pick the best live-capable model, refit on all data, save it")
    t.add_argument("table")
    t.add_argument("--lead", type=int, default=24)
    t.add_argument("--val-start", type=int, default=2014)
    t.add_argument("--test-start", type=int, default=2019)
    t.add_argument("-o", "--output", default="models/live.joblib")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("serve", help="HTTP server with live forecasts for active storms")
    s.add_argument("--model", default="models/live.joblib")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--refresh", type=float, default=30, help="minutes between realtime fetches")
    s.add_argument("--source", default="gfs", help="history environment for a transformer .pt model")
    s.add_argument("--future-source", help="future-token source, e.g. aigfs (default: --source)")
    s.add_argument("--feed", default="tcvitals", choices=["tcvitals", "nhc"],
                   help="active storms: TC vitals from the GFS bucket, or NHC via tropycal")
    s.set_defaults(func=cmd_serve)

    tv = sub.add_parser("tcvitals", help="track table from operational TC vitals (GFS bucket, 2021-)")
    tv.add_argument("--start", default="2021-01-01")
    tv.add_argument("--end")
    tv.add_argument("--folder", default="data/tcvitals")
    tv.add_argument("-o", "--output", default="data/table.csv")
    tv.set_defaults(func=cmd_tcvitals)

    su = sub.add_parser("site-update", help="write the website's live forecast and results pages")
    su.add_argument("--model", default="models/live.joblib")
    su.add_argument("--scores", default="runs/tcv_rolling.csv")
    su.add_argument("--tracks", default="data/table_tracks.csv")
    su.add_argument("--out", default="website/content")
    su.set_defaults(func=cmd_site_update)

    es = sub.add_parser("env-store", help="environment patches (history + 24 h future) for the transformer")
    es.add_argument("table")
    es.add_argument("--tracks", help="default: <table>_tracks.csv written by build-dataset")
    es.add_argument("--source", default="era5",
                    help="era5 | gfs | aigfs | graphcastgfs | xarray:<path.zarr>")
    es.add_argument("--future-source", help="source for the 24 h future token (default: --source)")
    es.add_argument("--years", nargs=2, type=int)
    es.add_argument("--workers", type=int, default=16, help="parallel downloads")
    es.add_argument("-o", "--output", default="data/env_era5.npz")
    es.set_defaults(func=cmd_env_store)

    for name, fn, helptext in (("pretrain", cmd_pretrain, "train the transformer from scratch"),
                               ("finetune", cmd_finetune, "continue training a transformer on a new source")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("table")
        q.add_argument("--tracks")
        q.add_argument("--store", required=True)
        q.add_argument("--years", nargs=2, type=int)
        q.add_argument("--test-years", type=int, default=2, help="last N seasons held out for scoring")
        q.add_argument("--epochs", type=int, default=40)
        q.add_argument("--batch-size", type=int, default=64)
        q.add_argument("--device", help="cpu | cuda | mps (default: cuda if available)")
        q.add_argument("-o", "--output", required=True)
        if name == "pretrain":
            q.add_argument("--size", default="small", choices=["small", "base", "large"])
            q.add_argument("--lr", type=float, default=3e-4)
        else:
            q.add_argument("--model", required=True, help="pre-trained .pt")
        q.set_defaults(func=fn)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
