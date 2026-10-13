"""Command line: tropcyc build-dataset | era5 | evaluate

    tropcyc build-dataset --basin north_atlantic --years 1982 2024 -o data/table.csv
    tropcyc era5 data/table.csv -o data/patches.npz
    tropcyc evaluate data/table.csv --val-start 2014 --test-start 2019 [--patches data/patches.npz --cnn]
    tropcyc train data/table.csv -o models/live.joblib
    tropcyc serve --model models/live.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__


def _read_table(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def cmd_build(a):
    from .features import build_table
    from .tracks import clean, from_tropycal, load_tropycal

    ds = load_tropycal(basin=a.basin, source=a.source)
    tracks = clean(from_tropycal(ds, tuple(a.years)), include_subtropical=a.subtropical)
    table = build_table(tracks, leads=tuple(a.leads))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.output, index=False)
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

    run(a.model, host=a.host, port=a.port, refresh_minutes=a.refresh)


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
    s.set_defaults(func=cmd_serve)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
