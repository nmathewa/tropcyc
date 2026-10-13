import numpy as np
import pandas as pd

from tropcyc import (
    Climatology,
    GradientBoosting,
    Persistence,
    Trend,
    build_table,
    clean,
    compare,
    from_tropycal,
    scores,
    split_by_year,
)
from tropcyc.evaluate import ri_scores
from tropcyc.features import TRACK_FEATURES


def test_from_tropycal_and_clean(fake_dataset):
    tracks = from_tropycal(fake_dataset)
    assert tracks["sid"].nunique() == 48
    cleaned = clean(tracks)
    assert set(cleaned["type"]) <= {"TD", "TS", "HU"}  # EX fix dropped
    assert (cleaned["time"].dt.hour % 6 == 0).all()


def test_year_range(fake_dataset):
    tracks = from_tropycal(fake_dataset, year_range=(2005, 2006))
    assert sorted(tracks["time"].dt.year.unique()) == [2005, 2006]


def test_targets_use_time_not_rows(fake_dataset):
    tracks = clean(from_tropycal(fake_dataset))
    # remove one fix in the middle of a storm: the 24 h target must skip, not shift
    sid = tracks["sid"].iloc[0]
    gap_time = tracks[tracks["sid"] == sid]["time"].iloc[10]
    gapped = tracks[~((tracks["sid"] == sid) & (tracks["time"] == gap_time))]
    table = build_table(gapped)
    s = gapped[gapped["sid"] == sid].set_index("time")["vmax"]
    for r in table[table["sid"] == sid].itertuples():
        assert r.dv_24 == s[r.time + pd.Timedelta(hours=24)] - s[r.time]
        assert r.time + pd.Timedelta(hours=24) != gap_time


def test_features_present_and_finite(fake_dataset):
    table = build_table(clean(from_tropycal(fake_dataset)))
    assert set(TRACK_FEATURES) <= set(table.columns)
    assert np.isfinite(table[[c for c in TRACK_FEATURES if c != "mslp"]].to_numpy()).all()
    # westward, northward motion in the synthetic tracks
    assert (table["u_motion"] < 0).all() and (table["v_motion"] > 0).all()


def test_split_is_by_season(fake_dataset):
    table = build_table(clean(from_tropycal(fake_dataset)))
    train, val, test = split_by_year(table, 2008, 2010)
    assert not set(train["sid"]) & set(val["sid"])
    assert not set(val["sid"]) & set(test["sid"])
    assert train["time"].max() < val["time"].min()


def test_models_and_compare(fake_dataset):
    table = build_table(clean(from_tropycal(fake_dataset)))
    train, _, test = split_by_year(table, 2008, 2010)
    models = [Persistence(), Trend(), Climatology(), GradientBoosting(n_estimators=50)]
    result = compare(models, train, test)
    assert list(result.index) == ["persistence", "trend", "climatology", "gbm"]
    assert abs(result.loc["persistence", "skill_pct"]) < 1e-9
    assert (result["mae"] > 0).all()


def test_scores():
    s = scores([0, 10, 20], [0, 10, 30], baseline_pred=[0, 0, 0])
    assert s["mae"] == 10 / 3 and s["bias"] == 10 / 3
    assert np.isclose(s["skill_pct"], 100 * (1 - (10 / 3) / 10))
    r = ri_scores([35, 0, 40, 5], [31, 32, 0, 0])
    assert (r["pod"], r["far"]) == (0.5, 0.5)
