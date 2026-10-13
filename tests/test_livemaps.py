import pandas as pd

from tropcyc.livemaps import _to_tropycal_dict, ace


def test_ace_counts_only_synoptic_tropical_fixes_at_ts_strength():
    times = pd.date_range("2026-10-10", periods=4, freq="6h").tolist() + [pd.Timestamp("2026-10-11 03:00")]
    expected = round((50**2 + 100**2) * 1e-4, 1)  # TD below 34 kt, EX and the 03 UTC fix skipped
    assert ace(times, [30, 50, 100, 100, 100], ["TD", "TS", "HU", "EX", "HU"]) == expected


def test_tcvitals_rows_become_tropycal_storm():
    g = pd.DataFrame({"sid": "20E2026", "name": "SIMON",
                      "time": pd.date_range("2026-10-10", periods=3, freq="6h"),
                      "lat": [16.0, 16.5, 17.0], "lon": [-104.0, -104.5, -105.0], "vmax": [60, 70, 80],
                      "mslp": [990.0, 985.0, 980.0], "type": ["TS", "HU", "HU"]})
    d = _to_tropycal_dict(g)
    assert d["id"] == "EP202026" and d["basin"] == "east_pacific" and len(d["time"]) == 3
    assert d["ace"] == round((60**2 + 70**2 + 80**2) * 1e-4, 1)
