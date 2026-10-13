import pandas as pd

from tropcyc.tcvitals import parse_line, parse_text, to_tracks

LINES = """\
NHC  20E SIMON     20261012 0600 199N 1061W 320 021 0958 1007 0148 51 009 0074 0093 0074 0074 D
NHC  20E SIMON     20261012 1200 200N 1064W 310 015 0962 1008C0400 49 009 0074 0093 0074 0074 D
JTWC 28W TWENTY-EI:20261012 0000 164N 1508E 020 016 1006 1010 0222 13 064 -999 -999 -999 -999 S
NHC  91L INVEST    20261012 0000 120N 0450W 270 010 1009 1012 0200 13 100 -999 -999 -999 -999 S
"""


def test_fixed_width_fields_that_touch():
    r = parse_line(LINES.splitlines()[1])  # "1008C0400": fields joined by a flag letter
    assert r["vmax"] == round(49 * 1.943844) and r["mslp"] == 962
    assert (r["lat"], r["lon"]) == (20.0, -106.4)
    assert r["time"] == pd.Timestamp("2026-10-12 12:00")


def test_parse_and_tracks():
    recs = parse_text(LINES)
    assert [r["id"] for r in recs] == ["20E", "20E", "28W"]  # invest 91L dropped
    tw = recs[2]
    assert tw["name"] == "TWENTY-EI" and tw["lon"] == 150.8 and tw["lat"] == 16.4
    tracks = to_tracks(recs)
    assert set(tracks["sid"]) == {"20E2026", "28W2026"}
    assert list(tracks[tracks["sid"] == "20E2026"]["type"]) == ["HU", "HU"]


def test_season_crossing_new_year_keeps_one_id():
    a = parse_line("JTWC 05S ALPHA     20251231 1800 150S 0700E 180 010 0980 1005 0200 30 020 -999 -999 -999 -999 D")
    b = parse_line("JTWC 05S ALPHA     20260101 0000 152S 0698E 180 010 0978 1005 0200 32 020 -999 -999 -999 -999 D")
    assert set(to_tracks([a, b])["sid"]) == {"05S2025"}
