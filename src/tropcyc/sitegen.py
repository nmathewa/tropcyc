"""Write the website's live-forecast and results pages from real model output.

    tropcyc site-update --model models/live.joblib --scores runs/tcv_rolling.csv

writes website/content/forecast.md (active storms, 24 h forecasts, one chart
each) and website/content/results.md (held-out scores by season). Pages are
regenerated from data every time, so numbers on the site always match a run.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Saffir-Simpson thresholds (kt) drawn as faint reference lines
CATEGORIES = [(34, "TS"), (64, "Cat 1"), (96, "Cat 3")]


def storm_chart(hist: pd.DataFrame, fc_row) -> str:
    """Inline SVG: observed max wind (solid) and the 24 h forecast (dashed), themed via CSS tokens."""
    w, h, left, right, top, bottom = 560, 190, 44, 70, 14, 30
    t0 = hist["time"].min()
    t_end = fc_row.valid_time
    hours = lambda t: (t - t0).total_seconds() / 3600  # noqa: E731
    span = max(hours(t_end), 1)
    vals = list(hist["vmax"]) + [fc_row.vmax_24]
    vmax_axis = max(80, int(np.ceil((max(vals) + 10) / 20) * 20))
    vmin_axis = 0

    def x(t):
        return left + (w - left - right) * hours(t) / span

    def y(v):
        return top + (h - top - bottom) * (1 - (v - vmin_axis) / (vmax_axis - vmin_axis))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" style="width:100%;height:auto;'
             f'font-family:var(--sans);font-size:11px;font-variant-numeric:tabular-nums">',
             f"<title>Maximum wind of {html.escape(_name(fc_row))}: observed and 24 h forecast</title>"]
    for v in range(0, vmax_axis + 1, 20):  # recessive grid + y labels
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke:var(--grid)"/>')
        parts.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" style="fill:var(--muted)">{v}</text>')
    for v, lab in CATEGORIES:
        if v < vmax_axis and abs(v - fc_row.vmax_24) > 0.1 * vmax_axis:  # keep clear of the forecast label
            parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                         f'style="stroke:var(--line);stroke-dasharray:2 3"/>')
            parts.append(f'<text x="{w - right + 6}" y="{y(v) + 4:.1f}" style="fill:var(--muted)">{lab}</text>')
    # x labels: each day at 00 UTC, plus the forecast valid time
    for day in pd.date_range(t0.ceil("D"), t_end, freq="D"):
        parts.append(f'<text x="{x(day):.1f}" y="{h - 10}" text-anchor="middle" style="fill:var(--muted)">'
                     f'{day:%d %b}</text>')
    pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in zip(hist["time"], hist["vmax"], strict=True))
    parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ink);stroke-width:2;'
                 f'stroke-linejoin:round;stroke-linecap:round"/>')
    last_t, last_v = hist["time"].iloc[-1], hist["vmax"].iloc[-1]
    parts.append(f'<line x1="{x(last_t):.1f}" y1="{y(last_v):.1f}" x2="{x(t_end):.1f}" y2="{y(fc_row.vmax_24):.1f}" '
                 f'style="stroke:var(--accent-2);stroke-width:2;stroke-dasharray:5 4"/>')
    for t, v in zip(hist["time"], hist["vmax"], strict=True):  # hover targets bigger than the marks
        parts.append(f'<g><title>{t:%d %b %H} UTC · {v:.0f} kt (observed)</title>'
                     f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="9" style="fill:transparent"/>'
                     f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="2.5" style="fill:var(--ink)"/></g>')
    parts.append(f'<g><title>{t_end:%d %b %H} UTC · {fc_row.vmax_24:.0f} kt (forecast)</title>'
                 f'<circle cx="{x(t_end):.1f}" cy="{y(fc_row.vmax_24):.1f}" r="9" style="fill:transparent"/>'
                 f'<circle cx="{x(t_end):.1f}" cy="{y(fc_row.vmax_24):.1f}" r="4.5" '
                 f'style="fill:var(--bg);stroke:var(--accent-2);stroke-width:2"/></g>')
    parts.append(f'<text x="{x(t_end) + 10:.1f}" y="{y(fc_row.vmax_24) + 4:.1f}" '
                 f'style="fill:var(--ink);font-weight:600">{fc_row.vmax_24:.0f} kt</text>')
    parts.append(f'<text x="{x(last_t) - 6:.1f}" y="{y(last_v) - 14:.1f}" text-anchor="end" '
                 f'style="fill:var(--ink)">{last_v:.0f} kt now</text>')
    parts.append("</svg>")
    return "".join(parts)


NUMBER_WORDS = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN",
                "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN",
                "NINETEEN", "TWENTY", "THIRTY", "FORTY", "FIFTY")


def _name(row) -> str:
    """Storm name; unnamed depressions carry a spelled-out number (often truncated) instead."""
    n = re.sub(r"[^A-Z0-9 \-]", "", str(row.name).upper()).strip()  # outside data: plain text only
    if not n or n.split("-")[0] in NUMBER_WORDS:
        return f"Depression {_sid(row)}"
    return n.title()


def _sid(row) -> str:
    """Short storm id for display: ATCF 'EP202026' -> 'EP20', TC vitals '20E2026' -> '20E'."""
    s = str(row.sid)
    return s[:4] if s[:2].isalpha() else s[:3]


BASIN_NAMES = {"north_atlantic": "North Atlantic", "east_pacific": "East Pacific",
               "west_pacific": "West Pacific", "north_indian": "North Indian",
               "south_indian": "South Indian", "south_pacific": "South Pacific",
               "south_atlantic": "South Atlantic", "australia": "Australian region"}


def _signed(v: float) -> str:
    """+5 / −12 with a true minus sign."""
    s = f"{v:+.0f}".replace("-", "−")
    return "0" if s in ("+0", "−0") else s


def _position(lat, lon) -> str:
    return f"{abs(lat):.1f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.1f}° {'E' if lon >= 0 else 'W'}"


def forecast_page(bundle: dict, tracks: pd.DataFrame, fc: pd.DataFrame, issued: pd.Timestamp,
                  storms: dict | None = None, maps: dict | None = None, sources=()) -> str:
    """`storms`: tropycal storm dicts by id; `maps`: {'overview' | sid: image path relative to the page}."""
    storms, maps = storms or {}, maps or {}
    lines = ["---", "title: Live forecast",
             f"subtitle: 24-hour intensity forecasts for active tropical cyclones, issued {issued:%d %b %Y %H} UTC",
             f"date: {issued:%d %B %Y}", "contents: false", "---", "",
             "::: caveat",
             "**Experimental research output, not an official forecast.** For decisions use the "
             "National Hurricane Center, the Joint Typhoon Warning Center, or your national "
             "weather service.", ":::", ""]
    if "overview" in maps:
        names = ", ".join(sorted(str(d.get("name", k)).title() for k, d in storms.items()))
        lines += ["```{=html}", '<figure id="fig:overview">',
                  f'<img src="{maps["overview"]}" alt="Map of the tracks of all active tropical cyclones: {html.escape(names)}" width="100%">',
                  "<figcaption>Every active tropical cyclone, track coloured by Saffir–Simpson category "
                  "(map drawn with tropycal).</figcaption>", "</figure>", "```", ""]
    if fc.empty:
        lines += ["No active tropical cyclone has 24 h of track yet, so there is no forecast this cycle.", ""]
        return "\n".join(lines)
    basin = {k: BASIN_NAMES.get(str(d.get("basin", "")), "") for k, d in storms.items()}
    if not basin and "basin" in tracks:
        basin = tracks.groupby("sid")["basin"].last().to_dict()
    feeds = " and ".join(sources) if sources else "NHC / JTWC TC vitals"
    model = {"gbm": "gradient boosting"}.get(bundle["name"], bundle["name"])
    lines += [f"Model: **{model}**, from track predictors only (see [Results](results.html)). "
              f"Storm positions and intensities: {feeds}, latest fix {issued:%d %b %H} UTC.", "",
              '::: {style="overflow-x:auto"}',  # 7 columns: scroll inside on phones, never the page
              "| storm | basin | last fix (UTC) | position | now (kt) | 24 h change (kt) | forecast (kt) |",
              "|:--|:--|:--|:--|--:|--:|--:|"]
    for r in fc.sort_values("vmax", ascending=False).itertuples():
        label = _name(r) if _name(r).startswith("Depression") else f"{_name(r)} ({_sid(r)})"
        lines.append(f"| {label} | {basin.get(r.sid, '')} | {r.time:%d %b %H}:00 | "
                     f"{_position(r.lat, r.lon)} | {r.vmax:.0f} | {_signed(r.dv_24)} | **{r.vmax_24:.0f}** |")
    lines += ["", "Table: Active storms and their forecast maximum wind 24 h after the last fix. {#tbl:live}",
              ":::", ""]
    for r in fc.sort_values("vmax", ascending=False).itertuples():
        hist = tracks[(tracks["sid"] == r.sid) & (tracks["time"] <= r.time)].tail(9)
        title = _name(r) if _name(r).startswith("Depression") else f"{_name(r)} ({_sid(r)})"
        storm_map = []
        if r.sid in maps:
            storm_map = ["```{=html}", f'<figure id="fig:map-{r.sid}">',
                         f'<img src="{maps[r.sid]}" alt="Track map of {html.escape(_name(r))}" width="100%">',
                         f"<figcaption>Track of {html.escape(_name(r))} coloured by Saffir–Simpson category; "
                         "the star is the last fix and the label gives the tropcyc 24 h forecast "
                         "(map drawn with tropycal).</figcaption>", "</figure>", "```", ""]
        lines += [f"## {title}", "",
                  f"Last fix {r.time:%d %b %H} UTC at {_position(r.lat, r.lon)}: {r.vmax:.0f} kt"
                  + (f", {r.mslp:.0f} hPa" if pd.notna(r.mslp) else "")
                  + f"; {_signed(r.dvmax_24)} kt over the previous 24 h. "
                  f"Forecast for {r.valid_time:%d %b %H} UTC: **{r.vmax_24:.0f} kt** ({_signed(r.dv_24)} kt).", "",
                  *storm_map,
                  "```{=html}", f'<figure id="fig:{r.sid}">', storm_chart(hist, r),
                  f"<figcaption>Maximum sustained wind of {html.escape(_name(r))}: observed over the last "
                  "48 h (solid) and the 24 h forecast (dashed). Dotted lines mark tropical-storm, "
                  "category 1 and category 3 strength. Hover a point for its value.</figcaption>",
                  "</figure>", "```", ""]
    return "\n".join(lines)


def results_page(scores: pd.DataFrame, n_storms: int, years: str) -> str:
    mae = scores.pivot(index="model", columns="test_year", values="mae")
    skill = scores.pivot(index="model", columns="test_year", values="skill_pct")
    order = [m for m in ["persistence", "trend", "climatology", "gbm"] if m in mae.index]
    names = {"persistence": "persistence", "trend": "damped trend", "climatology": "climatology",
             "gbm": "gradient boosting"}
    yrs = list(mae.columns)
    head = "| model | " + " | ".join(str(y) for y in yrs) + " |"
    sep = "|:--|" + "--:|" * len(yrs)
    rows_mae = [f"| {names[m]} | " + " | ".join(f"{mae.loc[m, y]:.1f}" for y in yrs) + " |" for m in order]
    rows_sk = [f"| {names[m]} | " + " | ".join(f"{_signed(skill.loc[m, y])} %" for y in yrs) + " |"
               for m in order if m != "persistence"]
    g = scores[scores["model"] == "gbm"].set_index("test_year")
    rows_ri = [f"| {y} | {int(g.loc[y, 'n'])} | {int(g.loc[y, 'ri_events'])} | {g.loc[y, 'pod']:.2f} | "
               f"{g.loc[y, 'far']:.2f} | {g.loc[y, 'csi']:.2f} |" for y in yrs]
    best = skill.loc["gbm"]
    return "\n".join([
        "---", "title: Results", "subtitle: Held-out skill by season", "contents: true", "---", "",
        "## Setup", "",
        f"Tracks are operational TC vitals from all basins, {years} ({n_storms} storms), not the "
        "post-season best track, which NOAA's servers did not allow us to download for this run. "
        "Every score is **rolling-origin**: for each test season, models are trained only on the "
        "seasons before it, and every storm belongs to the season it formed in. The target is "
        "the 24 h change in maximum sustained wind.", "",
        "## Intensity error", "",
        head, sep, *rows_mae, "",
        "Table: Mean absolute error of the 24 h wind change (kt), by test season. {#tbl:mae}", "",
        head, sep, *rows_sk, "",
        "Table: Skill over persistence, 100 (1 − MAE / MAE~persistence~). {#tbl:skill}", "",
        "::: callout",
        f"**Gradient boosting beats persistence in every season**, by {best.min():.0f}–{best.max():.0f} %, "
        "using track predictors alone.", ":::", "",
        "## Rapid intensification", "",
        "| season | samples | RI events | POD | FAR | CSI |", "|:--|--:|--:|--:|--:|--:|", *rows_ri, "",
        "Table: Gradient boosting as an RI detector (forecast change ≥ 30 kt). POD: share of RI "
        "events caught; FAR: share of RI forecasts that were wrong; CSI: critical success index. "
        "{#tbl:ri}", "",
        "::: caveat",
        "**RI is rarely caught.** A model trained on average error forecasts the middle of the "
        "distribution, so it seldom predicts +30 kt. The storm transformer, with an RI head and "
        "environmental fields, is built to address this; it is not trained yet.", ":::", ""])


def draw_maps(storms: dict, fc: pd.DataFrame, img_dir) -> dict:
    """tropycal maps for the live page; returns {'overview' | sid: path relative to the page}."""
    from .livemaps import plot_overview, plot_storm

    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):  # storms that are gone should not linger
        old.unlink()
    rel = f"img/{img_dir.name}"
    maps = {}
    if storms:
        plot_overview(storms, img_dir / "overview.png")
        maps["overview"] = f"{rel}/overview.png"
    for r in fc.itertuples():
        if r.sid in storms:
            plot_storm(storms[r.sid], img_dir / f"{r.sid}.png", r)
            maps[r.sid] = f"{rel}/{r.sid}.png"
    return maps


def write_site(bundle, tracks, fc, scores, out_dir, issued, n_storms, years, storms=None, sources=()):
    out = Path(out_dir)
    maps = draw_maps(storms, fc, out / "img" / "live") if storms is not None else {}
    (out / "forecast.md").write_text(forecast_page(bundle, tracks, fc, issued, storms, maps, sources))
    (out / "results.md").write_text(results_page(scores, n_storms, years))
