"""HTTP server: 24 h intensity forecasts for the storms active right now.

    tropcyc train data/table.csv -o models/live.joblib
    tropcyc serve --model models/live.joblib            # http://localhost:8000

Endpoints
    GET  /                 small HTML page with the current forecasts
    GET  /storms           forecasts for every active storm (JSON)
    GET  /storms/{sid}     one storm
    POST /predict          forecast from a track you send
    GET  /model            which model is loaded and how it scored
    GET  /health

Active storms come from tropycal's realtime feed (NHC best tracks). The feed
is fetched at most once per `refresh_minutes`; between fetches the cached
result is served, so the server never hammers NOAA.
"""

from __future__ import annotations

import html
import math
import threading
import time
from collections.abc import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .serving import fetch_active_storms, forecast, forecast_active
from .tracks import from_storm_dicts


class Fix(BaseModel):
    time: str = Field(examples=["2024-10-07T12:00:00"])
    lat: float
    lon: float = Field(description="degrees east; west is negative")
    vmax: float = Field(description="max sustained wind, kt")
    mslp: float | None = None
    type: str = "TS"


class Track(BaseModel):
    name: str = "storm"
    fixes: list[Fix] = Field(description="6-hourly fixes, at least the last 24 h")


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in ("time", "valid_time"):
        if col in out:
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%MZ")
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict(orient="records")


def _json_safe(obj):
    """NaN/inf -> None, recursively (scores like FAR are NaN when nothing is forecast)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def create_app(bundle: dict, fetch: Callable[[], dict] = fetch_active_storms,
               refresh_minutes: float = 30) -> FastAPI:
    app = FastAPI(title="tropcyc", summary="ML tropical cyclone intensity forecasts")
    cache = {"at": 0.0, "forecasts": None, "error": None}
    lock = threading.Lock()

    def current() -> pd.DataFrame:
        with lock:
            stale = time.time() - cache["at"] > refresh_minutes * 60
            if stale or cache["forecasts"] is None:
                try:
                    cache["forecasts"] = forecast_active(bundle, fetch())
                    cache["error"] = None
                except Exception as e:  # keep serving the last good result
                    cache["error"] = f"{type(e).__name__}: {e}"
                    if cache["forecasts"] is None:
                        raise HTTPException(503, f"realtime feed unavailable: {cache['error']}") from e
                cache["at"] = time.time()
            return cache["forecasts"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model": bundle["name"], "feed_error": cache["error"]}

    @app.get("/model")
    def model_info():
        return _json_safe({k: v for k, v in bundle.items() if k != "model"})

    @app.get("/storms")
    def storms():
        fc = current()
        updated = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(cache["at"]))
        return {"model": bundle["name"], "updated": updated, "storms": _records(fc)}

    @app.get("/storms/{sid}")
    def storm(sid: str):
        fc = current()
        row = fc[fc["sid"].str.upper() == sid.upper()]
        if row.empty:
            raise HTTPException(404, f"{sid} is not active, or has under 24 h of tropical track")
        return _records(row)[0]

    @app.post("/predict")
    def predict(track: Track):
        f = [x.model_dump() for x in track.fixes]
        storm_dict = {k: [x[k] for x in f] for k in ("time", "lat", "lon", "vmax", "mslp", "type")}
        storm_dict["mslp"] = [float("nan") if v is None else v for v in storm_dict["mslp"]]
        storm_dict["name"] = track.name
        fc = forecast(bundle, from_storm_dicts({"USER": storm_dict}))
        if fc.empty:
            raise HTTPException(422, "need tropical 6-hourly fixes covering at least the last 24 h")
        return _records(fc)[0]

    @app.get("/", response_class=HTMLResponse)
    def index():
        try:
            fc, err = current(), cache["error"]
        except HTTPException as e:
            fc, err = pd.DataFrame(), e.detail
        rows = "".join(
            f"<tr><td>{html.escape(str(r['name']))}</td><td>{r['sid']}</td><td>{r['time']}</td>"
            f"<td>{r['vmax']:.0f}</td><td>{r['dv_24']:+.0f}</td><td><b>{r['vmax_24']:.0f}</b></td>"
            f"<td>{'yes' if r['rapid_intensification'] else ''}</td></tr>"
            for r in _records(fc)
        ) or '<tr><td colspan="7">No active storms with 24 h of tropical track.</td></tr>'
        note = f'<p class="err">Feed problem: {html.escape(str(err))}</p>' if err else ""
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>tropcyc live</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--mute:#666;--rule:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mute:#999;--rule:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:860px;
margin:2rem auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 8px;border-bottom:1px solid var(--rule);
text-align:left}}th{{color:var(--mute);font-weight:500}}.err{{color:#c33}}p.m{{color:var(--mute)}}
</style></head><body><h1>Active tropical cyclones: 24 h intensity forecast</h1>
<p class="m">Model: {bundle['name']}. Winds in kt. Experimental research output, not an official
forecast; see the National Hurricane Center or your national weather service.</p>{note}
<div style="overflow-x:auto"><table><tr><th>Name</th><th>ID</th><th>Last fix (UTC)</th><th>Now</th>
<th>Change</th><th>In 24 h</th><th>Rapid int.</th></tr>{rows}</table></div>
<p class="m"><a href="/docs">API</a> · <a href="/model">model details</a></p></body></html>"""

    return app


def run(model_path: str, host: str = "0.0.0.0", port: int = 8000, refresh_minutes: float = 30,
        source: str = "gfs", future_source: str | None = None):
    import uvicorn

    from .serving import load_bundle

    bundle = load_bundle(model_path, source=source, future_source=future_source)
    uvicorn.run(create_app(bundle, refresh_minutes=refresh_minutes), host=host, port=port)
