---
title: tropcyc
subtitle: Machine-learning forecasts of tropical cyclone intensity change, built on tropycal
date: 13 October 2026
abstract: >-
  tropcyc forecasts how much a tropical cyclone's maximum wind will change over the next
  24 hours, and the chance it rapidly intensifies. Storm tracks come from tropycal; the
  environment around each storm comes from ERA5 for training and from GFS or NOAA's
  GraphCast-based AI models in real time. Models range from persistence baselines to a
  temporal transformer pre-trained on four decades of reanalysis. Every model is scored
  on whole seasons it never saw.
contents: true
---

## What it forecasts

For a storm at time $t$ the target is the change in maximum sustained wind over the next
24 hours,

$$ \Delta v_{24} = v_{\max}(t + 24\,\mathrm{h}) - v_{\max}(t) \quad \text{(kt)}, $$

and *rapid intensification* (RI) is $\Delta v_{24} \ge 30$ kt, the threshold of
@kaplan2003. RI is rare but behind many of the worst forecast busts, so it gets its own
output: a probability, not just a yes or no.

## Why another intensity model

Track forecasts have improved steadily for decades; intensity forecasts much less so. The
operational statistical model SHIPS [@demaria2005] gets much of its skill from the
environment *along the forecast track*: vertical wind shear, ocean temperature and
mid-level moisture. tropcyc keeps that idea and asks two questions:

1. How much does a learned model that sees the environment as images, over the storm's
   recent history, add over tabular predictors?
2. Does the future environment from an AI weather model (GraphCast, [@lam2023]) help
   more than the same field from GFS?

## The models

All models share one `fit` / `predict` interface and are compared on the same seasons.

| model | inputs | runs live? |
|:--|:--|:--:|
| persistence | none: no change | yes |
| damped trend | last 24 h change | yes |
| climatology | mean change for the current intensity class | yes |
| gradient boosting | 14 track predictors | yes |
| CNN | ERA5 patch at $t$ + track predictors | no (ERA5 lags ~5 days) |
| **storm transformer** | 24 h of environment + track, plus the 24 h forecast environment | yes, on GFS / AIGFS |

Table: Models in tropcyc. Track predictors: current intensity and pressure, the 6, 12 and
24 h intensity change, latitude, longitude, storm motion, storm age and day of year.
{#tbl:models}

```{=html}
<figure id="fig:arch">
<svg viewBox="0 0 660 300" role="img" aria-labelledby="arch-title arch-desc" style="width:100%;height:auto;font-family:var(--sans);font-size:12px">
  <title id="arch-title">Storm transformer architecture</title>
  <desc id="arch-desc">Five history tokens from t minus 24 hours to t, each an environment patch plus track features, and one future token with the 24 hour forecast environment, pass with a class token through a transformer encoder to two outputs: the 24 hour wind change and the probability of rapid intensification.</desc>
  <defs>
    <marker id="arw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" style="fill:var(--muted)"/></marker>
  </defs>
  <g style="fill:var(--muted)"><text x="20" y="24">tokens</text><text x="292" y="24">attention over all tokens</text><text x="520" y="24">outputs</text></g>
  <g style="font-family:var(--mono);font-size:11px">
    <g style="fill:var(--panel);stroke:var(--line)">
      <rect x="20" y="40" width="200" height="30" rx="3"/><rect x="20" y="78" width="200" height="30" rx="3"/>
      <rect x="20" y="116" width="200" height="30" rx="3"/><rect x="20" y="154" width="200" height="30" rx="3"/>
      <rect x="20" y="192" width="200" height="30" rx="3"/>
    </g>
    <rect x="20" y="240" width="200" height="30" rx="3" style="fill:var(--panel);stroke:var(--accent-2);stroke-width:1.5"/>
    <g style="fill:var(--ink)">
      <text x="30" y="60">t−24 h  env + track</text><text x="30" y="98">t−18 h  env + track</text>
      <text x="30" y="136">t−12 h  env + track</text><text x="30" y="174">t−6 h   env + track</text>
      <text x="30" y="212">t       env + track</text>
    </g>
    <text x="30" y="260" style="fill:var(--accent-2)">t+24 h  forecast env</text>
  </g>
  <g style="stroke:var(--muted);fill:none" marker-end="url(#arw)">
    <path d="M222,55 L286,140"/><path d="M222,93 L286,145"/><path d="M222,131 L286,150"/>
    <path d="M222,169 L286,155"/><path d="M222,207 L286,160"/><path d="M222,255 L286,168"/>
  </g>
  <rect x="290" y="96" width="190" height="104" rx="4" style="fill:var(--panel);stroke:var(--accent);stroke-width:1.5"/>
  <g style="fill:var(--ink)" text-anchor="middle">
    <text x="385" y="124" style="font-weight:600">transformer encoder</text>
    <text x="385" y="146" style="fill:var(--muted)">[CLS] + 6 tokens</text>
    <text x="385" y="166" style="fill:var(--muted)">masked if missing</text>
    <text x="385" y="186" style="fill:var(--muted)">2–6 layers</text>
  </g>
  <g style="stroke:var(--muted);fill:none" marker-end="url(#arw)"><path d="M482,135 L516,108"/><path d="M482,160 L516,190"/></g>
  <g style="fill:var(--panel);stroke:var(--line)"><rect x="520" y="84" width="124" height="44" rx="3"/><rect x="520" y="170" width="124" height="44" rx="3"/></g>
  <g style="fill:var(--ink)" text-anchor="middle">
    <text x="582" y="104" style="font-weight:600">Δv₂₄ (kt)</text><text x="582" y="120" style="fill:var(--muted)">Huber loss</text>
    <text x="582" y="190" style="font-weight:600">P(RI)</text><text x="582" y="206" style="fill:var(--muted)">weighted BCE</text>
  </g>
</svg>
<figcaption>Figure 1. The storm transformer. Each history token is an environment patch (SST,
200–850 hPa shear, 700 hPa humidity, 850 hPa vorticity; 1°, ±10° around the storm) encoded by
a small CNN, plus that fix's track predictors. The future token (vermillion) is the 24 h
forecast environment where the storm is heading, which is SHIPS's key ingredient.</figcaption>
</figure>
```

[Figure 1](#fig:arch) shows the transformer [@vaswani2017]. It is pre-trained on ERA5
[@hersbach2020] for 1980–2020, where the "forecast" environment is simply the reanalysis
24 h later. It is then fine-tuned on GFS from 2021 onward and run live on GFS, optionally
with the future environment from NOAA's AIGFS. The [Method](method.html) page has the
details.

## Results

The track-only models are trained, on operational TC vitals from all basins
(2021–2026). Scored season by season on storms they never saw, gradient boosting beats
persistence in every season, by 13–20 % in mean absolute error. It rarely catches rapid
intensification. The [Results](results.html) page has the full tables, and the
[Live forecast](forecast.html) page runs the model on today's storms.

::: caveat
**The storm transformer is not trained yet.** It needs the long ERA5 record of best-track
storms for pre-training. Its results will be added here when that run is done.
:::

## Data

| source | role | period |
|:--|:--|:--|
| HURDAT2 [@landsea2013] / IBTrACS [@knapp2010], via tropycal [@burg2020] | tracks, targets (transformer) | 1980– |
| TC vitals (NHC / JTWC operational messages, GFS bucket on AWS) | tracks for the live model, live storm feed | 2021– |
| ERA5, WeatherBench2 1° copy [@hersbach2020; @rasp2024] | pre-training environment | 1980–2020 |
| GFS 1°, NOAA open data on AWS | fine-tuning and live environment | 2021– |
| GraphCast-GFS, AIGFS, NOAA open data on AWS | AI-model future environment | 2024– |

Table: Data sources. All are public and read with anonymous access. {#tbl:data}

::: note
tropcyc is experimental research software. Its output is not an official forecast; for
decisions, use the National Hurricane Center or your national weather service.
:::

## References {#references}

::: {#refs}
:::
