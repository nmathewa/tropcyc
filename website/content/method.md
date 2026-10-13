---
title: Method
subtitle: Tracks, predictors, environment sources, models and verification
date: 13 October 2026
contents: true
---

## Tracks and targets

Best tracks come from HURDAT2 [@landsea2013] or IBTrACS [@knapp2010], read with tropycal
[@burg2020]. tropcyc flattens them into one table, one row per storm per fix, and keeps
only:

- tropical fixes (TD, TS, HU; subtropical optionally), since extratropical transition
  changes intensity for different reasons;
- synoptic times (00, 06, 12, 18 UTC), dropping the extra landfall and peak records, so
  every step is exactly 6 h.

Lags and leads are looked up **by time, not by row**. A track with a gap, for example a
storm that is briefly extratropical, never pairs two fixes that are not really 6, 12 or
24 h apart. The target is $\Delta v_{24}$ (kt); RI is $\Delta v_{24} \ge 30$ kt
[@kaplan2003].

## Track predictors

Current maximum wind and central pressure; the 6, 12 and 24 h change in maximum wind;
latitude (signed and absolute) and longitude; eastward and northward storm motion and
speed; hours since genesis; day of year as sine and cosine. These feed the
gradient-boosting model directly and enter every history token of the transformer.

## Environment

Each environment patch has five channels on a storm-centred box of 21×21 points (1°
spacing, ±10°, about 2,000 km across):

| channel | meaning | why |
|:--|:--|:--|
| SST | sea-surface temperature (K) | energy supply |
| shear u, v | 200 minus 850 hPa wind (m s⁻¹) | shear ventilates and tilts the vortex |
| RH 700 | 700 hPa relative humidity (%) | dry air entrainment |
| ζ 850 | 850 hPa relative vorticity (10⁻⁵ s⁻¹) | low-level spin-up |

Table: Environment channels. {#tbl:channels}

**Why 1°.** Every cloud chunk of 0.25° ERA5 is a whole global field, about 170 MB per
time step, so pre-training at 0.25° would stream about 1.7 TB. The 1° WeatherBench2 copy
[@rasp2024] is read one day at a time (about 35 MB) and shared by every storm active that
day: about 150 GB in all. The model needs the large-scale environment, which 1° resolves.
The inner core is summarised by the current intensity and its recent change.

### One grid, several sources

| source | used for | how |
|:--|:--|:--|
| ERA5 (WeatherBench2, 1°, 6-hourly) | pre-training | day-chunks, cached |
| GFS 1° (`pgrb2.1p00`) | fine-tuning, live | HTTP range requests for the 8 GRIB messages needed (~1 MB per cycle) |
| GraphCast-GFS (Feb 2024 – May 2026), AIGFS (Apr 2026–) | AI-model future environment | 0.25° averaged to 1°; RH and ζ derived from q, T and winds; SST from GFS |
| any xarray dataset | your own GraphCast / GenCast / WeatherNext runs | variable mapping |

Table: Environment sources. All return the same channels on the same points. {#tbl:sources}

The sources agree closely where they overlap. Box means at the same place and time:

| case | source | SST | shear u | shear v | RH 700 |
|:--|:--|--:|--:|--:|--:|
| A · analysis | ERA5 | 302.6 | 6.5 | 4.5 | 61.7 |
| A · analysis | GFS | 302.2 | 6.2 | 4.3 | 63.4 |
| B · 24 h forecast | GraphCast-GFS | 302.0 | 8.8 | 6.2 | 48.3 |
| B · 24 h forecast | GFS | 301.9 | 8.8 | 5.9 | 51.7 |
| C · 24 h forecast | AIGFS | 301.5 | 15.0 | −4.0 | 54.6 |
| C · 24 h forecast | GFS | 301.4 | 14.1 | −3.4 | 55.5 |

Table: Cross-source check, 21×21 box means (K, m s⁻¹, %). A: Hurricane Ian, 22.4° N 83.6° W,
27 Sep 2022 06 UTC. B: Gulf of Mexico, 22.7° N 90.6° W, from 8 Oct 2024 00 UTC.
C: tropical Atlantic, 15° N 45° W, from 12 Oct 2026 00 UTC. {#tbl:agree}

Vorticity is where they differ. At Ian's centre GFS has a stronger vortex than ERA5
(121 vs 79 × 10⁻⁵ s⁻¹ at 1°). That kind of systematic difference is what fine-tuning on
GFS is for.

## The storm transformer

A sample is one storm at one fix time $t$. Its sequence has a class token, five history
tokens at $t-24, t-18, t-12, t-6, t$ h, and one future token. Each history token is the
sum of (i) the environment patch passed through three strided convolutions to a
$d$-dimensional vector, (ii) a linear projection of that fix's track predictors, and
(iii) a learned position embedding. The future token is the environment valid at $t+24$
h, at the storm's position then, with no track predictors.

| setting | where it comes from |
|:--|:--|
| future token, pre-training | ERA5 at $t+24$ at the real future position ("perfect prognosis") |
| future token, fine-tuning | GFS 24 h forecast from $t$, at the real future position |
| future token, live | GFS or AIGFS 24 h forecast, at a position extrapolated from recent motion |

Table: What the future token sees in each phase. {#tbl:future}

A pre-norm transformer encoder [@vaswani2017] attends over the tokens. Missing tokens
(a young storm, a missing forecast cycle) are masked. The class output feeds two heads:

$$ \begin{aligned}
\mathcal{L} &= \mathrm{Huber}_{\delta = 10\,\mathrm{kt}}\big(\widehat{\Delta v}_{24},\, \Delta v_{24}\big) \\
&\quad + \lambda\, \mathrm{BCE}_{w}\big(\hat p_\mathrm{RI},\, \mathbb{1}[\Delta v_{24} \ge 30]\big),
\end{aligned} $$

with $\lambda = 0.5$ and the RI class up-weighted by its rarity ($w = (1-r)/r$, with $r$
the training RI rate). Training uses AdamW with cosine decay and early stopping on the
most recent 15 % of *storms*, not rows. Fine-tuning continues from the pre-trained
weights at a tenth of the learning rate.

| size | $d$ | layers | heads | parameters | CPU time per epoch, 25k samples |
|:--|--:|--:|--:|--:|--:|
| small | 64 | 2 | 4 | 0.13 M | ~20 s |
| base | 128 | 4 | 8 | 0.9 M | ~30 s |
| large | 256 | 6 | 8 | 5 M | ~80 s |

Table: Model sizes. Training is cheap; downloading the environment is the slow step. {#tbl:sizes}

## Verification

Consecutive 6-hourly fixes of one storm are nearly identical, so a random split would let
a model memorise storms and look far better than it is. tropcyc splits by **season**:
training years come before validation years, which come before test years. Each storm
belongs to the season it formed in. The test seasons are scored once, at the end.

Scores: mean absolute error, RMSE and bias of $\Delta v_{24}$; MAE skill over persistence;
and for RI, the probability of detection, false-alarm ratio and critical success index.

## Limitations

::: caveat
- **Short GFS archive.** 2021 onward is about five Atlantic seasons. Held-out scores on
  one or two seasons are noisy; a 1–2 kt difference needs several seasons to trust.
- **Optimistic pre-training.** The ERA5 future token is the truth at the real future
  position. Live, it is a forecast at an extrapolated position. NHC's official forecast
  track would be a better position and is a planned improvement.
- **Uncalibrated RI probability.** Check a reliability diagram on held-out seasons
  before quoting it.
- **No inner core.** At 1° the patches see the environment, not the eyewall.
:::

## References {#references}

::: {#refs}
:::
