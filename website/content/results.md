---
title: Results
subtitle: Held-out skill by season
contents: true
---

## Setup

Tracks are operational TC vitals from all basins, 2021–2026 (537 storms), not the post-season best track, which NOAA's servers did not allow us to download for this run. Every score is **rolling-origin**: for each test season, models are trained only on the seasons before it, and every storm belongs to the season it formed in. The target is the 24 h change in maximum sustained wind.

## Intensity error

| model | 2023 | 2024 | 2025 | 2026 |
|:--|--:|--:|--:|--:|
| persistence | 16.1 | 16.8 | 14.2 | 15.8 |
| damped trend | 15.8 | 16.5 | 14.0 | 15.4 |
| climatology | 15.5 | 16.1 | 13.7 | 15.0 |
| gradient boosting | 13.9 | 13.8 | 12.3 | 12.7 |

Table: Mean absolute error of the 24 h wind change (kt), by test season. {#tbl:mae}

| model | 2023 | 2024 | 2025 | 2026 |
|:--|--:|--:|--:|--:|
| damped trend | +2 % | +2 % | +1 % | +2 % |
| climatology | +4 % | +4 % | +4 % | +5 % |
| gradient boosting | +14 % | +18 % | +13 % | +20 % |

Table: Skill over persistence, 100 (1 − MAE / MAE~persistence~). {#tbl:skill}

::: callout
**Gradient boosting beats persistence in every season**, by 13–20 %, using track predictors alone.
:::

## Rapid intensification

| season | samples | RI events | POD | FAR | CSI |
|:--|--:|--:|--:|--:|--:|
| 2023 | 1190 | 90 | 0.02 | 0.71 | 0.02 |
| 2024 | 1018 | 74 | 0.09 | 0.53 | 0.09 |
| 2025 | 1275 | 74 | 0.11 | 0.69 | 0.09 |
| 2026 | 1225 | 95 | 0.08 | 0.62 | 0.07 |

Table: Gradient boosting as an RI detector (forecast change ≥ 30 kt). POD: share of RI events caught; FAR: share of RI forecasts that were wrong; CSI: critical success index. {#tbl:ri}

::: caveat
**RI is rarely caught.** A model trained on average error forecasts the middle of the distribution, so it seldom predicts +30 kt. The storm transformer, with an RI head and environmental fields, is built to address this; it is not trained yet.
:::
