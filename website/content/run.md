---
title: Run it
subtitle: Install, train, and serve live forecasts
date: 13 October 2026
contents: true
---

## Install

```bash
git clone --recurse-submodules https://github.com/nmathewa/tropcyc && cd tropcyc
pip install -e ".[all]"       # tracks, models, ERA5/GFS access, PyTorch, server
pytest -q
```

Everything runs on a CPU. A GPU is used automatically if PyTorch finds one.

## Train everything

```bash
scripts/train_all.sh                           # tracks via tropycal (needs NOAA access)
HURDAT=data/hurdat2.txt scripts/train_all.sh   # or a HURDAT2 file you downloaded
SIZE=base WORKERS=16 scripts/train_all.sh      # bigger model, more parallel downloads
```

The script builds the track table, scores the baselines and gradient boosting, downloads
the ERA5 environment for 1980–2020, pre-trains the transformer, downloads GFS for 2021
onward, and fine-tunes. Every step is cached and resumable. The ERA5 download takes a
few hours; training takes minutes.

## Step by step

| step | command |
|:--|:--|
| tracks → table | `tropcyc build-dataset --years 1980 2025 -o data/table.csv` |
| baselines | `tropcyc evaluate data/table.csv --val-start 2014 --test-start 2019` |
| ERA5 environment | `tropcyc env-store data/table.csv --source era5 --years 1980 2020 -o data/env_era5.npz` |
| pre-train | `tropcyc pretrain data/table.csv --store data/env_era5.npz --years 1980 2020 -o models/tf_era5.pt` |
| GFS environment | `tropcyc env-store data/table.csv --source gfs --years 2021 2025 -o data/env_gfs.npz` |
| fine-tune | `tropcyc finetune data/table.csv --model models/tf_era5.pt --store data/env_gfs.npz --years 2021 2025 -o models/tf_gfs.pt` |

Table: The pipeline, one command per step. {#tbl:steps}

To compare GFS with an AI model for the future environment, build a second GFS store
with `--future-source graphcastgfs` (2024–2025) and fine-tune from the same pre-trained
checkpoint.

## Live forecasts

```bash
tropcyc serve --model models/tf_gfs.pt --source gfs --future-source aigfs
# or the track-only model:
tropcyc train data/table.csv -o models/live.joblib && tropcyc serve --model models/live.joblib
```

The server reads the active storms from tropycal's realtime feed (NHC) at most every
30 minutes and keeps serving the last good forecast if the feed is down.

| endpoint | returns |
|:--|:--|
| `GET /` | a page listing active storms with their 24 h forecast |
| `GET /storms`, `GET /storms/{id}` | the same as JSON, including the RI probability |
| `POST /predict` | a forecast for any track you send (6-hourly fixes, at least 24 h) |
| `GET /model` | which model is loaded and how it scored |
| `GET /docs` | interactive API documentation |

Table: Server endpoints. {#tbl:api}

## Build this website

```bash
cd website && make serve       # needs pandoc ≥ 3.1; http://localhost:8010
```

The pages are Markdown in `website/content/`, built with the
[academic-site-template](https://github.com/nmathewa/academic-site-template), which is
included as the `website/theme` submodule.
