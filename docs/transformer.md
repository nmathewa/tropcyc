# The storm transformer: pre-train on ERA5, fine-tune on GFS, run live

This is the model behind `tropcyc pretrain`, `tropcyc finetune` and
`tropcyc serve --model *.pt`. It forecasts the 24 h change in maximum wind and
the probability of rapid intensification (+30 kt in 24 h).

## Why this design

ERA5 is the best record of the past (1959 onward) but arrives about 5 days late,
so a model trained only on ERA5 can't run on today's storms. Operational models
(GFS, and NOAA's GraphCast-based AI models) are available in real time but have
only a few years of archive. So:

1. **Pre-train on ERA5, 1980–2020.** About 40 seasons, so the model learns how
   shear, SST and moisture drive intensity change.
2. **Fine-tune on GFS, 2021 onward.** A few seasons at a 10× lower learning rate,
   enough to adjust to GFS's slightly different fields (its vortex is stronger
   than ERA5's, for example).
3. **Run live on GFS**, optionally taking the future environment from an AI model
   (AIGFS).

## Inputs

For a storm at time `t` the model reads 6 tokens, each with 5 environment
channels on a storm-centred box (1° spacing, ±10°, 21×21 points, about 2,000 km
across):

| token | environment | track features |
|---|---|---|
| t−24, t−18, t−12, t−6, t | analysis at that time and position | intensity, its 6/12/24 h change, motion, latitude, season… |
| future | 24 h forecast valid at t+24, at the storm's position then | — |

Channels: SST, 200–850 hPa shear (u and v), 700 hPa relative humidity, 850 hPa
relative vorticity.

The **future token** is what forecasters call the environment along the
forecast track. SHIPS, the operational statistical model, gets a large share of
its skill from it.
- **Training on ERA5:** "the forecast" is ERA5 at t+24, at the storm's real
  position then (known as "perfect prognosis").
- **Live:** it's the GFS (or AIGFS) 24 h forecast at a position extrapolated
  from the storm's recent motion.

Missing tokens are masked, for example a storm that is younger than 24 h or a
missing GFS cycle.

## Network

```
each token:  patch (5×21×21) → 3 strided convs → vector (d_model)
           + Linear(track features) + learned position embedding
[CLS] + 6 tokens → TransformerEncoder (pre-norm, GELU) → CLS output
                 → dv_24 head  (Huber loss, δ = 10 kt)
                 → RI head     (BCE, positive class up-weighted by its rarity)
```

| `--size` | parameters | CPU (4 threads), per epoch of 25k samples |
|---|---|---|
| small | 0.13 M | ~20 s |
| base  | 0.9 M  | ~30 s |
| large | 5 M    | ~80 s |

Training is cheap on any laptop. **Downloading the environment is the slow part**
(see below), and it is cached and resumable.

## Environment sources

| `--source` | what | years | notes |
|---|---|---|---|
| `era5` | ERA5 from WeatherBench2, 1°, 6-hourly | 1959–2023 | read one day-chunk (~35 MB) at a time, shared by all storms that day |
| `gfs` | NOAA GFS 1° (`pgrb2.1p00`) on AWS | 2021–now | only the 8 needed GRIB messages are downloaded (~1 MB per cycle) |
| `graphcastgfs` | NOAA's GraphCast run from GFS, on AWS | Feb 2024 – May 2026 | 0.25° averaged to 1°; RH and vorticity derived; SST from GFS |
| `aigfs` | NOAA's operational AI GFS (GraphCast family) | Apr 2026–now | same handling as `graphcastgfs` |
| `xarray:<path>` | any Zarr/NetCDF with init + lead time | — | your own GraphCast/GenCast runs, WeatherNext exports, WeatherBench2 forecasts |

All sources return the same channels on the same grid. Checked on real data at
the same place and time, ERA5, GFS, GraphCast-GFS and AIGFS agree within about
0.5 K for SST, 1 m/s for shear and a few % for RH. Vorticity differs most, which
is what the fine-tuning absorbs.

Why 1° and not 0.25°: every cloud chunk of ERA5 at 0.25° is a whole global field
(about 170 MB per time step), so pre-training would stream about 1.7 TB. At 1°
it is about 150 GB for 1980–2020 Atlantic storms, read once. The model looks at
the large-scale environment, which 1° resolves. The inner core is summarised by
the track features instead (intensity and its trend).

## Step by step

```bash
pip install -e ".[all]"

# 1. tracks + targets (writes data/table.csv and data/table_tracks.csv)
tropcyc build-dataset --years 1980 2025 -o data/table.csv

# 2. ERA5 environment for pre-training (the long download; resumable, cached)
tropcyc env-store data/table.csv --source era5 --years 1980 2020 -o data/env_era5.npz

# 3. pre-train; prints held-out skill vs persistence and gradient boosting
tropcyc pretrain data/table.csv --store data/env_era5.npz --years 1980 2020 \
    --size base -o models/tf_era5.pt

# 4. GFS environment for 2021+ and fine-tune
tropcyc env-store data/table.csv --source gfs --years 2021 2025 -o data/env_gfs.npz
tropcyc finetune data/table.csv --model models/tf_era5.pt --store data/env_gfs.npz \
    --years 2021 2025 -o models/tf_gfs.pt

# 5. live: today's storms, GFS history + GFS (or AIGFS) future
tropcyc serve --model models/tf_gfs.pt --source gfs                      # GFS only
tropcyc serve --model models/tf_gfs.pt --source gfs --future-source aigfs
```

To compare GFS with an AI model for the future environment, build a second
fine-tuning store with `--future-source graphcastgfs` (2024–2025) and fine-tune
from the same pre-trained checkpoint. Then compare the held-out scores the two
runs print.

## Honest caveats

- **The GFS archive is short.** 2021 onward is about 5 Atlantic seasons and about
  100 storms. Fine-tuning helps, but the held-out test is only the last 1–2
  seasons, so expect noisy scores. Look at several seasons before trusting a
  difference of 1–2 kt.
- **The future token is optimistic during pre-training.** In ERA5 pre-training the
  "future" is the truth at the real future position. Live, it is a forecast at an
  extrapolated position. Fine-tuning on GFS f024 at real positions closes part of
  that gap. Extrapolated positions are the remaining part (NHC's forecast track
  would be better, and is an easy next step).
- **Your RI probability needs calibrating** before you quote it. Check a
  reliability diagram on held-out seasons first.
- **This is not an official forecast.** For decisions, use the National Hurricane
  Center or your national weather service.
