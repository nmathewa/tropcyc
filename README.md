# tropcyc

[![tests](https://github.com/nmathewa/tropcyc/actions/workflows/tests.yml/badge.svg)](https://github.com/nmathewa/tropcyc/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Machine-learning forecasts of tropical cyclone **intensity change**, built on
[tropycal](https://tropycal.github.io/tropycal/).

tropycal handles the best-track data (HURDAT2, IBTrACS) and the maps. tropcyc adds
everything after that:

- **predictors** from the track itself (current intensity, recent trend, motion, season)
  and from **ERA5** around the storm (SST, deep-layer shear, mid-level humidity, low-level
  vorticity), read directly from the public ARCO-ERA5 cloud store with no bulk download;
- **models** behind one `fit` / `predict` interface: persistence, damped trend and
  climatology baselines, gradient boosting, and a CNN on ERA5 patches (PyTorch);
- **verification that doesn't leak**: splits by whole season, MAE/RMSE/bias, skill against
  persistence, and rapid-intensification POD/FAR/CSI.

The target is the change in maximum sustained wind over the next 24 h (`dv_24`, kt). Rapid
intensification is `dv_24 >= 30` kt.

## Install

```bash
pip install tropcyc                 # tracks, features, baselines, gradient boosting
pip install "tropcyc[all]"          # + ERA5 access, the PyTorch CNN, plotting
```

From source:

```bash
git clone https://github.com/nmathewa/tropcyc && cd tropcyc
pip install -e ".[dev]"
pytest -q
```

## Quick start

```bash
# 1. best tracks -> feature table (tropycal downloads HURDAT2 the first time)
tropcyc build-dataset --basin north_atlantic --years 1982 2024 -o data/table.csv

# 2. compare models on the validation seasons (2014-2018)
tropcyc evaluate data/table.csv --val-start 2014 --test-start 2019

# 3. add the ERA5 environment: patches first (slow once, then cached), then the CNN
tropcyc era5 data/table.csv -o data/patches.npz
tropcyc evaluate data/table.csv --patches data/patches.npz --cnn

# 4. once the design is settled, score once on the held-out test seasons
tropcyc evaluate data/table.csv --patches data/patches.npz --cnn --final
```

In Python:

```python
import tropcyc as tc

ds = tc.load_tropycal("north_atlantic")                 # a tropycal TrackDataset
tracks = tc.clean(tc.from_tropycal(ds, (1982, 2024)))
table = tc.build_table(tracks)

train, val, test = tc.split_by_year(table, val_start=2014, test_start=2019)
print(tc.compare([tc.Persistence(), tc.Trend(), tc.Climatology(), tc.GradientBoosting()],
                 train, val))

ds.get_storm(("ian", 2022)).plot()                      # maps stay with tropycal
```

## Layout

```
src/tropcyc/
  tracks.py      tropycal TrackDataset -> one tidy table; tropical, 6-hourly fixes only
  features.py    track predictors and dv/RI targets (looked up by time, so gaps are safe)
  era5.py        storm-centred ERA5 patches from ARCO-ERA5, cached to .npz
  models.py      Persistence, Trend, Climatology, GradientBoosting
  cnn.py         IntensityNet (PyTorch): ERA5 patch + track features -> dv_24
  evaluate.py    season splits, scores, RI scores, compare()
  plotting.py    forecast-vs-best-track and scatter plots
  cli.py         tropcyc build-dataset | era5 | evaluate
docs/
  tensorflow_to_pytorch.md   the CNN explained for Keras users
```

## Notes

- **Data access.** tropycal downloads from NOAA (NHC for HURDAT2, NCEI for IBTrACS).
  ERA5 comes from `gs://gcp-public-data-arco-era5` with anonymous access.
- **Earlier work.** The 2023–24 Atlantic experiments (TensorFlow CNN/RNN, ERA5
  preprocessing, Manim animations) are preserved on the
  [`legacy-2024`](https://github.com/nmathewa/tropcyc/tree/legacy-2024) branch.

## License

MIT. See [LICENSE](LICENSE).
