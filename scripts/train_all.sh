#!/usr/bin/env bash
# End-to-end training: tracks -> baselines -> ERA5 pre-training -> GFS fine-tuning.
# Every step is cached/resumable: re-running skips work already done.
#
#   scripts/train_all.sh                         # tracks via tropycal (needs NOAA access)
#   HURDAT=data/hurdat2.txt scripts/train_all.sh # or a local HURDAT2 file
#   SIZE=base WORKERS=16 scripts/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${SIZE:-small}            # small | base | large
WORKERS=${WORKERS:-16}         # parallel environment downloads
PRE_YEARS=${PRE_YEARS:-"1980 2020"}
FT_YEARS=${FT_YEARS:-"2021 2025"}
mkdir -p data models runs

log() { printf '\n== %s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [[ ! -f data/table.csv ]]; then
  log "1/6 best tracks -> data/table.csv"
  if [[ -n "${HURDAT:-}" ]]; then
    python - "$HURDAT" <<'PY'
import sys
from pathlib import Path
from tropycal import tracks
import tropcyc as tc
ds = tracks.TrackDataset(basin="north_atlantic", source="hurdat", atlantic_url=sys.argv[1], include_btk=False)
t = tc.clean(tc.from_tropycal(ds, (1980, 2030)))
tc.build_table(t).to_csv("data/table.csv", index=False)
t.to_csv("data/table_tracks.csv", index=False)
print("tracks from", sys.argv[1])
PY
  else
    tropcyc build-dataset --years 1980 2030 -o data/table.csv
  fi
fi

log "2/6 baselines + gradient boosting (track features only)"
tropcyc evaluate data/table.csv --val-start 2014 --test-start 2019 -o runs/baselines_val.csv
tropcyc train data/table.csv -o models/live.joblib | tee runs/train_live.txt

log "3/6 ERA5 environment $PRE_YEARS (long; resumable)"
tropcyc env-store data/table.csv --source era5 --years $PRE_YEARS --workers "$WORKERS" -o data/env_era5.npz

log "4/6 pre-train transformer ($SIZE)"
tropcyc pretrain data/table.csv --store data/env_era5.npz --years $PRE_YEARS --size "$SIZE" \
  -o models/tf_era5.pt | tee runs/pretrain.txt

log "5/6 GFS environment $FT_YEARS"
tropcyc env-store data/table.csv --source gfs --years $FT_YEARS --workers "$WORKERS" -o data/env_gfs.npz

log "6/6 fine-tune on GFS"
tropcyc finetune data/table.csv --model models/tf_era5.pt --store data/env_gfs.npz --years $FT_YEARS \
  -o models/tf_gfs.pt | tee runs/finetune.txt

log "done: models/ and runs/"
