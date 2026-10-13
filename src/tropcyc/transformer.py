"""Temporal transformer for 24 h intensity change and rapid intensification.

Each sample is one storm at one fix time t. The model reads a short sequence
of tokens:

    [CLS] [t-24] [t-18] [t-12] [t-6] [t] [future: environment at t+24]

A history token = CNN encoding of the environment patch at that fix
+ projection of that fix's track features + a learned time embedding.
The future token is the environment the storm is heading into: reanalysis
at t+24 when pre-training ("perfect prognosis"), a GFS / GraphCast 24 h
forecast when fine-tuning and running live. Missing fixes are masked.
Self-attention mixes the tokens; the CLS output feeds two heads:

    dv_24  regression (kt)               Huber loss
    ri_24  rapid intensification logit   BCE loss, up-weighted (RI is rare)

Workflow:  pre-train on ERA5 (1980-2020) -> fine-tune on GFS (2021-) -> live on GFS.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import RI_THRESHOLD_KT, TRACK_FEATURES, add_track_features

HISTORY_HOURS = (-24, -18, -12, -6, 0)
FUTURE_LEAD = 24

PRESETS = {  # "small" trains on a laptop CPU; "base" wants a GPU
    "small": {"d_model": 64, "layers": 2, "heads": 4, "cnn_width": 16},
    "base": {"d_model": 128, "layers": 4, "heads": 8, "cnn_width": 32},
    "large": {"d_model": 256, "layers": 6, "heads": 8, "cnn_width": 48},
}


# ---------------------------------------------------------------- environment store

def env_key(sid, time, lead: int = 0):
    return (str(sid), pd.Timestamp(time), int(lead))


def future_positions(tracks: pd.DataFrame, lead: int = FUTURE_LEAD,
                     extrapolate: bool = False) -> pd.DataFrame:
    """Where each storm will be at t+lead: best track if known (training), else extrapolated (live)."""
    t = tracks[["sid", "time", "lat", "lon"]].copy()
    ahead = t.assign(time=t["time"] - pd.Timedelta(hours=lead)).rename(columns={"lat": "flat", "lon": "flon"})
    out = t.merge(ahead, on=["sid", "time"], how="left")
    if extrapolate:
        f = add_track_features(tracks)[["sid", "time", "u_motion", "v_motion"]]
        out = out.merge(f, on=["sid", "time"], how="left")
        km_n = out["v_motion"].fillna(0) * lead
        km_e = out["u_motion"].fillna(0) * lead
        est_lat = out["lat"] + km_n / 111.0
        est_lon = out["lon"] + km_e / (111.0 * np.cos(np.deg2rad(out["lat"])))
        out["flat"] = out["flat"].fillna(est_lat)
        out["flon"] = out["flon"].fillna(est_lon)
    return out[["sid", "time", "flat", "flon"]]


def build_env_store(table: pd.DataFrame, tracks: pd.DataFrame, source, cache=None,
                    extrapolate: bool = False, progress: bool = True, workers: int = 1,
                    future_source=None) -> dict:
    """Environment patches every sample in `table` needs, from `source`.

    `future_source` (default: `source`) supplies the 24 h future token, e.g.
    history from GFS analyses and the future from an AI model (AIGFS).

    History patches are analyses (lead 0) at each fix; the future patch is
    the source's 24 h forecast from t at the storm's t+24 position. Resumable
    via `cache` (.npz).
    """
    store = load_store(cache) if cache and Path(cache).exists() else {}
    fixes = tracks.set_index(["sid", "time"])[["lat", "lon"]]
    fut = future_positions(tracks, extrapolate=extrapolate).set_index(["sid", "time"])
    jobs = {}
    for r in table[["sid", "time"]].itertuples(index=False):
        for h in HISTORY_HOURS:
            t = r.time + pd.Timedelta(hours=h)
            if (r.sid, t) in fixes.index:
                jobs[env_key(r.sid, t, 0)] = (t, *fixes.loc[(r.sid, t)], 0)
        if (r.sid, r.time) in fut.index:
            flat, flon = fut.loc[(r.sid, r.time)]
            if np.isfinite(flat):
                jobs[env_key(r.sid, r.time, FUTURE_LEAD)] = (r.time, flat, flon, FUTURE_LEAD)
    todo = [k for k in jobs if k not in store]

    def fetch(k):
        init, lat, lon, lead = jobs[k]
        try:
            src = future_source if (lead and future_source is not None) else source
            return k, src.patch(init, float(lat), float(lon), lead)
        except Exception as e:  # a missing cycle/file should not stop the whole build
            if progress:
                print(f"  skip {k}: {type(e).__name__}: {e}")
            return k, None

    # Cloud reads are latency-bound, so threads give a near-linear speed-up. Sorting by
    # valid time keeps requests for the same ERA5 day-chunk / GFS cycle together, so
    # each is downloaded once and shared by every storm and token that needs it.
    todo.sort(key=lambda k: (jobs[k][0] + pd.Timedelta(hours=jobs[k][3]), jobs[k][3]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (k, x) in enumerate(pool.map(fetch, todo), 1):
            if x is not None:
                store[k] = x
            if cache and i % 500 == 0:
                save_store(store, cache)
                if progress:
                    print(f"  {i}/{len(todo)} patches")
    if cache and todo:
        save_store(store, cache)
    return store


def save_store(store: dict, path) -> None:
    keys = list(store)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sid=np.array([k[0] for k in keys]),
                        time=np.array([k[1].value for k in keys], dtype=np.int64),
                        lead=np.array([k[2] for k in keys]), x=np.stack([store[k] for k in keys]))


def load_store(path) -> dict:
    f = np.load(path)
    return {env_key(s, t, ld): x for s, t, ld, x in
            zip(f["sid"], pd.to_datetime(f["time"]), f["lead"], f["x"], strict=True)}


# ---------------------------------------------------------------- dataset

class SequenceDataset(Dataset):
    """Token sequences for each row of `table`.

    Patches are normalised once into a single array and each sample only
    stores indices into it, so a patch shared by six sequences is kept once.
    """

    def __init__(self, table, tracks, store, stats, target="dv_24"):
        ft = add_track_features(tracks)
        f_norm = (ft[list(TRACK_FEATURES)].to_numpy(np.float32) - stats["f_mean"]) / stats["f_std"]
        feats = dict(zip(zip(ft["sid"], ft["time"], strict=True), f_norm, strict=True))  # fast lookup
        keys = list(store)
        pos = {k: i for i, k in enumerate(keys)}
        x = (np.stack([store[k] for k in keys]) - stats["x_mean"]) / stats["x_std"]
        x = np.concatenate([np.nan_to_num(x), np.zeros_like(x[:1])]).astype(np.float32)
        self.patches = torch.from_numpy(x)
        blank = len(keys)  # index of the all-zero patch used for missing tokens

        n, n_tok = len(table), len(HISTORY_HOURS) + 1
        idx = np.full((n, n_tok), blank, np.int64)
        f = np.zeros((n, n_tok, len(TRACK_FEATURES)), np.float32)
        mask = np.ones((n, n_tok), bool)  # True = missing (PyTorch convention)
        for i, r in enumerate(table[["sid", "time"]].itertuples(index=False)):
            for j, hrs in enumerate(HISTORY_HOURS):
                t = r.time + pd.Timedelta(hours=hrs)
                k = env_key(r.sid, t, 0)
                if k in pos and (r.sid, t) in feats:
                    idx[i, j] = pos[k]
                    f[i, j] = feats[(r.sid, t)]
                    mask[i, j] = False
            k = env_key(r.sid, r.time, FUTURE_LEAD)
            if k in pos:  # the future token has environment only, no track features
                idx[i, -1] = pos[k]
                mask[i, -1] = False
        y = table[target].to_numpy(np.float32) if target in table else np.zeros(n, np.float32)
        y = np.nan_to_num(y)
        self.idx, self.mask = torch.from_numpy(idx), torch.from_numpy(mask)
        self.f = torch.from_numpy(np.nan_to_num(f))
        self.y, self.ri = torch.from_numpy(y), torch.from_numpy((y >= RI_THRESHOLD_KT).astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.patches[self.idx[i]], self.f[i], self.mask[i], self.y[i], self.ri[i]


def compute_stats(store: dict, tracks: pd.DataFrame) -> dict:
    x = np.stack(list(store.values()))
    f = add_track_features(tracks)[list(TRACK_FEATURES)].to_numpy(np.float32)
    return {
        "x_mean": np.nanmean(x, axis=(0, 2, 3))[:, None, None].astype(np.float32),
        "x_std": (np.nanstd(x, axis=(0, 2, 3))[:, None, None] + 1e-6).astype(np.float32),
        "f_mean": np.nanmean(f, axis=0).astype(np.float32),
        "f_std": (np.nanstd(f, axis=0) + 1e-6).astype(np.float32),
    }


# ---------------------------------------------------------------- network

class StormTransformer(nn.Module):
    def __init__(self, n_channels=5, n_features=None, d_model=64, layers=2, heads=4,
                 cnn_width=16, dropout=0.1):
        super().__init__()
        n_features = n_features or len(TRACK_FEATURES)
        w = cnn_width
        self.encoder = nn.Sequential(  # one patch -> one d_model vector
            nn.Conv2d(n_channels, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, d_model),
        )
        self.feat = nn.Linear(n_features, d_model)
        n_tok = len(HISTORY_HOURS) + 1
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, d_model))  # +1 for CLS
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dv_head = nn.Linear(d_model, 1)
        self.ri_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x, f, mask):
        b, n = x.shape[:2]
        env = self.encoder(x.flatten(0, 1)).view(b, n, -1)
        tok = env + self.feat(f)
        tok = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1) + self.pos
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.norm(self.transformer(tok, src_key_padding_mask=pad)[:, 0])
        return self.dv_head(z).squeeze(1), self.ri_head(z).squeeze(1)


# ---------------------------------------------------------------- fit / predict wrapper

class TransformerModel:
    """Same fit/predict interface as tropcyc.models, plus pre-train -> fine-tune."""

    name = "transformer"

    def __init__(self, store: dict, tracks: pd.DataFrame, target="dv_24", size="small", epochs=40,
                 batch_size=64, lr=3e-4, patience=6, ri_weight=0.5, val_frac=0.15, device=None,
                 seed=0, verbose=True):
        self.store, self.tracks, self.target = store, tracks, target
        self.size, self.epochs, self.batch_size, self.lr = size, epochs, batch_size, lr
        self.patience, self.ri_weight, self.val_frac = patience, ri_weight, val_frac
        self.seed, self.verbose = seed, verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net_, self.stats_, self.history_ = None, None, []

    def _loader(self, table, shuffle):
        ds = SequenceDataset(table, self.tracks, self.store, self.stats_, self.target)
        return DataLoader(ds, batch_size=self.batch_size if shuffle else 256, shuffle=shuffle)

    def fit(self, table: pd.DataFrame, finetune: bool = False):
        """Train from scratch, or (finetune=True) continue from the loaded weights at lr/10."""
        torch.manual_seed(self.seed)
        sids = table.groupby("sid")["time"].min().sort_values().index
        is_val = table["sid"].isin(sids[-max(1, int(len(sids) * self.val_frac)):])
        if len(sids) < 2:  # one storm: nothing to hold out, early-stop on the training rows
            is_val[:] = False
        tr = table[~is_val].reset_index(drop=True)
        va = table[is_val].reset_index(drop=True) if is_val.any() else tr

        if not finetune or self.net_ is None:
            self.stats_ = compute_stats(self.store, self.tracks)
            self.net_ = StormTransformer(n_channels=next(iter(self.store.values())).shape[0],
                                         **PRESETS[self.size])
        self.net_.to(self.device)
        lr = self.lr / 10 if finetune else self.lr
        opt = torch.optim.AdamW(self.net_.parameters(), lr=lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        huber = nn.HuberLoss(delta=10.0)
        ri_rate = float((tr[self.target] >= RI_THRESHOLD_KT).mean()) or 0.05
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - ri_rate) / ri_rate, device=self.device))

        train_dl, val_dl = self._loader(tr, True), self._loader(va, False)
        best, best_state, bad = np.inf, copy.deepcopy(self.net_.state_dict()), 0
        for epoch in range(self.epochs):
            self.net_.train()
            total = 0.0
            for x, f, m, y, ri in train_dl:
                x, f, m, y, ri = (t.to(self.device) for t in (x, f, m, y, ri))
                opt.zero_grad()
                dv, logit = self.net_(x, f, m)
                loss = huber(dv, y) + self.ri_weight * bce(logit, ri)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(y)
            sched.step()
            dv_val, _ = self._predict_loader(val_dl)
            val_mae = float(np.mean(np.abs(dv_val - val_dl.dataset.y.numpy())))
            loss_avg = total / len(tr)
            self.history_.append({"epoch": epoch, "finetune": finetune, "loss": loss_avg, "val_mae": val_mae})
            if self.verbose:
                tag = "ft " if finetune else ""
                print(f"{tag}epoch {epoch:3d}  loss {loss_avg:7.3f}  val_mae {val_mae:6.2f} kt")
            if val_mae < best:
                best, best_state, bad = val_mae, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _predict_loader(self, dl):
        self.net_.eval()
        dv, p = [], []
        for x, f, m, _, _ in dl:
            a, b = self.net_(x.to(self.device), f.to(self.device), m.to(self.device))
            dv.append(a.cpu().numpy())
            p.append(torch.sigmoid(b).cpu().numpy())
        return np.concatenate(dv), np.concatenate(p)

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return self._predict_loader(self._loader(table.reset_index(drop=True), False))[0]

    def predict_ri_probability(self, table: pd.DataFrame) -> np.ndarray:
        return self._predict_loader(self._loader(table.reset_index(drop=True), False))[1]

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net_.state_dict(), "stats": self.stats_, "size": self.size,
                    "target": self.target, "history": self.history_}, path)

    @classmethod
    def load(cls, path, store=None, tracks=None, **kwargs):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(store or {}, tracks, target=ckpt["target"], size=ckpt["size"], **kwargs)
        m.stats_, m.history_ = ckpt["stats"], ckpt["history"]
        m.net_ = StormTransformer(n_channels=len(m.stats_["x_mean"]), **PRESETS[m.size])
        m.net_.load_state_dict(ckpt["state"])
        m.net_.to(m.device)
        return m
