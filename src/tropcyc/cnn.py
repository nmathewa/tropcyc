"""A CNN on ERA5 patches + track features, in PyTorch.

Two inputs, one output:

    ERA5 patch (5, 41, 41) --conv blocks--> pooled vector --+
                                                             +--> dense --> dv_24 (kt)
    track features (14,)   --dense-------------------------+

The track branch carries what persistence knows (current intensity, recent
trend); the conv branch adds the environment. docs/tensorflow_to_pytorch.md
walks through this file next to its Keras equivalent.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import TRACK_FEATURES


class PatchDataset(Dataset):
    """(patch, track features, target) for each row of a feature table."""

    def __init__(self, table, store, features, target, stats):
        keys = [(r.sid, pd.Timestamp(r.time)) for r in table.itertuples()]
        x = np.stack([store[k] for k in keys]).astype(np.float32)
        x = (x - stats["x_mean"]) / stats["x_std"]
        self.x = torch.from_numpy(np.nan_to_num(x))  # land points in SST -> channel mean
        f = (table[features].to_numpy(np.float32) - stats["f_mean"]) / stats["f_std"]
        self.f = torch.from_numpy(np.nan_to_num(f))
        self.y = torch.from_numpy(table[target].to_numpy(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.f[i], self.y[i]


class IntensityNet(nn.Module):
    def __init__(self, n_channels: int, n_features: int, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.conv = nn.Sequential(
            block(n_channels, width), block(width, 2 * width), block(2 * width, 4 * width),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.track = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(4 * width + 64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, patch, features):
        z = torch.cat([self.conv(patch), self.track(features)], dim=1)
        return self.head(z).squeeze(1)


class CNNModel:
    """fit/predict wrapper matching the other models in tropcyc.models."""

    name = "cnn"

    def __init__(self, store: dict, target: str = "dv_24", features=None, epochs: int = 40,
                 batch_size: int = 64, lr: float = 1e-3, patience: int = 6, val_frac: float = 0.15,
                 device: str | None = None, seed: int = 0, verbose: bool = True):
        self.store, self.target = store, target
        self.features = features or list(TRACK_FEATURES)
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.patience, self.val_frac, self.seed, self.verbose = patience, val_frac, seed, verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _stats(self, table):
        x = np.stack([self.store[(r.sid, pd.Timestamp(r.time))] for r in table.itertuples()])
        f = table[self.features].to_numpy(np.float32)
        return {
            "x_mean": np.nanmean(x, axis=(0, 2, 3), keepdims=True)[0],
            "x_std": np.nanstd(x, axis=(0, 2, 3), keepdims=True)[0] + 1e-6,
            "f_mean": np.nanmean(f, axis=0),
            "f_std": np.nanstd(f, axis=0) + 1e-6,
        }

    def fit(self, table: pd.DataFrame):
        torch.manual_seed(self.seed)
        # Hold out the most recent storms for early stopping (never random rows).
        sids = table.groupby("sid")["time"].min().sort_values().index
        n_val = max(1, int(len(sids) * self.val_frac))
        is_val = table["sid"].isin(sids[-n_val:])
        tr, va = table[~is_val], table[is_val]

        self.stats_ = self._stats(tr)
        train_dl = DataLoader(PatchDataset(tr, self.store, self.features, self.target, self.stats_),
                              batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(PatchDataset(va, self.store, self.features, self.target, self.stats_),
                            batch_size=256)

        n_channels = next(iter(self.store.values())).shape[0]
        self.net_ = IntensityNet(n_channels, len(self.features)).to(self.device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.HuberLoss(delta=10.0)  # robust to the few huge RI cases

        best, best_state, bad = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.net_.train()
            total = 0.0
            for x, f, y in train_dl:
                x, f, y = x.to(self.device), f.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net_(x, f), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            val_mae = self._mae(val_dl)
            self.history_.append({"epoch": epoch, "train_loss": total / len(tr), "val_mae": val_mae})
            if self.verbose:
                print(f"epoch {epoch:3d}  train_loss {total / len(tr):7.3f}  val_mae {val_mae:6.2f} kt")
            if val_mae < best:
                best, best_state, bad = val_mae, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _predict_loader(self, dl) -> np.ndarray:
        self.net_.eval()
        return np.concatenate([self.net_(x.to(self.device), f.to(self.device)).cpu().numpy()
                               for x, f, _ in dl])

    def _mae(self, dl) -> float:
        pred = self._predict_loader(dl)
        return float(np.mean(np.abs(pred - dl.dataset.y.numpy())))

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        t = table.copy()
        if self.target not in t:
            t[self.target] = 0.0
        ds = PatchDataset(t, self.store, self.features, self.target, self.stats_)
        return self._predict_loader(DataLoader(ds, batch_size=256))

    def save(self, path) -> None:
        torch.save({"state": self.net_.state_dict(), "stats": self.stats_,
                    "features": self.features, "target": self.target}, path)
