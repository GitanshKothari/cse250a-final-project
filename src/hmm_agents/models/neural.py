from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..encoding import encode_run
from .base import FailurePredictor, group_holdout
from .feature_lr import (HMM_DEFAULTS, HMM_VARIANT_DEFAULTS, LDS_DEFAULTS, WithinModelScaler,
                         fit_or_load_hmm, fit_or_load_lds)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

INPUTS = ("steps", "hmm", "lds")
ARCHS = ("gru", "mlp", "cnn")
CNN_DILATIONS = (1, 2, 4, 8)


if torch is not None:
    class GRUNet(nn.Module):
        def __init__(self, d_in: int, hidden: int, dropout: float):
            super().__init__()
            self.rnn = nn.GRU(d_in, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, ctx):
            h, _ = self.rnn(x)
            c = ctx[:, None, :].expand(-1, h.shape[1], -1)
            return self.head(torch.cat([h, c], -1)).squeeze(-1)

    class MeanMLPNet(nn.Module):
        def __init__(self, d_in: int, hidden: int, dropout: float):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * d_in + 1, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, ctx):
            n = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)[None, :, None]
            cum_mean = x.cumsum(1) / n
            c = ctx[:, None, :].expand(-1, x.shape[1], -1)
            return self.mlp(torch.cat([cum_mean, x, c], -1)).squeeze(-1)

    class CausalCNNNet(nn.Module):
        def __init__(self, d_in: int, hidden: int, dropout: float, kernel_size: int,
                     dilations: tuple[int, ...]):
            super().__init__()
            self.inp = nn.Conv1d(d_in, hidden, 1)
            self.pads = [(kernel_size - 1) * d for d in dilations]
            self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel_size, dilation=d)
                                       for d in dilations)
            self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x, ctx):
            h = torch.relu(self.inp(x.transpose(1, 2)))
            for pad, conv in zip(self.pads, self.convs):
                h = h + torch.relu(conv(nn.functional.pad(h, (pad, 0))))
            h = h.transpose(1, 2)
            c = ctx[:, None, :].expand(-1, h.shape[1], -1)
            return self.head(torch.cat([h, c], -1)).squeeze(-1)


class NeuralSeq(FailurePredictor):
    stochastic = True

    def __init__(self, name: str, arch: str = "gru", inputs: list[str] | None = None,
                 hidden: int = 32, dropout: float = 0.1, lr: float = 2e-3,
                 weight_decay: float = 1e-4, batch_size: int = 256, max_epochs: int = 30,
                 patience: int = 4, holdout_frac: float = 0.1, n_threads: int = 4,
                 hmm: dict | None = None, lds: dict | None = None,
                 kernel_size: int = 3, dilations: list[int] | None = None, seed: int = 0):
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
        if arch != "cnn" and dilations is not None:
            raise ValueError("dilations / kernel_size apply to arch 'cnn' only")
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in (dilations or CNN_DILATIONS))
        if self.kernel_size < 2 or min(self.dilations) < 1:
            raise ValueError("cnn needs kernel_size >= 2 and dilations >= 1")
        inputs = list(inputs or ["steps"])
        if "steps" not in inputs:
            inputs = ["steps", *inputs]
        unknown = set(inputs) - set(INPUTS)
        if unknown:
            raise ValueError(f"unknown inputs {sorted(unknown)}; known: {INPUTS}")
        self.name, self.arch, self.inputs, self.seed = name, arch, inputs, seed
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.holdout_frac, self.n_threads = holdout_frac, n_threads
        self.hmm_settings = ({**HMM_DEFAULTS, **HMM_VARIANT_DEFAULTS, **(hmm or {})}
                             if "hmm" in inputs else None)
        if self.hmm_settings and (self.hmm_settings["alphabet"] != "base"
                                  or self.hmm_settings["n_phases"] != 1):
            raise ValueError("neural `hmm` input uses the plain 12-symbol HMM")
        self.lds_settings = {**LDS_DEFAULTS, **(lds or {})} if "lds" in inputs else None

    def _step_inputs(self, runs: pd.DataFrame, max_len: int) -> list[np.ndarray]:
        recs = runs.to_dict("records")
        enc = [encode_run(r, max_len) for r in recs]
        parts = [enc]
        if "hmm" in self.inputs:
            parts.append(self.hmm_.filter([np.asarray(r["symbols"][:max_len]) for r in recs]))
        if "lds" in self.inputs:
            parts.append(self.lds_.filter(enc)[0])
        return [np.hstack([p[i] for p in parts]) for i in range(len(enc))]

    def _context(self, runs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"log_issue_len": np.log1p(runs["issue_len"].to_numpy(float))})

    def _tensorise(self, seqs: list[np.ndarray], T: int, dtype) -> np.ndarray:
        X = np.zeros((len(seqs), T, len(self.mu_)), dtype)
        for i, s in enumerate(seqs):
            X[i, : len(s)] = (s - self.mu_) / self.sd_
        return X

    @property
    def receptive_field(self) -> int | None:
        if self.arch != "cnn":
            return None
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def _new_net(self, d_in: int):
        if self.arch == "cnn":
            return CausalCNNNet(d_in, self.hidden, self.dropout, self.kernel_size, self.dilations)
        cls = GRUNet if self.arch == "gru" else MeanMLPNet
        return cls(d_in, self.hidden, self.dropout)

    def fit(self, runs, y_fail, prefixes):
        if torch is None:
            raise ImportError("NeuralSeq needs torch (see dev_docs/15 for the local install)")
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_threads)
        rng = np.random.default_rng(self.seed)
        runs = runs.reset_index(drop=True)
        y = np.asarray(y_fail, bool).astype(np.float32)
        self.T_ = int(max(prefixes))

        if "hmm" in self.inputs:
            self.hmm_ = fit_or_load_hmm(runs, self.hmm_settings, self.seed)
        if "lds" in self.inputs:
            self.lds_ = fit_or_load_lds(runs, self.lds_settings, self.seed)

        seqs = self._step_inputs(runs, self.T_)
        flat = np.vstack(seqs)
        self.mu_, sd = flat.mean(0), flat.std(0)
        self.sd_ = np.where(sd < 1e-8, 1.0, sd)
        self.ctx_scaler_ = WithinModelScaler().fit(self._context(runs), runs["model_name"])
        ctx = self.ctx_scaler_.transform(self._context(runs), runs["model_name"]).astype(np.float32)
        X = self._tensorise(seqs, self.T_, np.float32)

        n_steps = runs["n_steps"].to_numpy()
        W = np.zeros((len(runs), self.T_), np.float32)
        for t in prefixes:
            W[n_steps > t, t - 1] = 1.0
        has_loss = W.sum(1) > 0
        fit_mask, hold_mask = group_holdout(runs, self.holdout_frac, self.seed)
        fit_idx = np.flatnonzero(fit_mask & has_loss)
        hold_idx = np.flatnonzero(hold_mask & has_loss)

        Xt, Ct = torch.from_numpy(X), torch.from_numpy(ctx)
        Yt = torch.from_numpy(np.repeat(y[:, None], self.T_, 1))
        Wt = torch.from_numpy(W)
        net = self._new_net(X.shape[2])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        def loss_on(idx):
            out = net(Xt[idx], Ct[idx])
            w = Wt[idx]
            return (bce(out, Yt[idx]) * w).sum() / w.sum()

        best, best_state, bad, self.history_ = np.inf, None, 0, []
        for epoch in range(self.max_epochs):
            net.train()
            perm = rng.permutation(fit_idx)
            for b in range(0, len(perm), self.batch_size):
                idx = torch.from_numpy(perm[b:b + self.batch_size])
                opt.zero_grad()
                loss = loss_on(idx)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                hl = float(sum(loss_on(torch.from_numpy(hold_idx[b:b + 4096])) * len(hold_idx[b:b + 4096])
                               for b in range(0, len(hold_idx), 4096)) / max(len(hold_idx), 1))
            self.history_.append(hl)
            if hl < best - 1e-4:
                best, bad = hl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.double().eval()
        self.best_epoch_ = int(np.argmin(self.history_)) + 1
        self.best_holdout_loss_ = float(best)
        return self

    def predict_proba(self, prefix_runs, t):
        if t > self.T_:
            raise ValueError(f"trained for prefixes up to {self.T_}, asked for t={t}")
        seqs = self._step_inputs(prefix_runs, t)
        L = np.array([len(s) for s in seqs])
        X = self._tensorise(seqs, int(L.max()), np.float64)
        ctx = self.ctx_scaler_.transform(self._context(prefix_runs), prefix_runs["model_name"])
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(X), torch.from_numpy(ctx.astype(np.float64)))
            out = torch.sigmoid(logits[torch.arange(len(L)), torch.from_numpy(L - 1)])
        return out.numpy()

    def get_config(self) -> dict[str, Any]:
        n_params = (sum(p.numel() for p in self.net_.parameters())
                    if hasattr(self, "net_") else None)
        return {"name": self.name, "type": "neural", "arch": self.arch, "inputs": self.inputs,
                "hidden": self.hidden, "dropout": self.dropout, "lr": self.lr,
                "weight_decay": self.weight_decay, "batch_size": self.batch_size,
                "max_epochs": self.max_epochs, "patience": self.patience,
                "holdout_frac": self.holdout_frac, "hmm": self.hmm_settings,
                "lds": self.lds_settings, "seed": self.seed, "n_params": n_params,
                **({"kernel_size": self.kernel_size, "dilations": list(self.dilations),
                    "receptive_field": self.receptive_field} if self.arch == "cnn" else {}),
                "best_epoch": getattr(self, "best_epoch_", None),
                "best_holdout_loss": getattr(self, "best_holdout_loss_", None),
                "holdout_loss_history": [round(h, 5) for h in getattr(self, "history_", [])]}
