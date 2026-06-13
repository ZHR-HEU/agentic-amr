"""AMR classifier (M1): length-agnostic CNN / CNN-GRU backbone + incremental trainer.

Exposes logits, penultimate features (for coreset/diversity), predict_proba, and ECE.
The classifier is the ONLY component that touches IQ. The controller never calls it
to classify on its behalf 鈥?it only reads deterministic summaries derived here.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AMRNet(nn.Module):
    def __init__(self, n_classes, hidden=128, dropout=0.3, backbone="cnn"):
        super().__init__()
        self.backbone = backbone
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        if backbone == "cnngru":
            self.gru = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
            feat_in = 2 * hidden
        else:
            self.gru = None
            feat_in = 128
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(feat_in, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def _features(self, x):                  # x: (B, 2, L)
        h = self.conv(x)                     # (B, 128, L')
        if self.gru is not None:
            seq = h.transpose(1, 2)          # (B, L', 128)
            out, _ = self.gru(seq)           # (B, L', 2*hidden)
            h = out.transpose(1, 2)          # (B, 2*hidden, L')
        h = self.pool(h).squeeze(-1)         # (B, feat_in)
        feat = F.relu(self.fc1(h))           # (B, hidden)
        return feat

    def forward(self, x):
        feat = self._features(x)
        return self.fc2(self.drop(feat))

    def penultimate(self, x):
        return self._features(x)


class _ResUnit(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=k // 2); self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2); self.b2 = nn.BatchNorm1d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class _ResStack(nn.Module):
    def __init__(self, cin, c, k=5):
        super().__init__()
        self.proj = nn.Conv1d(cin, c, 1)
        self.u1 = _ResUnit(c, k); self.u2 = _ResUnit(c, k)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.u2(self.u1(self.proj(x))))


class ResNet1D(nn.Module):
    """O'Shea-2018-style 1D residual network for AMR (RML2018 standard)."""
    def __init__(self, n_classes, hidden=128, dropout=0.3, c=32, n_stacks=6):
        super().__init__()
        self.stacks = nn.ModuleList([_ResStack(2 if i == 0 else c, c) for i in range(n_stacks)])
        self.gap = nn.AdaptiveAvgPool1d(2)
        self.fc1 = nn.Linear(c * 2, hidden); self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden); self.fc3 = nn.Linear(hidden, n_classes)

    def _feat(self, x):
        for s in self.stacks:
            if x.shape[-1] >= 2:
                x = s(x)
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x)); x = self.drop(x)
        return F.relu(self.fc2(x))

    def forward(self, x):
        return self.fc3(self._feat(x))

    def penultimate(self, x):
        return self._feat(x)


class CLDNN(nn.Module):
    """CNN + LSTM + DNN (classic strong RML2016 baseline)."""
    def __init__(self, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 8, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 8, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, hidden); self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def _feat(self, x):
        h = self.conv(x).transpose(1, 2)         # (B, L', 64)
        out, _ = self.lstm(h)
        return F.relu(self.fc1(out[:, -1, :]))

    def forward(self, x):
        return self.fc2(self.drop(self._feat(x)))

    def penultimate(self, x):
        return self._feat(x)


def _make_net(cfg, n_classes):
    m = cfg.model
    b = m.backbone
    if b in ("cnn", "cnngru"):
        return AMRNet(n_classes, m.hidden, m.dropout, b)
    if b == "resnet":
        return ResNet1D(n_classes, m.hidden, m.dropout)
    if b == "cldnn":
        return CLDNN(n_classes, m.hidden, m.dropout)
    raise ValueError(f"unknown backbone {b!r} (cnn|cnngru|resnet|cldnn)")


class Classifier:
    """Wraps the backbone with batched fit / predict_proba / features."""

    def __init__(self, cfg, n_classes):
        m = cfg.model
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.net = _make_net(cfg, n_classes).to(self.device)
        self.lr = m.lr
        self.weight_decay = m.weight_decay
        self.batch_size = m.batch_size
        self.n_classes = n_classes
        self.temperature = 1.0
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        self._head_opt = None

    def reset_optimizer(self):
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

    def reset(self):
        """Reinit weights + optimizer + temperature (adaptation action 'reset')."""
        self.net = _make_net(self.cfg, self.n_classes).to(self.device)
        self.temperature = 1.0
        self._head_opt = None
        self.reset_optimizer()

    def _head_params(self):
        return list(self.net.fc1.parameters()) + list(self.net.fc2.parameters())

    def fit(self, X, y, epochs, head_only=False, lr=None):
        if len(X) == 0 or epochs <= 0:
            return
        self.net.train()
        head_ids = {id(p) for p in self._head_params()}
        if head_only:
            if self._head_opt is None:
                self._head_opt = torch.optim.Adam(self._head_params(),
                                                  lr=lr or self.lr, weight_decay=self.weight_decay)
            opt = self._head_opt
            for p in self.net.parameters():          # freeze backbone (no wasted grads)
                if id(p) not in head_ids:
                    p.requires_grad_(False)
        else:
            opt = self.opt
            if lr is not None:
                for g in opt.param_groups:
                    g["lr"] = lr
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        n = len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xt[idx].to(self.device)
                yb = yt[idx].to(self.device)
                opt.zero_grad()
                loss = F.cross_entropy(self.net(xb), yb)
                loss.backward()
                opt.step()
        if head_only:                                # unfreeze backbone
            for p in self.net.parameters():
                p.requires_grad_(True)
        elif lr is not None:                         # restore base lr
            for g in self.opt.param_groups:
                g["lr"] = self.lr

    @torch.no_grad()
    def _logits(self, X):
        self.net.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        out = []
        for i in range(0, len(Xt), 1024):
            out.append(self.net(Xt[i:i + 1024].to(self.device)).cpu())
        return torch.cat(out, dim=0) if out else torch.zeros((0, self.n_classes))

    def recalibrate(self, X, y):
        """Temperature scaling on (X,y): fit T minimizing NLL (adaptation action 'recalibrate')."""
        if len(X) < 10:
            return
        logits = self._logits(X)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / torch.exp(logT), yt)
            loss.backward()
            return loss
        opt.step(closure)
        self.temperature = float(torch.exp(logT).item())

    @torch.no_grad()
    def predict_proba(self, X):
        self.net.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        out = []
        for i in range(0, len(Xt), 1024):
            xb = Xt[i:i + 1024].to(self.device)
            out.append(F.softmax(self.net(xb) / self.temperature, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_classes))

    @torch.no_grad()
    def features(self, X):
        self.net.eval()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        out = []
        for i in range(0, len(Xt), 1024):
            xb = Xt[i:i + 1024].to(self.device)
            out.append(self.net.penultimate(xb).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.cfg.model.hidden))


def accuracy(proba, y):
    if len(y) == 0:
        return float("nan")
    return float((proba.argmax(axis=1) == np.asarray(y)).mean())


def expected_calibration_error(proba, y, n_bins=15):
    """Standard ECE: |confidence - accuracy| averaged over confidence bins."""
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
