"""Dataset loaders for RML2016.10A and RML2018.01A.

Confirmed formats (probed 2026-06-04):
  RML2016.10A: pickle (latin1) dict keyed (mod_str, snr_int) -> (1000, 2, 128) f32.
               11 mods, SNR -20..18 step 2.
  RML2018.01A: hdf5 with X (N,1024,2) f32, Y (N,24) one-hot int64, Z (N,1) int64 SNR.
               Block-ordered: 24 classes x 26 SNRs x 4096 frames (N=2,555,904).
               SNR -20..30 step 2.

Returns a `SignalData` with X (N,2,L) f32, y (N,) int64 class idx, snr (N,) int64,
plus class names and a stratified train/test split.
"""
from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np

RML2018_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


@dataclass
class SignalData:
    X: np.ndarray            # (N, 2, L) float32
    y: np.ndarray            # (N,) int64  class index
    snr: np.ndarray          # (N,) int64
    classes: list            # class names, index = label
    train_idx: np.ndarray    # indices into X
    test_idx: np.ndarray

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def length(self):
        return self.X.shape[2]


def _normalize_per_sample(X: np.ndarray) -> np.ndarray:
    # standardize each sample across its 2*L values (zero mean, unit std)
    flat = X.reshape(X.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    flat = (flat - mean) / std
    return flat.reshape(X.shape).astype(np.float32)


def _stratified_test_split(y, snr, test_per_class, rng):
    """Pick test indices per (class, SNR) bucket so the test set spans SNRs.

    train/test are complementary index sets -> disjoint by construction.
    """
    test_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        snrs_c = np.unique(snr[cls_idx])
        per_bucket = max(1, test_per_class // max(1, len(snrs_c)))
        for s in snrs_c:
            bucket = cls_idx[snr[cls_idx] == s]
            k = int(min(per_bucket, len(bucket)))
            if k <= 0:
                continue
            chosen = rng.choice(bucket, size=k, replace=False)
            test_mask[chosen] = True
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def load_rml2016(path, normalize="per_sample", test_per_class=100, seed=42) -> SignalData:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d.keys()})
    mod2idx = {m: i for i, m in enumerate(mods)}
    Xs, ys, snrs = [], [], []
    for (mod, snr), arr in d.items():
        arr = np.asarray(arr, dtype=np.float32)   # (n, 2, 128)
        Xs.append(arr)
        ys.append(np.full(arr.shape[0], mod2idx[mod], dtype=np.int64))
        snrs.append(np.full(arr.shape[0], int(snr), dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    snr = np.concatenate(snrs, axis=0)
    if normalize == "per_sample":
        X = _normalize_per_sample(X)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _stratified_test_split(y, snr, test_per_class, rng)
    return SignalData(X, y, snr, mods, train_idx, test_idx)


def load_rml2018(path, normalize="per_sample", max_per_class_snr=200,
                 test_per_class=100, seed=42) -> SignalData:
    import h5py
    n_class, n_snr, frames = 24, 26, 4096
    snr_values = list(range(-20, 31, 2))   # 26 values
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        Xd, Yd, Zd = f["X"], f["Y"], f["Z"]
        N = Xd.shape[0]
        block_ordered = (N == n_class * n_snr * frames)
        sel = []   # (global_idx, class_idx, snr_value)
        if block_ordered:
            for c in range(n_class):
                for si in range(n_snr):
                    start = (c * n_snr + si) * frames
                    k = min(max_per_class_snr, frames)
                    off = rng.choice(frames, size=k, replace=False)
                    for o in off:
                        sel.append((start + int(o), c, snr_values[si]))
        else:
            # Fallback: read labels/snr fully and subsample per (class, snr).
            Yfull = np.asarray(Yd[:]).argmax(axis=1)
            Zfull = np.asarray(Zd[:]).reshape(-1)
            for c in range(n_class):
                for s in np.unique(Zfull):
                    idx = np.where((Yfull == c) & (Zfull == s))[0]
                    if len(idx) == 0:
                        continue
                    k = min(max_per_class_snr, len(idx))
                    for gi in rng.choice(idx, size=k, replace=False):
                        sel.append((int(gi), c, int(s)))
        sel.sort(key=lambda t: t[0])            # hdf5 fancy-index must be increasing
        gidx = np.array([t[0] for t in sel])
        y = np.array([t[1] for t in sel], dtype=np.int64)
        snr = np.array([t[2] for t in sel], dtype=np.int64)
        Xsel = Xd[gidx]                          # (M, 1024, 2)
    X = np.transpose(Xsel, (0, 2, 1)).astype(np.float32)   # -> (M, 2, 1024)
    if normalize == "per_sample":
        X = _normalize_per_sample(X)
    train_idx, test_idx = _stratified_test_split(y, snr, test_per_class, rng)
    return SignalData(X, y, snr, RML2018_CLASSES, train_idx, test_idx)


def load_dataset(cfg) -> SignalData:
    ds = cfg.dataset
    if ds.name == "rml2016":
        return load_rml2016(ds.rml2016_path, ds.normalize, ds.test_per_class, cfg.seed)
    if ds.name == "rml2018":
        return load_rml2018(ds.rml2018_path, ds.normalize,
                            ds.rml2018_max_per_class_snr, ds.test_per_class, cfg.seed)
    raise ValueError(f"unknown dataset {ds.name!r}")
