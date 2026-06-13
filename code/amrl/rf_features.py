"""Interpretable RF descriptors for modulation signals (direction C: LLM knowledge).

Classic, textbook AMR features an LLM can reason about semantically (unlike raw IQ
or CNN embeddings): higher-order cumulants, amplitude / phase / instantaneous-freq
statistics, spectral shape. Used to test whether an LLM's modulation knowledge
grounds to these descriptors for zero-shot / open-set recognition.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "amp_std", "papr_dB", "phase_resid_std", "ifreq_std", "ifreq_absmean",
    "|C20|", "|C40|", "|C41|", "|C42|", "spec_kurtosis", "spec_flatness", "spec_peak2mean",
]

FEATURE_LEGEND = (
    "Legend (signal power-normalized to 1):\n"
    "- amp_std: std of instantaneous amplitude (~0 => constant envelope, as in PSK/FSK/FM; "
    "large => amplitude-bearing, as in QAM/PAM/ASK/AM).\n"
    "- papr_dB: peak-to-average power (dB) (low => constant envelope).\n"
    "- phase_resid_std: std of instantaneous phase after removing a linear trend, rad "
    "(captures #phase states / phase spread; PSK has discrete phase jumps).\n"
    "- ifreq_std / ifreq_absmean: instantaneous-frequency variation (large => FSK/FM/freq-mod).\n"
    "- |C20|,|C40|,|C41|,|C42|: magnitudes of normalized higher-order cumulants "
    "(classic modulation discriminators; e.g. |C40| separates PSK orders / QAM).\n"
    "- spec_kurtosis / spec_flatness / spec_peak2mean: power-spectrum shape "
    "(tones vs broadband; SSB asymmetry, analog vs digital).\n"
)


def _descr_one(z):
    z = np.asarray(z, dtype=np.complex128)
    p = np.mean(np.abs(z) ** 2)
    if p <= 0:
        z = z + 1e-9
        p = np.mean(np.abs(z) ** 2)
    z = z / np.sqrt(p)                       # unit power
    a = np.abs(z)
    amp_std = float(a.std())
    papr_dB = float(10 * np.log10((a ** 2).max() / np.mean(a ** 2) + 1e-12))
    ph = np.unwrap(np.angle(z))
    n = np.arange(len(ph))
    # remove linear (carrier-offset) trend
    A = np.vstack([n, np.ones_like(n)]).T
    coef, *_ = np.linalg.lstsq(A, ph, rcond=None)
    resid = ph - A @ coef
    phase_resid_std = float(resid.std())
    ifreq = np.diff(ph) / (2 * np.pi)
    ifreq_std = float(ifreq.std())
    ifreq_absmean = float(np.mean(np.abs(ifreq - ifreq.mean())))
    # cumulants (unit power)
    M20 = np.mean(z ** 2)
    M21 = np.mean(np.abs(z) ** 2)
    M40 = np.mean(z ** 4)
    M41 = np.mean(z ** 3 * np.conj(z))
    M42 = np.mean(np.abs(z) ** 4)
    C20 = M20
    C40 = M40 - 3 * M20 ** 2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21 ** 2
    # spectrum
    P = np.abs(np.fft.fftshift(np.fft.fft(z))) ** 2
    P = P / (P.sum() + 1e-12)
    spec_kurt = float(((P - P.mean()) ** 4).mean() / ((P.var() + 1e-12) ** 2))
    spec_flat = float(np.exp(np.mean(np.log(P + 1e-12))) / (P.mean() + 1e-12))
    spec_p2m = float(P.max() / (P.mean() + 1e-12))
    return np.array([
        amp_std, papr_dB, phase_resid_std, ifreq_std, ifreq_absmean,
        abs(C20), abs(C40), abs(C41), abs(C42),
        spec_kurt, spec_flat, spec_p2m,
    ], dtype=np.float64)


def compute_descriptors(X):
    """X: (n, 2, L) -> (n, D) descriptor matrix."""
    out = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float64)
    for i in range(len(X)):
        z = X[i, 0, :] + 1j * X[i, 1, :]
        out[i] = _descr_one(z)
    return out


def render_descriptor_text(vec):
    parts = [f"{nm}={v:.3g}" for nm, v in zip(FEATURE_NAMES, vec)]
    return "  ".join(parts)
