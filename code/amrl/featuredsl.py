"""Safe symbolic feature DSL for the LLM-as-feature-program-synthesizer gate.

A "feature program" is a short closed-form algebraic expression over a fixed set of
IQ-derived PRIMITIVES (amplitude/phase/frequency stats, higher-order cumulants,
spectral shape) combined with +,-,*,/,**, and the unary funcs abs/sqrt/log/sq.
Primitives are computed batched ([N,L] complex -> [N]); programs are evaluated by a
RESTRICTED-AST interpreter (no arbitrary code execution). The SAME DSL is searched by
the LLM and by every baseline (random / enumeration / GP / textbook bank), so the only
difference is the SEARCH POLICY, which is exactly the LLM's hypothesized edge.
"""
import ast
import numpy as np


def _kurt(x):
    m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-9
    return (((x - m) / s) ** 4).mean(1) - 3.0


def _skew(x):
    m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True) + 1e-9
    return (((x - m) / s) ** 3).mean(1)


def primitives(Z):
    """Z: complex [N, L]. Returns dict name -> [N] feature values (NaN/inf scrubbed)."""
    Z = Z.astype(np.complex128)
    p = np.sqrt(np.mean(np.abs(Z) ** 2, axis=1, keepdims=True)) + 1e-9
    Z = Z / p                                  # unit-power normalize
    a = np.abs(Z); L = Z.shape[1]; eps = 1e-9
    ph = np.unwrap(np.angle(Z), axis=1)
    t = np.arange(L); tc = t - t.mean(); vt = (tc ** 2).sum() + eps
    slope = ((ph - ph.mean(1, keepdims=True)) * tc).sum(1) / vt          # linear phase fit
    resid = ph - (slope[:, None] * tc + ph.mean(1, keepdims=True))
    ifr = np.diff(ph, axis=1)
    C21 = np.mean(np.abs(Z) ** 2, axis=1)
    C20 = np.abs(np.mean(Z ** 2, axis=1))
    m20 = np.mean(Z ** 2, axis=1)
    C40 = np.abs(np.mean(Z ** 4, axis=1) - 3 * m20 ** 2)
    C42 = np.mean(np.abs(Z) ** 4, axis=1) - np.abs(m20) ** 2 - 2 * C21 ** 2
    S = np.abs(np.fft.fft(Z, axis=1)) ** 2
    spec_flat = np.exp(np.mean(np.log(S + eps), axis=1)) / (np.mean(S, axis=1) + eps)
    spec_peak = np.max(S, axis=1) / (np.mean(S, axis=1) + eps)
    papr = np.max(a ** 2, axis=1) / (np.mean(a ** 2, axis=1) + eps)
    acf1 = np.mean((a[:, :-1] - a.mean(1, keepdims=True)) * (a[:, 1:] - a.mean(1, keepdims=True)), axis=1) / (a.var(1) + eps)
    d = {
        "amp_mean": a.mean(1), "amp_std": a.std(1), "amp_kurt": _kurt(a), "amp_skew": _skew(a),
        "ifreq_std": ifr.std(1), "ifreq_kurt": _kurt(ifr),
        "phase_std": resid.std(1),
        "C20": C20, "C21": C21, "C40": C40 / (C21 ** 2 + eps), "C42": C42 / (C21 ** 2 + eps),
        "spec_flat": spec_flat, "spec_peak": spec_peak, "papr": papr, "env_acf1": acf1,
    }
    return {k: np.nan_to_num(v.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) for k, v in d.items()}


PRIM_NAMES = ["amp_mean", "amp_std", "amp_kurt", "amp_skew", "ifreq_std", "ifreq_kurt",
              "phase_std", "C20", "C21", "C40", "C42", "spec_flat", "spec_peak", "papr", "env_acf1"]

_FUNCS = {"abs": np.abs, "sqrt": lambda x: np.sqrt(np.abs(x)),
          "log": lambda x: np.log(np.abs(x) + 1e-9), "sq": np.square}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: lambda a, b: a / (b + np.sign(b) * 1e-9 + (b == 0) * 1e-9), ast.Pow: lambda a, b: np.power(np.abs(a) + 1e-9, b)}


def safe_eval(expr, ns):
    """Evaluate a feature-program string against namespace ns (name->[N] array).
    Restricted AST: names in ns, numeric constants, +,-,*,/,**, unary -, whitelisted funcs."""
    tree = ast.parse(expr, mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        if isinstance(n, ast.Name) and n.id in ns: return ns[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
        raise ValueError(f"disallowed node: {ast.dump(n)[:40]}")
    out = ev(tree)
    return np.nan_to_num(np.asarray(out, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def auc_1d(feat, y):
    """Direction-agnostic 1-D separability (AUC) of a scalar feature for binary labels y."""
    from sklearn.metrics import roc_auc_score
    f = np.nan_to_num(feat)
    if np.std(f) < 1e-12: return 0.5
    try:
        a = roc_auc_score(y, f)
    except Exception:
        return 0.5
    return max(a, 1 - a)


TEXTBOOK = [  # 50-years-of-theory AMR discriminator bank (fair knowledge baseline, no LLM)
    "C40", "C42", "C42 / (C21 * C21)", "C40 / C42", "amp_kurt", "amp_std / amp_mean",
    "ifreq_std", "phase_std", "papr", "spec_flat", "spec_peak", "C20", "env_acf1",
    "amp_std * amp_kurt", "C40 / (C20 + 0.01)",
]
