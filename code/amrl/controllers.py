"""Acquisition-policy controllers (M4).

Every controller maps the two state cards -> a weight dict over CRITERIA.
The episode runner then calls selection.select_topk. The controller NEVER sees
raw IQ and NEVER classifies.

B0 implements: random, entropy, margin, coreset, class_balance, fixed_uniform,
fixed_hybrid, rule, llm, llm_hardselect. Trained/baseline controllers
(weighted_ensemble, aggregated_af, mlp, bandit, oracle, llm_distilled,
mlp_oracle_imitation) are registered in B1-B3 鈥?see EXPERIMENT_PLAN.md.
"""
from __future__ import annotations
import json
import re
import numpy as np
from .state import CRITERIA


class Controller:
    name = "base"

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

    def weights(self, rf_card, cand_card):
        raise NotImplementedError

    # bookkeeping hook (trained controllers override; no-op here)
    def update(self, *a, **k):
        pass


class SingleCriterion(Controller):
    def __init__(self, cfg, rng, criterion):
        super().__init__(cfg, rng)
        self.name = criterion
        self.criterion = criterion

    def weights(self, rf_card, cand_card):
        return {self.criterion: 1.0}


class FixedUniform(Controller):
    name = "fixed_uniform"

    def weights(self, rf_card, cand_card):
        return {c: 1.0 for c in self.cfg.controller.criteria}


class FixedHybrid(Controller):
    """A reasonable static blend (the tuned-hybrid baseline)."""
    name = "fixed_hybrid"
    BLEND = {"entropy": 0.4, "margin": 0.2, "coreset": 0.2, "class_balance": 0.2, "random": 0.0}

    def weights(self, rf_card, cand_card):
        return dict(self.BLEND)


class StaticMixture(Controller):
    """A fixed weight vector over criteria 鈥?used for the B1 oracle grid search
    (the 'best static mixture in hindsight' envelope). Never calls the LLM."""

    def __init__(self, cfg, rng, weights, label="static"):
        super().__init__(cfg, rng)
        self.name = label
        self._w = {k: float(v) for k, v in weights.items()}

    def weights(self, rf_card, cand_card):
        return dict(self._w)


class BinPolicyController(Controller):
    """State-conditioned switching policy (B1b adaptive-oracle diagnostic): bin the
    current state by SNR / drift / phase and apply a per-bin static mixture. The
    best such policy in hindsight is the ADAPTIVE oracle (upper bound for a
    state-conditioned controller); compare to the best homogeneous (static) policy."""

    def __init__(self, cfg, rng, policy, axis, label="binpolicy"):
        super().__init__(cfg, rng)
        self.name = label
        self.policy = {int(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in policy.items()}
        self.axis = axis

    def _bin(self, rf):
        if self.axis == "snr":
            return 0 if rf.snr_est < 3 else 1
        if self.axis == "drift":
            return 0 if rf.drift_level < 0.5 else 1
        return 0 if rf.step < rf.n_steps // 2 else 1   # phase

    def weights(self, rf_card, cand_card):
        return dict(self.policy[self._bin(rf_card)])


class RuleController(Controller):
    """Strong hand-designed RF rule with an ECE gate (the C5 non-LLM competitor)."""
    name = "rule"

    def weights(self, rf_card, cand_card):
        thr = self.cfg.controller.ece_gate_threshold
        ece = rf_card.ece
        miscalibrated = (np.isfinite(ece) and ece > thr) or rf_card.snr_est < 0
        unseen = any(c == 0 for c in rf_card.class_counts)
        if miscalibrated:
            # uncertainty untrustworthy -> lean on diversity + class balance
            w = {"entropy": 0.05, "margin": 0.05, "coreset": 0.45, "class_balance": 0.45, "random": 0.0}
        else:
            # well-calibrated -> uncertainty is informative
            w = {"entropy": 0.45, "margin": 0.25, "coreset": 0.2, "class_balance": 0.1, "random": 0.0}
        if unseen:
            w["class_balance"] += 0.2
        if rf_card.drift_level > 0.5:
            w["coreset"] += 0.2          # after drift, cover novel regions
        return w


_LLM_SYSTEM = (
    "You are an acquisition-policy controller for ONLINE automatic modulation "
    "recognition (AMR) active learning under RF distribution shift. You read deterministic, "
    "IQ-free state cards and decide how to spend a small label budget. You NEVER classify "
    "signals. Output ONLY a JSON object: {\"weights\": {\"entropy\": w, \"margin\": w, "
    "\"coreset\": w, \"class_balance\": w, \"random\": w}} with non-negative numbers. "
    "Guidance: when classifier_ECE is high or SNR is low, the classifier is miscalibrated, "
    "so DOWN-weight uncertainty criteria (entropy, margin) and prefer coreset/diversity and "
    "class_balance; when well-calibrated, uncertainty is informative; if classes are unseen, "
    "raise class_balance; after channel drift, raise coreset. No prose, JSON only."
)


class LLMController(Controller):
    """Frozen prompted LLM emits SOFT weights over criteria (the flagship)."""
    name = "llm"
    hard_select = False

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self._client = None
        self.n_fallback = 0
        self.n_calls = 0
        # build_controller is defined at module level (resolved at call time).
        # Never let the fallback recurse into an LLM controller.
        fb = cfg.controller.llm_fallback
        if fb in ("llm", "llm_hardselect"):
            fb = "rule"
        self._fallback = build_controller(fb, cfg, rng)

    def _client_lazy(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.cfg.controller.endpoint, api_key="EMPTY")
        return self._client

    def _parse(self, txt):
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        w = obj.get("weights", obj)
        if not isinstance(w, dict):
            return None
        out = {c: float(w[c]) for c in CRITERIA if c in w and _is_num(w[c])}
        return out or None

    def weights(self, rf_card, cand_card):
        self.n_calls += 1
        c = self.cfg.controller
        prompt = rf_card.to_text() + "\n" + cand_card.to_text() + (
            "\nEmit the weight vector now (JSON only)."
        )
        try:
            r = self._client_lazy().chat.completions.create(
                model=c.model,
                messages=[{"role": "system", "content": _LLM_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=c.temperature, max_tokens=c.max_tokens,
                timeout=c.llm_timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": c.enable_thinking}},
            )
            w = self._parse(r.choices[0].message.content)
        except Exception:
            w = None
        if w is None:
            self.n_fallback += 1
            return self._fallback.weights(rf_card, cand_card)
        if self.hard_select:                      # LMABO-style: keep only the argmax criterion
            top = max(w, key=w.get)
            return {top: 1.0}
        return w


class LLMHardSelect(LLMController):
    """LMABO-style ablation: same cards, but pick ONE acquisition criterion."""
    name = "llm_hardselect"
    hard_select = True


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# ---- registry ----------------------------------------------------------
def build_controller(name, cfg, rng) -> Controller:
    if name in ("entropy", "margin", "coreset", "class_balance", "random"):
        return SingleCriterion(cfg, rng, name)
    table = {
        "fixed_uniform": FixedUniform,
        "fixed_hybrid": FixedHybrid,
        "rule": RuleController,
        "llm": LLMController,
        "llm_hardselect": LLMHardSelect,
    }
    if name not in table:
        raise ValueError(
            f"unknown / not-yet-implemented controller {name!r}. "
            f"B0 supports: random,entropy,margin,coreset,class_balance,"
            f"fixed_uniform,fixed_hybrid,rule,llm,llm_hardselect"
        )
    return table[name](cfg, rng)
