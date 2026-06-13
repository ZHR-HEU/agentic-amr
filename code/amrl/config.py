"""Config loading: YAML -> nested dict with attribute access + dotted overrides.

Usage:
    cfg = load_config("configs/default.yaml", overrides=["controller.name=llm",
                                                         "stream.n_steps=8"])
    cfg.controller.name  ->  "llm"
"""
from __future__ import annotations
import copy
import ast
import yaml


class Cfg(dict):
    """dict with attribute access; nested dicts are wrapped on access."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _coerce(val: str):
    """Parse an override string value as a Python literal, else keep as str."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def apply_override(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def load_config(path: str, overrides=None) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"bad override (need key=value): {ov!r}")
        key, val = ov.split("=", 1)
        apply_override(raw, key.strip(), _coerce(val.strip()))
    return Cfg(raw)
