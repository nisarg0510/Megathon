"""Split conformal suggestion sets.

Nonconformity of a calibration image = 1 - (highest final score among its true
categories), or 1 when none of them is a candidate. With q the
ceil((n + 1)(1 - alpha))-th smallest calibration score, the set

    S(x) = {c : score(c | x) >= 1 - q}

contains at least one correct category with probability >= 1 - alpha for an
image exchangeable with the calibration images. Confident images get short
sets, ambiguous ones longer sets, and an empty set is an abstention (manual
categorisation). Uploads with and without a title/description have different
score scales, so each group is calibrated separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5)


def group_of(has_meta: bool) -> str:
    return "with_metadata" if has_meta else "image_only"


def nonconformity(final: np.ndarray, cand_names: list[str], gold: set[str]) -> float:
    hits = [float(s) for s, n in zip(final, cand_names) if n in gold]
    return 1.0 - max(hits) if hits else 1.0


class ConformalCalibrator:
    def __init__(self, scores: dict[str, list[float]], alpha: float = 0.3):
        self.scores = {g: np.sort(np.asarray(v, dtype=np.float64)) for g, v in scores.items()}
        self.alpha = alpha

    def threshold(self, alpha: float | None = None, group: str = "with_metadata") -> float:
        """Score a category needs to enter the set. 0 means every candidate is included."""
        alpha = self.alpha if alpha is None else alpha
        s = self.scores.get(group)
        if s is None or not len(s):
            s = np.concatenate([v for v in self.scores.values()]) if self.scores else np.zeros(0)
        n = len(s)
        rank = math.ceil((n + 1) * (1 - alpha))
        if n == 0 or rank > n:
            return 0.0
        return float(1.0 - np.sort(s)[rank - 1])

    def predict_set(self, final: np.ndarray, has_meta: bool, alpha: float | None = None) -> np.ndarray:
        """Indices of candidates in the set, best first (empty = abstain)."""
        t = self.threshold(alpha, group_of(has_meta))
        idx = np.where(np.asarray(final) >= t)[0]
        return idx[np.argsort(-np.asarray(final)[idx])]

    def choose_alpha(self, set_sizes: dict[float, float], max_median_size: float = 10) -> float:
        """Smallest alpha (strongest guarantee) whose median set size stays reviewable."""
        ok = [a for a in sorted(set_sizes) if set_sizes[a] <= max_median_size]
        self.alpha = ok[0] if ok else max(set_sizes)
        return self.alpha

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "scores": {g: v.round(5).tolist() for g, v in self.scores.items()}}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCalibrator":
        d = json.loads(Path(path).read_text())
        return cls(d["scores"], d.get("alpha", 0.3))


def set_report(cal: ConformalCalibrator, scored: list[dict], has_meta: list[bool], gold: list[set[str]],
               names: list[str], alphas=ALPHAS) -> dict:
    """Coverage (a correct category in the set), set size and abstention per alpha."""
    out = {}
    for a in alphas:
        covered, sizes, abstain = [], [], []
        for res, m, g in zip(scored, has_meta, gold):
            idx = cal.predict_set(res["final"], m, a)
            sizes.append(len(idx))
            abstain.append(len(idx) == 0)
            covered.append(any(names[res["cands"][i]] in g for i in idx))
        answered = [c for c, ab in zip(covered, abstain) if not ab]
        out[f"{a:g}"] = {
            "target": 1 - a,
            "coverage": float(np.mean(covered)) if covered else None,
            "mean_size": float(np.mean(sizes)) if sizes else None,
            "median_size": float(np.median(sizes)) if sizes else None,
            "abstention": float(np.mean(abstain)) if abstain else None,
            "coverage_when_answered": float(np.mean(answered)) if answered else None,
        }
    return out


def topk_hit_rate(scored: list[dict], gold: list[set[str]], names: list[str], k: int) -> float:
    """Share of images with a correct category in a fixed top-k (the baseline for conformal sets)."""
    hits = [any(names[res["cands"][i]] in g for i in np.argsort(-res["final"])[:k]) for res, g in zip(scored, gold)]
    return float(np.mean(hits)) if hits else 0.0
