"""Candidate generation, feature fusion and hierarchy post-processing.

For one query the scorers each propose their top-N categories. Fusion runs only
on the union of these candidates (a few hundred, never all of C). Each
candidate gets a feature vector; a logistic-regression stacker fitted on the
validation split turns it into a calibrated probability. Finally, an ancestor
is pushed down when one of its descendants already scores high
("most specific category" convention on Commons).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data_loader import is_maintenance_category
from .geo import is_country
from .hierarchy import CategoryGraph
from .metadata import LexicalScorer, extract_years, year_feature

FEATURE_NAMES = [
    "knn",        # neighbour vote s_knn
    "knn_hit",    # at least one neighbour voted
    "txt",        # zero-shot cosine image . t_c
    "txt_z",      # same, standardised over this query's top-N
    "head",       # supervised head probability
    "in_head",    # category is covered by the head
    "meta",       # cosine metadata-text . t_c
    "meta_z",     # same, standardised over this query's top-N
    "lex",        # lexical overlap with title/description
    "year",       # year agreement (+1 / -1 / 0)
    "log_freq",   # log(1 + training frequency)
    "has_meta",   # metadata was supplied
    "lowq_knn",   # low-quality image x knn
    "lowq_txt",   # low-quality image x txt_z
    "lowq_lex",   # low-quality image x lex
]


@dataclass
class QueryContext:
    """Everything the individual scorers produced for one query."""

    img_emb: np.ndarray
    knn: dict[int, float]
    neighbor_sims: np.ndarray
    neighbor_idx: np.ndarray
    txt_ids: np.ndarray
    txt_sims: np.ndarray
    head_probs: np.ndarray | None = None
    meta_emb: np.ndarray | None = None
    meta_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    meta_sims: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    meta_tokens: set[str] = field(default_factory=set)
    meta_years: set[int] = field(default_factory=set)
    low_quality: bool = False

    @property
    def has_meta(self) -> bool:
        return self.meta_emb is not None


class FusionModel:
    """Logistic regression over standardised candidate features."""

    def __init__(self, coef: np.ndarray, intercept: float, mean: np.ndarray, scale: np.ndarray,
                 feature_names: list[str] = FEATURE_NAMES, fitted: bool = False):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.fitted = fitted

    @classmethod
    def default(cls) -> "FusionModel":
        """Hand-set weights, used before a fusion model has been fitted."""
        w = dict(knn=5.0, knn_hit=0.3, txt_z=0.5, head=3.0, meta_z=0.3, lex=2.0, year=0.5, log_freq=0.1)
        coef = np.array([w.get(n, 0.0) for n in FEATURE_NAMES])
        return cls(coef, -4.0, np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, c: float = 1.0) -> "FusionModel":
        from sklearn.linear_model import LogisticRegression

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit((x - mean) / scale, y)
        return cls(model.coef_[0], model.intercept_[0], mean, scale, fitted=True)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.coef + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logits(x), -30, 30)))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Logit added by each feature relative to that feature being absent (0).

        Used for explanations, so a missing signal never shows up as a reason.
        """
        return (x / self.scale) * self.coef

    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names, "coef": self.coef.tolist(), "intercept": self.intercept,
                "mean": self.mean.tolist(), "scale": self.scale.tolist(), "fitted": self.fitted}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "FusionModel":
        d = json.loads(Path(path).read_text())
        if d["feature_names"] != FEATURE_NAMES:
            raise ValueError("Fusion model was trained with a different feature set; re-run train.py")
        return cls(d["coef"], d["intercept"], d["mean"], d["scale"], d["feature_names"], d.get("fitted", True))


def suppress_ancestors(names: list[str], scores: np.ndarray, graph: CategoryGraph, depth: int = 3,
                       ratio: float = 0.5, penalty: float = 0.3, n_confident: int = 20,
                       window: int = 60) -> tuple[np.ndarray, dict[int, int]]:
    """Down-weight a candidate when a descendant of it is already confident.

    names/scores are aligned candidate arrays. An ancestor that scores high
    while no descendant does is kept unchanged.
    Returns (adjusted scores, {suppressed index: descendant index}).
    """
    adjusted = scores.astype(np.float64).copy()
    suppressed: dict[int, int] = {}
    if not len(graph) or not len(scores):
        return adjusted, suppressed
    order = np.argsort(-scores)
    confident = order[:n_confident]
    anc = {int(d): graph.ancestors(names[d], depth) for d in confident}
    for i in order[:window]:
        i = int(i)
        for d in confident:
            d = int(d)
            if d != i and scores[d] >= ratio * scores[i] and names[i] in anc[d]:
                adjusted[i] = scores[i] * penalty
                suppressed[i] = d
                break
    return adjusted, suppressed


class Ranker:
    def __init__(self, names: list[str], cat_freq: np.ndarray, prototypes, head, lexical: LexicalScorer,
                 fusion: FusionModel, graph: CategoryGraph, top_n: int = 100, hierarchy_depth: int = 3,
                 ancestor_ratio: float = 0.5, ancestor_penalty: float = 0.3, hide_countries: bool = False):
        self.names = names
        self.cat_freq = cat_freq
        self.prototypes = prototypes
        self.head = head
        self.lexical = lexical
        self.fusion = fusion
        self.graph = graph
        self.top_n = top_n
        self.hierarchy_depth = hierarchy_depth
        self.ancestor_ratio = ancestor_ratio
        self.ancestor_penalty = ancestor_penalty
        self._cat_years = [extract_years(n) for n in names]
        self.hide_countries = hide_countries
        self._blocked = {i for i, n in enumerate(names) if self._is_blocked(n)}

    def on_categories_added(self) -> None:
        """Keep per-category caches in sync after the vocabulary grows."""
        start = len(self._cat_years)
        self._cat_years.extend(extract_years(n) for n in self.names[start:])
        self._blocked.update(i for i in range(start, len(self.names)) if self._is_blocked(self.names[i]))

    def _is_blocked(self, name: str) -> bool:
        return is_maintenance_category(name) or (self.hide_countries and is_country(name))

    # -- candidates and features ---------------------------------------------

    def candidates(self, ctx: QueryContext) -> np.ndarray:
        cands: set[int] = set()
        knn_top = sorted(((c, s) for c, s in ctx.knn.items() if s > 0), key=lambda x: -x[1])[:self.top_n]
        cands.update(c for c, _ in knn_top)
        cands.update(ctx.txt_ids[:self.top_n].tolist())
        if ctx.head_probs is not None and self.head is not None and len(self.head):
            n = min(self.top_n, len(self.head))
            top = np.argpartition(-ctx.head_probs, n - 1)[:n]
            cands.update(self.head.cat_ids[top].tolist())
        if ctx.has_meta:
            cands.update(ctx.meta_ids[:self.top_n].tolist())
            cands.update(self.lexical.top(ctx.meta_tokens, self.top_n).keys())
        cands.discard(-1)
        cands -= self._blocked
        return np.array(sorted(cands), dtype=np.int64)

    def features(self, ctx: QueryContext, cands: np.ndarray) -> np.ndarray:
        n = len(cands)
        x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
        if n == 0:
            return x
        lowq = 1.0 if ctx.low_quality else 0.0

        x[:, 0] = [ctx.knn.get(int(c), 0.0) for c in cands]
        x[:, 1] = [1.0 if int(c) in ctx.knn else 0.0 for c in cands]

        txt = self.prototypes.similarity(ctx.img_emb, cands)
        mu, sd = _stats(ctx.txt_sims)
        x[:, 2] = txt
        x[:, 3] = (txt - mu) / sd

        if ctx.head_probs is not None and self.head is not None:
            cols = [self.head.col.get(int(c), -1) for c in cands]
            x[:, 4] = [ctx.head_probs[j] if j >= 0 else 0.0 for j in cols]
            x[:, 5] = [1.0 if j >= 0 else 0.0 for j in cols]

        if ctx.has_meta:
            meta = self.prototypes.similarity(ctx.meta_emb, cands)
            mu, sd = _stats(ctx.meta_sims)
            x[:, 6] = meta
            x[:, 7] = (meta - mu) / sd
            x[:, 8] = [self.lexical.score_one(int(c), ctx.meta_tokens) for c in cands]
            x[:, 9] = [year_feature(self._cat_years[int(c)], ctx.meta_years) for c in cands]
            x[:, 11] = 1.0

        x[:, 10] = np.log1p(self.cat_freq[cands])
        x[:, 12] = lowq * x[:, 0]
        x[:, 13] = lowq * x[:, 3]
        x[:, 14] = lowq * x[:, 8]
        return x

    # -- scoring ---------------------------------------------------------------

    def score(self, ctx: QueryContext, use_hierarchy: bool = True) -> dict:
        cands = self.candidates(ctx)
        x = self.features(ctx, cands)
        prob = self.fusion.predict_proba(x) if len(cands) else np.zeros(0)
        names = [self.names[c] for c in cands]
        if use_hierarchy:
            final, suppressed = suppress_ancestors(names, prob, self.graph, self.hierarchy_depth,
                                                   self.ancestor_ratio, self.ancestor_penalty)
        else:
            final, suppressed = prob.copy(), {}
        return {"cands": cands, "features": x, "prob": prob, "final": final, "suppressed": suppressed}

    def rank(self, ctx: QueryContext, top_k: int = 10, use_hierarchy: bool = True, res: dict | None = None) -> list[dict]:
        res = res if res is not None else self.score(ctx, use_hierarchy)
        order = np.argsort(-res["final"])[:top_k]
        contrib = self.fusion.contributions(res["features"][order]) if len(order) else np.zeros((0, 0))
        out = []
        for row, i in enumerate(order.tolist()):
            c = int(res["cands"][i])
            feats = dict(zip(FEATURE_NAMES, res["features"][i].tolist()))
            out.append({
                "cat_id": c,
                "category": self.names[c],
                "score": float(res["final"][i]),
                "probability": float(res["prob"][i]),
                "signals": {k: round(feats[k], 4) for k in ("knn", "txt", "head", "meta", "lex", "year")},
                "contributions": {k: round(float(v), 3) for k, v in zip(FEATURE_NAMES, contrib[row]) if abs(v) > 0.05},
                "suppressed_by": self.names[int(res["cands"][res["suppressed"][i]])] if i in res["suppressed"] else None,
            })
        return out


def _stats(values: np.ndarray) -> tuple[float, float]:
    if values is None or len(values) < 2:
        return 0.0, 1.0
    sd = float(np.std(values))
    return float(np.mean(values)), (sd if sd > 1e-6 else 1.0)
