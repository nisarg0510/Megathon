"""Ranking metrics for multi-label category suggestion.

Sample-wise (per image, then averaged):
    P@k = |top-k ∩ Y| / k,   R@k = |top-k ∩ Y| / |Y|,   F1@k = 2PR / (P + R)
Category-wise: recall@k per category, grouped into head / torso / tail /
unseen by training frequency; mAP over categories present in the test set;
hierarchy-aware precision where an ancestor or descendant of a true label
counts as a hit.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .hierarchy import CategoryGraph

FREQ_BUCKETS = (("unseen (0)", 0, 0), ("tail (1-4)", 1, 4), ("torso (5-19)", 5, 19), ("head (20+)", 20, 10**12))


def prf_at_k(preds: list[list[str]], gold: list[set[str]], k: int) -> dict[str, float]:
    p = r = f = 0.0
    n = 0
    for pred, y in zip(preds, gold):
        if not y:
            continue
        hits = len(set(pred[:k]) & y)
        pi, ri = hits / k, hits / len(y)
        p += pi
        r += ri
        f += 2 * pi * ri / (pi + ri) if hits else 0.0
        n += 1
    n = max(n, 1)
    return {"P": p / n, "R": r / n, "F1": f / n}


def hierarchical_precision_at_k(preds: list[list[str]], gold: list[set[str]], graph: CategoryGraph,
                                k: int, depth: int = 3) -> float:
    total, n = 0.0, 0
    for pred, y in zip(preds, gold):
        if not y:
            continue
        hits = sum(1 for p in pred[:k] if any(graph.related(p, t, depth) for t in y))
        total += hits / k
        n += 1
    return total / max(n, 1)


def label_coverage(gold: list[set[str]], vocab: set[str]) -> float:
    """Share of test labels the system can output at all (upper bound on recall)."""
    total = sum(len(y) for y in gold)
    return sum(1 for y in gold for c in y if c in vocab) / max(total, 1)


def per_category_recall(preds: list[list[str]], gold: list[set[str]], k: int) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for pred, y in zip(preds, gold):
        top = set(pred[:k])
        for c in y:
            stats[c][1] += 1
            stats[c][0] += c in top
    return {c: (h, t) for c, (h, t) in stats.items()}


def bucket_report(per_cat: dict[str, tuple[int, int]], train_freq: dict[str, float]) -> dict[str, dict]:
    out = {}
    for name, lo, hi in FREQ_BUCKETS:
        cats = [c for c in per_cat if lo <= train_freq.get(c, 0) <= hi]
        if not cats:
            out[name] = {"categories": 0, "labels": 0, "macro_recall": None, "micro_recall": None}
            continue
        hits = sum(per_cat[c][0] for c in cats)
        labels = sum(per_cat[c][1] for c in cats)
        out[name] = {
            "categories": len(cats),
            "labels": labels,
            "macro_recall": float(np.mean([per_cat[c][0] / per_cat[c][1] for c in cats])),
            "micro_recall": hits / labels,
        }
    return out


def mean_average_precision(scores: list[dict[str, float]], gold: list[set[str]], min_positives: int = 1) -> float:
    """Macro mAP over categories with at least `min_positives` test images.

    Images where the category was not a candidate get score 0, so a category
    the system never proposes for a positive image is penalised.
    """
    from sklearn.metrics import average_precision_score

    cats = defaultdict(int)
    for y in gold:
        for c in y:
            cats[c] += 1
    aps = []
    for c, count in cats.items():
        if count < min_positives:
            continue
        y_true = np.array([c in y for y in gold], dtype=np.int8)
        y_score = np.array([s.get(c, 0.0) for s in scores], dtype=np.float64)
        aps.append(average_precision_score(y_true, y_score))
    return float(np.mean(aps)) if aps else 0.0


# Coarse, transparent heuristic used only to stratify the evaluation report.
_EVENT = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b|festival|protest|demonstration|election|championship|match(es)?\b|"
                    r"tournament|ceremony|parade|concert|conference|wikimania|olympic|war\b|battle|earthquake|"
                    r"flood|celebration|wedding|carnival|marathon|diwali|holi\b|christmas|ramadan|eid\b|games\b|"
                    r"visit|meeting|summit|exhibition|fair\b|race\b|cup\b|strike", re.IGNORECASE)
_SCENE = re.compile(r"landscape|view|panorama|skyline|street|interior|beach|sunset|sunrise|night|forest|mountain|"
                    r"river|lake|park|garden|square|market|harbou?r|port\b|coast|valley|aerial|cityscape|rural|"
                    r"village|town|city|desert|waterfall|field|sky|clouds|road|station|airport|campus|"
                    # buildings and landmarks are places (Scene), not movable objects
                    r"church|cathedral|basilica|chapel|mosque|temple|synagogue|monaster|castle|palace|fort\b|"
                    r"fortress|tower|bridge|lighthouse|skyscraper|building|\bhouses\b|monument|museum|stadium|"
                    r"fa[cç]ade|architecture|landmark|ruins",
                    re.IGNORECASE)
_ABSTRACT = re.compile(r"\bart\b|culture|religion|history|politic|science|economy|education|music|sport|"
                       r"technology|love|death|poverty|peace|freedom|emotion|symbol|pattern|colou?r|geometry|"
                       r"light\b|shadow|reflection|abstract|concept|style|tradition|humou?r|"
                       r"design|motion|silhouette|symmetry|minimalism|mythology|literature", re.IGNORECASE)


def category_type(name: str) -> str:
    if _EVENT.search(name):
        return "event"
    if _ABSTRACT.search(name):
        return "abstract"
    if _SCENE.search(name):
        return "scene"
    return "object"


def type_report(preds: list[list[str]], gold: list[set[str]], k: int) -> dict[str, dict]:
    """Recall of true labels and precision of predictions, per category type."""
    rec = defaultdict(lambda: [0, 0])
    prec = defaultdict(lambda: [0, 0])
    for pred, y in zip(preds, gold):
        top = pred[:k]
        for c in y:
            t = category_type(c)
            rec[t][1] += 1
            rec[t][0] += c in top
        for p in top:
            t = category_type(p)
            prec[t][1] += 1
            prec[t][0] += p in y
    out = {}
    for t in ("object", "scene", "event", "abstract"):
        out[t] = {
            "labels": rec[t][1],
            "recall": rec[t][0] / rec[t][1] if rec[t][1] else None,
            "predictions": prec[t][1],
            "precision": prec[t][0] / prec[t][1] if prec[t][1] else None,
        }
    return out


def percentiles(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    a = np.asarray(values_ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}
