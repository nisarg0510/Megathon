"""Metadata signals from the file title and description.

* `LexicalScorer`: IDF-weighted share of a category name's words that appear in
  the title/description. Strong for places, named objects and events, which
  uploaders usually name in the file title.
* `year_feature`: whether a year in the category name ("Diwali 2023 in India")
  agrees with years found in the metadata.

The dense metadata signal (text embedding of title + description against the
category prototypes) is computed in the pipeline with the shared encoder.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict

from .data_loader import title_to_text

STOPWORDS = frozenset("""
a an the and or of in on at to for from by with without into onto over under near
de la le les el los las da do das dos di del della der die das und von zu im
is are was were be been this that these those it its as
file files image images photo photos photograph photographs picture pictures
jpg jpeg png tif tiff gif svg webp view views
""".split())

_JUNK = re.compile(r"^(img|dsc|dscn|dscf|pxl|imgp|wp|cimg|sam|p\d+|\d{5,}|\d{1,3}|[a-f0-9]{8,})$")
_YEAR = re.compile(r"\b(1[89]\d\d|20[0-3]\d)\b")


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "xes", "sses")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("_", " ")
    out = []
    for tok in re.findall(r"\w+", text):
        if tok in STOPWORDS or _JUNK.match(tok) or len(tok) < 2:
            continue
        out.append(_stem(tok))
    return out


def extract_years(text: str) -> set[int]:
    return {int(y) for y in _YEAR.findall(text or "")}


def build_meta_text(title: str = "", description: str = "") -> str:
    parts = [title_to_text(title) if title else "", (description or "")[:300]]
    return ". ".join(p for p in parts if p).strip()


def year_feature(category_years: set[int], meta_years: set[int]) -> float:
    if not category_years or not meta_years:
        return 0.0
    return 1.0 if category_years & meta_years else -1.0


class LexicalScorer:
    def __init__(self, names: list[str] | None = None, max_postings: int = 5000):
        self.cat_tokens: list[tuple[str, ...]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.max_postings = max_postings
        self._idf: dict[str, float] = {}
        if names:
            self.add(names)

    def __len__(self) -> int:
        return len(self.cat_tokens)

    def add(self, names: list[str]) -> None:
        for name in names:
            cid = len(self.cat_tokens)
            toks = tuple(sorted(set(tokenize(name))))
            self.cat_tokens.append(toks)
            for t in toks:
                self.postings[t].append(cid)
        n = max(len(self.cat_tokens), 1)
        self._idf = {t: math.log(1 + n / (1 + len(p))) for t, p in self.postings.items()}

    def score_one(self, cid: int, tokens: set[str]) -> float:
        toks = self.cat_tokens[cid]
        if not toks or not tokens:
            return 0.0
        den = sum(self._idf[t] for t in toks)
        num = sum(self._idf[t] for t in toks if t in tokens)
        return num / den if den else 0.0

    def top(self, tokens: set[str], n: int) -> dict[int, float]:
        candidates: set[int] = set()
        for t in tokens:
            posting = self.postings.get(t)
            if posting and len(posting) <= self.max_postings:
                candidates.update(posting)
        scored = ((c, self.score_one(c, tokens)) for c in candidates)
        best = sorted((x for x in scored if x[1] > 0), key=lambda x: -x[1])[:n]
        return dict(best)
