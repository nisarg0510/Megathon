"""Over-specific category names that are hidden unless the system is very sure.

Editors want broadly useful suggestions first; date-stamped events
("Pacific Partnership 2022"), people counts ("3 men and 1 woman") and names in
a non-Latin script are only shown when their score is high.
"""
from __future__ import annotations

import re
import unicodedata

_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)s?\b")
_PEOPLE_COUNT = re.compile(r"\b\d+\s+(men|man|women|woman|people|persons?|boys?|girls?|children|child|adults?)\b",
                           re.IGNORECASE)
MAX_WORDS = 7


def _non_latin(name: str) -> bool:
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in name)


def specificity_reason(name: str) -> str | None:
    """Why a category name counts as very specific, or None."""
    if _YEAR.search(name):
        return "date"
    if _PEOPLE_COUNT.search(name):
        return "people count"
    if _non_latin(name):
        return "non-Latin script"
    if len(name.split()) > MAX_WORDS:
        return "very long name"
    return None
