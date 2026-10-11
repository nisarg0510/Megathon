"""Over-specific category names that are hidden unless the system is very sure.

Editors want broadly useful suggestions first; date-stamped events
("Pacific Partnership 2022"), counts ("3 men and 1 woman", "1 automobile in ..."),
collections and containers ("... by Geoff Charles", "Trees by fruit",
"Milimedia files", "(review needed)"), country-level variants ("Black
automobiles in Germany", unless the title names the country) and names in a
non-Latin script are only shown when their score is high.
"""
from __future__ import annotations

import re
import unicodedata

from .geo import COUNTRY_NAMES

_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)s?\b")
_PEOPLE_COUNT = re.compile(r"\b\d+\s+(men|man|women|woman|people|persons?|boys?|girls?|children|child|adults?)\b",
                           re.IGNORECASE)
_COUNT = re.compile(r"^\d+\s+[a-z]")
_COLLECTION = re.compile(r"\bby\s|\bfiles\b|review needed", re.IGNORECASE)
_CODE = re.compile(r"^(Letters?|Numbers?|Digits?)\s+[A-Z0-9]{1,5}\s")
_IN_PLACE = re.compile(r"\bin\s+(.+)$")
MAX_WORDS = 7


def _non_latin(name: str) -> bool:
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in name)


def specificity_reason(name: str, context: str = "") -> str | None:
    """Why a category name counts as very specific, or None. `context` is the
    upload's title and description: a country named there is not too specific."""
    if _YEAR.search(name):
        return "date"
    if _PEOPLE_COUNT.search(name):
        return "people count"
    if _COUNT.search(name):
        return "count"
    if _COLLECTION.search(name):
        return "collection"
    if _CODE.search(name):
        return "code"
    place = _IN_PLACE.search(name)
    if place:
        country = place.group(1).strip().lower()
        if country in COUNTRY_NAMES and country.removeprefix("the ") not in context.lower():
            return "country-level"
    if _non_latin(name):
        return "non-Latin script"
    if len(name.split()) > MAX_WORDS:
        return "very long name"
    return None
