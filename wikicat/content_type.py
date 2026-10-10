"""Trained content-type classifier: Object / Living thing / Scene / Event / Abstract concept.

A multinomial logistic regression on frozen image embeddings, trained on
openly licensed Wikimedia Commons photos collected per type (see
prepare_content_types.py). It replaces the prompt-only zero-shot check inside
the content-type scorer when present. Also evaluates on curated test packs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .families import FAMILIES

# Commons categories used to collect training photos for each type.
TYPE_SEEDS = {
    "Object": ["Tools", "Furniture", "Bags", "Bottles", "Cups", "Chairs", "Clocks", "Bicycles", "Automobiles",
               "Musical instruments", "Toys", "Shoes", "Computers", "Lamps", "Books", "Food", "Statues", "Vases"],
    # People are living things; they get a third of the type so portraits and group photos are not read as events.
    "Living thing": ["Cats", "Dogs", "Birds", "Insects", "Flowers", "Trees", "Mushrooms", "Horses", "Fish",
                     "Reptiles", "Butterflies", "Cattle", "Portrait photographs", "Smiling people",
                     "Group photographs", "Portrait photographs of men", "Portrait photographs of women",
                     "Cartes de visite", "Selfies"],
    "Scene": ["Landscapes", "Streets", "Beaches", "Mountains", "Cityscapes", "Forests", "Rivers", "Interiors",
              "Deserts", "Skylines", "Waterfalls", "Lakes", "Parks", "Markets", "Churches", "Bridges"],
    "Event": ["Festivals", "Protests", "Parades", "Concerts", "Weddings", "Association football matches",
              "Marathons", "Carnivals", "Fireworks", "Conferences", "Ceremonies", "Processions"],
    "Abstract concept": ["Patterns", "Textures", "Abstract art", "Symmetry", "Reflections", "Shadows", "Fractals",
                         "Geometric shapes", "Symbols", "Light painting", "Silhouettes", "Mosaics"],
}

# Test packs use four broad families; living things count as objects there.
TO_FOUR = {"Object": "objects", "Living thing": "objects", "Scene": "scenes", "Event": "events",
           "Abstract concept": "abstract concepts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class ContentTypeClassifier:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: list[str], model_name: str = ""):
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.classes = list(classes)
        self.model_name = model_name

    @classmethod
    def fit(cls, x: np.ndarray, labels: list[str], c: float = 2.0, model_name: str = "") -> "ContentTypeClassifier":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        model.fit(x, labels)
        coef, intercept = model.coef_, model.intercept_
        if len(model.classes_) == 2:  # sklearn stores one row for binary problems
            coef, intercept = np.vstack([-coef, coef]), np.array([-intercept[0], intercept[0]])
        return cls(coef, intercept, list(model.classes_), model_name)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilities in FAMILIES order (families without training data get 0)."""
        logits = np.atleast_2d(x).astype(np.float32) @ self.coef.T + self.intercept
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out = np.zeros((len(p), len(FAMILIES)), dtype=np.float32)
        for j, c in enumerate(self.classes):
            out[:, FAMILIES.index(c)] = p[:, j]
        return out

    def save(self, path: str | Path) -> None:
        np.savez(path, coef=self.coef, intercept=self.intercept, classes=np.array(self.classes),
                 model_name=np.array(self.model_name))

    @classmethod
    def load(cls, path: str | Path) -> "ContentTypeClassifier":
        d = np.load(path, allow_pickle=False)
        return cls(d["coef"], d["intercept"], [str(c) for c in d["classes"]], str(d["model_name"]))


def image_folder_dataset(root: str | Path) -> tuple[list[str], list[str]]:
    """Images under root/<Type name>/... (any depth). Folder names must match FAMILIES."""
    paths, labels = [], []
    for family in FAMILIES:
        folder = Path(root) / family
        if folder.is_dir():
            for p in sorted(folder.rglob("*")):
                if p.suffix.lower() in IMAGE_EXT:
                    paths.append(str(p))
                    labels.append(family)
    return paths, labels


def commons_title(url_or_title: str) -> str:
    """'https://commons.wikimedia.org/wiki/File:A_b.jpg' or 'File:A b.jpg' -> 'a b.jpg'."""
    t = unquote(url_or_title.split("/wiki/")[-1])
    if ":" in t:
        t = t.split(":", 1)[1]
    return t.replace("_", " ").strip().lower()


PACK_FILES = (("labels.csv", "primary"), ("answers.csv", "expected_category"))


def find_pack_file(pack: str | Path) -> Path:
    """labels.csv / answers.csv for a pack given as its folder, a parent folder, the CSV itself, or the .zip."""
    pack = Path(pack).expanduser()
    if pack.suffix.lower() == ".zip" and pack.is_file():
        import zipfile

        target = pack.with_suffix("")
        if not target.exists():
            with zipfile.ZipFile(pack) as zf:
                zf.extractall(target)
        pack = target
    if pack.is_file() and pack.name in dict(PACK_FILES):
        return pack
    if not pack.exists():
        raise FileNotFoundError(f"Test pack not found: {pack}")
    found = [f for name, _ in PACK_FILES for f in pack.rglob(name)]
    if not found:
        raise FileNotFoundError(f"No labels.csv or answers.csv in {pack} or its subfolders")
    return min(found, key=lambda f: len(f.parts))


def load_pack(pack: str | Path) -> list[dict]:
    """Rows of a test pack: photo-test-pack (labels.csv) or living-objects-dataset (answers.csv)."""
    f = find_pack_file(pack)
    pack_dir = f.parent
    for name, primary_col in PACK_FILES:
        if f.name == name:
            with f.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            out = []
            for r in rows:
                accepted = {a.strip() for a in (r.get("accepted_categories") or "").split(";") if a.strip()}
                out.append({
                    "path": str(pack_dir / r["path"]),
                    "primary": r[primary_col].strip(),
                    "accepted": accepted | {r[primary_col].strip()},
                    "confusing": str(r.get("confusing", "")).lower() == "true",
                    "living": str(r.get("living_subject", "")).lower() == "true",
                    "group": r.get("group", ""),
                    "title": commons_title(r.get("source_url", "")),
                })
            return out
    raise FileNotFoundError(f"Unrecognised test pack file: {f}")


def pack_scores(rows: list[dict], probs: np.ndarray) -> dict:
    """Strict (primary) and lenient (accepted) accuracy, split ordinary / confusing."""
    pred = [FAMILIES[i] for i in probs.argmax(axis=1)]
    res: dict = {}
    for name, idx in (("all", range(len(rows))),
                      ("ordinary", [i for i, r in enumerate(rows) if not r["confusing"]]),
                      ("confusing", [i for i, r in enumerate(rows) if r["confusing"]])):
        idx = list(idx)
        if not idx:
            continue
        res[name] = {
            "n": len(idx),
            "strict": float(np.mean([TO_FOUR[pred[i]] == rows[i]["primary"] for i in idx])),
            "lenient": float(np.mean([TO_FOUR[pred[i]] in rows[i]["accepted"] for i in idx])),
        }
    living = [i for i, r in enumerate(rows) if r["living"]]
    if living:
        res["living_as_living_thing"] = float(np.mean([pred[i] == "Living thing" for i in living]))
        people = [i for i in living if rows[i]["group"] == "people"]
        if people:
            res["people_as_living_thing"] = float(np.mean([pred[i] == "Living thing" for i in people]))
        groups = sorted({rows[i]["group"] for i in living if rows[i]["group"]})
        res["living_by_group"] = {g: {"n": sum(rows[i]["group"] == g for i in living),
                                      "as_living_thing": float(np.mean([pred[i] == "Living thing" for i in living
                                                                         if rows[i]["group"] == g]))}
                                  for g in groups}
    res["predicted"] = {f: pred.count(f) for f in FAMILIES}
    return res


def save_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1))
