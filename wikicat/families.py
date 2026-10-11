"""Content type of an image: Object, Living object, Scene, Event or Abstract concept.

The four families named in the problem statement plus living things (people,
animals, plants, fungi), which testers did not see as objects. An image's family scores
blend two signals:
  * a score-weighted vote of the families of its suggested categories, and
  * a zero-shot match between the image and short descriptions of each family,
    which keeps the answer sensible when the suggestions are weak.
"""
from __future__ import annotations

import numpy as np

from .metrics import category_type

FAMILIES = ("Object", "Living object", "Scene", "Event", "Abstract concept")
# Names used by earlier versions (saved classifiers, training folders).
LEGACY_FAMILIES = {"Living thing": "Living object"}
_TYPE_TO_FAMILY = {"object": "Object", "living": "Living object", "scene": "Scene", "event": "Event",
                   "abstract": "Abstract concept"}
FAMILY_PROMPTS = {
    "Object": ["a photo of a single object", "a close-up photo of a bag, tool, device or other thing",
               "a photo of food or a household item", "a photo of a vehicle, fence, wire or other man-made object"],
    "Living object": ["a photo of an animal", "a photo of a plant, flower or tree",
                     "a photo of a mushroom, insect or bird in nature", "a portrait photo of a person"],
    "Scene": ["a photo of a landscape", "a photo of a street, square, room or other place",
              "a wide view of a city or of nature", "a photo of a building, church, temple or famous landmark"],
    "Event": ["a photo of an event such as a festival, ceremony, parade or sports match",
              "a crowd of people at a celebration, race or protest"],
    "Abstract concept": ["an abstract painting or digital artwork", "a pattern or texture with no recognisable objects",
                         "a symbol, logo or diagram"],
}


# What the main subject is, used to adjust how the content type is shown: people are rarely the
# "type" an editor wants (a parade is an Event, a selfie at a beach a Scene), and statues are not alive.
SUBJECT_PROMPTS = {
    "person": ["a photo of a person", "a photo of people", "a selfie", "a portrait photo of a person",
               "a group of people posing"],
    "animal or plant": ["a photo of an animal", "a photo of a bird", "a photo of an insect",
                        "a photo of a plant or flower", "a photo of a mushroom"],
    # statues and other depictions: paintings and framed pictures of people or animals are not alive
    "statue": ["a photo of a statue", "a photo of a sculpture", "a bronze statue of a person",
               "a stone statue of an animal", "a monument with a statue", "a framed picture on a wall",
               "a painting of a person or an animal"],
    "other": ["a photo of a building", "a photo of a landscape", "a photo of an object", "an abstract pattern",
              "a photo of a street"],
}


def adjust_for_subject(families: list[dict], subject: dict[str, float]) -> tuple[list[dict], str]:
    """Statues are never living; when people are the main subject, the type of the scene leads and
    Living object comes second. Returns the families and 'statue', 'people' or ''."""
    scores = {f["family"]: f["score"] for f in families}
    living = scores.get("Living object", 0.0)
    if subject.get("statue", 0) >= 0.5 and subject["statue"] > subject.get("person", 0):
        scores["Object"] = scores.get("Object", 0.0) + living
        scores["Living object"] = 0.0
        order = sorted(scores, key=lambda f: -scores[f])
        return [{"family": f, "score": round(scores[f], 3)} for f in order], "statue"
    if subject.get("person", 0) >= 0.5 and families and families[0]["family"] == "Living object":
        rest = {f: v for f, v in scores.items() if f != "Living object"}
        total = sum(rest.values()) or 1.0
        order = sorted(rest, key=lambda f: -rest[f])
        out = [{"family": order[0], "score": round(rest[order[0]] / total, 3)},
               {"family": "Living object", "score": round(living, 3)}]
        out += [{"family": f, "score": round(rest[f] / total, 3)} for f in order[1:]]
        return out, "people"
    return families, ""


def family_of(category: str) -> str:
    return _TYPE_TO_FAMILY[category_type(category)]


class FamilyScorer:
    """Blends a vote of the suggested categories' types with an image-level check.

    The image-level check is the trained content-type classifier when available
    (artifacts/content_type.npz), otherwise a zero-shot match against prompts.
    """

    def __init__(self, encoder, vote_weight: float | None = None, temperature: float = 0.01, classifier=None):
        self.classifier = classifier
        self.vote_weight = vote_weight if vote_weight is not None else (0.35 if classifier is not None else 0.5)
        self.temperature = temperature
        emb = np.stack([encoder.encode_texts(FAMILY_PROMPTS[f]).mean(axis=0) for f in FAMILIES])
        self.prototypes = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sub = np.stack([encoder.encode_texts(v).mean(axis=0) for v in SUBJECT_PROMPTS.values()])
        self.subject_prototypes = sub / np.maximum(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12)

    def subject(self, img_emb: np.ndarray) -> dict[str, float]:
        """Zero-shot probabilities that the main subject is a person, an animal or plant, a statue or other."""
        logits = (self.subject_prototypes @ img_emb.astype(np.float32)) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(SUBJECT_PROMPTS, p.round(3).tolist()))

    def zero_shot(self, img_emb: np.ndarray) -> np.ndarray:
        logits = (self.prototypes @ img_emb.astype(np.float32)) / self.temperature
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def image_check(self, img_emb: np.ndarray) -> np.ndarray:
        if self.classifier is not None:
            return self.classifier.predict_proba(img_emb)[0]
        return self.zero_shot(img_emb)

    def score(self, img_emb: np.ndarray, suggestions: list[tuple[str, float]], top: int = 10) -> list[dict]:
        """[{'family': ..., 'score': share}] sorted by share; shares sum to 1."""
        zs = self.image_check(img_emb)
        votes = np.zeros(len(FAMILIES))
        for name, s in suggestions[:top]:
            votes[FAMILIES.index(family_of(name))] += np.sqrt(max(float(s), 0.0))
        votes = votes / votes.sum() if votes.sum() > 0 else zs
        mix = self.vote_weight * votes + (1 - self.vote_weight) * zs
        order = np.argsort(-mix)
        return [{"family": FAMILIES[i], "score": round(float(mix[i]), 3)} for i in order]
