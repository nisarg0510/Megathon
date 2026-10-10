"""Content type of an image: Object, Scene, Event or Abstract concept.

The four families named in the problem statement. An image's family scores
blend two signals:
  * a score-weighted vote of the families of its suggested categories, and
  * a zero-shot match between the image and short descriptions of each family,
    which keeps the answer sensible when the suggestions are weak.
"""
from __future__ import annotations

import numpy as np

from .metrics import category_type

FAMILIES = ("Object", "Scene", "Event", "Abstract concept")
_TYPE_TO_FAMILY = {"object": "Object", "scene": "Scene", "event": "Event", "abstract": "Abstract concept"}
FAMILY_PROMPTS = {
    "Object": ["a photo of a single object", "a close-up photo of a bag, tool, device or other thing",
               "a photo of an animal or a plant", "a photo of a vehicle, fence, wire, building or other structure"],
    "Scene": ["a photo of a landscape", "a photo of a street, room or other place",
              "a wide view of a city or of nature"],
    "Event": ["a photo of an event such as a festival, ceremony, parade or sports match",
              "a crowd of people at a celebration, race or protest"],
    "Abstract concept": ["an abstract painting or digital artwork", "a pattern or texture with no recognisable objects",
                         "a symbol, logo or diagram"],
}


def family_of(category: str) -> str:
    return _TYPE_TO_FAMILY[category_type(category)]


class FamilyScorer:
    def __init__(self, encoder, vote_weight: float = 0.5, temperature: float = 0.01):
        self.vote_weight = vote_weight
        self.temperature = temperature
        emb = np.stack([encoder.encode_texts(FAMILY_PROMPTS[f]).mean(axis=0) for f in FAMILIES])
        self.prototypes = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def zero_shot(self, img_emb: np.ndarray) -> np.ndarray:
        logits = (self.prototypes @ img_emb.astype(np.float32)) / self.temperature
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def score(self, img_emb: np.ndarray, suggestions: list[tuple[str, float]], top: int = 10) -> list[dict]:
        """[{'family': ..., 'score': share}] sorted by share; shares sum to 1."""
        zs = self.zero_shot(img_emb)
        votes = np.zeros(len(FAMILIES))
        for name, s in suggestions[:top]:
            votes[FAMILIES.index(family_of(name))] += max(float(s), 0.0)
        votes = votes / votes.sum() if votes.sum() > 0 else zs
        mix = self.vote_weight * votes + (1 - self.vote_weight) * zs
        order = np.argsort(-mix)
        return [{"family": FAMILIES[i], "score": round(float(mix[i]), 3)} for i in order]
