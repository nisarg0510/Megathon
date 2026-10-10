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
    "Object": ["a photo of an object", "a close-up photo of an animal, plant or thing",
               "a photo of a vehicle, building or artefact"],
    "Scene": ["a photo of a landscape", "a photo of a place or a street", "a photo of an indoor or outdoor scene"],
    "Event": ["a photo of an event", "a photo of a festival, ceremony, protest or sports match",
              "a photo of a crowd at a celebration"],
    "Abstract concept": ["an abstract image", "a pattern, symbol or texture", "an artwork expressing an idea or emotion"],
}


def family_of(category: str) -> str:
    return _TYPE_TO_FAMILY[category_type(category)]


class FamilyScorer:
    def __init__(self, encoder, vote_weight: float = 0.7, temperature: float = 0.01):
        self.vote_weight = vote_weight
        self.temperature = temperature
        prompts = [p for f in FAMILIES for p in FAMILY_PROMPTS[f]]
        emb = encoder.encode_texts(prompts).reshape(len(FAMILIES), -1, encoder.dim).mean(axis=1)
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
