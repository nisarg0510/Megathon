"""Broad category: a zero-shot match against a short list of broad Commons categories.

Testers asked for a general answer first ("Automobiles", not "Black automobiles
in Germany"). Choosing among about a hundred broad categories is something a
CLIP-style encoder does well without any training, unlike choosing among tens
of thousands. Shown for orientation next to the content type; it does not
change the ranked suggestions.
"""
from __future__ import annotations

import numpy as np

BROAD_CATEGORIES = (
    # people and living things
    "Selfies", "Group selfies", "Portraits", "Groups of people", "Children", "Crowds", "Animals", "Dogs", "Cats", "Birds", "Horses", "Cattle",
    "Insects", "Butterflies", "Fish", "Reptiles", "Plants", "Flowers", "Trees", "Mushrooms",
    # objects
    "Automobiles", "Buses", "Trucks", "Motorcycles", "Bicycles", "Trains", "Aircraft", "Ships", "Boats", "Food",
    "Fruits", "Vegetables", "Drinks", "Furniture", "Tools", "Musical instruments", "Computers", "Mobile phones",
    "Books", "Toys", "Clothing", "Shoes", "Bags", "Bottles", "Clocks", "Coins", "Jewellery", "Weapons", "Masks",
    "Signs", "Flags", "Statues",
    # places and scenes
    "Buildings", "Houses", "Churches", "Mosques", "Hindu temples", "Buddhist temples", "Castles", "Bridges",
    "Skyscrapers", "Monuments and memorials", "Streets", "Roads", "Markets", "Interiors", "Gardens", "Landscapes",
    "Mountains", "Beaches", "Rivers", "Lakes", "Waterfalls", "Forests", "Deserts", "Cityscapes", "Railway stations",
    "Sky", "Sunsets", "Night",
    # events
    "Association football", "Cricket", "Sports", "Running", "Concerts", "Festivals", "Parades", "Protests",
    "Weddings", "Ceremonies", "Conferences", "Fireworks",
    # art and abstract
    "Paintings", "Drawings", "Sculptures", "Maps", "Diagrams", "Logos", "Text", "Patterns", "Textures", "Shadows",
    "Abstract art", "Graffiti", "Mosaics",
)


# Broad categories that fit when the main subject is a person or a statue (see families.SUBJECT_PROMPTS).
PEOPLE_BROAD = ("Selfies", "Group selfies", "Portraits", "Groups of people", "Children", "Crowds")
STATUE_BROAD = ("Statues", "Sculptures", "Monuments and memorials", "Paintings")


class BroadCategoryScorer:
    def __init__(self, encoder, names=BROAD_CATEGORIES, temperature: float = 0.01):
        self.names = list(names)
        self.temperature = temperature
        emb = encoder.encode_categories(self.names)
        self.prototypes = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def top(self, img_emb: np.ndarray, k: int = 3, subject: dict[str, float] | None = None) -> list[dict]:
        """Best broad categories. When the subject check is confident that the main subject is a person
        (or a statue), the best matching people (statue) category is put first, so a portrait is never
        summarised by its background."""
        logits = (self.prototypes @ np.asarray(img_emb, dtype=np.float32)) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = list(np.argsort(-p))
        subject = subject or {}
        prefer = PEOPLE_BROAD if subject.get("person", 0) >= 0.5 else (
            STATUE_BROAD if subject.get("statue", 0) >= 0.5 else ())
        if prefer:
            first = next(i for i in order if self.names[i] in prefer)
            order = [first] + [i for i in order if i != first]
        return [{"category": self.names[i], "score": round(float(p[i]), 3)} for i in order[:k]]
