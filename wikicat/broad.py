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


class BroadCategoryScorer:
    def __init__(self, encoder, names=BROAD_CATEGORIES, temperature: float = 0.01):
        self.names = list(names)
        self.temperature = temperature
        emb = encoder.encode_categories(self.names)
        self.prototypes = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def top(self, img_emb: np.ndarray, k: int = 3) -> list[dict]:
        logits = (self.prototypes @ np.asarray(img_emb, dtype=np.float32)) / self.temperature
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[:k]
        return [{"category": self.names[i], "score": round(float(p[i]), 3)} for i in order]
