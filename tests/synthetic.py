"""Tiny synthetic Commons-like dataset for offline tests.

Coloured shapes on noisy backgrounds, each with a specific category
("Red circles"), its broader parents ("Circles", "Red objects") and sometimes
an event category with a year. Only for plumbing tests: no network, no weights.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wikicat.data_loader import assign_temporal_splits, save_records

COLORS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (40, 70, 220), "yellow": (235, 210, 30)}
SHAPES = ("circle", "square", "triangle")


def draw(color: str, shape: str, rng: np.random.Generator, size: int = 96) -> Image.Image:
    bg = rng.integers(200, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    r = int(rng.integers(size // 4, size // 2 - 4))
    cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, size=2))
    box = [cx - r, cy - r, cx + r, cy + r]
    fill = tuple(int(np.clip(c + rng.integers(-20, 20), 0, 255)) for c in COLORS[color])
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    return img


def make_dataset(root: Path, n: int = 240, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data_dir = Path(root) / "data"
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 1)
    records, parents = [], {}
    for i in range(n):
        color = str(rng.choice(list(COLORS)))
        shape = str(rng.choice(SHAPES))
        specific = f"{color.capitalize()} {shape}s"
        cats = [specific]
        parents[specific] = [f"{shape.capitalize()}s", f"{color.capitalize()} objects"]
        parents[f"{shape.capitalize()}s"] = ["Shapes"]
        parents[f"{color.capitalize()} objects"] = ["Objects by color"]
        if i % 3 == 0:
            cats.append(f"{shape.capitalize()}s")  # some editors also add the broad category
        year = 2020 + int(rng.integers(0, 3))
        if i % 5 == 0:
            cats.append(f"Shape festival {year}")
        path = data_dir / "images" / f"{i}.png"
        draw(color, shape, rng).save(path)
        records.append({
            "id": str(i),
            "title": f"File:{color}_{shape}_{year}_{i}.png",
            "description": f"A {color} {shape}" if i % 4 else "",
            "date_original": "",
            "timestamp": (start + timedelta(hours=i)).isoformat() + "Z",
            "uploader": f"user{i % 7}",
            "width": 96, "height": 96, "mime": "image/png",
            "thumb_url": "", "page_url": f"https://example.org/{i}",
            "image_path": str(path),
            "categories": cats,
            "seed": "Shapes",
            "split": "",
        })
    records = assign_temporal_splits(records, 0.2, 0.2)
    save_records(records, data_dir / "records.jsonl")
    (data_dir / "hierarchy.json").write_text(json.dumps(parents))
    (data_dir / "seeds.json").write_text(json.dumps(["Shapes"]))
    return data_dir
