"""Run configuration.

A copy of the config is written next to the built artifacts, so inference and
evaluation always use the same encoder and scoring settings the index was
built with.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # Encoder: any Hugging Face CLIP or SigLIP checkpoint, or "hashing" for offline tests
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"  # auto | cpu | cuda | mps
    batch_size: int = 32

    # Instance retrieval (k-NN voting)
    knn_k: int = 50
    knn_tau: float = 0.05
    knn_reject_weight: float = 1.0  # weight of editor-rejected labels as negative votes
    index_type: str = "flat"  # "flat" or any FAISS factory string, e.g. "HNSW32", "IVF1024,PQ64"

    # Candidate generation: top-N per scorer before fusion
    top_n: int = 100

    # Supervised head
    head_min_freq: int = 3
    head_epochs: int = 40
    head_lr: float = 2e-3
    head_l2_to_init: float = 1e-3

    # Fusion training augmentation on the validation split
    fusion_drop_meta_frac: float = 0.3
    fusion_degrade_frac: float = 0.3

    # Hierarchy post-processing
    hierarchy_depth: int = 3
    ancestor_ratio: float = 0.5  # descendant must reach this fraction of the ancestor's score
    ancestor_penalty: float = 0.3  # multiplier applied to a suppressed ancestor

    # Vocabulary
    vocab_include_parents: bool = True

    # Presentation of suggestions
    hide_countries: bool = True  # never suggest bare country / continent names
    unsure_threshold: float = 0.5  # content-type score below this marks the image "Unsure"
    center_focus: float = 0.4  # weight of the central-crop embedding for uploaded images (0 = off)
    center_crop: float = 0.6  # size of the central crop, as a fraction of width and height

    # Splits / deduplication
    dedup_threshold: float = 0.95
    index_val_after_fusion: bool = True

    seed: int = 13

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "Config":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        return cfg.update(**overrides)

    def update(self, **overrides) -> "Config":
        known = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in known:
                raise KeyError(f"Unknown config key: {key}")
            setattr(self, key, value)
        return self
