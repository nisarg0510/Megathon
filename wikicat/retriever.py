"""Vector search and the two retrieval-style scorers.

* `KNNScorer`     - instance retrieval: neighbours in the image index vote for
                    their categories with softmax(similarity / tau) weights.
* `PrototypeScorer` - zero-shot: cosine between the query and each category's
                    text prototype. Covers rare and newly added categories.

Both use `VectorIndex`, which runs on FAISS when installed (flat, HNSW or
IVF-PQ via a factory string) and falls back to exact NumPy search otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# torch and faiss both ship an OpenMP runtime on macOS; allow them to coexist.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    if os.environ.get("WIKICAT_NO_FAISS"):
        raise ImportError
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - depends on the environment
    faiss = None


class VectorIndex:
    """Inner-product search over L2-normalised float32 vectors."""

    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None

    def __len__(self) -> int:
        return len(self.vectors)

    def _make_faiss(self, sample: np.ndarray):
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dim)
        index = faiss.index_factory(self.dim, self.index_type, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            try:
                index.train(sample)
            except RuntimeError as exc:
                log.warning("Cannot train %s on %d vectors (%s); using a flat index", self.index_type, len(sample), exc)
                return faiss.IndexFlatIP(self.dim)
        if hasattr(index, "nprobe"):
            index.nprobe = 16
        return index

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, self.dim)
        if not len(x):
            return
        self.vectors = np.concatenate([self.vectors, x]) if len(self.vectors) else x
        if faiss is not None:
            if self._faiss is None:
                self._faiss = self._make_faiss(self.vectors)
                self._faiss.add(self.vectors)
            else:
                self._faiss.add(x)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, self.dim)
        k = min(k, len(self))
        if k <= 0:
            return np.zeros((len(q), 0), np.float32), np.zeros((len(q), 0), np.int64)
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, k)
            return sims, idx.astype(np.int64)
        sims = q @ self.vectors.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(idx, order, axis=1)

    def save(self, prefix: str | Path) -> None:
        np.save(f"{prefix}.npy", self.vectors.astype(np.float16))
        if self._faiss is not None and self.index_type != "flat":
            faiss.write_index(self._faiss, f"{prefix}.faiss")

    @classmethod
    def load(cls, prefix: str | Path, index_type: str = "flat") -> "VectorIndex":
        vectors = np.load(f"{prefix}.npy").astype(np.float32)
        index = cls(vectors.shape[1], index_type)
        faiss_path = Path(f"{prefix}.faiss")
        if faiss is not None and index_type != "flat" and faiss_path.exists():
            index.vectors = vectors
            index._faiss = faiss.read_index(str(faiss_path))
        else:
            index.add(vectors)
        return index


class KNNScorer:
    """s_knn(c|x) = sum_i softmax(z_x . z_i / tau) * (1[c in Y_i] - lambda * 1[c in R_i])

    Y_i are the categories of indexed image i, R_i the categories editors
    rejected for it (hard negatives collected through feedback).
    """

    def __init__(self, index: VectorIndex, labels: list[list[int]], negatives: list[list[int]] | None = None,
                 item_ids: list[str] | None = None):
        self.index = index
        self.labels = [list(map(int, l)) for l in labels]
        self.negatives = [list(map(int, n)) for n in (negatives or [[] for _ in labels])]
        self.item_ids = list(item_ids or [str(i) for i in range(len(labels))])

    def __len__(self) -> int:
        return len(self.labels)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(q, k)

    def vote(self, sims: np.ndarray, idx: np.ndarray, tau: float, reject_weight: float = 1.0,
             k: int | None = None) -> dict[int, float]:
        valid = idx >= 0
        sims, idx = sims[valid], idx[valid]
        if k is not None:
            sims, idx = sims[:k], idx[:k]
        if not len(idx):
            return {}
        w = np.exp((sims - sims.max()) / tau)
        w /= w.sum()
        scores: dict[int, float] = {}
        for wi, item in zip(w.tolist(), idx.tolist()):
            for c in self.labels[item]:
                scores[c] = scores.get(c, 0.0) + wi
            for c in self.negatives[item]:
                scores[c] = scores.get(c, 0.0) - reject_weight * wi
        return scores

    def add(self, embs: np.ndarray, labels: list[list[int]], item_ids: list[str],
            negatives: list[list[int]] | None = None) -> None:
        self.index.add(embs)
        self.labels.extend(list(map(int, l)) for l in labels)
        self.negatives.extend(list(map(int, n)) for n in (negatives or [[] for _ in labels]))
        self.item_ids.extend(item_ids)

    def save(self, directory: Path) -> None:
        self.index.save(directory / "knn_index")
        (directory / "knn_labels.json").write_text(json.dumps(
            {"labels": self.labels, "negatives": self.negatives, "item_ids": self.item_ids}))

    @classmethod
    def load(cls, directory: Path, index_type: str = "flat") -> "KNNScorer":
        meta = json.loads((directory / "knn_labels.json").read_text())
        return cls(VectorIndex.load(directory / "knn_index", index_type), meta["labels"], meta["negatives"],
                   meta["item_ids"])


class PrototypeScorer:
    """Zero-shot scorer: s_txt(c|x) = z_x . t_c over category text prototypes."""

    def __init__(self, prototypes: np.ndarray, index_type: str = "flat"):
        self.index = VectorIndex(prototypes.shape[1], index_type)
        self.index.add(prototypes)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def vectors(self) -> np.ndarray:
        return self.index.vectors

    def top(self, q: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(q, n)

    def similarity(self, q: np.ndarray, cat_ids: np.ndarray) -> np.ndarray:
        return self.index.vectors[cat_ids] @ q.astype(np.float32)

    def add(self, prototypes: np.ndarray) -> None:
        self.index.add(prototypes)

    def save(self, directory: Path) -> None:
        self.index.save(directory / "prototypes")

    @classmethod
    def load(cls, directory: Path, index_type: str = "flat") -> "PrototypeScorer":
        obj = cls.__new__(cls)
        obj.index = VectorIndex.load(directory / "prototypes", index_type)
        return obj
