"""Category graph (child -> parents). Commons intends a DAG but contains cycles,
so every walk is depth-limited and tracks visited nodes."""
from __future__ import annotations

import json
from collections import deque
from functools import lru_cache
from pathlib import Path


class CategoryGraph:
    def __init__(self, parents: dict[str, list[str]] | None = None):
        self.parents: dict[str, list[str]] = {k: list(v) for k, v in (parents or {}).items()}
        self._ancestors = lru_cache(maxsize=200_000)(self._ancestors_uncached)

    def __len__(self) -> int:
        return len(self.parents)

    def _ancestors_uncached(self, name: str, max_depth: int) -> frozenset[str]:
        seen: set[str] = set()
        queue = deque([(name, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in self.parents.get(node, ()):
                if parent not in seen and parent != name:
                    seen.add(parent)
                    queue.append((parent, depth + 1))
        return frozenset(seen)

    def ancestors(self, name: str, max_depth: int = 3) -> frozenset[str]:
        return self._ancestors(name, max_depth)

    def is_ancestor(self, ancestor: str, name: str, max_depth: int = 3) -> bool:
        return ancestor in self.ancestors(name, max_depth)

    def related(self, a: str, b: str, max_depth: int = 3) -> bool:
        """Same category, or one is an ancestor of the other."""
        return a == b or self.is_ancestor(a, b, max_depth) or self.is_ancestor(b, a, max_depth)

    def add_edges(self, child: str, parents: list[str]) -> None:
        current = self.parents.setdefault(child, [])
        current.extend(p for p in parents if p not in current)
        self._ancestors.cache_clear()

    def nodes(self) -> set[str]:
        return set(self.parents) | {p for ps in self.parents.values() for p in ps}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.parents, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CategoryGraph":
        path = Path(path)
        return cls(json.loads(path.read_text()) if path.exists() else {})
