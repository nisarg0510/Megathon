"""Graph-smoothed category prototypes.

    b_c = norm((1 - w_c) t_c + w_c m_c),            w_c = n_c / (n_c + kappa)
    p_c = norm((1 - g_c) b_c + g_c mean(b_j : j parent or sibling of c)),   g_c = gamma (1 - w_c)

t_c is the category's text embedding, m_c the centroid of the indexed images
labelled c (moved by the mean text-minus-image offset, so it lies in the same
region of the space as the text embeddings), and n_c their number.
Well-populated categories lean on their own images; data-sparse tail
categories borrow from their parents and siblings in the Commons graph.
"""
from __future__ import annotations

import numpy as np

from .hierarchy import CategoryGraph


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def neighbour_matrix(names: list[str], graph: CategoryGraph, max_children: int = 50):
    """Row-normalised sparse matrix averaging each category's parents and siblings
    (half weight each when both exist). Parents with more than `max_children`
    children in the vocabulary are too broad to define useful siblings."""
    from scipy import sparse

    index = {n: i for i, n in enumerate(names)}
    parent_col: dict[str, int] = {}
    inc_rows, inc_cols, par_rows, par_cols = [], [], [], []
    for i, name in enumerate(names):
        for p in graph.parents.get(name, ()):
            inc_rows.append(i)
            inc_cols.append(parent_col.setdefault(p, len(parent_col)))
            j = index.get(p)
            if j is not None and j != i:
                par_rows.append(i)
                par_cols.append(j)
    n = len(names)
    inc = sparse.csr_matrix((np.ones(len(inc_rows)), (inc_rows, inc_cols)), shape=(n, max(len(parent_col), 1)))
    inc.data[:] = 1.0
    narrow = np.asarray(inc.sum(axis=0)).ravel() <= max_children
    inc = inc[:, np.where(narrow)[0]]
    sib = (inc @ inc.T).tocsr()
    sib = (sib - sparse.diags(sib.diagonal())).tocsr()
    sib.eliminate_zeros()
    sib.data[:] = 1.0
    par = sparse.csr_matrix((np.ones(len(par_rows)), (par_rows, par_cols)), shape=(n, n))
    par.data[:] = 1.0

    def row_normalise(m):
        s = np.asarray(m.sum(axis=1)).ravel()
        return sparse.diags(1.0 / np.maximum(s, 1.0)) @ m, s > 0

    par, has_p = row_normalise(par)
    sib, has_s = row_normalise(sib)
    both = has_p & has_s
    w_p = np.where(both, 0.5, has_p.astype(float))
    w_s = np.where(both, 0.5, has_s.astype(float))
    return (sparse.diags(w_p) @ par + sparse.diags(w_s) @ sib).tocsr()


def smooth_prototypes(text: np.ndarray, names: list[str], graph: CategoryGraph, embs: np.ndarray,
                      labels: list[list[int]], kappa: float = 5.0, gamma: float = 0.3,
                      use_images: bool = True) -> np.ndarray:
    """Smoothed prototypes for every category in `names` (rows of `text`).

    embs / labels: embeddings of the indexed images and their category ids.
    """
    text = _normalize(np.asarray(text, dtype=np.float32))
    n, dim = text.shape
    counts = np.zeros(n, dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)
    base = text
    if use_images and len(embs):
        sums = np.zeros((n, dim), dtype=np.float32)
        for e, cats in zip(np.asarray(embs, dtype=np.float32), labels):
            for c in cats:
                if 0 <= c < n:
                    sums[c] += e
                    counts[c] += 1
        has = counts > 0
        gap = text.mean(axis=0) - _normalize(np.asarray(embs, dtype=np.float32)).mean(axis=0)
        centroid = np.zeros_like(text)
        centroid[has] = _normalize(_normalize(sums[has]) + gap)
        w = counts / (counts + kappa)
        base = _normalize((1 - w)[:, None] * text + w[:, None] * centroid)
    if gamma <= 0:
        return base.astype(np.float32)
    a = neighbour_matrix(names, graph)
    has_nb = np.asarray(a.sum(axis=1)).ravel() > 0
    nb = _normalize(np.asarray(a @ base))
    g = (gamma * (1 - w) * has_nb).astype(np.float32)
    return _normalize((1 - g)[:, None] * base + g[:, None] * nb).astype(np.float32)
