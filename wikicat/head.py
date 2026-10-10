"""Supervised multi-label head on frozen embeddings.

p_c = sigmoid(s * w_c . z_x + b_c), with w_c initialised from the category's
text prototype t_c (text-initialised classifier) and pulled back towards it by
an L2 penalty, so rare categories keep their zero-shot knowledge. Trained with
the asymmetric loss, which down-weights easy negatives and, through the
probability margin, tolerates the many missing labels on Commons.

Inference is pure NumPy; torch is only needed for training.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def asymmetric_loss(logits, targets, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                    margin: float = 0.05, eps: float = 1e-8):
    import torch

    p = torch.sigmoid(logits)
    p_neg = (p - margin).clamp(min=0)
    loss_pos = targets * (1 - p).pow(gamma_pos) * torch.log(p.clamp(min=eps))
    loss_neg = (1 - targets) * p_neg.pow(gamma_neg) * torch.log((1 - p_neg).clamp(min=eps))
    return -(loss_pos + loss_neg).sum() / logits.shape[0]


class LinearHead:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, scale: float, cat_ids: np.ndarray):
        self.weights = weights.astype(np.float32)
        self.bias = bias.astype(np.float32)
        self.scale = float(scale)
        self.cat_ids = np.asarray(cat_ids, dtype=np.int64)
        self.col = {int(c): j for j, c in enumerate(self.cat_ids)}

    def __len__(self) -> int:
        return len(self.cat_ids)

    def predict(self, q: np.ndarray) -> np.ndarray:
        """Probabilities, shape (n_queries, n_head_categories)."""
        logits = self.scale * (np.atleast_2d(q).astype(np.float32) @ self.weights.T) + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    @classmethod
    def fit(cls, x: np.ndarray, label_lists: list[list[int]], cat_ids: np.ndarray, init_weights: np.ndarray,
            epochs: int = 40, lr: float = 2e-3, l2_to_init: float = 1e-3, batch_size: int = 256,
            seed: int = 13, device: str = "cpu") -> "LinearHead":
        import torch

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        cat_ids = np.asarray(cat_ids, dtype=np.int64)
        col = {int(c): j for j, c in enumerate(cat_ids)}
        rows = [[col[c] for c in labels if c in col] for labels in label_lists]
        n, n_cats = len(x), len(cat_ids)

        counts = np.zeros(n_cats, dtype=np.float64)
        for r in rows:
            counts[r] += 1
        prior = np.clip(counts / max(n, 1), 1e-4, 0.5)

        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        w0 = torch.from_numpy(np.ascontiguousarray(init_weights, dtype=np.float32)).to(device)
        weights = torch.nn.Parameter(w0.clone())
        bias = torch.nn.Parameter(torch.from_numpy(np.log(prior / (1 - prior)).astype(np.float32)).to(device))
        log_scale = torch.nn.Parameter(torch.tensor(math.log(10.0), device=device))
        opt = torch.optim.AdamW([weights, bias, log_scale], lr=lr, weight_decay=0.0)

        for epoch in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                batch = perm[start:start + batch_size]
                y = torch.zeros(len(batch), n_cats, device=device)
                for i, item in enumerate(batch):
                    if rows[item]:
                        y[i, rows[item]] = 1.0
                logits = log_scale.exp() * (xt[batch] @ weights.T) + bias
                loss = asymmetric_loss(logits, y) + l2_to_init * ((weights - w0) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if epoch % 10 == 0 or epoch == epochs - 1:
                log.info("head epoch %d/%d loss %.4f", epoch + 1, epochs, total / max(n, 1))

        return cls(weights.detach().cpu().numpy(), bias.detach().cpu().numpy(),
                   float(log_scale.exp()), cat_ids)

    def save(self, path: str | Path) -> None:
        np.savez(path, weights=self.weights.astype(np.float16), bias=self.bias,
                 scale=np.array(self.scale), cat_ids=self.cat_ids)

    @classmethod
    def load(cls, path: str | Path) -> "LinearHead":
        data = np.load(path)
        return cls(data["weights"].astype(np.float32), data["bias"], float(data["scale"]), data["cat_ids"])
