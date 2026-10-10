"""Image and text embeddings in a shared vision-language space.

`HFEncoder` wraps any Hugging Face CLIP or SigLIP checkpoint. `HashingEncoder`
is a weight-free deterministic encoder used only by the offline test suite;
its image and text spaces are unrelated, so it is not a model.
"""
from __future__ import annotations

import io
import logging
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 200_000_000  # Commons hosts very large scans

PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}.",
    "a picture showing {}.",
    "an image from the category: {}.",
)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def category_to_text(name: str) -> str:
    """Turn a category title into a natural-language phrase for the text tower."""
    name = name.replace("_", " ")
    name = re.sub(r"\s*\(([^)]*)\)", r", \1", name)  # "Mercury (planet)" -> "Mercury, planet"
    return re.sub(r"\s+", " ", name).strip()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    """Load any common image into RGB: applies EXIF rotation, flattens alpha."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        import requests

        from .data_loader import default_user_agent

        resp = requests.get(source, timeout=60, headers={"User-Agent": default_user_agent()})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(source)
    if getattr(img, "is_animated", False):
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def _safe_load(path: str) -> Image.Image | None:
    try:
        img = load_image(path)
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001 - corrupt or missing files are skipped
        log.warning("Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class BaseEncoder:
    name: str = "base"
    dim: int = 0

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_categories(self, names: Sequence[str], templates: Sequence[str] = PROMPT_TEMPLATES,
                          chunk: int = 2048) -> np.ndarray:
        """Prompt-ensembled text prototype for each category name."""
        out = []
        for start in range(0, len(names), chunk):
            batch = [category_to_text(n) for n in names[start:start + chunk]]
            prompts = [t.format(n) for n in batch for t in templates]
            emb = self.encode_texts(prompts).reshape(len(batch), len(templates), -1).mean(axis=1)
            out.append(l2_normalize(emb))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(out)

    def encode_paths(self, paths: Sequence[str], workers: int = 4, chunk: int = 256,
                     progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Encode image files. Returns (embeddings, ok_mask); failed rows are zero."""
        embs = np.zeros((len(paths), self.dim), dtype=np.float32)
        ok = np.zeros(len(paths), dtype=bool)
        iterator = range(0, len(paths), chunk)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="encode images", unit="chunk")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in iterator:
                images = list(pool.map(_safe_load, paths[start:start + chunk]))
                idx = [i for i, im in enumerate(images) if im is not None]
                if idx:
                    embs[[start + i for i in idx]] = self.encode_images([images[i] for i in idx])
                    ok[[start + i for i in idx]] = True
        return embs, ok


class HFEncoder(BaseEncoder):
    """CLIP / SigLIP from Hugging Face transformers (v4 and v5 APIs)."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.name = model_name
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.batch_size = batch_size
        # Load in fp32 and cast afterwards: the dtype kwarg was renamed between transformers versions.
        self.model = AutoModel.from_pretrained(model_name).to(self.device, dtype=self.dtype).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.is_siglip = "siglip" in getattr(self.model.config, "model_type", "")
        with torch.inference_mode():
            self.dim = int(self.encode_texts(["a photo"]).shape[1])
        log.info("Loaded %s on %s (dim=%d)", model_name, self.device, self.dim)

    def _features(self, out):
        # transformers v4 returns a tensor; v5 returns a ModelOutput whose
        # pooler_output holds the projected embedding.
        return out if isinstance(out, self.torch.Tensor) else out.pooler_output

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self.torch
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = list(images[start:start + self.batch_size])
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
                feats = self._features(self.model.get_image_features(pixel_values=pixel_values))
                chunks.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, self.dim), dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        kwargs = {"padding": "max_length", "max_length": 64} if self.is_siglip else {"padding": True, "max_length": 77}
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size * 4):
                batch = list(texts[start:start + self.batch_size * 4])
                inputs = self.processor(text=batch, return_tensors="pt", truncation=True, **kwargs)
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
                if self.is_siglip:
                    inputs.pop("attention_mask", None)  # SigLIP was trained without a text mask
                feats = self._features(self.model.get_text_features(**inputs))
                chunks.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)


class HashingEncoder(BaseEncoder):
    """Deterministic, weight-free encoder for offline tests only.

    Images: 16x16 RGB thumbnail + colour histogram, randomly projected.
    Texts: hashed bag of words, randomly projected. The two spaces are not
    aligned, so zero-shot scores from this encoder are meaningless.
    """

    def __init__(self, dim: int = 128, seed: int = 0):
        self.name = "hashing"
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._img_proj = rng.standard_normal((16 * 16 * 3 + 48, dim)).astype(np.float32)
        self._txt_proj = rng.standard_normal((4096, dim)).astype(np.float32)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        feats = []
        for img in images:
            small = np.asarray(img.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            hist = np.concatenate([np.histogram(small[..., c], bins=16, range=(0, 1))[0] for c in range(3)])
            feats.append(np.concatenate([small.flatten() - 0.5, hist / 256.0]))
        if not feats:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack(feats) @ self._img_proj)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        bags = np.zeros((len(texts), 4096), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                bags[i, zlib.crc32(tok.encode()) % 4096] += 1.0
        return l2_normalize(bags @ self._txt_proj + 1e-6)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_name: str, device: str = "auto", batch_size: int = 32) -> BaseEncoder:
    if model_name == "hashing":
        return HashingEncoder()
    return HFEncoder(model_name, device=device, batch_size=batch_size)
