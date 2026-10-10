"""Image quality checks and synthetic degradations for robustness tests."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

MIN_SIDE = 224  # encoder input size; anything smaller gets upsampled
BLUR_THRESHOLD = 60.0  # variance of the Laplacian on a <=512 px grayscale copy


def laplacian_variance(img: Image.Image, max_side: int = 512) -> float:
    gray = img.convert("L")
    if max(gray.size) > max_side:
        gray.thumbnail((max_side, max_side))
    a = np.asarray(gray, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def assess_quality(img: Image.Image, original_size: tuple[int, int] | None = None) -> dict:
    """Flags used to adjust scoring and to warn the user.

    `original_size` is the full-resolution size when `img` is only a thumbnail.
    """
    w, h = original_size if original_size and all(original_size) else img.size
    sharpness = laplacian_variance(img)
    low_res = min(w, h) < MIN_SIDE
    blurry = sharpness < BLUR_THRESHOLD
    flags = [name for name, on in (("low_resolution", low_res), ("blurry", blurry)) if on]
    return {
        "width": int(w),
        "height": int(h),
        "sharpness": round(sharpness, 1),
        "low_resolution": low_res,
        "blurry": blurry,
        "low_quality": low_res or blurry,
        "flags": flags,
    }


def resolution_band(width: int, height: int) -> str:
    side = max(width or 0, height or 0)
    if side == 0:
        return "unknown"
    if side < 800:
        return "low (<800px)"
    if side < 2000:
        return "medium (800-2000px)"
    return "high (>=2000px)"


def degrade(img: Image.Image, kind: str, level: float) -> Image.Image:
    """Apply one synthetic degradation.

    kind: blur (radius px), jpeg (quality 1-95), downscale (factor 0-1), gray.
    """
    img = img.convert("RGB")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "downscale":
        w, h = img.size
        small = img.resize((max(8, int(w * level)), max(8, int(h * level))), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)
    if kind == "gray":
        return img.convert("L").convert("RGB")
    raise ValueError(f"Unknown degradation: {kind}")


def parse_degradations(spec: str) -> list[tuple[str, float]]:
    """'blur:2,jpeg:15,downscale:0.25' -> [('blur', 2.0), ('jpeg', 15.0), ('downscale', 0.25)]"""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        kind, _, level = part.partition(":")
        out.append((kind, float(level or 0)))
    return out


def random_degradation(rng: np.random.Generator) -> tuple[str, float]:
    choice = rng.integers(3)
    if choice == 0:
        return "blur", float(rng.uniform(2.0, 5.0))
    if choice == 1:
        return "jpeg", float(rng.integers(5, 20))
    return "downscale", float(rng.uniform(0.08, 0.25))
