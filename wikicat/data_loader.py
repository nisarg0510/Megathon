"""Wikimedia Commons data pipeline.

Crawls files and their categories through the MediaWiki API, removes
maintenance / process categories, fetches the category hierarchy, downloads
thumbnails, removes near-duplicates and assigns a temporal train/val/test split.

Every record is one JSON line with this schema:

    id, title, description, timestamp, uploader, width, height, mime,
    thumb_url, page_url, image_path, categories, seed, split
"""
from __future__ import annotations

import csv
import html
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import requests

log = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
RASTER_MIMES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/gif"}

# Wikimedia asks every client to identify itself. Set WIKIMEDIA_CONTACT to an
# email or URL so operators can reach you if the crawler misbehaves.
USER_AGENT = "WikiCatSuggest/0.1 (Megathon X hackathon research; contact: {contact}) python-requests"

# ---------------------------------------------------------------------------
# Category cleaning
# ---------------------------------------------------------------------------

# Hidden categories are already excluded server-side (clshow=!hidden). These
# patterns catch the visible ones that still describe process, licensing,
# provenance or file properties rather than image content.
_MAINTENANCE_PATTERNS = [
    r"^(cc|pd|gfdl|fal|ogl|cecill)([- ]|$)",
    r"licen[cs]e",
    r"copyright",
    r"self-published",
    r"own work",
    r"^uploaded (with|by|from|via|using)",
    r"upload ?wizard",
    r"^(files|media|images|photographs|photos|pictures) (uploaded|by|from|with|without|needing|lacking|created|moved|transferred|taken|of user)",
    r"\b(needing|needs|requiring|to be (checked|categori[sz]ed|deleted|renamed|reviewed))\b",
    r"^(quality|valued|featured|assessed) (images|pictures|media|sounds)",
    r"pictures? of the (day|year)|media of the day|^potd",
    r"^(supported|funded|sponsored) by",
    r"\bwiki ?loves\b|\bwiki science competition\b|^wlm\b|^wle\b",
    r"^pages? (with|using|that|in)",
    r"^(uncategori[sz]ed|unidentified|unknown|unsorted)\b",
    r"\buser:|^users?\b|^files? by user",
    r"\b(geocoded|coordinates|gps|exif|metadata|location not applicable)\b",
    r"\b(retouched|cropped|derivative works?|annotations?|watermark(ed)?)\b",
    r"\b(svg|png|jpe?g|gif|tiff?|webp|djvu|pdf)\b",
    r"^(deletion|duplicate|redirect|disambiguation|hidden|tracking|maintenance|backlog|clean-?up|templates?)\b",
    r"\b(focal length|exposure time|iso speed|f-number|aperture value|lens model)\b",
    r"^taken (with|on|by)|taken on \d|^photographs taken on",
    r"\bflickr\b|\bpanoramio\b|\bgeograph\b",
    r"freedom of panorama|^fop\b",
    r"died more than|^author died",
    r"^(media|files|images) (contributed|donated|released)",
    r"\bbot uploads?\b|^bots?\b",
    r"^commons[: ]",
    r"^(categories|category) (requiring|needing|for)",
]
_MAINTENANCE_RE = re.compile("|".join(f"(?:{p})" for p in _MAINTENANCE_PATTERNS), re.IGNORECASE)


def normalize_category(name: str) -> str:
    """'Category:Churches_in_Paris' -> 'Churches in Paris'."""
    name = name.strip()
    if name.lower().startswith("category:"):
        name = name[len("category:"):]
    name = name.replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def is_maintenance_category(name: str) -> bool:
    name = normalize_category(name)
    return not name or bool(_MAINTENANCE_RE.search(name))


def clean_categories(names: Iterable[str]) -> list[str]:
    """Normalize, drop maintenance categories and duplicates, keep order."""
    seen: set[str] = set()
    out = []
    for raw in names:
        name = normalize_category(raw)
        if name and name not in seen and not is_maintenance_category(name):
            seen.add(name)
            out.append(name)
    return out


def strip_html(text) -> str:
    text = re.sub(r"<[^>]+>", " ", "" if text is None else str(text))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def title_to_text(title: str) -> str:
    """'File:Eiffel_Tower_at_night.jpg' -> 'Eiffel Tower at night'."""
    if ":" in title:
        title = title.split(":", 1)[1]
    title = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", title)
    return re.sub(r"[_\s]+", " ", title).strip()


# ---------------------------------------------------------------------------
# MediaWiki API client
# ---------------------------------------------------------------------------


def default_user_agent() -> str:
    return USER_AGENT.format(contact=os.environ.get("WIKIMEDIA_CONTACT", "unset"))


class CommonsClient:
    """Minimal, polite MediaWiki API client with retries and continuation."""

    def __init__(self, user_agent: str | None = None, min_interval: float = 0.2, max_retries: int = 6):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent or default_user_agent()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, params: dict) -> dict:
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(API_URL, params=params, timeout=60)
            except requests.RequestException as exc:
                log.warning("API request failed (%s), retrying", exc)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                continue
            resp.raise_for_status()
            data = resp.json()
            error = data.get("error")
            if error:
                if error.get("code") == "maxlag":
                    time.sleep(float(resp.headers.get("Retry-After", 5)))
                    continue
                raise RuntimeError(f"MediaWiki API error: {error}")
            return data
        raise RuntimeError(f"MediaWiki API request failed after {self.max_retries} attempts: {params}")

    def query(self, params: dict) -> Iterator[dict]:
        """Yield each response of an action=query call, following continuation."""
        cont: dict = {}
        while True:
            data = self.get({"action": "query", **params, **cont})
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]

    # -- high level helpers -------------------------------------------------

    def subcategories(self, category: str, limit: int) -> list[str]:
        out: list[str] = []
        params = {
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmtype": "subcat",
            "cmlimit": min(500, max(limit, 1)),
        }
        for data in self.query(params):
            for member in data.get("query", {}).get("categorymembers", []):
                out.append(normalize_category(member["title"]))
                if len(out) >= limit:
                    return out
        return out

    def files_in_category(self, category: str, limit: int, thumb_width: int = 330) -> list[dict]:
        """Most recently added raster files in a category, with their visible categories."""
        params = {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": min(50, max(limit, 1)),
            "gcmsort": "timestamp",
            "gcmdir": "desc",
            "prop": "imageinfo|categories",
            "iiprop": "url|timestamp|user|size|mime|extmetadata",
            "iiurlwidth": thumb_width,
            "iiextmetadatafilter": "ImageDescription|DateTimeOriginal",
            "iiextmetadatalanguage": "en",
            "clshow": "!hidden",
            "cllimit": "max",
        }
        pages: dict[int, dict] = {}
        for data in self.query(params):
            for page in data.get("query", {}).get("pages", []):
                merged = pages.setdefault(page["pageid"], {"title": page["title"], "categories": []})
                if page.get("imageinfo"):
                    merged["imageinfo"] = page["imageinfo"][0]
                merged["categories"].extend(c["title"] for c in page.get("categories", []))
            # Category lists for the current batch are only complete once the
            # continuation no longer carries a categories cursor.
            if len(pages) >= limit and "clcontinue" not in data.get("continue", {}):
                break
        records = []
        for pageid, page in pages.items():
            record = _page_to_record(pageid, page, seed=category)
            if record is not None:
                records.append(record)
        return records[:limit]

    def random_files(self, n: int, thumb_width: int = 330, max_batches: int | None = None) -> list[dict]:
        """Uniformly random raster files from all of Commons, with their visible categories.

        Unlike seed-based crawling this follows the real upload distribution,
        so it gives an unbiased estimate of performance on what is uploaded.
        """
        records: dict[str, dict] = {}
        max_batches = max_batches or max(10, n // 10)
        for _ in range(max_batches):
            if len(records) >= n:
                break
            params = {
                "generator": "random",
                "grnnamespace": 6,
                "grnlimit": 50,
                "grnfilterredir": "nonredirects",
                "prop": "imageinfo|categories",
                "iiprop": "url|timestamp|user|size|mime|extmetadata",
                "iiurlwidth": thumb_width,
                "iiextmetadatafilter": "ImageDescription|DateTimeOriginal",
                "iiextmetadatalanguage": "en",
                "clshow": "!hidden",
                "cllimit": "max",
            }
            pages: dict[int, dict] = {}
            for data in self.query(params):
                for page in data.get("query", {}).get("pages", []):
                    merged = pages.setdefault(page["pageid"], {"title": page["title"], "categories": []})
                    if page.get("imageinfo"):
                        merged["imageinfo"] = page["imageinfo"][0]
                    merged["categories"].extend(c["title"] for c in page.get("categories", []))
                if "clcontinue" not in data.get("continue", {}):
                    break  # this random batch is complete; the next loop draws a new one
            for pageid, page in pages.items():
                rec = _page_to_record(pageid, page, seed="random")
                if rec is not None:
                    records.setdefault(rec["id"], rec)
            log.info("random sample: %d usable files", len(records))
        return list(records.values())[:n]

    def recent_files(self, n: int, newest_days: float = 7, oldest_days: float = 30, points: int = 25,
                     per_uploader: int = 5, thumb_width: int = 330, seed: int = 0) -> list[dict]:
        """Files uploaded between `oldest_days` and `newest_days` ago, with their visible categories.

        The deployment distribution: what the system would actually be asked about. Files are taken
        from `points` random moments in the window (consecutive uploads are often one batch) with at
        most `per_uploader` files per uploader. The newest week is skipped so that editors have had
        time to categorise the files.
        """
        rng = random.Random(seed)
        now = datetime.now(timezone.utc)
        starts = sorted((now - timedelta(days=rng.uniform(newest_days, oldest_days)) for _ in range(points)),
                        reverse=True)
        per_point = math.ceil(n / points)
        records: dict[str, dict] = {}
        per_user: Counter = Counter()
        for start in starts:
            params = {
                "generator": "allimages",
                "gaisort": "timestamp",
                "gaidir": "older",
                "gaistart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gailimit": 50,
                "prop": "imageinfo|categories",
                "iiprop": "url|timestamp|user|size|mime|extmetadata",
                "iiurlwidth": thumb_width,
                "iiextmetadatafilter": "ImageDescription|DateTimeOriginal",
                "iiextmetadatalanguage": "en",
                "clshow": "!hidden",
                "cllimit": "max",
            }
            pages: dict[int, dict] = {}
            for data in self.query(params):
                for page in data.get("query", {}).get("pages", []):
                    merged = pages.setdefault(page["pageid"], {"title": page["title"], "categories": []})
                    if page.get("imageinfo"):
                        merged["imageinfo"] = page["imageinfo"][0]
                    merged["categories"].extend(c["title"] for c in page.get("categories", []))
                if len(pages) >= per_point * 4 and "clcontinue" not in data.get("continue", {}):
                    break
            taken = 0
            for pageid, page in pages.items():
                rec = _page_to_record(pageid, page, seed="recent")
                if rec is None or rec["id"] in records or per_user[rec["uploader"]] >= per_uploader:
                    continue
                records[rec["id"]] = rec
                per_user[rec["uploader"]] += 1
                taken += 1
                if taken >= per_point:
                    break
            log.info("recent uploads before %s: %d usable files", start.date(), len(records))
        return list(records.values())[:n]

    def category_parents(self, categories: list[str]) -> dict[str, list[str]]:
        """Visible parent categories of each category (one level)."""
        parents: dict[str, list[str]] = {}
        for start in range(0, len(categories), 50):
            batch = categories[start:start + 50]
            params = {
                "titles": "|".join(f"Category:{c}" for c in batch),
                "prop": "categories",
                "clshow": "!hidden",
                "cllimit": "max",
            }
            for data in self.query(params):
                for page in data.get("query", {}).get("pages", []):
                    child = normalize_category(page["title"])
                    found = [normalize_category(c["title"]) for c in page.get("categories", [])]
                    parents.setdefault(child, []).extend(found)
        return {c: clean_categories(p) for c, p in parents.items()}


def _meta_value(meta: dict, key: str):
    entry = meta.get(key)
    return entry.get("value", "") if isinstance(entry, dict) else ""


def _page_to_record(pageid: int, page: dict, seed: str) -> dict | None:
    info = page.get("imageinfo")
    if not info or info.get("mime") not in RASTER_MIMES:
        return None
    # PHP serialises an empty map as [], so extmetadata can arrive as a list.
    meta = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    categories = clean_categories(page.get("categories", []))
    if not categories:
        return None
    return {
        "id": str(pageid),
        "title": page["title"],
        "description": strip_html(_meta_value(meta, "ImageDescription"))[:500],
        "date_original": strip_html(_meta_value(meta, "DateTimeOriginal")),
        "timestamp": info.get("timestamp", ""),
        "uploader": info.get("user", ""),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "mime": info.get("mime", ""),
        "thumb_url": info.get("thumburl") or info.get("url", ""),
        "page_url": info.get("descriptionurl", ""),
        "image_path": "",
        "categories": categories,
        "seed": seed,
        "split": "",
    }


# ---------------------------------------------------------------------------
# Crawl / download / hierarchy
# ---------------------------------------------------------------------------


def read_seed_file(path: str | Path) -> list[str]:
    return list(read_seed_groups(path))


def read_seed_groups(*paths: str | Path) -> dict[str, str]:
    """Seed category -> group, where a group is set by a '# --- Name: ... ---' header."""
    groups: dict[str, str] = {}
    for path in paths:
        group = "other"
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("# ---"):
                group = stripped.strip("#- ").split(":")[0].strip() or "other"
                continue
            line = stripped.split("#", 1)[0].strip()
            if line:
                groups.setdefault(normalize_category(line), group)
    return groups


def crawl(
    client: CommonsClient,
    seeds: list[str],
    files_per_category: int = 40,
    subcats_per_category: int = 5,
    files_per_subcategory: int = 10,
    max_files: int | None = None,
    thumb_width: int = 330,
    random_files: int = 0,
) -> list[dict]:
    """Collect files from each seed category and a few of its subcategories,
    plus optionally `random_files` uniformly random files (on top of max_files)."""
    records: dict[str, dict] = {}

    def take(category: str, limit: int, seed: str) -> None:
        try:
            batch = client.files_in_category(category, limit, thumb_width)
        except Exception as exc:  # noqa: BLE001 - one bad category must not kill the crawl
            log.warning("Skipping %s: %s", category, exc)
            return
        for rec in batch:
            rec["seed"] = seed
            records.setdefault(rec["id"], rec)

    for i, seed in enumerate(seeds):
        if max_files and len(records) >= max_files:
            break
        take(seed, files_per_category, seed)
        if subcats_per_category > 0:
            try:
                subcats = client.subcategories(seed, subcats_per_category)
            except Exception as exc:  # noqa: BLE001
                log.warning("No subcategories for %s: %s", seed, exc)
                subcats = []
            for sub in subcats:
                if max_files and len(records) >= max_files:
                    break
                if not is_maintenance_category(sub):
                    take(sub, files_per_subcategory, seed)
        log.info("[%d/%d] %s -> %d files total", i + 1, len(seeds), seed, len(records))

    out = list(records.values())
    out = out[:max_files] if max_files else out
    if random_files > 0:
        seen = {r["id"] for r in out}
        try:
            out += [r for r in client.random_files(random_files, thumb_width) if r["id"] not in seen]
        except Exception as exc:  # noqa: BLE001
            log.warning("Random sampling failed: %s", exc)
    return out


def fetch_hierarchy(client: CommonsClient, categories: Iterable[str], depth: int = 2,
                    max_nodes: int = 50000) -> dict[str, list[str]]:
    """Breadth-first walk up the category graph from the given categories."""
    parents: dict[str, list[str]] = {}
    frontier = sorted(set(categories))
    for level in range(depth):
        todo = [c for c in frontier if c not in parents]
        if not todo:
            break
        found = client.category_parents(todo)
        for c in todo:
            parents[c] = found.get(c, [])
        frontier = sorted({p for c in todo for p in parents[c]} - parents.keys())
        log.info("hierarchy level %d: %d categories, %d new parents", level + 1, len(todo), len(frontier))
        if len(parents) >= max_nodes:
            break
    return parents


def download_images(records: list[dict], image_dir: str | Path, workers: int = 4,
                    user_agent: str | None = None, max_retries: int = 5) -> list[dict]:
    """Download thumbnails. Sets record['image_path']; failed downloads keep ''."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = user_agent or default_user_agent()

    def fetch(rec: dict) -> None:
        url = rec.get("thumb_url")
        if not url:
            return
        ext = os.path.splitext(url.split("?")[0])[1].lower() or ".jpg"
        path = image_dir / f"{rec['id']}{ext}"
        if path.exists() and path.stat().st_size > 0:
            rec["image_path"] = str(path)
            return
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("Retry-After", 2 ** (attempt + 1))))
                continue
            if resp.ok:
                path.write_bytes(resp.content)
                rec["image_path"] = str(path)
            return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, records))
    ok = sum(1 for r in records if r.get("image_path"))
    log.info("Downloaded %d/%d images", ok, len(records))
    return records


# ---------------------------------------------------------------------------
# Deduplication and splits
# ---------------------------------------------------------------------------


def dhash(path: str | Path, size: int = 8) -> int | None:
    """64-bit difference hash; robust to rescaling and recompression."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
            px = np.asarray(gray, dtype=np.int16)
    except Exception:  # noqa: BLE001 - unreadable image
        return None
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming_min(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Minimum Hamming distance from each query hash to any reference hash."""
    if len(reference) == 0:
        return np.full(len(query), 64)
    ref_bytes = reference.view(np.uint8).reshape(len(reference), 8)
    out = np.empty(len(query), dtype=np.int64)
    for i, q in enumerate(query):
        x = np.bitwise_xor(ref_bytes, np.array([q], dtype=np.uint64).view(np.uint8).reshape(1, 8))
        out[i] = np.unpackbits(x, axis=1).sum(axis=1).min()
    return out


def assign_temporal_splits(records: list[dict], val_frac: float = 0.15, test_frac: float = 0.15,
                           uploader_disjoint: bool = False) -> list[dict]:
    """Oldest uploads -> train, newer -> val, newest -> test."""
    records = sorted(records, key=lambda r: r.get("timestamp", ""))
    n = len(records)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    for i, rec in enumerate(records):
        rec["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    if uploader_disjoint:
        train_uploaders = {r["uploader"] for r in records if r["split"] == "train"}
        before = len(records)
        records = [r for r in records if r["split"] == "train" or r["uploader"] not in train_uploaders]
        log.info("Uploader-disjoint split dropped %d val/test files", before - len(records))
    return records


def remove_near_duplicates(records: list[dict], max_distance: int = 4) -> list[dict]:
    """Drop val/test files whose thumbnail perceptually matches a training file."""
    hashes = {r["id"]: dhash(r["image_path"]) for r in records if r.get("image_path")}
    train = np.array([h for r in records if r["split"] == "train" and (h := hashes.get(r["id"])) is not None],
                     dtype=np.uint64)
    held = [r for r in records if r["split"] != "train" and hashes.get(r["id"]) is not None]
    if not held:
        return records
    dist = _hamming_min(np.array([hashes[r["id"]] for r in held], dtype=np.uint64), train)
    drop = {r["id"] for r, d in zip(held, dist) if d <= max_distance}
    log.info("Perceptual-hash dedup dropped %d val/test files", len(drop))
    return [r for r in records if r["id"] not in drop]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_records(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records(path: str | Path, split: str | None = None) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if split is None or rec.get("split") == split:
                    out.append(rec)
    return out


def import_table(path: str | Path, image_root: str | Path | None = None) -> list[dict]:
    """Import an externally provided dataset (CSV or JSONL).

    Required columns: image_path (or file) and categories (list, or a string
    separated by '|' or ';'). Optional: id, title, description, timestamp,
    uploader, width, height.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    records = []
    for i, row in enumerate(rows):
        cats = row.get("categories", [])
        if isinstance(cats, str):
            cats = re.split(r"[|;]", cats)
        image_path = row.get("image_path") or row.get("file") or ""
        if image_root and image_path and not Path(image_path).is_absolute():
            image_path = str(Path(image_root) / image_path)
        categories = clean_categories(cats)
        if not categories or not image_path:
            continue
        records.append({
            "id": str(row.get("id") or i),
            "title": row.get("title") or Path(image_path).name,
            "description": strip_html(row.get("description", "")),
            "date_original": row.get("date_original", ""),
            "timestamp": row.get("timestamp", ""),
            "uploader": row.get("uploader", ""),
            "width": int(row.get("width") or 0),
            "height": int(row.get("height") or 0),
            "mime": row.get("mime", ""),
            "thumb_url": row.get("thumb_url", ""),
            "page_url": row.get("page_url", ""),
            "image_path": image_path,
            "categories": categories,
            "seed": row.get("seed", ""),
            "split": row.get("split", ""),
        })
    return records


def category_counts(records: list[dict], split: str | None = "train") -> Counter:
    counts: Counter = Counter()
    for rec in records:
        if split is None or rec.get("split") == split:
            counts.update(rec["categories"])
    return counts
