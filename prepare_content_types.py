"""Collect openly licensed Wikimedia Commons photos per content type, for train_content_type.py.

    python prepare_content_types.py                    # ~200 photos per type into content_type_data/
    python prepare_content_types.py --per-type 300

Writes content_type_data/<Type>/<id>.jpg and content_type_data/manifest.json
(title and page URL of every photo, for attribution and test-pack overlap checks).
You may add your own folders of images under content_type_data/<Type>/ as well.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from wikicat import data_loader as dl
from wikicat.content_type import TYPE_SEEDS, save_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="content_type_data")
    parser.add_argument("--per-type", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = dl.CommonsClient()
    out = Path(args.out)
    # Keep entries from earlier runs: their photos stay in the folders and are trained on too.
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else []
    manifest = {m["path"]: m for m in old}
    for family, seeds in TYPE_SEEDS.items():
        # An equal share per seed category, so seeds late in the list are not crowded out.
        per_seed = max(5, math.ceil(args.per_type / len(seeds)))
        seen: set[str] = set()
        for seed in seeds:
            records = dl.crawl(client, [seed], files_per_category=math.ceil(per_seed * 0.6), subcats_per_category=2,
                               files_per_subcategory=math.ceil(per_seed * 0.3), max_files=per_seed)
            records = [r for r in records if r["id"] not in seen]
            seen.update(r["id"] for r in records)
            records = dl.download_images(records, out / family, workers=args.workers)
            got = [r for r in records if r.get("image_path")]
            for r in got:
                manifest[r["image_path"]] = {"type": family, "path": r["image_path"], "title": r["title"],
                                             "page_url": r["page_url"], "seed": r["seed"]}
            if not got:
                logging.warning("%s / %s: no photos (category missing or empty on Commons)", family, seed)
        logging.info("%s: %d photos", family, sum(1 for m in manifest.values() if m["type"] == family))
    manifest = list(manifest.values())
    save_manifest(out / "manifest.json", manifest)
    print(f"Saved {len(manifest)} photos to {out}/")


if __name__ == "__main__":
    main()
