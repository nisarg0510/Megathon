"""Collect openly licensed Wikimedia Commons photos per content type, for train_content_type.py.

    python prepare_content_types.py                    # ~200 photos per type into content_type_data/
    python prepare_content_types.py --per-type 300

Writes content_type_data/<Type>/<id>.jpg and content_type_data/manifest.json
(title and page URL of every photo, for attribution and test-pack overlap checks).
You may add your own folders of images under content_type_data/<Type>/ as well.
"""
from __future__ import annotations

import argparse
import logging
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
    manifest = []
    for family, seeds in TYPE_SEEDS.items():
        per_seed = max(5, args.per_type // len(seeds) + 1)
        records = dl.crawl(client, seeds, files_per_category=per_seed, subcats_per_category=2,
                           files_per_subcategory=max(3, per_seed // 3), max_files=args.per_type)
        records = dl.download_images(records, out / family, workers=args.workers)
        for r in records:
            if r.get("image_path"):
                manifest.append({"type": family, "path": r["image_path"], "title": r["title"],
                                 "page_url": r["page_url"], "seed": r["seed"]})
        logging.info("%s: %d photos", family, sum(1 for m in manifest if m["type"] == family))
    save_manifest(out / "manifest.json", manifest)
    print(f"Saved {len(manifest)} photos to {out}/")


if __name__ == "__main__":
    main()
