"""Build the dataset: crawl Commons, download thumbnails, fetch hierarchy, split.

Examples
--------
    python prepare_data.py all --seeds configs/seed_categories.txt --max-files 6000
    python prepare_data.py crawl --seeds configs/seed_categories.txt
    python prepare_data.py all --seeds configs/seed_categories.txt configs/seed_regions.txt --random-files 1500
    python prepare_data.py download --workers 4
    python prepare_data.py hierarchy --depth 2
    python prepare_data.py split --val-frac 0.15 --test-frac 0.15
    python prepare_data.py import --table my_dataset.csv --image-root /path/to/images
    python prepare_data.py --data-dir data_recent recent --n 500     # test set of the newest uploads
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wikicat import data_loader as dl

log = logging.getLogger("prepare_data")


def records_path(data_dir: Path) -> Path:
    return data_dir / "records.jsonl"


def cmd_crawl(args) -> None:
    client = dl.CommonsClient()
    groups = dl.read_seed_groups(*args.seeds)
    seeds = list(groups)
    records = dl.crawl(
        client, seeds,
        files_per_category=args.files_per_cat,
        subcats_per_category=args.subcats,
        files_per_subcategory=args.files_per_subcat,
        max_files=args.max_files,
        thumb_width=args.thumb_width,
        random_files=args.random_files,
    )
    dl.save_records(records, records_path(args.data_dir))
    (args.data_dir / "seeds.json").write_text(json.dumps(seeds, indent=1))
    log.info("Saved %d records to %s", len(records), records_path(args.data_dir))


def cmd_download(args) -> None:
    records = dl.load_records(records_path(args.data_dir))
    records = dl.download_images(records, args.data_dir / "images", workers=args.workers)
    records = [r for r in records if r.get("image_path")]
    dl.save_records(records, records_path(args.data_dir))


def cmd_hierarchy(args) -> None:
    records = dl.load_records(records_path(args.data_dir))
    cats = {c for r in records for c in r["categories"]}
    seeds_file = args.data_dir / "seeds.json"
    if seeds_file.exists():
        cats.update(json.loads(seeds_file.read_text()))
    parents = dl.fetch_hierarchy(dl.CommonsClient(), cats, depth=args.depth)
    (args.data_dir / "hierarchy.json").write_text(json.dumps(parents, ensure_ascii=False))
    log.info("Saved parents for %d categories", len(parents))


def cmd_split(args) -> None:
    records = dl.load_records(records_path(args.data_dir))
    records = [r for r in records if r.get("image_path") and Path(r["image_path"]).exists()]
    records = dl.assign_temporal_splits(records, args.val_frac, args.test_frac, args.uploader_disjoint)
    if not args.no_dedup:
        records = dl.remove_near_duplicates(records, max_distance=args.hash_distance)
    dl.save_records(records, records_path(args.data_dir))
    counts = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    n_cats = len(dl.category_counts(records, split=None))
    log.info("Split sizes %s, %d distinct categories", counts, n_cats)


def cmd_import(args) -> None:
    records = dl.import_table(args.table, args.image_root)
    dl.save_records(records, records_path(args.data_dir))
    log.info("Imported %d records", len(records))


def cmd_recent(args) -> None:
    """Test set of files uploaded 7-30 days ago: later than every training image, never seen in training."""
    known = set()
    main = records_path(Path(args.exclude_from))
    if main.exists():
        known = {r["id"] for r in dl.load_records(main)}
    records = dl.CommonsClient().recent_files(args.n, args.newest_days, args.oldest_days,
                                              per_uploader=args.per_uploader, thumb_width=args.thumb_width)
    skipped = sum(r["id"] in known for r in records)
    records = [r for r in records if r["id"] not in known]
    records = dl.download_images(records, args.data_dir / "images", workers=args.workers)
    records = [dict(r, split="test") for r in records if r.get("image_path")]
    dl.save_records(records, records_path(args.data_dir))
    log.info("Saved %d recent uploads to %s (%d already in %s were skipped)", len(records),
             records_path(args.data_dir), skipped, main)


def cmd_all(args) -> None:
    cmd_crawl(args)
    cmd_download(args)
    if args.depth > 0:
        cmd_hierarchy(args)
    cmd_split(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)

    def crawl_args(p):
        p.add_argument("--seeds", nargs="+", default=["configs/seed_categories.txt"],
                       help="One or more seed files, e.g. configs/seed_categories.txt configs/seed_regions.txt")
        p.add_argument("--random-files", type=int, default=0,
                       help="Also sample this many uniformly random Commons files (unbiased sample)")
        p.add_argument("--files-per-cat", type=int, default=40)
        p.add_argument("--subcats", type=int, default=5)
        p.add_argument("--files-per-subcat", type=int, default=10)
        p.add_argument("--max-files", type=int, default=6000)
        p.add_argument("--thumb-width", type=int, default=330,
                       help="Use a standard Wikimedia thumbnail width (120, 250, 330, 500, 960)")

    def download_args(p):
        p.add_argument("--workers", type=int, default=4)

    def hierarchy_args(p):
        p.add_argument("--depth", type=int, default=2)

    def split_args(p):
        p.add_argument("--val-frac", type=float, default=0.15)
        p.add_argument("--test-frac", type=float, default=0.15)
        p.add_argument("--uploader-disjoint", action="store_true",
                       help="Drop val/test files whose uploader also appears in train")
        p.add_argument("--no-dedup", action="store_true")
        p.add_argument("--hash-distance", type=int, default=4)

    p = sub.add_parser("crawl"); crawl_args(p); p.set_defaults(func=cmd_crawl)
    p = sub.add_parser("download"); download_args(p); p.set_defaults(func=cmd_download)
    p = sub.add_parser("hierarchy"); hierarchy_args(p); p.set_defaults(func=cmd_hierarchy)
    p = sub.add_parser("split"); split_args(p); p.set_defaults(func=cmd_split)
    p = sub.add_parser("import")
    p.add_argument("--table", required=True, help="CSV or JSONL with image_path and categories")
    p.add_argument("--image-root", default=None)
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("recent", help="Test set of the newest Commons uploads")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--newest-days", type=float, default=7, help="Skip uploads newer than this (not yet categorised)")
    p.add_argument("--oldest-days", type=float, default=30)
    p.add_argument("--per-uploader", type=int, default=5)
    p.add_argument("--thumb-width", type=int, default=330)
    p.add_argument("--exclude-from", default="data", help="Dataset whose files must not appear in this test set")
    download_args(p)
    p.set_defaults(func=cmd_recent)
    p = sub.add_parser("all")
    crawl_args(p); download_args(p); hierarchy_args(p); split_args(p)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
