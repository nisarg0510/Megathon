"""Update a built index without retraining.

Examples
--------
    # New categories become suggestible immediately (zero-shot + title words)
    python update.py add-category "Durga Puja 2026 in Kolkata" --parent "Durga Puja in Kolkata"

    # Index newly categorised files (records.jsonl schema: image_path, categories, title, ...)
    python update.py add-images --records data/new_uploads.jsonl

    # Editor feedback: accepted labels index the image, rejected ones become hard negatives
    python update.py feedback photo.jpg --accept "Lighthouses in Kerala" --reject "Churches"
"""
from __future__ import annotations

import argparse
import logging

from wikicat.data_loader import load_records
from wikicat.pipeline import CategorySuggester
from wikicat.training import item_meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-category")
    p.add_argument("names", nargs="+")
    p.add_argument("--parent", action="append", default=[], help="Parent category (repeatable)")

    p = sub.add_parser("add-images")
    p.add_argument("--records", required=True)

    p = sub.add_parser("feedback")
    p.add_argument("image")
    p.add_argument("--accept", nargs="*", default=[])
    p.add_argument("--reject", nargs="*", default=[])
    p.add_argument("--title", default="")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    suggester = CategorySuggester.load(args.artifacts_dir, device=args.device)
    before = (len(suggester.vocab), len(suggester.knn))

    if args.command == "add-category":
        suggester.add_categories(args.names, {n: args.parent for n in args.names} if args.parent else None)
    elif args.command == "add-images":
        records = load_records(args.records)
        embs, ok = suggester.encoder.encode_paths([r["image_path"] for r in records])
        good = [i for i, k in enumerate(ok) if k]
        suggester.add_images(embs[good], [records[i]["categories"] for i in good],
                             [item_meta(records[i]) for i in good])
    elif args.command == "feedback":
        if not args.accept and not args.reject:
            parser.error("feedback needs --accept and/or --reject")
        suggester.record_feedback(args.image, args.accept, args.reject, args.title, args.artifacts_dir)

    suggester.save(args.artifacts_dir)
    print(f"categories {before[0]} -> {len(suggester.vocab)}, indexed images {before[1]} -> {len(suggester.knn)}")


if __name__ == "__main__":
    main()
