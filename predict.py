"""Suggest Commons categories for one or more images.

Examples
--------
    python predict.py photo.jpg
    python predict.py photo.jpg --title "Diwali lights in Jaipur 2023" --top-k 10
    python predict.py a.jpg b.jpg --json > suggestions.json
    python predict.py https://example.org/image.jpg --no-hierarchy
"""
from __future__ import annotations

import argparse
import json
import sys

from wikicat.pipeline import CategorySuggester

SIGNAL_NAMES = {"knn": "similar images", "head": "classifier", "txt": "zero-shot", "lex": "title words",
                "meta": "description", "year": "year match"}


def explain(s: dict) -> str:
    """One-line reason: the fusion features that pushed this category up the most."""
    totals: dict[str, float] = {}
    for key, value in s["contributions"].items():
        label = SIGNAL_NAMES.get(key.replace("_z", "").replace("lowq_", "").replace("_hit", ""))
        if label:
            totals[label] = totals.get(label, 0.0) + value
    top = sorted(((v, k) for k, v in totals.items() if v > 0.1), reverse=True)[:3]
    return ", ".join(k for _, k in top) or "prior"


def print_result(path: str, result: dict, show_evidence: bool) -> None:
    print(f"\n=== {path}")
    q = result["quality"]
    print(f"    {q['width']}x{q['height']} px, sharpness {q['sharpness']}"
          + (f"  [{', '.join(q['flags'])}]" if q["flags"] else ""))
    for note in result["notes"]:
        print(f"    note: {note}")
    top, *rest = result["content_type"]
    others = ", ".join(f"{f['family']} {f['score']:.0%}" for f in rest)
    print(f"    {'#':>4}  {'score':>6}  {'category':<50} why")
    label = f"Unsure (best guess: {top['family']})" if result["unsure"] else f"Content type: {top['family']}"
    print(f"    {'★':>4}  {top['score']:6.3f}  {label:<50} other types: {others}")
    for rank, s in enumerate(result["suggestions"], 1):
        cat = s["category"] if len(s["category"]) <= 50 else s["category"][:47] + "..."
        extra = f"  (broader than '{s['suppressed_by']}')" if s["suppressed_by"] else ""
        print(f"    {rank:>4}  {s['score']:6.3f}  {cat:<50} {explain(s)}{extra}")
        if show_evidence:
            for ev in s["evidence"]:
                print(f"            similar: {ev['title']} (cos {ev['similarity']:.2f}) {ev['page_url']}")
    t = result["timing_ms"]
    print(f"    time: encode {t['encode_image']} ms, score+rank {t['score_rank']} ms")


def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image paths or http(s) URLs")
    parser.add_argument("--title", default="", help="File title (optional metadata)")
    parser.add_argument("--description", default="", help="File description (optional metadata)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-hierarchy", action="store_true", help="Disable ancestor suppression")
    parser.add_argument("--unsure-threshold", type=float, default=None,
                        help="Mark the image Unsure when the top score is below this (default 0.6)")
    parser.add_argument("--center-focus", type=float, default=None,
                        help="Weight of the central crop, 0-1 (default 0.4; 0 = whole image only)")
    parser.add_argument("--no-evidence", action="store_true", help="Hide neighbour evidence")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    suggester = CategorySuggester.load(args.artifacts_dir, device=args.device, unsure_threshold=args.unsure_threshold,
                                       center_focus=args.center_focus)
    outputs = []
    for path in args.images:
        try:
            result = suggester.suggest(path, args.title, args.description, args.top_k,
                                       use_hierarchy=not args.no_hierarchy)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next image
            print(f"Could not process {path}: {exc}", file=sys.stderr)
            continue
        result["image"] = path
        outputs.append(result)
        if not args.json:
            print_result(path, result, not args.no_evidence)
    if args.json:
        print(json.dumps(outputs, indent=1, ensure_ascii=False))
    return outputs


if __name__ == "__main__":
    main()
