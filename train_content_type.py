"""Train the content-type classifier and test it on curated test packs.

    python train_content_type.py
    python train_content_type.py --eval-pack ~/Downloads/photo-test-pack --eval-pack ~/Downloads/living-objects-dataset

Trains on content_type_data/<Type>/ images, saves artifacts/content_type.npz (used
automatically by predict.py and app.py), and compares it with the prompt-only
check on each test pack. Test-pack images are never used for training; any
training photo that is also in a test pack is removed first.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from wikicat.config import Config
from wikicat.content_type import ContentTypeClassifier, commons_title, image_folder_dataset, load_pack, pack_scores
from wikicat.encoder import load_encoder
from wikicat.families import FAMILIES, FamilyScorer

log = logging.getLogger("train_content_type")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="content_type_data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--eval-pack", action="append", default=[], help="Test pack folder (repeatable)")
    parser.add_argument("--model", default=None, help="Encoder (default: the one in artifacts/config.json)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--holdout", type=float, default=0.2, help="Share of training photos held out for a check")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    art = Path(args.artifacts_dir)
    cfg = Config.load(art / "config.json") if (art / "config.json").exists() else Config()
    encoder = load_encoder(args.model or cfg.model_name, args.device or cfg.device, cfg.batch_size)

    paths, labels = image_folder_dataset(args.data)
    packs = {p: load_pack(p) for p in args.eval_pack}
    pack_titles = {r["title"] for rows in packs.values() for r in rows if r["title"]}
    manifest_file = Path(args.data) / "manifest.json"
    if pack_titles and manifest_file.exists():
        overlap = {m["path"] for m in json.loads(manifest_file.read_text()) if commons_title(m["title"]) in pack_titles}
        keep = [i for i, p in enumerate(paths) if p not in overlap]
        log.info("Removed %d training photos that also appear in a test pack", len(paths) - len(keep))
        paths, labels = [paths[i] for i in keep], [labels[i] for i in keep]
    if len(set(labels)) < 2:
        raise SystemExit(f"Need images for at least two types under {args.data}/<Type>/ (types: {', '.join(FAMILIES)})")

    embs, ok = encoder.encode_paths(paths, focus=cfg.center_focus, crop=cfg.center_crop)
    x, y = embs[ok], [l for l, good in zip(labels, ok) if good]
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(y))
    n_hold = int(len(y) * args.holdout)
    hold, train = order[:n_hold], order[n_hold:]
    report: dict = {"train_images": {f: y.count(f) for f in FAMILIES}}
    if n_hold:
        clf = ContentTypeClassifier.fit(x[train], [y[i] for i in train])
        pred = clf.predict_proba(x[hold]).argmax(axis=1)
        report["holdout_accuracy"] = float(np.mean([FAMILIES[p] == y[i] for p, i in zip(pred, hold)]))
    clf = ContentTypeClassifier.fit(x, y, model_name=encoder.name)
    art.mkdir(parents=True, exist_ok=True)
    clf.save(art / "content_type.npz")

    prompts = FamilyScorer(encoder)
    for pack, rows in packs.items():
        pe, pok = encoder.encode_paths([r["path"] for r in rows], focus=cfg.center_focus, crop=cfg.center_crop)
        rows = [r for r, good in zip(rows, pok) if good]
        pe = pe[pok]
        report[f"pack:{Path(pack).name}"] = {
            "prompt_only": pack_scores(rows, np.stack([prompts.zero_shot(e) for e in pe])),
            "trained_classifier": pack_scores(rows, clf.predict_proba(pe)),
        }
    (art / "content_type_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"Saved {art / 'content_type.npz'}; the demo and predict.py use it automatically.")
    return report


if __name__ == "__main__":
    main()
