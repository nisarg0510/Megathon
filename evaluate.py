"""Evaluate category suggestions on the held-out temporal split.

Reports sample-wise P@k / R@k / F1@k (with and without metadata), a per-scorer
ablation, hierarchy-aware precision, mAP, recall by category frequency
(head / torso / tail / unseen), by category type, by resolution band, under
synthetic degradations, plus latency and index size.

Examples
--------
    python evaluate.py
    python evaluate.py --split test --k 1 3 5 10 --degrade blur:2,jpeg:10,downscale:0.15 --latency 50
    python evaluate.py --split val --limit 300
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from wikicat.data_loader import load_records, read_seed_groups
from wikicat.encoder import load_image
from wikicat.metrics import (bucket_report, hierarchical_precision_at_k, label_coverage, mean_average_precision,
                             per_category_recall, percentiles, prf_at_k, type_report)
from wikicat.pipeline import CategorySuggester
from wikicat.quality import assess_quality, degrade, parse_degradations, resolution_band
from wikicat.ranker import FEATURE_NAMES
from wikicat.training import embed_images, embed_meta, model_slug, quality_flags, record_meta_text

log = logging.getLogger("evaluate")

# Single-scorer rankings for the ablation: (feature column, keep only positive scores)
ABLATIONS = {
    "k-NN only": ("knn", True),
    "zero-shot only": ("txt", False),
    "head only": ("head", True),
    "metadata text only": ("meta", False),
    "metadata lexical only": ("lex", True),
}


def ranked_names(scored: list[dict], names: list[str], max_k: int) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = defaultdict(list)
    for res in scored:
        cands, x = res["cands"], res["features"]
        out["fusion + hierarchy"].append([names[cands[i]] for i in np.argsort(-res["final"])[:max_k]])
        out["fusion"].append([names[cands[i]] for i in np.argsort(-res["prob"])[:max_k]])
        for label, (feat, positive_only) in ABLATIONS.items():
            col = x[:, FEATURE_NAMES.index(feat)] if len(cands) else np.zeros(0)
            if feat == "meta" and len(cands) and not x[0, FEATURE_NAMES.index("has_meta")]:
                out[label].append([])
                continue
            order = [i for i in np.argsort(-col) if not positive_only or col[i] > 0][:max_k]
            out[label].append([names[cands[i]] for i in order])
    return out


def score_all(suggester, embs, texts, metas, lowq):
    contexts = suggester.contexts(embs, texts, metas, lowq)
    t0 = time.perf_counter()
    scored = [suggester.ranker.score(ctx) for ctx in contexts]
    return scored, (time.perf_counter() - t0) / max(len(scored), 1) * 1000


def prf_table(preds, gold, ks) -> dict:
    return {f"@{k}": prf_at_k(preds, gold, k) for k in ks}


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def markdown_report(res: dict, ks: list[int]) -> str:
    lines = [f"# Evaluation: {res['split']} split ({res['n_images']} images)", "",
             f"Label coverage (test labels present in the vocabulary): {res['label_coverage']:.3f}", ""]
    lines += ["## Main results (fusion + hierarchy)", "", "| setting | k | P@k | R@k | F1@k | hier. P@k |",
              "|---|---|---|---|---|---|"]
    for setting, table in res["main"].items():
        for k in ks:
            m = table[f"@{k}"]
            hp = res["hierarchical_precision"].get(setting, {}).get(f"@{k}")
            lines.append(f"| {setting} | {k} | {fmt(m['P'])} | {fmt(m['R'])} | {fmt(m['F1'])} | {fmt(hp)} |")
    lines += ["", f"mAP (macro over test categories): {res['mAP']:.3f}", ""]
    k_mid = 5 if 5 in ks else ks[0]
    lines += [f"## Ablation (with metadata), F1@{k_mid} and R@{max(ks)}", "", "| ranking | F1 | R |", "|---|---|---|"]
    for name, table in res["ablation"].items():
        lines.append(f"| {name} | {fmt(table[f'@{k_mid}']['F1'])} | {fmt(table[f'@{max(ks)}']['R'])} |")
    lines += ["", f"## Recall@{k_mid} by category frequency in the index", "",
              "| bucket | categories | labels | macro recall | micro recall |", "|---|---|---|---|---|"]
    for b, v in res["frequency_buckets"].items():
        lines.append(f"| {b} | {v['categories']} | {v['labels']} | {fmt(v['macro_recall'])} | {fmt(v['micro_recall'])} |")
    lines += ["", f"## By category type (k={k_mid})", "", "| type | labels | recall | predictions | precision |",
              "|---|---|---|---|---|"]
    for t, v in res["category_types"].items():
        lines.append(f"| {t} | {v['labels']} | {fmt(v['recall'])} | {v['predictions']} | {fmt(v['precision'])} |")
    lines += ["", f"## By original resolution (F1@{k_mid})", "", "| band | images | F1 |", "|---|---|---|"]
    for band, v in res["resolution_bands"].items():
        lines.append(f"| {band} | {v['images']} | {fmt(v['F1'])} |")
    if res.get("seed_groups"):
        lines += ["", "## By seed group / region (bias audit)", "",
                  f"| group | images | F1@{k_mid} | R@{max(ks)} | F1@{k_mid} image only |", "|---|---|---|---|---|"]
        for g, v in res["seed_groups"].items():
            lines.append(f"| {g} | {v['images']} | {fmt(v['F1'])} | {fmt(v['R'])} | {fmt(v['F1_image_only'])} |")
    if res.get("degradations"):
        lines += ["", f"## Robustness to degradation (F1@{k_mid})", "",
                  "| degradation | flagged low-quality | with metadata | image only |", "|---|---|---|---|"]
        for name, v in res["degradations"].items():
            lines.append(f"| {name} | {v['flagged_low_quality']:.2f} | {fmt(v['with_metadata'])} | {fmt(v['image_only'])} |")
    eff = res["efficiency"]
    lines += ["", "## Efficiency", "", f"- Scoring + ranking (batch, excludes encoding): {eff['rank_ms_per_image']:.2f} ms/image"]
    if eff.get("end_to_end"):
        e = eff["end_to_end"]
        lines.append(f"- End-to-end single image (load, encode, score, rank, explain): p50 {e['p50_ms']:.1f} ms, "
                     f"p95 {e['p95_ms']:.1f} ms over {eff['latency_images']} images")
    lines.append(f"- Index: {eff['indexed_images']} images, {eff['categories']} categories, "
                 f"{eff['memory_mb']:.1f} MB embeddings in memory, {eff['artifacts_mb']:.1f} MB on disk")
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    parser.add_argument("--degrade", default="blur:2,blur:5,jpeg:10,downscale:0.15",
                        help='Comma list of kind:level, or "" to skip')
    parser.add_argument("--latency", type=int, default=30, help="Images for the end-to-end latency test (0 = skip)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed-dir", default="configs", help="Folder with seed*.txt files (for group labels)")
    parser.add_argument("--min-group", type=int, default=10, help="Smallest group shown in the bias audit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ks = sorted(set(args.k))
    max_k = max(ks)

    suggester = CategorySuggester.load(args.artifacts_dir, device=args.device)
    cfg = suggester.cfg
    data_dir = Path(args.data_dir or cfg.data_dir)
    excluded_file = Path(args.artifacts_dir) / "excluded_ids.json"
    excluded = set(json.loads(excluded_file.read_text())) if excluded_file.exists() else set()
    records = [r for r in load_records(data_dir / "records.jsonl", args.split) if r["id"] not in excluded]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise SystemExit(f"No records in split '{args.split}'")

    slug = model_slug(cfg.model_name)
    embs, ok = embed_images(suggester.encoder, records, data_dir / "cache" / f"img_{slug}.npz")
    records = [r for r, good in zip(records, ok) if good]
    embs = embs[ok]
    gold = [set(r["categories"]) for r in records]
    texts = [record_meta_text(r) for r in records]
    metas = embed_meta(suggester, records, data_dir / "cache" / f"meta_{slug}.npz")
    lowq = quality_flags(records)
    names = suggester.vocab.names
    graph = suggester.graph

    res: dict = {"split": args.split, "n_images": len(records),
                 "label_coverage": label_coverage(gold, set(names)), "main": {}, "hierarchical_precision": {}}

    scored_meta, rank_ms = score_all(suggester, embs, texts, metas, lowq)
    scored_img, _ = score_all(suggester, embs, None, None, lowq)
    rankings = ranked_names(scored_meta, names, max_k)
    full = rankings["fusion + hierarchy"]
    full_img = ranked_names(scored_img, names, max_k)["fusion + hierarchy"]
    for setting, preds in (("image + metadata", full), ("image only", full_img)):
        res["main"][setting] = prf_table(preds, gold, ks)
        res["hierarchical_precision"][setting] = {
            f"@{k}": hierarchical_precision_at_k(preds, gold, graph, k, cfg.hierarchy_depth) for k in ks}
    res["ablation"] = {name: prf_table(preds, gold, ks) for name, preds in rankings.items()}

    score_dicts = [{names[c]: float(s) for c, s in zip(r["cands"], r["final"])} for r in scored_meta]
    res["mAP"] = mean_average_precision(score_dicts, gold)

    k_mid = 5 if 5 in ks else ks[0]
    freq = {n: float(f) for n, f in zip(names, suggester.vocab.freq)}
    res["frequency_buckets"] = bucket_report(per_category_recall(full, gold, k_mid), freq)
    res["category_types"] = type_report(full, gold, k_mid)

    bands: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        bands[resolution_band(r.get("width", 0), r.get("height", 0))].append(i)
    res["resolution_bands"] = {
        b: {"images": len(idx), "F1": prf_at_k([full[i] for i in idx], [gold[i] for i in idx], k_mid)["F1"]}
        for b, idx in sorted(bands.items())}

    seed_files = sorted(Path(args.seed_dir).glob("seed*.txt")) if Path(args.seed_dir).is_dir() else []
    seed_group = read_seed_groups(*seed_files) if seed_files else {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        seed = r.get("seed", "")
        groups["random sample" if seed == "random" else seed_group.get(seed, "other")].append(i)
    res["seed_groups"] = {}
    for g, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(idx) >= args.min_group:
            sub_gold = [gold[i] for i in idx]
            res["seed_groups"][g] = {
                "images": len(idx),
                "F1": prf_at_k([full[i] for i in idx], sub_gold, k_mid)["F1"],
                "R": prf_at_k([full[i] for i in idx], sub_gold, max_k)["R"],
                "F1_image_only": prf_at_k([full_img[i] for i in idx], sub_gold, k_mid)["F1"],
            }

    res["degradations"] = {}
    for kind, level in parse_degradations(args.degrade or ""):
        images, flags = [], []
        for r in records:
            img = degrade(load_image(r["image_path"]), kind, level)
            images.append(img)
            flags.append(assess_quality(img)["low_quality"])
        d_embs = suggester.encoder.encode_images(images)
        with_meta = ranked_names(score_all(suggester, d_embs, texts, metas, flags)[0], names, max_k)
        img_only = ranked_names(score_all(suggester, d_embs, None, None, flags)[0], names, max_k)
        res["degradations"][f"{kind}:{level:g}"] = {
            "flagged_low_quality": float(np.mean(flags)),
            "with_metadata": prf_at_k(with_meta["fusion + hierarchy"], gold, k_mid)["F1"],
            "image_only": prf_at_k(img_only["fusion + hierarchy"], gold, k_mid)["F1"],
        }

    latencies = []
    for r in records[:args.latency]:
        t0 = time.perf_counter()
        suggester.suggest(r["image_path"], r.get("title", ""), r.get("description", ""), top_k=10,
                          original_size=(r.get("width", 0), r.get("height", 0)))
        latencies.append((time.perf_counter() - t0) * 1000)
    art = Path(args.artifacts_dir)
    res["efficiency"] = {
        "rank_ms_per_image": rank_ms,
        "end_to_end": percentiles(latencies),
        "latency_images": len(latencies),
        "indexed_images": len(suggester.knn),
        "categories": len(names),
        "memory_mb": (suggester.knn.index.vectors.nbytes + suggester.prototypes.vectors.nbytes) / 2**20,
        "artifacts_mb": sum(p.stat().st_size for p in art.iterdir() if p.is_file()) / 2**20,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{args.split}.json").write_text(json.dumps(res, indent=1))
    report = markdown_report(res, ks)
    (out / f"report_{args.split}.md").write_text(report)
    print(report)
    print(f"Saved {out / f'metrics_{args.split}.json'} and {out / f'report_{args.split}.md'}")
    return res


if __name__ == "__main__":
    main()
