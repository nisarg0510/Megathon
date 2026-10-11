"""Build every artifact from data/records.jsonl.

1. Encode images (cached per model) and drop val/test near-duplicates of train.
2. Vocabulary = train categories, seed categories and their parents. Val
   categories are not added yet, so validation sees the same share of unknown
   categories that test will (otherwise fusion is tuned on an easy set).
3. Category prototypes: text embeddings (matched against images and
   metadata text) plus a second, smoothed set that blends in image centroids
   and parent/sibling prototypes (blend chosen on val), matched against images
   only; k-NN index over train; supervised head on train.
4. Tune k-NN (K, tau) and fit the fusion stacker on val, with copies of val
   that have metadata removed or images degraded, so the stacker learns how
   to weigh signals in those conditions. Then calibrate conformal suggestion
   sets on val.
5. Add val categories and val images (they predate the test period),
   recompute the smoothed prototypes with them, and save.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .conformal import ALPHAS, ConformalCalibrator, nonconformity
from .data_loader import assign_temporal_splits, is_maintenance_category, load_records
from .encoder import BaseEncoder, load_encoder, load_image
from .head import LinearHead
from .hierarchy import CategoryGraph
from .metadata import build_meta_text
from .metrics import per_category_recall, prf_at_k
from .pipeline import CategorySuggester, Vocabulary
from .quality import assess_quality, degrade, random_degradation
from .ranker import FEATURE_NAMES, FusionModel
from .retriever import KNNScorer, PrototypeScorer, VectorIndex
from .smoothing import smooth_prototypes

log = logging.getLogger(__name__)


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def item_meta(rec: dict) -> dict:
    return {k: rec.get(k, "") for k in ("id", "title", "page_url", "thumb_url", "image_path")}


def record_meta_text(rec: dict) -> str:
    return build_meta_text(rec.get("title", ""), rec.get("description", ""))


# ---------------------------------------------------------------------------
# Embedding caches
# ---------------------------------------------------------------------------


def _load_cache(path: Path, model: str) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    if str(data["model"]) != model:
        return {}
    return dict(zip(data["ids"].tolist(), data["embs"].astype(np.float32)))


def _save_cache(path: Path, model: str, cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(cache)
    np.savez(path, ids=np.array(ids), embs=np.stack([cache[i] for i in ids]).astype(np.float16), model=np.array(model))


def embed_images(encoder: BaseEncoder, records: list[dict], cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Image embeddings aligned with `records`; ok[i] is False if the image is unreadable."""
    cache = _load_cache(cache_path, encoder.name)
    missing = [r for r in records if r["id"] not in cache]
    if missing:
        log.info("Encoding %d images (%d cached)", len(missing), len(cache))
        embs, ok = encoder.encode_paths([r["image_path"] for r in missing])
        for rec, emb, good in zip(missing, embs, ok):
            if good:
                cache[rec["id"]] = emb
        _save_cache(cache_path, encoder.name, cache)
    out = np.zeros((len(records), encoder.dim), dtype=np.float32)
    ok = np.zeros(len(records), dtype=bool)
    for i, rec in enumerate(records):
        if rec["id"] in cache:
            out[i] = cache[rec["id"]]
            ok[i] = True
    return out, ok


def embed_meta(suggester: CategorySuggester, records: list[dict], cache_path: Path) -> list[np.ndarray | None]:
    model = suggester.encoder.name
    cache = _load_cache(cache_path, model)
    texts = [record_meta_text(r) for r in records]
    todo = [i for i, r in enumerate(records) if texts[i] and r["id"] not in cache]
    if todo:
        for i, emb in zip(todo, suggester.encode_meta([texts[i] for i in todo])):
            cache[records[i]["id"]] = emb
        _save_cache(cache_path, model, cache)
    return [cache.get(r["id"]) if texts[i] else None for i, r in enumerate(records)]


def quality_flags(records: list[dict]) -> list[bool]:
    flags = []
    for rec in records:
        try:
            img = load_image(rec["image_path"])
            flags.append(assess_quality(img, (rec.get("width", 0), rec.get("height", 0)))["low_quality"])
        except Exception:  # noqa: BLE001
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# Tuning and fusion
# ---------------------------------------------------------------------------


def tune_knn(suggester: CategorySuggester, q: np.ndarray, gold: list[set[str]],
             ks=(10, 25, 50, 100), taus=(0.01, 0.02, 0.05, 0.1), k_eval: int = 5) -> dict:
    sims, idx = suggester.knn.search(q, max(ks))
    names = suggester.vocab.names
    results = []
    for k in ks:
        for tau in taus:
            preds = []
            for i in range(len(q)):
                votes = suggester.knn.vote(sims[i], idx[i], tau, suggester.cfg.knn_reject_weight, k=k)
                top = sorted(votes, key=lambda c: -votes[c])[:k_eval]
                preds.append([names[c] for c in top])
            results.append({"k": k, "tau": tau, f"F1@{k_eval}": prf_at_k(preds, gold, k_eval)["F1"]})
    best = max(results, key=lambda r: r[f"F1@{k_eval}"])
    return {"best": best, "grid": results}


def tune_hierarchy(suggester: CategorySuggester, val_records: list[dict], val_embs: np.ndarray,
                   val_meta: list[np.ndarray | None], penalties=(0.3, 0.6, 1.0), k_eval: int = 5) -> dict:
    """Pick the ancestor penalty on validation (1.0 = suppression off), so the
    hierarchy rule is only applied as strongly as it helps."""
    texts = [record_meta_text(r) for r in val_records]
    gold = [set(r["categories"]) for r in val_records]
    contexts = suggester.contexts(val_embs, texts, val_meta)
    names = suggester.vocab.names
    results = []
    for penalty in penalties:
        suggester.ranker.ancestor_penalty = penalty
        preds = []
        for ctx in contexts:
            res = suggester.ranker.score(ctx, use_hierarchy=penalty < 1.0)
            preds.append([names[res["cands"][i]] for i in np.argsort(-res["final"])[:k_eval]])
        results.append({"penalty": penalty, f"F1@{k_eval}": prf_at_k(preds, gold, k_eval)["F1"]})
    best = max(results, key=lambda r: r[f"F1@{k_eval}"])
    suggester.ranker.ancestor_penalty = suggester.cfg.ancestor_penalty = best["penalty"]
    return {"best": best, "grid": results}


def fusion_training_data(suggester: CategorySuggester, contexts, golds: list[set[str]]):
    """Candidate features and labels. Unlabelled ancestors of true labels are
    treated as unknown (dropped), not as negatives."""
    ranker, graph, names = suggester.ranker, suggester.graph, suggester.vocab.names
    xs, ys = [], []
    covered = total = 0
    for ctx, gold in zip(contexts, golds):
        cands = ranker.candidates(ctx)
        x = ranker.features(ctx, cands)
        cand_names = [names[c] for c in cands]
        y = np.array([n in gold for n in cand_names], dtype=np.int8)
        covered += int(y.sum())
        total += len(gold)
        ancestors = set()
        for g in gold:
            ancestors |= graph.ancestors(g, suggester.cfg.hierarchy_depth)
        keep = np.array([bool(y[j]) or cand_names[j] not in ancestors for j in range(len(cands))], dtype=bool)
        xs.append(x[keep])
        ys.append(y[keep])
    x = np.concatenate(xs) if xs else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    y = np.concatenate(ys) if ys else np.zeros(0, np.int8)
    return x, y, covered / max(total, 1)


def fit_fusion(suggester: CategorySuggester, val_records: list[dict], val_embs: np.ndarray,
               val_meta: list[np.ndarray | None], rng: np.random.Generator) -> tuple[FusionModel, dict]:
    cfg = suggester.cfg
    texts = [record_meta_text(r) for r in val_records]
    lowq = quality_flags(val_records)
    gold = [set(r["categories"]) for r in val_records]
    n = len(val_records)

    contexts = suggester.contexts(val_embs, texts, val_meta, lowq)
    golds = list(gold)
    candidate_recall = fusion_training_data(suggester, contexts, gold)[2]

    # Copy without metadata: the stacker must cope with image-only uploads.
    sel = rng.choice(n, int(n * cfg.fusion_drop_meta_frac), replace=False) if n else []
    if len(sel):
        contexts += suggester.contexts(val_embs[sel], None, None, [lowq[j] for j in sel])
        golds += [gold[j] for j in sel]

    # Copy with degraded images: blur, heavy JPEG, or extreme downscaling.
    sel = rng.choice(n, int(n * cfg.fusion_degrade_frac), replace=False) if n else []
    if len(sel):
        images, flags, keep = [], [], []
        for j in sel:
            try:
                img = degrade(load_image(val_records[j]["image_path"]), *random_degradation(rng))
            except Exception:  # noqa: BLE001
                continue
            images.append(img)
            flags.append(assess_quality(img)["low_quality"])
            keep.append(j)
        if images:
            embs = suggester.encoder.encode_images(images)
            contexts += suggester.contexts(embs, [texts[j] for j in keep], [val_meta[j] for j in keep], flags)
            golds += [gold[j] for j in keep]

    x, y, _ = fusion_training_data(suggester, contexts, golds)
    report = {"rows": int(len(y)), "positives": int(y.sum()), "val_candidate_recall": candidate_recall}
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        log.warning("Too few fusion training rows; keeping default fusion weights")
        return FusionModel.default(), report
    fusion = FusionModel.fit(x, y)
    report["coefficients"] = dict(zip(FEATURE_NAMES, np.round(fusion.coef, 3).tolist()))
    return fusion, report


# (name, use image centroids, gamma for the parent/sibling average)
SMOOTHING_GRID = (("text only", False, 0.0), ("text + graph", False, 0.3), ("text + images", True, 0.0),
                  ("text + images + graph", True, 0.3), ("text + images + graph (strong)", True, 0.6))


def prototype_recall(protos: np.ndarray, names: list[str], q: np.ndarray, gold: list[set[str]],
                     train_freq: dict[str, float], k: int = 10, tail_max: int = 4) -> dict:
    """Recall@k of the prototype scorer alone, overall and for tail categories
    (in the vocabulary with at most `tail_max` training images)."""
    preds = []
    for start in range(0, len(q), 256):
        sims = q[start:start + 256] @ protos.T
        kk = min(k, sims.shape[1])
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        preds += [[names[j] for j in row] for row in top]
    per_cat = per_category_recall(preds, gold, k)
    tail = [c for c in per_cat if c in train_freq and train_freq[c] <= tail_max]
    hits, labels = sum(per_cat[c][0] for c in tail), sum(per_cat[c][1] for c in tail)
    return {f"R@{k}": prf_at_k(preds, gold, k)["R"], f"tail_R@{k}": hits / labels if labels else None,
            "tail_labels": labels}


def tune_smoothing(cfg: Config, text: np.ndarray, vocab: Vocabulary, graph: CategoryGraph, train_embs: np.ndarray,
                   train_labels: list[list[int]], val_embs: np.ndarray, gold: list[set[str]]) -> dict:
    """Pick how prototypes are smoothed by the prototype scorer's recall on val."""
    freq = {n: float(f) for n, f in zip(vocab.names, vocab.freq)}
    results = []
    for name, use_images, gamma in SMOOTHING_GRID:
        protos = smooth_prototypes(text, vocab.names, graph, train_embs, train_labels, cfg.smooth_kappa, gamma,
                                   use_images)
        results.append({"setting": name, "use_images": use_images, "gamma": gamma,
                        **prototype_recall(protos, vocab.names, val_embs, gold, freq)})
    best = max(results, key=lambda r: r["R@10"])
    cfg.smooth_use_images, cfg.smooth_gamma = best["use_images"], best["gamma"]
    return {"best": best, "grid": results}


def calibrate_conformal(suggester: CategorySuggester, val_records: list[dict], val_embs: np.ndarray,
                        val_meta: list[np.ndarray | None]) -> tuple[ConformalCalibrator, dict]:
    """Nonconformity scores on val, kept separately for uploads with and without metadata."""
    texts = [record_meta_text(r) for r in val_records]
    gold = [set(r["categories"]) for r in val_records]
    lowq = quality_flags(val_records)
    names = suggester.vocab.names
    scores: dict[str, list[float]] = {"with_metadata": [], "image_only": []}
    scored = []
    # Only images with a category the system can output: no set can cover a category it does not know,
    # and those images need a new category, not a longer list.
    known = [i for i, g in enumerate(gold) if any(c in suggester.vocab for c in g)]
    val_embs, texts, gold, lowq = val_embs[known], [texts[i] for i in known], [gold[i] for i in known], \
        [lowq[i] for i in known]
    val_meta = [val_meta[i] for i in known]
    for contexts in (suggester.contexts(val_embs, texts, val_meta, lowq),
                     suggester.contexts(val_embs, None, None, lowq)):
        for ctx, g in zip(contexts, gold):
            res = suggester.ranker.score(ctx)
            group = "with_metadata" if ctx.has_meta else "image_only"
            scores[group].append(nonconformity(res["final"], [names[c] for c in res["cands"]], g))
            scored.append((res["final"], ctx.has_meta))
    cal = ConformalCalibrator(scores, suggester.cfg.conformal_alpha)
    median_size = {}
    for a in ALPHAS:
        sizes = {True: [], False: []}
        for final, has_meta in scored:
            sizes[has_meta].append(len(cal.predict_set(final, has_meta, a)))
        median_size[a] = max(float(np.median(v)) for v in sizes.values() if v)
    suggester.cfg.conformal_alpha = cal.choose_alpha(median_size)
    report = {"calibration_images": {g: len(v) for g, v in scores.items()}, "alpha": cal.alpha,
              "thresholds": {f"{a:g}": {g: round(cal.threshold(a, g), 4) for g in scores} for a in ALPHAS},
              "val_median_set_size": {f"{a:g}": m for a, m in median_size.items()}}
    return cal, report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_vocabulary(cfg: Config, records: list[dict], train_idx, val_idx, graph: CategoryGraph,
                     seeds: list[str]) -> Vocabulary:
    train_counts = Counter(c for i in train_idx for c in records[i]["categories"])
    names = list(train_counts)
    names += sorted({c for i in val_idx for c in records[i]["categories"]} - set(names))
    names += [s for s in seeds if not is_maintenance_category(s)]
    if cfg.vocab_include_parents:
        names += [p for c in list(names) for p in graph.parents.get(c, [])]
    names = list(dict.fromkeys(names))
    return Vocabulary(names, [float(train_counts.get(n, 0)) for n in names])


def train(cfg: Config, encoder: BaseEncoder | None = None) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    data_dir, out_dir = cfg.data_path, cfg.artifacts_path
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(data_dir / "records.jsonl")
    if not records:
        raise RuntimeError(f"No records in {data_dir / 'records.jsonl'}; run prepare_data.py first")
    if not all(r.get("split") for r in records):
        records = assign_temporal_splits(records)
    encoder = encoder or load_encoder(cfg.model_name, cfg.device, cfg.batch_size)
    cfg.model_name = encoder.name

    embs, ok = embed_images(encoder, records, data_dir / "cache" / f"img_{model_slug(encoder.name)}.npz")
    records = [r for r, good in zip(records, ok) if good]
    embs = embs[ok]
    split = np.array([r["split"] for r in records], dtype=object)
    tr = np.where(split == "train")[0]

    # Embedding-level near-duplicate removal (complements the perceptual hash).
    excluded: list[str] = []
    held = np.where(split != "train")[0]
    if len(held) and len(tr):
        index = VectorIndex(embs.shape[1])
        index.add(embs[tr])
        sims, _ = index.search(embs[held], 1)
        dup = held[sims[:, 0] >= cfg.dedup_threshold]
        excluded = [records[i]["id"] for i in dup]
        split[dup] = "duplicate"
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]
    log.info("train=%d val=%d test=%d (excluded %d near-duplicates)", len(tr), len(va), len(te), len(excluded))

    graph = CategoryGraph.load(data_dir / "hierarchy.json")
    seeds_file = data_dir / "seeds.json"
    seeds = json.loads(seeds_file.read_text()) if seeds_file.exists() else []
    vocab = build_vocabulary(cfg, records, tr, [], graph, seeds)
    log.info("Vocabulary: %d categories", len(vocab))

    text_protos = encoder.encode_categories(vocab.names)
    train_labels = [vocab.ids(records[i]["categories"]) for i in tr]
    prototypes = PrototypeScorer(text_protos, cfg.index_type)
    smoothing_report, smoothed = None, None
    if cfg.graph_smoothing:
        if len(va):
            smoothing_report = tune_smoothing(cfg, text_protos, vocab, graph, embs[tr], train_labels, embs[va],
                                              [set(records[i]["categories"]) for i in va])
            log.info("Prototype smoothing: %s", smoothing_report["best"]["setting"])
        cfg.graph_smoothing = cfg.smooth_use_images or cfg.smooth_gamma > 0  # "text only" won: nothing to add
    if cfg.graph_smoothing:
        smoothed = PrototypeScorer(smooth_prototypes(text_protos, vocab.names, graph, embs[tr], train_labels,
                                                     cfg.smooth_kappa, cfg.smooth_gamma, cfg.smooth_use_images),
                                   cfg.index_type)
    knn = KNNScorer(VectorIndex(encoder.dim, cfg.index_type), [], [], [])
    knn.add(embs[tr], train_labels, [records[i]["id"] for i in tr])
    items = [item_meta(records[i]) for i in tr]

    head_cats = np.array([c for c in range(len(vocab)) if vocab.freq[c] >= cfg.head_min_freq], dtype=np.int64)
    head = None
    if len(head_cats):
        log.info("Training head on %d categories", len(head_cats))
        head = LinearHead.fit(embs[tr], train_labels, head_cats, prototypes.vectors[head_cats],
                              epochs=cfg.head_epochs, lr=cfg.head_lr, l2_to_init=cfg.head_l2_to_init,
                              seed=cfg.seed)

    suggester = CategorySuggester(cfg, vocab, prototypes, knn, head, FusionModel.default(), graph, items, encoder,
                                  smoothed=smoothed)
    report: dict = {
        "model": cfg.model_name,
        "counts": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                   "near_duplicates_excluded": len(excluded)},
        "vocabulary": len(vocab),
        "head_categories": int(len(head_cats)),
    }
    if smoothing_report:
        report["smoothing_tuning"] = smoothing_report

    if len(va):
        val_records = [records[i] for i in va]
        gold = [set(r["categories"]) for r in val_records]
        tuning = tune_knn(suggester, embs[va], gold)
        cfg.knn_k, cfg.knn_tau = int(tuning["best"]["k"]), float(tuning["best"]["tau"])
        report["knn_tuning"] = tuning
        log.info("k-NN tuned: K=%d tau=%.3f", cfg.knn_k, cfg.knn_tau)

        val_meta = embed_meta(suggester, val_records, data_dir / "cache" / f"meta_{model_slug(encoder.name)}.npz")
        fusion, fusion_report = fit_fusion(suggester, val_records, embs[va], val_meta, rng)
        suggester.fusion = suggester.ranker.fusion = fusion
        report["fusion"] = fusion_report
        report["hierarchy_tuning"] = tune_hierarchy(suggester, val_records, embs[va], val_meta)
        log.info("Ancestor penalty tuned: %.1f", cfg.ancestor_penalty)
        suggester.conformal, report["conformal"] = calibrate_conformal(suggester, val_records, embs[va], val_meta)
        log.info("Conformal sets: alpha=%.2f", cfg.conformal_alpha)

        if cfg.index_val_after_fusion:
            val_cats = sorted({c for r in val_records for c in r["categories"]})
            if cfg.vocab_include_parents:
                val_cats += [p for c in val_cats for p in graph.parents.get(c, [])]
            suggester.add_categories(val_cats)
            report["vocabulary_after_val"] = len(suggester.vocab)
            suggester.add_images(embs[va], [r["categories"] for r in val_records],
                                 [item_meta(r) for r in val_records])

    if suggester.smoothed is not None:
        # Categories added after fusion got plain text vectors; smooth everything again
        # with the validation images now in the index.
        suggester.smoothed.set_vectors(smooth_prototypes(
            suggester.prototypes.vectors, suggester.vocab.names, graph, suggester.knn.index.vectors,
            suggester.knn.labels, cfg.smooth_kappa, cfg.smooth_gamma, cfg.smooth_use_images))

    suggester.save(out_dir)
    (out_dir / "excluded_ids.json").write_text(json.dumps(excluded))
    report["seconds"] = round(time.time() - t_start, 1)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=1))
    return report
