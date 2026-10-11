"""End-to-end suggester: load artifacts, score, rank, explain, and update.

Artifacts directory layout (written by train.py):

    config.json          run configuration
    vocab.json           category names and training frequencies
    prototypes.npy       category text prototypes (float16), matched against images and metadata text
    smoothed_prototypes.npy  graph-smoothed prototypes (text + image centroid), matched against images
    knn_index.npy        indexed image embeddings (float16)
    knn_labels.json      labels / rejected labels / ids of indexed images
    items.jsonl          title and URLs of indexed images (explanations)
    head.npz             supervised head
    fusion.json          fusion weights
    conformal.json       conformal calibration scores (suggestion sets)
    hierarchy.json       category parents
    feedback.jsonl       editor accept/reject log (appended at run time)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .conformal import ConformalCalibrator, group_of
from .encoder import BaseEncoder, load_encoder, load_image
from .families import FamilyScorer, adjust_for_subject
from .geo import is_country
from .head import LinearHead
from .hierarchy import CategoryGraph
from .metadata import LexicalScorer, build_meta_text, extract_years, tokenize
from .quality import assess_quality
from .specificity import specificity_reason
from .ranker import FusionModel, QueryContext, Ranker
from .retriever import KNNScorer, PrototypeScorer

log = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, names: list[str] | None = None, freq: list[float] | None = None):
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.freq = np.zeros(0, dtype=np.float32)
        if names:
            self.add(names, freq)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def add(self, names: list[str], freq: list[float] | None = None) -> list[int]:
        new_ids = []
        new_freq = []
        for i, name in enumerate(names):
            if name in self.index:
                continue
            self.index[name] = len(self.names)
            new_ids.append(len(self.names))
            self.names.append(name)
            new_freq.append(freq[i] if freq is not None else 0.0)
        if new_freq:
            self.freq = np.concatenate([self.freq, np.asarray(new_freq, dtype=np.float32)])
        return new_ids

    def ids(self, names: list[str]) -> list[int]:
        return [self.index[n] for n in names if n in self.index]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"names": self.names, "freq": self.freq.tolist()}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        d = json.loads(path.read_text())
        return cls(d["names"], d["freq"])


class CategorySuggester:
    def __init__(self, cfg: Config, vocab: Vocabulary, prototypes: PrototypeScorer, knn: KNNScorer,
                 head: LinearHead | None, fusion: FusionModel, graph: CategoryGraph, items: list[dict],
                 encoder: BaseEncoder | None = None, conformal: ConformalCalibrator | None = None,
                 smoothed: PrototypeScorer | None = None):
        self.cfg = cfg
        self.conformal = conformal
        self.smoothed = smoothed
        self.vocab = vocab
        self.prototypes = prototypes
        self.knn = knn
        self.head = head
        self.fusion = fusion
        self.graph = graph
        self.items = items
        self.lexical = LexicalScorer(vocab.names)
        self._encoder = encoder
        self._family_scorer: FamilyScorer | None = None
        self._broad_scorer = None
        self.ranker = Ranker(vocab.names, vocab.freq, prototypes, head, self.lexical, fusion, graph,
                             top_n=cfg.top_n, hierarchy_depth=cfg.hierarchy_depth,
                             ancestor_ratio=cfg.ancestor_ratio, ancestor_penalty=cfg.ancestor_penalty,
                             smoothed=smoothed)

    # -- loading -----------------------------------------------------------------

    @property
    def encoder(self) -> BaseEncoder:
        if self._encoder is None:
            self._encoder = load_encoder(self.cfg.model_name, self.cfg.device, self.cfg.batch_size)
        return self._encoder

    @property
    def family_scorer(self) -> FamilyScorer:
        if self._family_scorer is None:
            from .content_type import ContentTypeClassifier

            path = Path(self.cfg.artifacts_dir) / "content_type.npz"
            classifier = ContentTypeClassifier.load(path) if path.exists() else None
            if classifier is not None and classifier.model_name not in ("", self.encoder.name):
                log.warning("content_type.npz was trained with %s, not %s; ignoring it",
                            classifier.model_name, self.encoder.name)
                classifier = None
            self._family_scorer = FamilyScorer(self.encoder, classifier=classifier)
        return self._family_scorer

    @property
    def broad_scorer(self):
        if self._broad_scorer is None:
            from .broad import BroadCategoryScorer

            self._broad_scorer = BroadCategoryScorer(self.encoder)
        return self._broad_scorer

    @classmethod
    def load(cls, artifacts_dir: str | Path, encoder: BaseEncoder | None = None, **overrides) -> "CategorySuggester":
        d = Path(artifacts_dir)
        if not (d / "config.json").exists():
            raise FileNotFoundError(f"No artifacts in {d}. Run train.py first (see README).")
        cfg = Config.load(d / "config.json", artifacts_dir=str(d), **overrides)
        head_path = d / "head.npz"
        items = [json.loads(l) for l in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        fusion_path = d / "fusion.json"
        conformal_path = d / "conformal.json"
        return cls(
            cfg,
            Vocabulary.load(d / "vocab.json"),
            PrototypeScorer.load(d, cfg.index_type),
            KNNScorer.load(d, cfg.index_type),
            LinearHead.load(head_path) if head_path.exists() else None,
            FusionModel.load(fusion_path) if fusion_path.exists() else FusionModel.default(),
            CategoryGraph.load(d / "hierarchy.json"),
            items,
            encoder,
            ConformalCalibrator.load(conformal_path) if conformal_path.exists() else None,
            PrototypeScorer.load(d, cfg.index_type, "smoothed_prototypes")
            if (d / "smoothed_prototypes.npy").exists() else None,
        )

    def save(self, artifacts_dir: str | Path | None = None) -> Path:
        d = Path(artifacts_dir or self.cfg.artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.cfg.save(d / "config.json")
        self.vocab.save(d / "vocab.json")
        self.prototypes.save(d)
        if self.smoothed is not None:
            self.smoothed.save(d, "smoothed_prototypes")
        self.knn.save(d)
        if self.head is not None:
            self.head.save(d / "head.npz")
        self.fusion.save(d / "fusion.json")
        if self.conformal is not None:
            self.conformal.save(d / "conformal.json")
        self.graph.save(d / "hierarchy.json")
        with (d / "items.jsonl").open("w", encoding="utf-8") as fh:
            for item in self.items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return d

    # -- scoring -------------------------------------------------------------------

    def encode_meta(self, texts: list[str]) -> list[np.ndarray | None]:
        """Text embeddings for non-empty metadata strings; None where empty."""
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        out: list[np.ndarray | None] = [None] * len(texts)
        if idx:
            embs = self.encoder.encode_texts([texts[i] for i in idx])
            for row, i in enumerate(idx):
                out[i] = embs[row]
        return out

    def contexts(self, img_embs: np.ndarray, meta_texts: list[str] | None = None,
                 meta_embs: list[np.ndarray | None] | None = None, low_quality: list[bool] | None = None,
                 knn_k: int | None = None) -> list[QueryContext]:
        n = len(img_embs)
        meta_texts = meta_texts or [""] * n
        meta_embs = meta_embs or [None] * n
        low_quality = low_quality or [False] * n
        k = knn_k or self.cfg.knn_k

        nb_sims, nb_idx = self.knn.search(img_embs, k)
        txt_sims, txt_ids = self.prototypes.top(img_embs, self.cfg.top_n)
        if self.smoothed is not None:
            sm_sims, sm_ids = self.smoothed.top(img_embs, self.cfg.top_n)
        head_probs = self.head.predict(img_embs) if self.head is not None and len(self.head) else None

        meta_rows = [i for i in range(n) if meta_embs[i] is not None]
        meta_top: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if meta_rows:
            m_sims, m_ids = self.prototypes.top(np.stack([meta_embs[i] for i in meta_rows]), self.cfg.top_n)
            meta_top = {i: (m_sims[r], m_ids[r]) for r, i in enumerate(meta_rows)}

        out = []
        for i in range(n):
            ctx = QueryContext(
                img_emb=img_embs[i],
                knn=self.knn.vote(nb_sims[i], nb_idx[i], self.cfg.knn_tau, self.cfg.knn_reject_weight),
                neighbor_sims=nb_sims[i],
                neighbor_idx=nb_idx[i],
                txt_ids=txt_ids[i],
                txt_sims=txt_sims[i],
                head_probs=head_probs[i] if head_probs is not None else None,
                low_quality=bool(low_quality[i]),
            )
            if self.smoothed is not None:
                ctx.sm_sims, ctx.sm_ids = sm_sims[i], sm_ids[i]
            if meta_embs[i] is not None:
                ctx.meta_emb = meta_embs[i]
                ctx.meta_sims, ctx.meta_ids = meta_top[i]
                ctx.meta_tokens = set(tokenize(meta_texts[i]))
                ctx.meta_years = extract_years(meta_texts[i])
            out.append(ctx)
        return out

    def evidence(self, ctx: QueryContext, cat_id: int, limit: int = 3) -> list[dict]:
        """Indexed neighbours that carry the category: the explanation shown to editors."""
        out = []
        for sim, item in zip(ctx.neighbor_sims.tolist(), ctx.neighbor_idx.tolist()):
            if item >= 0 and cat_id in self.knn.labels[item]:
                meta = self.items[item] if item < len(self.items) else {}
                out.append({"similarity": round(sim, 3), "title": meta.get("title", ""),
                            "page_url": meta.get("page_url", ""), "thumb_url": meta.get("thumb_url", ""),
                            "image_path": meta.get("image_path", "")})
                if len(out) >= limit:
                    break
        return out

    def suggest(self, image, title: str = "", description: str = "", top_k: int = 10,
                use_hierarchy: bool = True, original_size: tuple[int, int] | None = None) -> dict:
        t0 = time.perf_counter()
        img = load_image(image)
        quality = assess_quality(img, original_size)
        emb = self.encoder.encode_focused([img], self.cfg.center_focus, self.cfg.center_crop)
        t1 = time.perf_counter()
        meta_text = build_meta_text(title, description)
        meta_embs = self.encode_meta([meta_text])
        ctx = self.contexts(emb, [meta_text], meta_embs, [quality["low_quality"]])[0]
        hidden = 0
        res = self.ranker.score(ctx, use_hierarchy)
        # Presentation filters (the model itself is scored without them): bare country names, and
        # very specific categories unless the score is high.
        ranked = []
        for s in self.ranker.rank(ctx, top_k * 3, use_hierarchy, res):
            if self.cfg.hide_countries and is_country(s["category"]):
                continue
            if (self.cfg.hide_specific and s["score"] < self.cfg.specific_min_score
                    and specificity_reason(s["category"], meta_text)):
                hidden += 1
            elif len(ranked) < top_k:
                ranked.append(s)
        suggestion_set = None
        if self.conformal is not None:
            alpha = self.cfg.conformal_alpha
            t = self.conformal.threshold(alpha, group_of(ctx.has_meta))
            size = int((res["final"] >= t).sum())
            for s in ranked:
                s["in_set"] = t > 0 and s["score"] >= t
            suggestion_set = {"alpha": alpha, "coverage_target": round(1 - alpha, 2), "threshold": round(t, 4),
                              "size": size, "abstain": size == 0}
        for s in ranked:
            s["evidence"] = self.evidence(ctx, s["cat_id"])
        families = self.family_scorer.score(emb[0], [(s["category"], s["score"]) for s in ranked])
        families, subject = adjust_for_subject(families, self.family_scorer.subject(emb[0]))
        broad = self.broad_scorer.top(emb[0])
        # Two types this close (e.g. a fruit market: Object and Scene) are both shown, unless the type is
        # uncertain anyway: then a single best guess is clearer.
        type_unsure = families[0]["score"] < self.cfg.unsure_threshold if families else True
        mixed = (not type_unsure and len(families) > 1
                 and families[1]["score"] >= 0.5 * families[0]["score"])
        type_label = f"{families[0]['family']} + {families[1]['family']}" if mixed else families[0]["family"]
        if subject == "people" and not type_unsure:
            type_label = f"{families[0]['family']} + Living object"  # what the photo shows, with people in it
        t2 = time.perf_counter()
        notes = []
        # "Unsure" follows the content type (the first row): below the threshold, ask for human feedback.
        confidence = ranked[0]["score"] if ranked else 0.0
        type_confidence = families[0]["score"] if families else 0.0
        unsure = type_confidence < self.cfg.unsure_threshold
        if unsure:
            notes.append(f"Unsure ({type_confidence:.0%} sure): human feedback is suggested.")
        elif confidence < self.cfg.unsure_threshold:
            notes.append(f"The content type is clear, but the specific categories below are low-confidence "
                         f"(best {confidence:.0%}). Adding a title usually helps.")
        if suggestion_set is not None:
            target = f"{suggestion_set['coverage_target']:.0%}"
            if suggestion_set["abstain"]:
                notes.append(f"No category reaches the calibrated threshold ({suggestion_set['threshold']:.0%}): "
                             f"manual categorisation recommended.")
            elif suggestion_set["threshold"] > 0:
                notes.append(f"Suggestion set: the top {suggestion_set['size']} categories (marked ✓) contain a "
                             f"correct one with at least {target} probability when the image's category is known "
                             f"(conformal calibration).")
        if hidden:
            notes.append(f"Hid {hidden} very specific categories (dates, counts, collections, country-level "
                         f"variants, non-Latin names) scoring below {self.cfg.specific_min_score:.0%}.")
        if quality["low_quality"]:
            notes.append("Low-quality image (" + ", ".join(quality["flags"]) + "): visual scores are less reliable.")
        if not meta_text:
            notes.append("No title/description given: suggestions use the image only.")
        return {
            "unsure": unsure,
            "confidence": round(confidence, 3),
            "type_confidence": round(type_confidence, 3),
            "content_type": families,
            "type_label": type_label,
            "subject": subject,
            "broad": broad,
            "suggestions": ranked,
            "suggestion_set": suggestion_set,
            "quality": quality,
            "notes": notes,
            "timing_ms": {"encode_image": round((t1 - t0) * 1000, 1), "score_rank": round((t2 - t1) * 1000, 1)},
        }

    # -- continual learning ----------------------------------------------------------

    def _sync(self) -> None:
        self.ranker.cat_freq = self.vocab.freq
        self.ranker.on_categories_added()

    def add_categories(self, names: list[str], parents: dict[str, list[str]] | None = None) -> list[int]:
        """Make new categories suggestible immediately (zero-shot + lexical), no retraining."""
        from .data_loader import normalize_category

        names = [normalize_category(n) for n in names]
        new = [n for n in dict.fromkeys(names) if n not in self.vocab]
        if new:
            vectors = self.encoder.encode_categories(new)
            self.prototypes.add(vectors)
            if self.smoothed is not None:  # keep ids aligned; a new category has no images yet
                self.smoothed.add(vectors)
            self.vocab.add(new)
            self.lexical.add(new)
            self._sync()
        for child, ps in (parents or {}).items():
            self.graph.add_edges(normalize_category(child), [normalize_category(p) for p in ps])
        return self.vocab.ids(names)

    def add_images(self, embs: np.ndarray, label_names: list[list[str]], items: list[dict],
                   rejected: list[list[str]] | None = None) -> None:
        """Append newly categorised images to the k-NN index (no retraining)."""
        rejected = rejected or [[] for _ in label_names]
        self.add_categories(sorted({c for labels in label_names + rejected for c in labels}))
        labels = [self.vocab.ids(l) for l in label_names]
        negatives = [self.vocab.ids(r) for r in rejected]
        self.knn.add(np.asarray(embs, dtype=np.float32), labels, [str(it.get("id", "")) for it in items], negatives)
        for ids in labels:
            self.vocab.freq[ids] += 1
        self.items.extend(items)

    def record_feedback(self, image, accepted: list[str], rejected: list[str], title: str = "",
                        artifacts_dir: str | Path | None = None) -> None:
        """Editor accept/reject: accepted labels index the image, rejects become hard negatives."""
        img = load_image(image)
        emb = self.encoder.encode_images([img])
        item_id = f"feedback-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        self.add_images(emb, [accepted], [{"id": item_id, "title": title or item_id}], [rejected])
        d = Path(artifacts_dir or self.cfg.artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "feedback.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": item_id, "title": title, "accepted": accepted, "rejected": rejected,
                                 "time": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
