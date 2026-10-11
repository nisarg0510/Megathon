"""Build the index, head and fusion model from data/records.jsonl.

Examples
--------
    python train.py
    python train.py --model google/siglip-base-patch16-224 --device cuda
    python train.py --index-type HNSW32 --artifacts-dir artifacts_siglip
    python train.py --no-graph-smoothing --artifacts-dir artifacts_text   # ablation
"""
from __future__ import annotations

import argparse
import json
import logging

from wikicat.config import Config
from wikicat.training import train


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--model", dest="model_name", help="Hugging Face CLIP/SigLIP id (default: CLIP ViT-B/32)")
    parser.add_argument("--device", help="auto | cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--index-type", help='"flat", "HNSW32", "IVF1024,PQ64", ...')
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--head-min-freq", type=int)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--no-parents", dest="vocab_include_parents", action="store_false", default=None,
                        help="Do not add parent categories to the vocabulary")
    parser.add_argument("--no-index-val", dest="index_val_after_fusion", action="store_false", default=None,
                        help="Keep validation images out of the final k-NN index")
    parser.add_argument("--no-graph-smoothing", dest="graph_smoothing", action="store_false", default=None,
                        help="Plain text prototypes (ablation)")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config().update(**{k: v for k, v in vars(args).items()})
    report = train(cfg)
    summary = {k: report[k] for k in ("model", "counts", "vocabulary", "head_categories", "seconds")}
    if "vocabulary_after_val" in report:
        summary["vocabulary_after_val"] = report["vocabulary_after_val"]
    if "knn_tuning" in report:
        summary["knn_best"] = report["knn_tuning"]["best"]
    if "fusion" in report:
        summary["val_candidate_recall"] = round(report["fusion"]["val_candidate_recall"], 3)
    if "hierarchy_tuning" in report:
        summary["hierarchy_best"] = report["hierarchy_tuning"]["best"]
    if "smoothing_tuning" in report:
        summary["smoothing_best"] = report["smoothing_tuning"]["best"]
    if "conformal" in report:
        summary["conformal_alpha"] = report["conformal"]["alpha"]
    print(json.dumps(summary, indent=1))
    print(f"Artifacts written to {cfg.artifacts_dir}/")
    return report


if __name__ == "__main__":
    main()
