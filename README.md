# CommonLens — Category Suggestions for Wikimedia Commons

## Requirements

- Python 3.10 – 3.13
- 8 GB RAM recommended; an NVIDIA GPU (CUDA) or Apple Silicon (MPS) is used automatically if present, CPU works
- Internet access to `commons.wikimedia.org` and `upload.wikimedia.org` (dataset) and `huggingface.co` (encoder weights, downloaded automatically on first use)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Optional, CPU-only machines: smaller PyTorch build
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Wikimedia asks API clients to identify themselves. Set a contact address before crawling:

```bash
export WIKIMEDIA_CONTACT="you@example.org"     # Windows: set WIKIMEDIA_CONTACT=you@example.org
```

## Run

All commands are run from the repository root.

### 1. Build the dataset

```bash
python prepare_data.py all --seeds configs/seed_categories.txt --max-files 6000
```

Writes `data/records.jsonl`, `data/images/`, `data/hierarchy.json` and `data/seeds.json`.
For a quick run use `--max-files 1500 --files-per-cat 15 --subcats 2`.
Stages can also be run one at a time: `crawl`, `download`, `hierarchy`, `split`.

Region-balanced seeds plus a uniformly random sample of Commons:

```bash
python prepare_data.py all --seeds configs/seed_categories.txt configs/seed_regions.txt --random-files 1500 --max-files 8000
```

To use a dataset you already have (CSV or JSONL with `image_path` and `categories`, optional `title`, `description`, `timestamp`, `uploader`, `width`, `height`):

```bash
python prepare_data.py import --table my_dataset.csv --image-root /path/to/images
python prepare_data.py split
python prepare_data.py hierarchy        # optional, needs internet
```

### 2. Train

```bash
python train.py
```

Writes everything needed for inference to `artifacts/`. Useful options:

| option | example |
|---|---|
| `--model` | `google/siglip-base-patch16-224`, `openai/clip-vit-base-patch16`, `google/siglip-base-patch16-256-multilingual` |
| `--device` | `cpu`, `cuda`, `mps` (default: auto) |
| `--index-type` | `flat` (default), `HNSW32`, `IVF1024,PQ64` |
| `--artifacts-dir` | `artifacts_siglip` (keep several models side by side) |

### 2b. Train the content-type classifier (Object / Living thing / Scene / Event / Abstract concept)

```bash
python prepare_content_types.py --per-type 200
python train_content_type.py --eval-pack path/to/photo-test-pack --eval-pack path/to/living-objects-dataset
```

Writes `artifacts/content_type.npz`, which `predict.py` and `app.py` use automatically. Extra images can be added
under `content_type_data/<Type>/`. Test packs are only evaluated on, never trained on.

### 3. Evaluate

```bash
python evaluate.py --split test
python evaluate.py --split test --k 1 3 5 10 --degrade blur:2,jpeg:10,downscale:0.15 --latency 50
python evaluate.py --center-focus 0.4                 # evaluate with centre-focused embeddings
```

Writes `results/metrics_test.json` and `results/report_test.md`.

### 4. Predict

```bash
python predict.py path/to/image.jpg
python predict.py path/to/image.jpg --title "Diwali lights in Jaipur 2023" --description "..." --top-k 10
python predict.py img1.jpg img2.jpg --json > suggestions.json
python predict.py https://upload.wikimedia.org/.../image.jpg
python predict.py photo.jpg --unsure-threshold 0.4   # mark as Unsure below 40% (default 50%)
python predict.py photo.jpg --center-focus 0        # use the whole frame equally (default 0.4 favours the centre)
```

### 5. Demo UI

```bash
python app.py                  # open http://127.0.0.1:7860
python app.py --share --read-only   # temporary public link for anyone; editing panels hidden
```

Problem reports sent by testers from the demo are saved in `feedback/reports.jsonl`. To summarise them:

```bash
python scripts/feedback_report.py > feedback_summary.md
```

### 6. Update a built index without retraining

```bash
python update.py add-category "Durga Puja 2026 in Kolkata" --parent "Durga Puja in Kolkata"
python update.py add-images --records data/new_uploads.jsonl
python update.py feedback photo.jpg --accept "Lighthouses in Kerala" --reject "Churches"
```

## Prebuilt artifacts and data

Large files are not stored in this repository.

| item | link | where to put it |
|---|---|---|
| Encoder weights | downloaded automatically from Hugging Face, e.g. [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | Hugging Face cache |
| Built index (`artifacts.zip`) | _link to be added_ | unzip to `artifacts/`, then run steps 4–5 directly |
| Crawled dataset (`data.zip`) | _link to be added_ | unzip to `data/`, needed for steps 2–3 |

## Tests

```bash
pytest -q
```

The tests run offline on a small synthetic dataset and need no model weights.

## Submission check

```bash
python scripts/check_submission.py --stage final            # file sizes, excluded folders, LICENSE, deliverables
python scripts/check_submission.py --stage final --make-zip # also writes code.zip from tracked files only
```

## Entry points

| file | purpose |
|---|---|
| `prepare_data.py` | crawl, import, split the dataset |
| `train.py` | build `artifacts/` |
| `prepare_content_types.py`, `train_content_type.py` | content-type classifier |
| `evaluate.py` | metrics on a split |
| `predict.py` | command-line suggestions |
| `app.py` | web demo |
| `update.py` | add categories, images or editor feedback |
| `wikicat/` | library code used by the scripts |

## Third-party software, models and data

- Images, titles, descriptions and categories from [Wikimedia Commons](https://commons.wikimedia.org) via the MediaWiki API; each file remains under the licence stated on its file page.
- [CLIP](https://github.com/openai/CLIP) (MIT) and optionally [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) (Apache-2.0) weights, loaded with [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0).
- [PyTorch](https://pytorch.org) (BSD-3-Clause), [FAISS](https://github.com/facebookresearch/faiss) (MIT), [scikit-learn](https://scikit-learn.org) (BSD-3-Clause), [NumPy](https://numpy.org) (BSD-3-Clause), [Pillow](https://python-pillow.org) (MIT-CMU), [Gradio](https://gradio.app) (Apache-2.0), [Requests](https://requests.readthedocs.io) (Apache-2.0), [tqdm](https://github.com/tqdm/tqdm) (MPL-2.0 / MIT).
- Parts of the code were written with the help of an AI coding assistant (Claude Code), as permitted by the hackathon rules.

## License

[MIT](LICENSE)
