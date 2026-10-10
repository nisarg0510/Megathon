"""End-to-end plumbing test on a synthetic dataset with the weight-free
HashingEncoder: train -> evaluate -> predict -> update. Checks that every
stage runs and produces well-formed output; the numbers are not model results."""
import json

import pytest

import evaluate
import predict
import update
from synthetic import make_dataset
from wikicat.config import Config
from wikicat.encoder import HashingEncoder
from wikicat.pipeline import CategorySuggester
from wikicat.training import train


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    data_dir = make_dataset(root)
    cfg = Config(data_dir=str(data_dir), artifacts_dir=str(root / "artifacts"), model_name="hashing", head_epochs=10)
    report = train(cfg, encoder=HashingEncoder())
    return root, cfg, report


def test_train_writes_artifacts(built):
    root, cfg, report = built
    art = root / "artifacts"
    for name in ("config.json", "vocab.json", "prototypes.npy", "knn_index.npy", "knn_labels.json", "items.jsonl",
                 "head.npz", "fusion.json", "hierarchy.json", "train_report.json"):
        assert (art / name).exists(), name
    assert report["counts"]["train"] > 0 and report["vocabulary"] > 0
    assert json.loads((art / "fusion.json").read_text())["fitted"]


def test_evaluate_runs(built):
    root, cfg, _ = built
    res = evaluate.main(["--artifacts-dir", cfg.artifacts_dir, "--out", str(root / "results"), "--latency", "3",
                         "--degrade", "blur:3"])
    assert res["n_images"] > 0
    for setting in ("image + metadata", "image only"):
        for k in ("@1", "@5"):
            assert 0.0 <= res["main"][setting][k]["F1"] <= 1.0
    assert "k-NN only" in res["ablation"] and "blur:3" in res["degradations"]
    assert (root / "results" / "report_test.md").exists()


def test_predict_returns_ranked_suggestions(built, capsys):
    root, cfg, _ = built
    image = str(root / "data" / "images" / "0.png")
    [out] = predict.main([image, "--title", "red circle", "--top-k", "4", "--json",
                          "--artifacts-dir", cfg.artifacts_dir])
    scores = [s["score"] for s in out["suggestions"]]
    assert len(scores) == 4 and scores == sorted(scores, reverse=True)
    assert json.loads(capsys.readouterr().out)[0]["image"] == image


def test_update_adds_categories_and_feedback(built):
    root, cfg, _ = built
    before = CategorySuggester.load(cfg.artifacts_dir)
    n_cats, n_items = len(before.vocab), len(before.knn)
    update.main(["--artifacts-dir", cfg.artifacts_dir, "add-category", "Purple hexagons", "--parent", "Hexagons"])
    image = str(root / "data" / "images" / "1.png")
    update.main(["--artifacts-dir", cfg.artifacts_dir, "feedback", image, "--accept", "Brand new category",
                 "--reject", "Squares"])
    after = CategorySuggester.load(cfg.artifacts_dir)
    assert len(after.vocab) == n_cats + 2 and len(after.knn) == n_items + 1
    assert after.graph.is_ancestor("Hexagons", "Purple hexagons")
    # The fed-back image is now its own nearest neighbour and carries the accepted label.
    result = after.suggest(image, top_k=50, use_hierarchy=False)
    assert "Brand new category" in [s["category"] for s in result["suggestions"]]
