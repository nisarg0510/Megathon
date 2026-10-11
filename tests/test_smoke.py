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
                 "head.npz", "fusion.json", "hierarchy.json", "train_report.json", "conformal.json",
                 "text_prototypes.npy"):
        assert (art / name).exists(), name
    assert report["counts"]["train"] > 0 and report["vocabulary"] > 0
    assert len(report["smoothing_tuning"]["grid"]) == 5 and report["conformal"]["alpha"] in (0.1, 0.2, 0.3, 0.4, 0.5)
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
    assert 0.0 <= res["content_type"]["top1_hit"] <= 1.0
    assert sum(g["images"] for g in res["seed_groups"].values()) == res["n_images"]
    assert (root / "results" / "report_test.md").exists()
    assert set(res["prototype_smoothing"]) == {"text only", "graph-smoothed"}
    cf = res["conformal"]
    assert all(0 <= v["coverage"] <= 1 and v["mean_size"] >= 0 for v in cf["image only"].values())
    assert set(cf["fixed_top_k"]) == {"@1", "@3", "@5", "@10"}


def test_predict_returns_ranked_suggestions(built, capsys):
    root, cfg, _ = built
    image = str(root / "data" / "images" / "0.png")
    [out] = predict.main([image, "--title", "red circle", "--top-k", "4", "--json",
                          "--artifacts-dir", cfg.artifacts_dir])
    assert len(out["content_type"]) == 5 and out["type_label"]
    assert all(s["score"] >= 0.7 for s in out["suggestions"] if "festival" in s["category"])  # dated ones hidden
    assert out["unsure"] == (out["type_confidence"] < 0.5)
    scores = [s["score"] for s in out["suggestions"]]
    assert len(scores) == 4 and scores == sorted(scores, reverse=True)
    sset = out["suggestion_set"]
    assert sset is not None and sset["abstain"] == (sset["size"] == 0)
    # The set is the top of the ranking: marked rows come first.
    marks = [s["in_set"] for s in out["suggestions"]]
    assert marks == sorted(marks, reverse=True)
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
    after.add_categories(["Kenya"])
    assert "Kenya" not in [s["category"] for s in after.suggest(image, title="Kenya", top_k=50)["suggestions"]]
    assert "Brand new category" in [s["category"] for s in result["suggestions"]]
