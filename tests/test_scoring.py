import numpy as np
import pytest

from wikicat import retriever
from wikicat.encoder import HashingEncoder, category_to_text, l2_normalize
from wikicat.head import LinearHead
from wikicat.hierarchy import CategoryGraph
from wikicat.metadata import LexicalScorer, extract_years, tokenize, year_feature
from wikicat.metrics import (bucket_report, category_type, hierarchical_precision_at_k, label_coverage,
                             mean_average_precision, per_category_recall, prf_at_k)
from wikicat.quality import assess_quality, degrade
from wikicat.ranker import FusionModel, suppress_ancestors

RNG = np.random.default_rng(0)
X = l2_normalize(RNG.standard_normal((300, 32)))


@pytest.mark.parametrize("index_type", ["flat", "HNSW16", "IVF4,Flat"])
def test_vector_index_finds_exact_match(index_type):
    if retriever.faiss is None and index_type != "flat":
        pytest.skip("faiss not installed")
    index = retriever.VectorIndex(32, index_type)
    index.add(X[:200])
    index.add(X[200:])
    sims, idx = index.search(X[[0, 250]], 3)
    assert idx[:, 0].tolist() == [0, 250]
    assert np.allclose(sims[:, 0], 1.0, atol=1e-4)


def test_numpy_fallback_matches_faiss(monkeypatch):
    monkeypatch.setattr(retriever, "faiss", None)
    index = retriever.VectorIndex(32)
    index.add(X)
    sims, idx = index.search(X[:5], 4)
    exact = np.argsort(-(X[:5] @ X.T), axis=1)[:, :4]
    assert (idx == exact).all()


def test_vector_index_save_load(tmp_path):
    index = retriever.VectorIndex(32)
    index.add(X)
    index.save(tmp_path / "v")
    loaded = retriever.VectorIndex.load(tmp_path / "v")
    assert len(loaded) == len(X)
    assert loaded.search(X[:1], 1)[1][0, 0] == 0


def test_knn_vote_with_rejected_labels():
    knn = retriever.KNNScorer(retriever.VectorIndex(32), [], [], [])
    knn.add(X[:3], [[0, 1], [0], [2]], ["a", "b", "c"], negatives=[[], [1], []])
    query = ((X[0] + X[1]) / np.linalg.norm(X[0] + X[1]))[None]
    sims, idx = knn.search(query, 3)
    votes = knn.vote(sims[0], idx[0], tau=1.0)
    assert votes[0] > votes[2]
    assert votes[1] < votes[0]  # item b rejected label 1


def test_head_learns_separable_labels():
    labels = [[int(X[j, 0] > 0), 2 + int(X[j, 1] > 0)] for j in range(len(X))]
    protos = l2_normalize(RNG.standard_normal((4, 32)))
    head = LinearHead.fit(X, labels, np.arange(4), protos, epochs=60, lr=0.01)
    p = head.predict(X)
    acc = np.mean([(p[j, 1] > p[j, 0]) == bool(labels[j][0]) for j in range(len(X))])
    assert acc > 0.9


def test_hierarchy_cycles_and_suppression():
    g = CategoryGraph({"Churches in Paris": ["Churches in France", "Buildings in Paris"],
                       "Churches in France": ["Churches"], "Churches": ["Churches in France"]})
    assert g.ancestors("Churches in Paris", 3) == {"Churches in France", "Buildings in Paris", "Churches"}
    names = ["Churches", "Churches in Paris", "Eiffel Tower"]
    adj, sup = suppress_ancestors(names, np.array([0.9, 0.7, 0.2]), g)
    assert sup == {0: 1} and adj[0] < adj[1]
    adj, sup = suppress_ancestors(names, np.array([0.9, 0.1, 0.2]), g)
    assert 0 not in sup  # descendant too weak: ancestor kept


def test_lexical_and_years():
    lex = LexicalScorer(["Churches in Paris", "Eiffel Tower", "Diwali 2023 in India", "Paris"])
    toks = set(tokenize("File:Eiffel_Tower_from_Champ_de_Mars,_Paris_IMG_20190512.jpg"))
    top = lex.top(toks, 5)
    assert top[1] == 1.0 and top[3] == 1.0 and 0 < top[0] < 1
    assert "img" not in toks and "church" in tokenize("Churches")
    assert extract_years("Diwali 2023; taken 2023-11-12") == {2023}
    assert year_feature({2023}, {2023}) == 1.0 and year_feature({2022}, {2023}) == -1.0
    assert year_feature(set(), {2023}) == 0.0


def test_metrics_values():
    preds = [["a", "b", "c"], ["x", "y", "z"]]
    gold = [{"a", "c", "d", "e"}, {"q"}]
    m = prf_at_k(preds, gold, 2)
    assert m["P"] == pytest.approx(0.25) and m["R"] == pytest.approx(0.125)
    assert m["F1"] == pytest.approx((2 * 0.5 * 0.25 / 0.75) / 2)
    g = CategoryGraph({"a": ["A"]})
    assert hierarchical_precision_at_k([["A"]], [{"a"}], g, 1) == 1.0
    assert label_coverage(gold, {"a", "c"}) == pytest.approx(2 / 5)
    per = per_category_recall(preds, gold, 3)
    assert per["a"] == (1, 1) and per["d"] == (0, 1)
    buckets = bucket_report(per, {"a": 25, "c": 3})
    assert buckets["head (20+)"]["categories"] == 1 and buckets["unseen (0)"]["categories"] == 3
    assert mean_average_precision([{"a": 0.9}, {"a": 0.1}], [{"a"}, set()]) == 1.0


def test_category_types():
    assert category_type("Diwali 2023 in India") == "event"
    assert category_type("Sunsets in Goa") == "scene"
    assert category_type("Abstract art") == "abstract"
    assert category_type("Bicycles") == "object"


def test_fusion_fit_and_roundtrip(tmp_path):
    from wikicat.ranker import FEATURE_NAMES

    x = RNG.standard_normal((400, len(FEATURE_NAMES))).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 4] > 0).astype(int)
    fusion = FusionModel.fit(x, y)
    assert fusion.coef[0] > 0 and fusion.coef[4] > 0
    fusion.save(tmp_path / "f.json")
    assert np.allclose(FusionModel.load(tmp_path / "f.json").predict_proba(x), fusion.predict_proba(x))
    assert (FusionModel.default().contributions(np.zeros((1, len(FEATURE_NAMES)))) == 0).all()


def test_quality_flags_and_encoder():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    for i in range(0, 400, 20):
        d.line([(i, 0), (i, 300)], fill="black", width=3)
    assert not assess_quality(img)["low_quality"]
    assert assess_quality(degrade(img, "blur", 6))["blurry"]
    assert assess_quality(img, original_size=(120, 90))["low_resolution"]
    enc = HashingEncoder()
    e = enc.encode_images([img, degrade(img, "jpeg", 10)])
    assert e.shape == (2, enc.dim) and float(e[0] @ e[1]) > 0.9
    assert enc.encode_categories(["Cats", "Dogs"]).shape == (2, enc.dim)
    assert category_to_text("Mercury (planet)") == "Mercury, planet"


@pytest.mark.parametrize("family", ["clip", "siglip"])
def test_hf_encoder_with_tiny_random_model(family):
    """Exercises HFEncoder's calls into transformers (v4 tensors / v5 ModelOutputs)
    with a tiny randomly initialised model, so no download is needed."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    from wikicat.encoder import HFEncoder

    tiny = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2)
    text_cfg = dict(tiny, vocab_size=100, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    vision_cfg = dict(tiny, image_size=32, patch_size=16)
    if family == "clip":
        model = transformers.CLIPModel(transformers.CLIPConfig(text_config=text_cfg, vision_config=vision_cfg,
                                                               projection_dim=16))
    else:
        model = transformers.SiglipModel(transformers.SiglipConfig(text_config=text_cfg, vision_config=vision_cfg))

    class FakeProcessor:
        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            if images is not None:
                return {"pixel_values": torch.randn(len(images), 3, 32, 32)}
            ids = torch.randint(2, 100, (len(text), 6))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    enc = HFEncoder.__new__(HFEncoder)
    enc.torch, enc.name, enc.device, enc.dtype, enc.batch_size = torch, family, "cpu", torch.float32, 2
    enc.model, enc.processor, enc.is_siglip = model.eval(), FakeProcessor(), family == "siglip"
    enc.dim = enc.encode_texts(["a photo"]).shape[1]

    imgs = enc.encode_images([Image.new("RGB", (40, 30))] * 3)
    txts = enc.encode_categories(["Cats", "Churches in Paris"])
    assert imgs.shape == (3, enc.dim) and txts.shape == (2, enc.dim)
    assert np.allclose(np.linalg.norm(imgs, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(txts, axis=1), 1, atol=1e-5)


def test_content_type_families():
    from wikicat.families import FAMILIES, FamilyScorer, family_of

    assert family_of("Diwali 2023 in India") == "Event" and family_of("Beaches in Goa") == "Scene"
    assert family_of("Abstract art") == "Abstract concept" and family_of("Bicycles") == "Object"
    enc = HashingEncoder()
    scorer = FamilyScorer(enc)
    img = l2_normalize(RNG.standard_normal(enc.dim))
    out = scorer.score(img, [("Beaches in Goa", 0.9), ("Bicycles", 0.1)])
    assert [f["family"] for f in out][0] == "Scene" and set(f["family"] for f in out) == set(FAMILIES)
    assert abs(sum(f["score"] for f in out) - 1) < 0.01
    assert len(scorer.score(img, [])) == 4  # falls back to zero-shot only


def test_country_names():
    from wikicat.geo import is_country

    assert is_country("India") and is_country("united states") and is_country("Asia")
    assert not is_country("Beaches in Goa") and not is_country("Indian food")
