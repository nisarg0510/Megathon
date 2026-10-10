import json

import numpy as np
from PIL import Image

import train_content_type
from wikicat.content_type import ContentTypeClassifier, commons_title, load_pack, pack_scores
from wikicat.families import FAMILIES, FamilyScorer

COLOURS = {"Object": (200, 30, 30), "Living thing": (30, 160, 60), "Scene": (40, 70, 220)}


def make_images(folder, colour, n, rng):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        arr = np.clip(np.array(colour) + rng.integers(-25, 25, size=(48, 48, 3)), 0, 255).astype(np.uint8)
        p = folder / f"{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def test_commons_title():
    assert commons_title("https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg") == "felis catus-cat on snow.jpg"
    assert commons_title("File:Felis catus-cat on snow.jpg") == "felis catus-cat on snow.jpg"


def test_pack_scores_strict_and_lenient():
    rows = [{"primary": "objects", "accepted": {"objects"}, "confusing": False, "living": True, "group": "people"},
            {"primary": "scenes", "accepted": {"scenes", "objects"}, "confusing": True, "living": False, "group": ""}]
    probs = np.zeros((2, len(FAMILIES)))
    probs[0, FAMILIES.index("Living thing")] = 1   # living thing counts as objects in four-family packs
    probs[1, FAMILIES.index("Object")] = 1         # wrong primary, but an accepted alternative
    res = pack_scores(rows, probs)
    assert res["all"]["strict"] == 0.5 and res["all"]["lenient"] == 1.0
    assert res["people_as_living_thing"] == 1.0 and res["confusing"]["n"] == 1


def test_train_and_evaluate_on_pack(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "content_type_data"
    manifest = []
    for family, colour in COLOURS.items():
        for p in make_images(data / family, colour, 12, rng):
            manifest.append({"type": family, "path": str(p), "title": f"File:{family} {p.stem}.png"})
    (data / "manifest.json").write_text(json.dumps(manifest))
    # Test pack whose first image is also in the training data: it must be dropped from training.
    pack = tmp_path / "pack"
    make_images(pack / "images", COLOURS["Scene"], 3, rng)
    rows = ["path,primary,accepted_categories,confusing,source_url"]
    rows.append("images/0.png,scenes,scenes,False,https://commons.wikimedia.org/wiki/File:Scene_0.png")
    rows += [f"images/{i}.png,scenes,scenes,True,https://commons.wikimedia.org/wiki/File:x{i}.png" for i in (1, 2)]
    (pack / "labels.csv").write_text("\n".join(rows) + "\n")
    assert len(load_pack(pack)) == 3

    art = tmp_path / "artifacts"
    report = train_content_type.main(["--data", str(data), "--artifacts-dir", str(art), "--model", "hashing",
                                      "--eval-pack", str(pack)])
    assert report["train_images"]["Scene"] == 11  # one overlapping photo removed
    assert report["holdout_accuracy"] > 0.8
    scores = report["pack:pack"]["trained_classifier"]
    assert scores["all"]["strict"] == 1.0 and "prompt_only" in report["pack:pack"]

    clf = ContentTypeClassifier.load(art / "content_type.npz")
    from wikicat.encoder import HashingEncoder

    enc = HashingEncoder()
    img = Image.open(data / "Living thing" / "0.png")
    emb = enc.encode_images([img])[0]
    out = FamilyScorer(enc, classifier=clf).score(emb, [])
    assert out[0]["family"] == "Living thing" and abs(sum(f["score"] for f in out) - 1) < 0.01
