import json

import numpy as np
from PIL import Image

import train_content_type
from wikicat.content_type import ContentTypeClassifier, commons_title, find_pack_file, load_pack, pack_scores
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


def test_pack_found_when_nested_or_zipped(tmp_path):
    import zipfile

    inner = tmp_path / "Downloads" / "pack-zip-name" / "photo-test-pack"
    (inner / "images").mkdir(parents=True)
    (inner / "labels.csv").write_text("path,primary,accepted_categories\nimages/a.jpg,scenes,scenes;objects\n")
    for given in (tmp_path / "Downloads" / "pack-zip-name", inner, inner / "labels.csv"):
        rows = load_pack(given)
        assert rows[0]["path"] == str(inner / "images" / "a.jpg") and rows[0]["accepted"] == {"scenes", "objects"}
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(inner / "labels.csv", "photo-test-pack/labels.csv")
    assert find_pack_file(archive) == tmp_path / "pack" / "photo-test-pack" / "labels.csv"
    try:
        load_pack(tmp_path / "missing")
    except FileNotFoundError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("missing pack should raise")


def test_every_seed_gets_a_share(tmp_path, monkeypatch):
    import prepare_content_types as pct

    seeds = {"Object": ["A", "B"], "Living thing": ["Cats", "Dogs", "Portrait photographs"]}
    calls = []

    def fake_crawl(client, seed_list, max_files=None, **kw):
        calls.append((seed_list[0], max_files))
        return [{"id": f"{seed_list[0]}{i}", "title": f"File:{seed_list[0]}{i}.jpg", "page_url": "", "seed": seed_list[0]}
                for i in range(max_files)]

    def fake_download(records, folder, workers=4):
        return [dict(r, image_path=str(folder / f"{r['id']}.jpg")) for r in records]

    monkeypatch.setattr(pct, "TYPE_SEEDS", seeds)
    monkeypatch.setattr(pct.dl, "CommonsClient", lambda: None)
    monkeypatch.setattr(pct.dl, "crawl", fake_crawl)
    monkeypatch.setattr(pct.dl, "download_images", fake_download)
    out = tmp_path / "ct"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps([{"type": "Object", "path": "old.jpg", "title": "File:old.jpg"}]))
    pct.main(["--out", str(out), "--per-type", "30"])
    assert [s for s, _ in calls] == ["A", "B", "Cats", "Dogs", "Portrait photographs"]
    assert dict(calls)["Portrait photographs"] == 10 and dict(calls)["A"] == 15
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest) == 1 + 30 + 30 and any(m["path"] == "old.jpg" for m in manifest)
