import json

import numpy as np
import pytest

from wikicat import data_loader as dl


@pytest.mark.parametrize("name", [
    "Churches in Paris", "Eiffel Tower", "Diwali 2023 in India", "Cats lying on the back", "Sunsets in Goa",
    "Abstract art", "Lighthouses in Kerala", "Contact lens", "Panoramics of Mumbai", "Women of India",
])
def test_content_categories_are_kept(name):
    assert not dl.is_maintenance_category(name)


@pytest.mark.parametrize("name", [
    "CC-BY-SA-4.0", "Self-published work", "Uploaded with UploadWizard", "Files by User:Foo",
    "Media needing categories", "Quality images", "Pictures of the day (2023)",
    "Images from Wiki Loves Monuments 2023 in India", "Wiki Loves Earth 2022", "Taken with Canon EOS 5D",
    "Photographs taken on 2019-05-04", "PNG files", "Retouched pictures", "Supported by Wikimedia Deutschland",
    "Category:License migration redundant", "Files with coordinates", "Unidentified locations",
    "Flickr images reviewed by FlickreviewR", "Lens focal length 50 mm", "GFDL", "PD-old-70", "",
])
def test_maintenance_categories_are_dropped(name):
    assert dl.is_maintenance_category(name)


def test_clean_categories_normalises_and_dedupes():
    assert dl.clean_categories(["Category:Churches_in_Paris", "Churches in Paris", "CC-BY-4.0"]) == ["Churches in Paris"]


def test_title_and_html_cleaning():
    assert dl.title_to_text("File:Eiffel_Tower_at_night_2019.JPG") == "Eiffel Tower at night 2019"
    assert dl.strip_html("<p>A <b>red</b> &amp; blue</p>") == "A red & blue"
    assert dl.strip_html(None) == ""


class FakeClient(dl.CommonsClient):
    """Replays a categories continuation: the second response only carries more categories."""

    def __init__(self):
        pass

    def query(self, params):
        info = {"mime": "image/jpeg", "timestamp": "2020-01-01T00:00:00Z", "user": "u", "thumburl": "http://x/a.jpg",
                "width": 10, "height": 10, "extmetadata": {"ImageDescription": {"value": "<i>A cat</i>"}}}
        yield {"query": {"pages": [{"pageid": 1, "title": "File:A.jpg", "imageinfo": [info],
                                    "categories": [{"title": "Category:Cats"}]}]},
               "continue": {"clcontinue": "1|x", "continue": "||"}}
        yield {"query": {"pages": [{"pageid": 1, "title": "File:A.jpg",
                                    "categories": [{"title": "Category:Cats in Kerala"}, {"title": "Category:CC-BY-4.0"}]}]}}


def test_files_in_category_merges_continuations():
    [rec] = FakeClient().files_in_category("Cats", 1)
    assert rec["categories"] == ["Cats", "Cats in Kerala"]
    assert rec["description"] == "A cat"
    assert rec["thumb_url"] == "http://x/a.jpg"


def test_temporal_split_orders_by_time():
    recs = [{"id": str(i), "timestamp": f"2020-01-{i + 1:02d}", "uploader": f"u{i % 3}", "categories": ["a"]}
            for i in range(20)]
    recs = dl.assign_temporal_splits(list(reversed(recs)), 0.15, 0.15)
    splits = [r["split"] for r in recs]
    assert splits == ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    assert max(r["timestamp"] for r in recs if r["split"] == "train") < min(
        r["timestamp"] for r in recs if r["split"] == "test")


def test_uploader_disjoint_split():
    recs = [{"id": str(i), "timestamp": f"2020-01-{i + 1:02d}", "uploader": "same" if i < 18 else f"new{i}",
             "categories": ["a"]} for i in range(20)]
    recs = dl.assign_temporal_splits(recs, 0.1, 0.1, uploader_disjoint=True)
    train_up = {r["uploader"] for r in recs if r["split"] == "train"}
    assert all(r["uploader"] not in train_up for r in recs if r["split"] != "train")


def test_hamming_distance():
    from wikicat.data_loader import _hamming_min

    d = _hamming_min(np.array([0, 2**64 - 1], dtype=np.uint64), np.array([1, 3], dtype=np.uint64))
    assert d.tolist() == [1, 62]


def test_import_table_csv(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("image_path,categories,title\nimg/a.jpg,Cats|CC-BY-4.0|Cats in Kerala,A cat\nimg/b.jpg,,empty\n")
    recs = dl.import_table(csv, image_root=tmp_path)
    assert len(recs) == 1
    assert recs[0]["categories"] == ["Cats", "Cats in Kerala"]
    assert recs[0]["image_path"] == str(tmp_path / "img/a.jpg")


def test_records_roundtrip(tmp_path):
    recs = [{"id": "1", "split": "train", "categories": ["Ä"]}, {"id": "2", "split": "test", "categories": ["B"]}]
    dl.save_records(recs, tmp_path / "r.jsonl")
    assert dl.load_records(tmp_path / "r.jsonl", "test") == [recs[1]]
    assert json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()[0])["categories"] == ["Ä"]
