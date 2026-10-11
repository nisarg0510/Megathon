import json
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
from PIL import Image  # noqa: E402

import app  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import feedback_report  # noqa: E402

SHOWN = {"title": "cathedral", "description": "", "unsure": False,
         "content_type": [{"family": "Object", "score": 0.6}, {"family": "Scene", "score": 0.3},
                          {"family": "Event", "score": 0.05}, {"family": "Abstract concept", "score": 0.05}],
         "suggestions": [{"category": "Churches in Paris", "score": 0.41}], "notes": []}


def test_report_requires_a_result_and_content(tmp_path):
    assert "Run a suggestion first" in app.save_report(None, None, [], "", "", False, str(tmp_path))
    assert "Tick what is wrong" in app.save_report(SHOWN, None, [], " ", "", False, str(tmp_path))
    assert not (tmp_path / "reports.jsonl").exists()


def test_report_saved_and_summarised(tmp_path):
    img = Image.new("RGB", (2000, 1500), "gray")
    msg = app.save_report(SHOWN, img, [app.ISSUE_TYPES[0]], "Cathedral is a scene", "Scene", True, str(tmp_path))
    assert "saved" in msg
    [record] = [json.loads(l) for l in (tmp_path / "reports.jsonl").read_text().splitlines()]
    assert record["expected"] == "Scene" and record["shown"]["suggestions"][0]["category"] == "Churches in Paris"
    assert Image.open(record["image_file"]).size == (1024, 768)  # downscaled copy only
    assert "already received" in app.save_report(SHOWN, img, [app.ISSUE_TYPES[0]], "Cathedral is a scene", "Scene",
                                                  True, str(tmp_path))
    app.save_report(SHOWN, img, [], "no image please", "", False, str(tmp_path))
    reports = feedback_report.load(tmp_path / "reports.jsonl")
    assert reports[1]["image_file"] == ""
    text = feedback_report.render(reports)
    assert "Cathedral is a scene" in text and "Object 60%" in text and "2 unique, 2 received" in text


def test_unsure_banner():
    assert app.unsure_banner({"unsure": False}) == ""
    banner = app.unsure_banner({"unsure": True, "type_confidence": 0.35, "content_type": SHOWN["content_type"],
                                "type_label": "Object"})
    assert "Unsure" in banner and "35%" in banner and "human feedback is suggested" in banner
