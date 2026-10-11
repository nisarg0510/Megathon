"""Web demo: upload an image, get category suggestions with evidence, give feedback.

    python app.py                          # http://127.0.0.1:7860
    python app.py --share --read-only      # public link for anyone; editing panels hidden

Problem reports from testers are saved to feedback/reports.jsonl;
summarise them with: python scripts/feedback_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from predict import explain
from wikicat.pipeline import CategorySuggester

SUGGESTER: CategorySuggester | None = None
ARTIFACTS = "artifacts"
REPORTS_DIR = "feedback"
READ_ONLY = False
_SENT_REPORTS: set[str] = set()

ISSUE_TYPES = [
    "Wrong content type (Object / Scene / Event / Abstract)",
    "Wrong or irrelevant categories",
    "A correct category is missing",
    "Marked Unsure but it is obvious (or the reverse)",
    "Slow, error or display problem",
    "Other",
]


def _thumb(ev: dict):
    path = ev.get("image_path")
    if path and Path(path).exists():
        return path
    return ev.get("thumb_url") or None


def unsure_banner(result: dict) -> str:
    if not result.get("unsure"):
        return ""
    best = result["type_label"]
    return ("<div style='padding:12px 16px;border-radius:8px;background:#fff4e5;color:#7a4100;"
            "border:1px solid #f0b46c;font-size:1.15em;font-weight:600'>"
            f"⚠ Unsure: the content type is only {result['type_confidence']:.0%} certain "
            f"(best guess: {best}). Treat the suggestions below as guesses.</div>")


def summarise(result: dict, title: str, description: str) -> dict:
    """What the tester saw, stored with any problem report."""
    return {
        "title": title or "",
        "description": description or "",
        "unsure": result["unsure"],
        "content_type": result["content_type"],
        "suggestions": [{"category": s["category"], "score": round(s["score"], 3)} for s in result["suggestions"]],
        "suggestion_set": result.get("suggestion_set"),
        "notes": result["notes"],
    }


def run(image, title, description, top_k, use_hierarchy):
    if image is None:
        return "", gr.Label(value=None, label="Content type"), [], [], "Upload an image first.", \
            gr.CheckboxGroup(choices=[], value=[]), None, None
    result = SUGGESTER.suggest(image, title or "", description or "", int(top_k), use_hierarchy=use_hierarchy)
    rows, gallery, seen = [], [], set()
    top, *rest = result["content_type"]
    others = ", ".join(f"{f['family']} {f['score']:.0%}" for f in rest)
    if result["unsure"]:
        label = f"Unsure (best guess: {result['type_label']})"
        others = f"content type only {result['type_confidence']:.0%} sure; other types: {others}"
    else:
        label = f"Content type: {result['type_label']}"
        others = f"other types: {others}"
    rows.append(["★", label, f"{top['score']:.0%}", others, ""])
    for rank, s in enumerate(result["suggestions"], 1):
        note = f"broader than '{s['suppressed_by']}'" if s["suppressed_by"] else ""
        mark = f"{rank} ✓" if s.get("in_set") else str(rank)
        rows.append([mark, s["category"], f"{s['score']:.0%}", explain(s), note])
        for ev in s["evidence"]:
            thumb = _thumb(ev)
            if thumb and ev["title"] not in seen:
                seen.add(ev["title"])
                gallery.append((thumb, f"{ev['title']} (cos {ev['similarity']:.2f}) -> {s['category']}"))
    q = result["quality"]
    info = [f"**Image:** {q['width']}x{q['height']} px, sharpness {q['sharpness']}"
            + (f" — flagged: {', '.join(q['flags'])}" if q["flags"] else "")]
    info += [f"- {n}" for n in result["notes"]]
    t = result["timing_ms"]
    info.append(f"\n*Encode {t['encode_image']} ms · score and rank {t['score_rank']} ms*")
    choices = [s["category"] for s in result["suggestions"]]
    box_title = (f"Content type: UNSURE (only {result['type_confidence']:.0%} sure)" if result["unsure"]
                 else f"Content type: {result['type_label']} ({result['type_confidence']:.0%} sure)")
    content_type = gr.Label(value={f["family"]: f["score"] for f in result["content_type"]}, label=box_title)
    return (unsure_banner(result), content_type, rows, gallery[:12], "\n".join(info),
            gr.CheckboxGroup(choices=choices, value=[]), image, summarise(result, title, description))


def feedback(image, title, accepted, extra, all_rows):
    if READ_ONLY:
        return "Editing is disabled on this public demo."
    if image is None:
        return "Run a suggestion first."
    extra_cats = [c.strip() for c in (extra or "").split("\n") if c.strip()]
    accepted = list(accepted or []) + extra_cats
    shown = [row[1] for row in (all_rows.values.tolist() if hasattr(all_rows, "values") else all_rows or [])
             if str(row[0]) != "★"]
    rejected = [c for c in shown if c not in accepted]
    if not accepted and not rejected:
        return "Nothing to record."
    SUGGESTER.record_feedback(image, accepted, rejected, title or "", ARTIFACTS)
    SUGGESTER.save(ARTIFACTS)
    return (f"Indexed this image with {len(accepted)} accepted categories; {len(rejected)} rejected "
            f"categories stored as hard negatives. Index now holds {len(SUGGESTER.knn)} images.")


def add_category(text):
    if READ_ONLY:
        return "Editing is disabled on this public demo."
    names = [c.strip() for c in (text or "").split("\n") if c.strip()]
    if not names:
        return "Enter one category per line."
    SUGGESTER.add_categories(names)
    SUGGESTER.save(ARTIFACTS)
    return f"Added {len(names)} categories; the vocabulary now has {len(SUGGESTER.vocab)}. They can be suggested right away."


def save_report(shown: dict | None, image, issues, comment, expected, attach_image, reports_dir: str | None = None):
    """Store a tester's problem report with what CommonLens showed. Never changes the model."""
    if shown is None:
        return "Run a suggestion first, then report what is wrong with it."
    comment, expected = (comment or "").strip(), (expected or "").strip()
    if not issues and not comment and not expected:
        return "Tick what is wrong or write a short description first."
    key = json.dumps([shown.get("suggestions"), sorted(issues or []), comment, expected], sort_keys=True)
    if key in _SENT_REPORTS:
        return "This report was already received — thank you!"
    _SENT_REPORTS.add(key)
    out = Path(reports_dir or REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    image_file = ""
    if attach_image and image is not None:
        (out / "images").mkdir(exist_ok=True)
        img = image.convert("RGB")
        img.thumbnail((1024, 1024))
        image_file = str(out / "images" / f"{report_id}.jpg")
        img.save(image_file, quality=85)
    record = {"id": report_id, "time": datetime.now(timezone.utc).isoformat(), "issues": list(issues or []),
              "comment": comment, "expected": expected, "image_file": image_file, "shown": shown}
    with (out / "reports.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Thank you! Report {report_id} saved. It will be reviewed and used to improve CommonLens."


def build_ui(read_only: bool = False) -> gr.Blocks:
    with gr.Blocks(title="CommonLens") as demo:
        gr.Markdown("## CommonLens — category suggestions for Wikimedia Commons\nSuggestions are for human review, "
                    "with the similar indexed images that support each one.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image")
                title = gr.Textbox(label="Title (optional)", placeholder="e.g. Diwali lights in Jaipur 2023")
                description = gr.Textbox(label="Description (optional)", lines=2)
                with gr.Row():
                    top_k = gr.Slider(1, 25, value=10, step=1, label="Suggestions")
                    use_hierarchy = gr.Checkbox(value=True, label="Prefer specific categories")
                go = gr.Button("Suggest categories", variant="primary")
            with gr.Column(scale=2):
                banner = gr.HTML()
                content_type = gr.Label(label="Content type", num_top_classes=5)
                info = gr.Markdown()
                table = gr.Dataframe(headers=["#", "category", "score", "why", "note"],
                                     datatype=["str", "str", "str", "str", "str"], interactive=False, wrap=True,
                                     column_widths=["5%", "33%", "9%", "28%", "25%"])
                gallery = gr.Gallery(label="Similar categorised images (evidence)", columns=4, height=300)
        with gr.Accordion("Review: accept or reject suggestions (updates the index live)", open=False,
                          visible=not read_only):
            accept = gr.CheckboxGroup(label="Accept these categories (unticked ones are recorded as rejected)")
            extra = gr.Textbox(label="Other correct categories (one per line)", lines=2)
            send = gr.Button("Save review")
            feedback_msg = gr.Markdown()
        with gr.Accordion("Add new categories (no retraining)", open=False, visible=not read_only):
            new_cats = gr.Textbox(label="Category names, one per line", lines=3)
            add_btn = gr.Button("Add categories")
            add_msg = gr.Markdown()
        with gr.Accordion("Report a problem with this result", open=True):
            gr.Markdown("Something wrong? Tell us. Reports are reviewed and used to improve CommonLens.")
            issues = gr.CheckboxGroup(ISSUE_TYPES, label="What is wrong?")
            comment = gr.Textbox(label="Describe the problem", lines=2,
                                 placeholder="e.g. A photo of a cathedral was labelled Object instead of Scene")
            expected = gr.Textbox(label="What should it say? (optional)", placeholder="e.g. Scene; Churches in Paris")
            attach = gr.Checkbox(value=False, label="Attach my image to the report (leave unticked for private photos)")
            report_btn = gr.Button("Send report")
            report_msg = gr.Markdown()
        last_image = gr.State()
        last_shown = gr.State()

        go.click(run, [image, title, description, top_k, use_hierarchy],
                 [banner, content_type, table, gallery, info, accept, last_image, last_shown])
        send.click(feedback, [last_image, title, accept, extra, table], feedback_msg)
        add_btn.click(add_category, new_cats, add_msg)
        report_btn.click(save_report, [last_shown, last_image, issues, comment, expected, attach], report_msg)
    return demo


def main() -> None:
    global SUGGESTER, ARTIFACTS, READ_ONLY, REPORTS_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="feedback", help="Where tester problem reports are saved")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    # Hosted platforms (e.g. Hugging Face Spaces) need the app to listen on all interfaces.
    parser.add_argument("--host", default="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    parser.add_argument("--read-only", action="store_true", default=bool(os.environ.get("SPACE_ID")),
                        help="Hide the review / add-category panels (use for public links)")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--show-specific", dest="hide_specific", action="store_false", default=None,
                        help="Also show dated / people-count / non-Latin categories below 70%%")
    parser.add_argument("--center-focus", type=float, default=None,
                        help="Weight of the central crop, 0-1 (default 0.4; 0 = whole image only)")
    parser.add_argument("--unsure-threshold", type=float, default=None,
                        help="Mark the image Unsure when the content-type score is below this (default 0.5)")
    parser.add_argument("--alpha", dest="conformal_alpha", type=float, default=None,
                        help="Suggestion set error rate: the set holds a correct category with probability >= 1 - alpha")
    args = parser.parse_args()
    ARTIFACTS, REPORTS_DIR = args.artifacts_dir, args.reports_dir
    SUGGESTER = CategorySuggester.load(args.artifacts_dir, device=args.device, unsure_threshold=args.unsure_threshold,
                                       center_focus=args.center_focus, hide_specific=args.hide_specific,
                                       conformal_alpha=args.conformal_alpha)
    _ = SUGGESTER.encoder  # load weights before the first request
    READ_ONLY = args.read_only
    build_ui(READ_ONLY).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
