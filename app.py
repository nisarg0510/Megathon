"""Web demo: upload an image, get category suggestions with evidence, give feedback.

    python app.py                      # http://127.0.0.1:7860
    python app.py --port 8080 --share  # temporary public link
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from predict import explain
from wikicat.pipeline import CategorySuggester

SUGGESTER: CategorySuggester | None = None
ARTIFACTS = "artifacts"


def _thumb(ev: dict):
    path = ev.get("image_path")
    if path and Path(path).exists():
        return path
    return ev.get("thumb_url") or None


def run(image, title, description, top_k, use_hierarchy):
    if image is None:
        return gr.Label(value=None, label="Content type"), [], [], "Upload an image first.", \
            gr.CheckboxGroup(choices=[], value=[]), None
    result = SUGGESTER.suggest(image, title or "", description or "", int(top_k), use_hierarchy=use_hierarchy)
    rows, gallery, seen = [], [], set()
    top, *rest = result["content_type"]
    others = ", ".join(f"{f['family']} {f['score']:.0%}" for f in rest)
    if result["unsure"]:
        label = f"Unsure (best guess: {top['family']})"
        others = f"content type only {result['type_confidence']:.0%} sure; other types: {others}"
    else:
        label = f"Content type: {top['family']}"
        others = f"other types: {others}"
    rows.append(["★", label, f"{top['score']:.0%}", others, ""])
    for rank, s in enumerate(result["suggestions"], 1):
        note = f"broader than '{s['suppressed_by']}'" if s["suppressed_by"] else ""
        rows.append([str(rank), s["category"], f"{s['score']:.0%}", explain(s), note])
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
    title = (f"Content type: UNSURE (only {result['type_confidence']:.0%} sure)" if result["unsure"]
             else f"Content type ({result['type_confidence']:.0%} sure)")
    content_type = gr.Label(value={f["family"]: f["score"] for f in result["content_type"]}, label=title)
    return content_type, rows, gallery[:12], "\n".join(info), gr.CheckboxGroup(choices=choices, value=[]), image


def feedback(image, title, accepted, extra, all_rows):
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
    names = [c.strip() for c in (text or "").split("\n") if c.strip()]
    if not names:
        return "Enter one category per line."
    SUGGESTER.add_categories(names)
    SUGGESTER.save(ARTIFACTS)
    return f"Added {len(names)} categories; the vocabulary now has {len(SUGGESTER.vocab)}. They can be suggested right away."


def build_ui() -> gr.Blocks:
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
                content_type = gr.Label(label="Content type", num_top_classes=4)
                info = gr.Markdown()
                table = gr.Dataframe(headers=["#", "category", "score", "why", "note"], datatype=["str", "str", "str", "str", "str"],
                                     interactive=False, wrap=True,
                                     column_widths=["5%", "33%", "9%", "28%", "25%"])
                gallery = gr.Gallery(label="Similar categorised images (evidence)", columns=4, height=300)
        with gr.Accordion("Review: accept or reject suggestions (updates the index live)", open=False):
            accept = gr.CheckboxGroup(label="Accept these categories (unticked ones are recorded as rejected)")
            extra = gr.Textbox(label="Other correct categories (one per line)", lines=2)
            send = gr.Button("Save review")
            feedback_msg = gr.Markdown()
        with gr.Accordion("Add new categories (no retraining)", open=False):
            new_cats = gr.Textbox(label="Category names, one per line", lines=3)
            add_btn = gr.Button("Add categories")
            add_msg = gr.Markdown()
        last_image = gr.State()

        go.click(run, [image, title, description, top_k, use_hierarchy], [content_type, table, gallery, info, accept, last_image])
        send.click(feedback, [last_image, title, accept, extra, table], feedback_msg)
        add_btn.click(add_category, new_cats, add_msg)
    return demo


def main() -> None:
    global SUGGESTER, ARTIFACTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--center-focus", type=float, default=None,
                        help="Weight of the central crop, 0-1 (default 0.4; 0 = whole image only)")
    parser.add_argument("--unsure-threshold", type=float, default=None,
                        help="Mark the image Unsure when the content-type score is below this (default 0.5)")
    args = parser.parse_args()
    ARTIFACTS = args.artifacts_dir
    SUGGESTER = CategorySuggester.load(args.artifacts_dir, device=args.device, unsure_threshold=args.unsure_threshold,
                                       center_focus=args.center_focus)
    _ = SUGGESTER.encoder  # load weights before the first request
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
