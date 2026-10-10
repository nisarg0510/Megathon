"""Render a deliverable HTML source to PDF with headless Chromium (Playwright).

    python deliverables/src/render.py M1 --team 42
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("name", help="M1, M2, final_report")
parser.add_argument("--team", required=True, help="Team number printed at the top")
parser.add_argument("--chromium", default=None, help="Path to a Chromium binary (optional)")
args = parser.parse_args()

src = Path(__file__).parent / f"{args.name}.html"
html = src.read_text(encoding="utf-8").replace("TEAM_NO", args.team)
out = Path(__file__).parent.parent / f"{args.name}.pdf"
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
    page = browser.new_page()
    page.set_content(html, wait_until="load")
    page.pdf(path=str(out), format="A4", print_background=True, prefer_css_page_size=True)
    browser.close()
print(f"Wrote {out}")
