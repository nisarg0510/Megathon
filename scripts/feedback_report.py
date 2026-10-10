"""Summarise tester problem reports from the web demo.

    python scripts/feedback_report.py                 # print a Markdown summary
    python scripts/feedback_report.py > feedback.md   # save it to share with the team
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedupe(reports: list[dict]) -> list[dict]:
    """Merge identical reports (e.g. the Send button clicked several times)."""
    merged: dict[str, dict] = {}
    for r in reports:
        key = json.dumps([r["shown"]["suggestions"], sorted(r["issues"]), r["comment"], r["expected"]], sort_keys=True)
        if key in merged:
            merged[key]["copies"] += 1
        else:
            merged[key] = {**r, "copies": 1}
    return list(merged.values())


def render(reports: list[dict]) -> str:
    if not reports:
        return "No reports yet."
    total = len(reports)
    reports = dedupe(reports)
    lines = [f"# Tester reports ({len(reports)} unique, {total} received)", "", "## Issues by type", ""]
    for issue, n in Counter(i for r in reports for i in r["issues"]).most_common():
        lines.append(f"- {issue}: {n}")
    lines += ["", "## Reports", ""]
    for r in reports:
        shown = r["shown"]
        types = ", ".join(f"{t['family']} {t['score']:.0%}" for t in shown["content_type"])
        top = "; ".join(f"{s['category']} ({s['score']:.0%})" for s in shown["suggestions"][:5])
        lines += [
            f"### {r['id']}" + (f" (sent {r['copies']} times)" if r["copies"] > 1 else ""),
            f"- Issues: {', '.join(r['issues']) or '-'}",
            f"- Comment: {r['comment'] or '-'}",
            f"- Expected: {r['expected'] or '-'}",
            f"- Shown content type: {types}{' (Unsure)' if shown['unsure'] else ''}",
            f"- Shown top categories: {top or '-'}",
            f"- Title given: {shown['title'] or '-'}",
            f"- Image: {r['image_file'] or 'not attached'}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", default="feedback")
    args = parser.parse_args()
    print(render(load(Path(args.reports_dir) / "reports.jsonl")))


if __name__ == "__main__":
    main()
