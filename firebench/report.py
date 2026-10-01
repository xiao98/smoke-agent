#!/usr/bin/env python
"""Aggregate firebench/runs/<tag>/<mode>/*/result.json into a Markdown table."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = HERE / "runs" / a.run_tag
    rows, summary = [], defaultdict(lambda: [0, 0])
    for res_file in sorted(root.glob("*/*/result.json")):
        r = json.loads(res_file.read_text())
        mode = res_file.parent.parent.name
        g = {x["name"]: x for x in r["gates"]}
        exp = g["experiment"]
        meta_f = res_file.parent / "agent_meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
        rows.append(f"| {r['case']} | {mode} | {'ok' if g['init']['passed'] else 'FAIL'} | {'ok' if g['run']['passed'] else 'FAIL'} | "
                    f"{exp['detail']} | {'ok' if g['criteria']['passed'] else 'FAIL'} | {meta.get('loops', '')} | {meta.get('wall_s', '')} | {'PASS' if r['passed'] else 'fail'} |")
        summary[mode][0] += r["passed"]; summary[mode][1] += 1
    lines = [f"# FireBench {a.run_tag}", "", "| case | mode | init | run | experiment | criteria | loops | wall s | all |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |", *rows, ""]
    for mode, (p, n) in summary.items():
        lines.append(f"- {mode}: {p}/{n} passed all gates")
    text = "\n".join(lines) + "\n"
    out = Path(a.out) if a.out else root / "report.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
