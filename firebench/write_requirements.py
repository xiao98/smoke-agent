#!/usr/bin/env python
"""Draft natural-language requirements for FireBench cases with the LLM, for human review.

The agent will later rebuild each case from this text alone (plus the injected
reference devices), so the text must contain everything SmokeAgent's ScenarioSpec
can express: domain, rectangular walls/doors, fire footprint + HRR + growth,
openings, exhaust, escape path, mesh cell size, duration.  Material details the
spec cannot express are omitted on purpose.

Usage: python firebench/write_requirements.py --only nbs_100a nbs_100o [--lang fr]
Writes requirement into case.yaml and sets requirement_status: llm_draft.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

SYSTEM = """You write the experiment description a fire safety engineer would give to an assistant that builds FDS smoke
simulations. The assistant can only model: a box domain; rectangular obstructions (walls, slabs, furniture blocks)
with rectangular holes (doors, windows); a rectangular burner footprint with peak HRR and t-squared growth (slow/medium/
fast/ultrafast/constant); permanently open vents on the domain boundary; mechanical exhaust vents with a volume flow;
evacuation paths as polylines; a uniform mesh cell size; a duration. Describe the official FDS input in those terms:
give every coordinate explicitly in metres (x0..x1, y0..y1, z0..z1), the fire HRR in kW and its growth, and when the
official input defines the burner with a RAMP (RAMP_Q or a time table), transcribe that curve as absolute HRR values at
each ramp time (e.g. "0 s: 0 kW, 20 s: 350 kW, 100 s: 1055 kW"), the ambient
openings, the mesh cell size that the official input uses, and the simulation duration. Do not mention material
properties, radiation settings or devices. Finish with one sentence naming a plausible evacuation path across the
space at 1.8 m height (two or three points). Write in {lang}, 120-220 words, no headings."""


class Req(BaseModel):
    requirement: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--lang", default="French")
    ap.add_argument("--fds_repo", default=str(Path.home() / "work" / "fds"))
    a = ap.parse_args()
    from config import Config
    from utils import LLMService

    llm = LLMService(Config())
    cases_dir = HERE / "cases"
    ids = a.only or sorted(p.name for p in cases_dir.iterdir() if (p / "case.yaml").is_file())
    for cid in ids:
        cpath = cases_dir / cid / "case.yaml"
        case = yaml.safe_load(cpath.read_text(encoding="utf-8"))
        if case.get("requirement_status") == "reviewed":
            continue
        src = Path(a.fds_repo) / case["source_fds"]
        text = src.read_text(errors="replace")[:12000]
        out = llm.invoke(f"Official FDS input ({case['chid']}):\n{text}\n\nWrite the requirement.",
                         SYSTEM.format(lang=a.lang), pydantic_obj=Req)
        req = out.requirement if hasattr(out, "requirement") else Req.model_validate(out).requirement
        case["requirement"] = req.strip()
        case["requirement_status"] = "llm_draft"
        cpath.write_text(yaml.safe_dump(case, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"=== {cid}\n{req}\n")


if __name__ == "__main__":
    main()
