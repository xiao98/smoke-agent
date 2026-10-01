#!/usr/bin/env python
"""FireBench runner.

  python firebench/run.py --mode official --only nbs_100a      # run NIST's own input, judge it (calibration ceiling)
  python firebench/run.py --mode agent    --only nbs_100a      # natural-language requirement -> SmokeAgent -> judge
  python firebench/run.py --mode judge    --only nbs_100a --bundle <dir>   # judge an existing bundle

Results land in firebench/runs/<date>/<mode>/<id>/ with result.json; report.py aggregates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from judge import judge, load_case  # noqa: E402


def run_official(case: dict, out: Path, fds_repo: Path, nproc: int, omp: int, timeout_s: int, t_end_override: float | None) -> Path:
    from fds import runner
    from fds.namelist import dump, parse

    src = fds_repo / case["source_fds"]
    out.mkdir(parents=True, exist_ok=True)
    text = src.read_text(errors="replace")
    if t_end_override:
        recs = parse(text)
        for r in recs:
            if r.group == "TIME":
                r.params["T_END"] = [str(t_end_override)]
        text = dump(recs)
    (out / f"{case['chid']}.fds").write_text(text, encoding="utf-8")
    res = runner.run(out, case["chid"], nproc=nproc, timeout_s=timeout_s, omp_threads=omp)
    print(f"official {case['id']}: rc={res.returncode} wall={res.wall_s:.0f}s")
    return out


def run_agent(case: dict, out: Path, omp: int, timeout_s: int, max_loop: int) -> Path:
    from config import Config
    from main import create_foam_agent_graph, initialize_state

    cfg = Config()
    cfg.case_dir = str(out)
    cfg.overwrite_case_dir = True
    cfg.max_loop = max_loop
    cfg.max_time_limit = timeout_s
    cfg.fds_omp_threads = omp
    cfg.extra_namelist_file = str(HERE / "cases" / case["id"] / "reference_devc.fds")
    app = create_foam_agent_graph().compile()
    req = case["requirement"]
    if case.get("rset_s"):
        req += f"\nRSET for the evacuation path: {case['rset_s']} s."
    state = initialize_state(req, cfg)
    t0 = time.time()
    result = app.invoke(state, config={"recursion_limit": cfg.recursion_limit})
    meta = {"workflow_status": result.get("workflow_status"), "termination": result.get("termination_reason"),
            "loops": result.get("loop_count"), "wall_s": round(time.time() - t0)}
    (out / "agent_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"agent {case['id']}: {meta}")
    return Path(result.get("case_dir") or out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["official", "agent", "judge"], required=True)
    ap.add_argument("--only", nargs="*", help="case ids")
    ap.add_argument("--group", default=None)
    ap.add_argument("--fds_repo", default=str(Path.home() / "work" / "fds"))
    ap.add_argument("--exp_repo", default=str(Path.home() / "work" / "exp"))
    ap.add_argument("--bundle", default=None, help="judge mode: bundle directory")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--omp", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--max_loop", type=int, default=5)
    ap.add_argument("--t_end", type=float, default=None, help="official mode: override T_END (smoke tests only)")
    ap.add_argument("--run_tag", default=date.today().isoformat())
    a = ap.parse_args()

    thresholds = ROOT / "configs" / "thresholds_fr_default.yaml"
    cases_dir = HERE / "cases"
    ids = a.only or sorted(p.name for p in cases_dir.iterdir() if (p / "case.yaml").is_file())
    results = []
    for cid in ids:
        case = load_case(cases_dir / cid)
        if a.group and case.get("group") != a.group:
            continue
        out = HERE / "runs" / a.run_tag / a.mode / cid
        if a.mode == "official":
            bundle = run_official(case, out, Path(a.fds_repo), a.nproc, a.omp, a.timeout, a.t_end)
        elif a.mode == "agent":
            if case.get("requirement_status") == "draft":
                print(f"skip {cid}: requirement still a draft"); continue
            bundle = run_agent(case, out, a.omp, a.timeout, a.max_loop)
        else:
            if a.bundle:
                bundle = Path(a.bundle)
            else:  # judge an existing official or agent bundle of this run tag
                cands = [HERE / "runs" / a.run_tag / m / cid for m in ("official", "agent")]
                bundle = next((c for c in cands if c.is_dir()), out)
        official = HERE / "runs" / a.run_tag / "official" / cid / "result.json"
        res = judge(bundle, case, Path(a.exp_repo), thresholds, official if official.is_file() else None)
        res["run_tag"], res["bundle"] = a.run_tag, str(bundle)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(res, indent=2))
        results.append(res)
        print(f"{cid}: passed={res['passed']} " + " | ".join(f"{g['name']}={'ok' if g['passed'] else 'FAIL'}" for g in res["gates"]))
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} cases passed all gates")


if __name__ == "__main__":
    main()
