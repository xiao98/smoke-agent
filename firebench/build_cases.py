#!/usr/bin/env python
"""Build FireBench case definitions from NIST's own validation mapping table.

Source of truth: ``Utilities/Python/FDS_validation_dataplot_inputs.csv`` in the
firemodels/fds repo.  Each row pairs one experimental column (d1_*) with one FDS
device column (d2_*) and names the metric NIST uses (max/min/mean/end).  We keep
rows whose FDS output belongs to a whitelisted building-smoke case and whose
quantity is one SmokeAgent can reproduce, then:

  firebench/cases/<chid>/case.yaml          id, source, references[], tolerance, requirement (draft)
  firebench/cases/<chid>/reference_devc.fds the official &DEVC records for those columns (injected into agent runs)

Usage: python firebench/build_cases.py --fds_repo ~/work/fds --exp_repo ~/work/exp
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from fds.namelist import Record, dump, parse, records_of  # noqa: E402

WHITELIST = ["NBS_Multi-Room", "ATF_Corridors", "Steckler_Compartment", "NIST_NRC", "UL_NIST_Vents",
             "Vettori_Flat_Ceiling", "NIST_Smoke_Alarms", "LLNL_Enclosure", "Restivo_Experiment", "Bryant_Doorway"]
QUANTITIES = {
    "HGL Temperature; Natural Ventilation": "hgl_temperature",
    "HGL Temperature; Forced Ventilation": "hgl_temperature",
    "HGL Temperature; No Ventilation": "hgl_temperature",
    "HGL Depth": "hgl_depth",
    "Ceiling Jet Temperature": "ceiling_jet_temperature",
    "Target Temperature": "target_temperature",
    "Plume Temperature": "plume_temperature",
}
# relative tolerance per quantity kind: 2 x experimental sigma from the scatterplot table, floor 15 %
TOL_FLOOR = 0.15


def load_sigma(scatter_csv: Path) -> dict[str, float]:
    out = {}
    with open(scatter_csv, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                out[r["Scatter_Plot_Title"].strip()] = float(r["Sigma_E"]) / 100.0
            except (ValueError, KeyError):
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fds_repo", required=True)
    ap.add_argument("--exp_repo", required=True)
    ap.add_argument("--out", default=str(HERE / "cases"))
    ap.add_argument("--max_refs", type=int, default=8, help="cap references per case")
    args = ap.parse_args()
    fds_repo, exp_repo, out = Path(args.fds_repo).expanduser(), Path(args.exp_repo).expanduser(), Path(args.out)
    table = fds_repo / "Utilities" / "Python" / "FDS_validation_dataplot_inputs.csv"
    sigma = load_sigma(fds_repo / "Utilities" / "Python" / "FDS_validation_scatterplot_inputs.csv")

    by_case: dict[str, list[dict]] = defaultdict(list)
    with open(table, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("switch_id", "d") != "d":
                continue
            d2 = r["d2_Filename"].strip()
            if not any(d2.startswith(w + "/") for w in WHITELIST):
                continue
            if r["Quantity"].strip() not in QUANTITIES:
                continue
            m = re.match(r"([^/]+)/(.+)_devc\.csv$", d2)
            if not m:
                continue
            dirname, chid = m.group(1), m.group(2)
            by_case[f"{dirname}/{chid}"].append(r)

    n_written = 0
    for key, rows in sorted(by_case.items()):
        dirname, chid = key.split("/", 1)
        src = fds_repo / "Validation" / dirname / "FDS_Input_Files" / f"{chid}.fds"
        if not src.is_file():
            print(f"skip {key}: no input file"); continue
        recs = parse(src.read_text(errors="replace"))
        devc_by_id = {r.get("ID"): r for r in records_of(recs, "DEVC") if r.get("ID")}
        head = records_of(recs, "HEAD")
        title = head[0].get("TITLE", "") if head else ""
        t_end = records_of(recs, "TIME")[0].floats("T_END")[0] if records_of(recs, "TIME") else None
        refs, ref_devc = [], []
        seen_devc: set[str] = set()
        for r in rows:
            if len(refs) >= args.max_refs:
                break
            exp_file = exp_repo / r["d1_Filename"].strip()
            if not exp_file.is_file():
                continue
            # NIST dataplot: '|' separates paired series, '+' sums columns within one series
            exp_parts = [c.strip() for c in r["d1_Dep_Col_Name"].split("|")]
            fds_parts = [c.strip() for c in r["d2_Dep_Col_Name"].split("|")]
            qkind = QUANTITIES[r["Quantity"].strip()]
            for exp_col, fds_col in zip(exp_parts, fds_parts):
                ids = [c.strip() for c in fds_col.split("+")]
                found = [devc_by_id[i] for i in ids if i in devc_by_id]
                refs.append({
                    "exp_file": str(Path(r["d1_Filename"].strip())),
                    "exp_col_name_row": int(float(r["d1_Col_Name_Row"] or 1)),
                    "exp_data_row": int(float(r["d1_Data_Row"] or 2)),
                    "exp_time_col": r["d1_Ind_Col_Name"].strip(),
                    "exp_col": exp_col,
                    "exp_initial": float(r["d1_Initial_Value"] or 0),
                    "exp_comp": [float(r["d1_Comp_Start"] or 0), float(r["d1_Comp_End"] or 1e9)],
                    "fds_col": fds_col,
                    "fds_initial": float(r["d2_Initial_Value"] or 0),
                    "fds_comp": [float(r["d2_Comp_Start"] or 0), float(r["d2_Comp_End"] or 1e9)],
                    "metric": r["Metric"].strip() or "max",
                    "quantity": r["Quantity"].strip(),
                    "kind": qkind,
                    "tol_rel": max(TOL_FLOOR, 2 * sigma.get(r["Quantity"].strip(), 0.0)),
                    "devc_in_source": len(found) == len(ids),
                })
                for d in found:
                    if d.get("ID") not in seen_devc:
                        seen_devc.add(d.get("ID")); ref_devc.append(d)
        if not refs:
            continue
        case_dir = out / chid.lower()
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "reference_devc.fds").write_text(dump(ref_devc), encoding="utf-8")
        case = {
            "id": chid.lower(),
            "chid": chid,
            "group": "basic",
            "source_dir": dirname,
            "source_fds": str(src.relative_to(fds_repo)),
            "title": title,
            "t_end_s": t_end,
            "requirement_status": "draft",
            "requirement": f"[DRAFT - write from the validation guide] {title}",
            "rset_s": 120,
            "references": refs,
            "gate3_min_fraction": 0.67,
            "expected_criteria_hash": None,
        }
        (case_dir / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False, allow_unicode=True), encoding="utf-8")
        n_written += 1
        print(f"{chid:32s} refs={len(refs):2d} devc_found={sum(r['devc_in_source'] for r in refs):2d} t_end={t_end}")
    print(f"wrote {n_written} cases to {out}")


if __name__ == "__main__":
    main()
