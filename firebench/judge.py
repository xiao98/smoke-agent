"""FireBench judge: four gates on a finished bundle directory."""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from fds import criteria as C  # noqa: E402
from fds.spec import ScenarioSpec  # noqa: E402


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""
    items: list[dict] = field(default_factory=list)


def load_case(case_dir: Path) -> dict:
    return yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- csv helpers

def read_table(path: Path, col_name_row: int = 1, data_row: int = 2) -> tuple[list[str], list[list[float]]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    names = [c.strip() for c in rows[col_name_row - 1]]
    data = []
    for r in rows[data_row - 1:]:
        if len(r) < len(names):
            continue
        try:
            data.append([float(x) if x.strip() not in ("", "NaN", "nan") else float("nan") for x in r[: len(names)]])
        except ValueError:
            continue
    return names, data


def series(path: Path, time_col: str, col: str, col_name_row: int = 1, data_row: int = 2) -> tuple[list[float], list[float]]:
    names, data = read_table(path, col_name_row, data_row)
    ti, ci = names.index(time_col), names.index(col)
    t = [r[ti] for r in data]
    v = [r[ci] for r in data]
    return t, v


def metric(t: list[float], v: list[float], kind: str, comp: list[float], initial: float) -> float:
    sel = [(x, y) for x, y in zip(t, v) if comp[0] <= x <= comp[1] and y == y]
    if not sel:
        return float("nan")
    ys = [y - initial for _, y in sel]
    if kind == "max":
        return max(ys)
    if kind == "min":
        return min(ys)
    if kind == "maxabs":
        return max(abs(y) for y in ys)
    if kind == "mean":
        return sum(ys) / len(ys)
    if kind == "end":
        return ys[-1]
    return max(ys)


# ---------------------------------------------------------------- gates

def gate1_init(bundle: Path, chid: str) -> GateResult:
    """The case produced an .smv (FDS got past setup) and the .fds exists."""
    ok = (bundle / f"{chid}.fds").is_file() and (bundle / f"{chid}.smv").is_file()
    return GateResult("init", ok, "fds+smv present" if ok else "missing .fds or .smv")


def gate2_run(bundle: Path, chid: str, spec: ScenarioSpec | None) -> GateResult:
    out = bundle / f"{chid}.out"
    if not out.is_file():
        return GateResult("run", False, "no .out")
    text = out.read_text(errors="replace")
    errs = re.findall(r"^\s*ERROR.*$", text, re.M)
    done = "STOP: FDS completed successfully" in text
    detail = f"completed={done} errors={len(errs)}"
    ok = done and not errs
    hrr_csv = bundle / f"{chid}_hrr.csv"
    if ok and spec is not None and hrr_csv.is_file():
        peak = C.read_hrr_peak(hrr_csv)
        exp = C.expected_hrr_peak(spec, spec.sim.t_end_s)
        tol = C.hrr_tolerance(spec, spec.sim.t_end_s)
        ok = abs(peak - exp) <= tol * exp
        detail += f" hrr_sim={peak:.0f} hrr_expected={exp:.0f} tol={tol:.0%}"
    return GateResult("run", ok, detail)


def gate3_experiment(bundle: Path, chid: str, case: dict, exp_repo: Path) -> GateResult:
    devc = bundle / f"{chid}_devc.csv"
    if not devc.is_file():
        return GateResult("experiment", False, "no _devc.csv")
    items, n_ok = [], 0
    for ref in case["references"]:
        try:
            te, ve = series(exp_repo / ref["exp_file"], ref["exp_time_col"], ref["exp_col"], ref["exp_col_name_row"], ref["exp_data_row"])
            tf, vf = series(devc, "Time", ref["fds_col"], 2, 3)
        except (ValueError, FileNotFoundError, IndexError) as exc:
            items.append({"fds_col": ref["fds_col"], "ok": False, "why": f"column missing: {exc}"[:120]})
            continue
        me = metric(te, ve, ref["metric"], ref["exp_comp"], ref["exp_initial"])
        mf = metric(tf, vf, ref["metric"], ref["fds_comp"], ref["fds_initial"])
        if me != me or mf != mf or me == 0:
            items.append({"fds_col": ref["fds_col"], "ok": False, "why": "nan or zero reference"})
            continue
        rel = abs(mf - me) / abs(me)
        ok = rel <= ref["tol_rel"]
        n_ok += ok
        items.append({"fds_col": ref["fds_col"], "quantity": ref["kind"], "metric": ref["metric"], "exp": round(me, 2),
                      "fds": round(mf, 2), "rel_err": round(rel, 3), "tol": ref["tol_rel"], "ok": ok})
    frac = n_ok / len(items) if items else 0.0
    passed = bool(items) and frac >= float(case.get("gate3_min_fraction", 0.67))
    return GateResult("experiment", passed, f"{n_ok}/{len(items)} references within tolerance", items)


def gate4_criteria(bundle: Path, spec: ScenarioSpec | None, thresholds: Path) -> GateResult:
    if spec is None:
        return GateResult("criteria", True, "official input: no spec, gate skipped")
    try:
        res = C.evaluate(bundle, spec, C.load_thresholds(thresholds))
    except Exception as exc:  # noqa: BLE001
        return GateResult("criteria", False, f"evaluate failed: {exc}"[:200])
    stored = bundle / "criteria.json"
    if stored.is_file():
        same = json.loads(stored.read_text())["paths"] == res.model_dump(mode="json")["paths"]
        return GateResult("criteria", same, "recomputed ASET matches stored criteria.json" if same else "criteria.json differs from recomputation")
    return GateResult("criteria", True, "computed (no stored criteria.json to compare)")


def judge(bundle: Path, case: dict, exp_repo: Path, thresholds: Path) -> dict:
    chid = case["chid"] if (bundle / f"{case['chid']}.fds").is_file() else next((p.stem for p in bundle.glob("*.fds")), case["chid"])
    spec = None
    if (bundle / "spec.json").is_file():
        spec = ScenarioSpec.model_validate_json((bundle / "spec.json").read_text())
        chid = spec.chid
    gates = [gate1_init(bundle, chid), gate2_run(bundle, chid, spec),
             gate3_experiment(bundle, chid, case, exp_repo), gate4_criteria(bundle, spec, thresholds)]
    return {"case": case["id"], "chid": chid, "mode": "agent" if spec else "official",
            "passed": all(g.passed for g in gates),
            "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail, "items": g.items} for g in gates]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle"); ap.add_argument("case_dir")
    ap.add_argument("--exp_repo", default=str(Path.home() / "work" / "exp"))
    ap.add_argument("--thresholds", default=str(HERE.parent / "configs" / "thresholds_fr_default.yaml"))
    a = ap.parse_args()
    res = judge(Path(a.bundle), load_case(Path(a.case_dir)), Path(a.exp_repo), Path(a.thresholds))
    print(json.dumps(res, indent=2))
