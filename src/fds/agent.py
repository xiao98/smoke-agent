"""FDS implementations of the Foam-Agent node bodies.

Each node in ``src/nodes`` keeps a one-line branch::

    if is_fds(state["config"]): return fds.agent.<fn>(state)

State keys used (all already in GraphState or added by SmokeAgent):
  user_requirement, config, llm_service, case_dir, case_name,
  scenario_spec (dict), error_logs (list[str]), fds_fingerprint, loop_count,
  error_fingerprints, review_analysis, rewrite_plan, input_writer_mode,
  workflow_status, termination_reason, criteria_result, report_path
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from fds import criteria as C
from fds import outparse, runner, validate, writer
from fds.spec import ScenarioSpec
from fds.target import DESCRIPTION

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _spec_from_state(state: dict) -> ScenarioSpec:
    return ScenarioSpec.model_validate(state["scenario_spec"])


def _thresholds_path(config: Any) -> Path:
    p = getattr(config, "thresholds_path", "") or ""
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent.parent / "configs" / "thresholds_fr_default.yaml"


def _retrieve_reference(user_requirement: str, config: Any, max_chars: int = 7000) -> tuple[str, str, str]:
    """Return (case_name, category, reference_text) from the FDS corpus."""
    from utils import retrieve_faiss  # lazy: utils pulls the whole LLM stack

    try:
        hits = retrieve_faiss("openfoam_tutorials_details", user_requirement, topk=3, config=config)
    except Exception as exc:  # corpus missing or empty
        print(f"<fds_plan>retrieval failed: {exc}</fds_plan>")
        return "", "", ""
    best = hits[0]
    text = str(best.get("tutorials") or best.get("full_content") or "")
    names = ", ".join(str(h.get("case_name")) for h in hits)
    print(f"<fds_plan>similar validation cases: {names}</fds_plan>")
    return str(best.get("case_name", "")), str(best.get("case_category", "")), text[:max_chars]


def apply_patch(obj: Any, patch: list[dict]) -> Any:
    """Minimal RFC 6902: add / replace / remove with JSON-pointer paths."""
    import copy

    doc = copy.deepcopy(obj)
    for op in patch:
        kind = op.get("op")
        parts = [p.replace("~1", "/").replace("~0", "~") for p in op.get("path", "").strip("/").split("/") if p != ""]
        if not parts:
            raise ValueError(f"empty path in patch op {op}")
        parent = doc
        for p in parts[:-1]:
            parent = parent[int(p)] if isinstance(parent, list) else parent[p]
        last = parts[-1]
        if isinstance(parent, list):
            idx = len(parent) if last == "-" else int(last)
            if kind == "add":
                parent.insert(idx, op["value"])
            elif kind == "replace":
                parent[idx] = op["value"]
            elif kind == "remove":
                del parent[idx]
            else:
                raise ValueError(f"unsupported op {kind}")
        else:
            if kind in ("add", "replace"):
                parent[last] = op["value"]
            elif kind == "remove":
                parent.pop(last, None)
            else:
                raise ValueError(f"unsupported op {kind}")
    return doc


# ----------------------------------------------------------------------------
# planner
# ----------------------------------------------------------------------------

PLAN_SYSTEM = """You are a fire safety engineer preparing an FDS (Fire Dynamics Simulator) smoke-control study.
Translate the user's request into a ScenarioSpec JSON. Rules:
- Units: metres, seconds, kW, degrees C. XB = [x0,x1,y0,y1,z0,z1].
- The domain must enclose every obstruction, the fire footprint and all escape-path points.
- The fire footprint is a horizontal rectangle (z0 == z1) on the floor or on top of an obstruction. Keep HRRPUA = peak/area between 250 and 2500 kW/m2.
- Walls are thin obstructions (0.2 m); doors are holes through them (height ~2.0 m).
- Openings to the outside are planar vents on the domain boundary.
- Mechanical exhaust: planar vent on the ceiling (z0 == z1 == domain z max) with volume flow in m3/s; make-up air refers to an opening id.
- Escape paths: polylines at 1.8 m height along the route occupants take, with rset_s if the user gives one.
- mesh.cell_size: "auto" unless the user specifies it. sim.t_end_s: 600 unless specified.
- chid: short lowercase identifier with underscores.
- Every numeric choice you make that the user did not state must be a conventional engineering default; list them in `assumptions`.
Return ONLY JSON matching the schema."""


class PlanOut(BaseModel):
    spec: ScenarioSpec
    assumptions: list[str] = Field(default_factory=list)


def plan(state: dict) -> dict:
    config = state["config"]
    req = state["user_requirement"]
    llm = state["llm_service"]
    ref_name, ref_cat, ref_text = _retrieve_reference(req, config)
    desc = json.dumps(DESCRIPTION, indent=1)
    base_prompt = (
        f"User requirement:\n{req}\n\n"
        f"Solver description:\n{desc}\n\n"
        + (f"A similar NIST validation case ({ref_name}, {ref_cat}) for reference only; do not copy its geometry:\n{ref_text}\n\n" if ref_text else "")
        + "Produce the ScenarioSpec."
    )
    last_err = ""
    out: Optional[PlanOut] = None
    for attempt in range(3):
        prompt = base_prompt + (f"\n\nYour previous answer failed validation:\n{last_err}\nFix it." if last_err else "")
        try:
            out = llm.invoke(prompt, PLAN_SYSTEM, pydantic_obj=PlanOut)
            if isinstance(out, dict):
                out = PlanOut.model_validate(out)
            break
        except (ValidationError, ValueError) as exc:
            last_err = str(exc)[:2000]
            print(f"<fds_plan attempt={attempt + 1}>validation error: {last_err[:300]}</fds_plan>")
    if out is None:
        return {"workflow_status": "failed", "termination_reason": "spec_validation_failed", "error_logs": [last_err]}

    spec = out.spec
    run_dir = Path(getattr(config, "run_directory", "") or "runs")
    case_dir = Path(getattr(config, "case_dir", "") or (run_dir / spec.chid))
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (case_dir / "assumptions.json").write_text(json.dumps(out.assumptions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"<fds_plan>chid={spec.chid} case_dir={case_dir} assumptions={len(out.assumptions)}</fds_plan>")
    case_info = f"case name: {spec.chid}\ncase domain: fire\ncase category: {ref_cat or 'building'}\ncase solver: fds"
    return {
        "case_name": spec.chid,
        "case_domain": "fire",
        "case_category": ref_cat or "building",
        "case_solver": "fds",
        "case_dir": str(case_dir),
        "case_info": case_info,
        "tutorial_reference": ref_text,
        "case_path_reference": str(case_dir / "reference.txt"),
        "dir_structure_reference": f"{spec.chid}.fds",
        "allrun_reference": f"mpiexec -n <n_mesh> fds {spec.chid}.fds",
        "subtasks": [],
        "similar_case_advice": None,
        "mesh_type": "standard_mesh",
        "requires_hpc": False,
        "requires_visualization": False,
        "scenario_spec": spec.model_dump(mode="json"),
        "input_writer_mode": "initial",
    }


# ----------------------------------------------------------------------------
# input writer (initial + rewrite)
# ----------------------------------------------------------------------------

def _write_and_validate(spec: ScenarioSpec, case_dir: Path, run_solver: bool = True) -> tuple[str, list[str]]:
    text = writer.write(spec)
    (case_dir / f"{spec.chid}.fds").write_text(text, encoding="utf-8")
    (case_dir / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    rep = validate.validate(spec, text, run_solver=run_solver)
    for w in rep.warnings:
        print(f"<fds_validate warning>{w}</fds_validate>")
    return text, [f"[validate] {e}" for e in rep.errors]


def write_initial(state: dict) -> dict:
    spec = _spec_from_state(state)
    case_dir = Path(state["case_dir"])
    text, errs = _write_and_validate(spec, case_dir)
    print(f"<fds_writer>wrote {spec.chid}.fds ({len(text.splitlines())} lines), validate errors={len(errs)}</fds_writer>")
    return {
        "dir_structure": {".": [f"{spec.chid}.fds"]},
        "foamfiles": {f"{spec.chid}.fds": text},
        "commands": [],
        "error_logs": errs,
        "fds_fingerprint": "validate:" + ";".join(errs)[:200] if errs else "",
    }


def rewrite(state: dict) -> dict:
    plan_ = state.get("rewrite_plan") or {}
    patch = plan_.get("patch") or []
    spec_dict = state["scenario_spec"]
    if patch:
        try:
            spec_dict = apply_patch(spec_dict, patch)
            spec = ScenarioSpec.model_validate(spec_dict)
        except (ValidationError, ValueError, KeyError, IndexError) as exc:
            msg = f"[patch] reviewer patch could not be applied: {str(exc)[:500]}"
            print(f"<fds_writer>{msg}</fds_writer>")
            return {"error_logs": [msg], "fds_fingerprint": "patch_failed"}
    else:
        spec = ScenarioSpec.model_validate(spec_dict)
    case_dir = Path(state["case_dir"])
    text, errs = _write_and_validate(spec, case_dir)
    print(f"<fds_writer mode=rewrite>applied {len(patch)} patch ops, validate errors={len(errs)}</fds_writer>")
    return {
        "scenario_spec": spec.model_dump(mode="json"),
        "dir_structure": {".": [f"{spec.chid}.fds"]},
        "foamfiles": {f"{spec.chid}.fds": text},
        "error_logs": errs,
        "fds_fingerprint": "validate:" + ";".join(errs)[:200] if errs else "",
    }


def write(state: dict) -> dict:
    if state.get("input_writer_mode") == "rewrite":
        return rewrite(state)
    return write_initial(state)


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------

def run(state: dict) -> dict:
    spec = _spec_from_state(state)
    case_dir = Path(state["case_dir"])
    config = state["config"]
    res = runner.run(case_dir, spec.chid, timeout_s=int(getattr(config, "max_time_limit", 3600)),
                     omp_threads=int(getattr(config, "fds_omp_threads", 1)))
    errs = outparse.errors(res.out_tail, res.stdout_tail, res.timed_out, res.returncode)
    logs = outparse.to_error_logs(errs)
    hrr_csv = case_dir / f"{spec.chid}_hrr.csv"
    if not errs and hrr_csv.exists():
        peak = C.read_hrr_peak(hrr_csv)
        expected = C.expected_hrr_peak(spec, spec.sim.t_end_s)
        if abs(peak - expected) > C.hrr_tolerance(spec, spec.sim.t_end_s) * expected:
            logs.append(f"[hrr_mismatch] simulated peak HRR {peak:.0f} kW vs expected {expected:.0f} kW by t_end\n"
                        "  cause: fire surface obstructed, under-ventilated, or ramp not reaching peak before t_end\n"
                        "  fix: check burner placement, make-up air and sim.t_end_s vs time to peak")
            errs.append(outparse.FdsError("hrr_mismatch", logs[-1], "", ""))
    print(f"<fds_runner>rc={res.returncode} wall={res.wall_s:.0f}s errors={len(logs)}</fds_runner>")
    out = {"error_logs": logs, "fds_fingerprint": outparse.fingerprint(errs)}
    if not logs:
        out["workflow_status"] = "success"
        out["termination_reason"] = None
    return out


# ----------------------------------------------------------------------------
# reviewer
# ----------------------------------------------------------------------------

REVIEW_SYSTEM = """You are debugging an FDS smoke simulation that failed. You will see the ScenarioSpec JSON that generated the
input file, the structured error list, and earlier attempts. Decide the smallest change to the ScenarioSpec that fixes the
root cause, and express it as RFC 6902 JSON patch operations (op: add|replace|remove, path: JSON pointer, value).
Never change the user's intent (fire size, building, escape paths) unless the error makes it physically impossible; prefer
mesh, timing, placement and vent fixes. If no change can fix it, return an empty patch and explain."""


class ReviewOut(BaseModel):
    analysis: str
    patch: list[dict] = Field(default_factory=list)


def review(state: dict) -> dict:
    from logger import log_review  # Foam-Agent's logger

    loop = int(state.get("loop_count", 0)) + 1
    fps = list(state.get("error_fingerprints") or [])
    fp = state.get("fds_fingerprint") or ""
    repeated = bool(fps and fp and fp == fps[-1])
    fps.append(fp)
    if repeated:
        print("<fds_reviewer>same error fingerprint as last loop; stopping</fds_reviewer>")
        return {"loop_count": loop, "error_fingerprints": fps, "workflow_status": "failed",
                "termination_reason": "repair_made_no_progress"}
    errs = state.get("error_logs") or []
    history = state.get("history_text") or ""
    prompt = (
        f"User requirement:\n{state.get('user_requirement', '')}\n\n"
        f"ScenarioSpec:\n{json.dumps(state['scenario_spec'], indent=1)}\n\n"
        f"Errors (loop {loop}):\n" + "\n".join(errs) + "\n\n"
        + (f"Earlier attempts:\n{history[-3000:]}\n\n" if history else "")
        + "Return the analysis and the JSON patch."
    )
    out = state["llm_service"].invoke(prompt, REVIEW_SYSTEM, pydantic_obj=ReviewOut)
    if isinstance(out, dict):
        out = ReviewOut.model_validate(out)
    log_review(out.analysis, f"fds_review_loop_{loop}")
    print(f"<fds_reviewer loop={loop}>{out.analysis[:300]}</fds_reviewer>")
    history += f"\n--- loop {loop} ---\nerrors: {errs[:3]}\nanalysis: {out.analysis[:500]}\npatch: {json.dumps(out.patch)[:500]}\n"
    return {
        "loop_count": loop,
        "error_fingerprints": fps,
        "review_analysis": out.analysis,
        "rewrite_plan": {"patch": out.patch, "target_files": [f"{state['case_name']}.fds"]},
        "input_writer_mode": "rewrite",
        "history_text": history,
    }


# ----------------------------------------------------------------------------
# criteria + report nodes (new)
# ----------------------------------------------------------------------------

def criteria_node(state: dict) -> dict:
    spec = _spec_from_state(state)
    case_dir = Path(state["case_dir"])
    th = C.load_thresholds(_thresholds_path(state["config"]))
    res = C.evaluate(case_dir, spec, th)
    (case_dir / "criteria.json").write_text(res.model_dump_json(indent=2), encoding="utf-8")
    for p in res.paths:
        print(f"<fds_criteria path={p.id}>ASET={p.aset_s:.0f}s RSET={p.rset_s} margin={p.margin} pass={p.passed}</fds_criteria>")
    return {"criteria_result": res.model_dump(mode="json")}


def report_node(state: dict) -> dict:
    """Placeholder until module 4: write a Markdown summary next to criteria.json."""
    case_dir = Path(state["case_dir"])
    cr = state.get("criteria_result") or {}
    lines = [f"# SmokeAgent summary: {state.get('case_name')}", ""]
    hrr = cr.get("hrr_check", {})
    lines.append(f"HRR peak: set {hrr.get('peak_kw_set')} kW, simulated {hrr.get('peak_kw_sim', 0):.0f} kW, ok={hrr.get('ok')}")
    lines.append("")
    lines.append("| path | RSET s | ASET s | margin | pass |")
    lines.append("| --- | --- | --- | --- | --- |")
    for p in cr.get("paths", []):
        m = p.get("margin")
        lines.append(f"| {p['id']} | {p.get('rset_s')} | {p['aset_s']:.0f} | {m if m is None else round(m, 2)} | {p.get('passed')} |")
    path = case_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"<fds_report>{path}</fds_report>")
    return {"report_path": str(path)}
