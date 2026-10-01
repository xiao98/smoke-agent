"""Device placement and ASET evaluation (module 3).  No LLM anywhere here."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import yaml

from fds.namelist import Record
from fds.spec import CriteriaResult, HrrCheck, PathResult, QuantityExceed, ScenarioSpec

QUANTITIES = {
    "T": ("TEMPERATURE", {}),
    "VIS": ("VISIBILITY", {}),
    "CO": ("VOLUME FRACTION", {"SPEC_ID": "CARBON MONOXIDE"}),
}
SPACING_M = 2.0


def _samples(polyline: list[list[float]], spacing: float = SPACING_M) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for (x0, y0, z0), (x1, y1, z1) in zip(polyline, polyline[1:]):
        seg = math.dist((x0, y0), (x1, y1))
        n = max(1, int(math.ceil(seg / spacing)))
        for i in range(n):
            t = i / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    pts.append(tuple(polyline[-1]))
    # de-duplicate consecutive identical points
    out: list[tuple[float, float, float]] = []
    for p in pts:
        if not out or math.dist(out[-1], p) > 1e-6:
            out.append(p)
    return out


def device_id(path_id: str, i: int, q: str) -> str:
    return f"{path_id}_{i:02d}_{q}"


def place_devices(spec: ScenarioSpec) -> list[Record]:
    """One point probe per quantity per sample point, plus a LAYER HEIGHT line."""
    z_lo, z_hi = spec.building.domain.z
    recs: list[Record] = []
    for p in spec.escape_paths:
        for i, (x, y, z) in enumerate(_samples(p.polyline)):
            for q, (quantity, extra) in QUANTITIES.items():
                params = {"ID": [device_id(p.id, i, q)], "XYZ": [f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"],
                          "QUANTITY": [quantity]}
                for k, v in extra.items():
                    params[k] = [v]
                recs.append(Record("DEVC", params))
            recs.append(Record("DEVC", {
                "ID": [device_id(p.id, i, "LH")],
                "XB": [f"{x:.3f}", f"{x:.3f}", f"{y:.3f}", f"{y:.3f}", f"{z_lo:.3f}", f"{z_hi:.3f}"],
                "QUANTITY": ["LAYER HEIGHT"],
            }))
    return recs


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def load_thresholds(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_devc(csv_path: Path) -> tuple[list[float], dict[str, list[float]]]:
    """Read CHID_devc.csv: row 1 units, row 2 column names, then data."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    names = [c.strip() for c in rows[1]]
    data = {n: [] for n in names}
    for r in rows[2:]:
        if len(r) != len(names):
            continue
        for n, v in zip(names, r):
            data[n].append(float(v))
    t = data.pop(names[0])
    return t, data


def _smooth(x: list[float], t: list[float], window_s: float) -> list[float]:
    if window_s <= 0 or len(x) < 3:
        return x
    out = []
    j0 = 0
    for i in range(len(x)):
        while t[i] - t[j0] > window_s:
            j0 += 1
        seg = x[j0:i + 1]
        out.append(sum(seg) / len(seg))
    return out


def _violates(op: str, value: float, limit: float) -> bool:
    return {">=": value < limit, "<=": value > limit, ">": value <= limit, "<": value >= limit}[op]


def _convert(q: str, v: float) -> float:
    return v * 1e6 if q == "co" else v  # volume fraction -> ppm


_COL = {"temperature": "T", "visibility": "VIS", "co": "CO", "layer_height": "LH"}


def evaluate(bundle_dir: Path, spec: ScenarioSpec, thresholds: dict, hrr_peak_sim: float | None = None) -> CriteriaResult:
    t, data = read_devc(bundle_dir / f"{spec.chid}_devc.csv")
    t_end = t[-1] if t else spec.sim.t_end_s
    window = float(thresholds.get("smoothing_s", 0))
    margin_min = float(thresholds.get("margin_min", 1.0))
    paths: list[PathResult] = []
    for p in spec.escape_paths:
        pts = _samples(p.polyline)
        qres: dict[str, QuantityExceed] = {}
        aset = t_end
        censored = True
        for qname, qcfg in thresholds["quantities"].items():
            suffix = _COL[qname]
            first, worst_pt, worst_val = None, None, None
            for i, pt in enumerate(pts):
                col = device_id(p.id, i, suffix)
                if col not in data:
                    continue
                series = _smooth([_convert(qname, v) for v in data[col]], t, window)
                for ti, v in zip(t, series):
                    if _violates(qcfg["op"], v, float(qcfg["value"])):
                        if first is None or ti < first:
                            first, worst_pt, worst_val = ti, list(pt), v
                        break
            qres[qname] = QuantityExceed(first_exceed_s=first, worst_point=worst_pt, worst_value=worst_val)
            if first is not None:
                censored = False
                aset = min(aset, first)
        margin = aset / p.rset_s if p.rset_s else None
        passed = (margin >= margin_min) if margin is not None else None
        paths.append(PathResult(id=p.id, rset_s=p.rset_s, aset_s=aset, censored=censored,
                                margin=margin, passed=passed, quantities=qres))
    peak_sim = hrr_peak_sim if hrr_peak_sim is not None else read_hrr_peak(bundle_dir / f"{spec.chid}_hrr.csv")
    expected = expected_hrr_peak(spec, t_end)
    hrr = HrrCheck(peak_kw_set=expected, peak_kw_sim=peak_sim, ok=abs(peak_sim - expected) <= hrr_tolerance(spec, t_end) * expected)
    return CriteriaResult(chid=spec.chid, thresholds_profile=spec.thresholds_profile, t_end_s=t_end, paths=paths, hrr_check=hrr)


HRR_TOL = 0.10        # relative tolerance on the smoothed peak
HRR_WINDOW_S = 10.0   # moving-average window; FDS instantaneous HRR fluctuates


def expected_hrr_peak(spec: ScenarioSpec, t_end: float, window_s: float = HRR_WINDOW_S) -> float:
    """Peak of the *smoothed* HRR the ramp can reach by t_end.

    During t-squared growth the moving average lags the instantaneous curve, so
    the comparison point is the middle of the averaging window.
    """
    tp = spec.fire.t_peak()
    if tp <= 0 or t_end >= tp + window_s:
        return spec.fire.hrr_peak_kw
    t_mid = max(0.0, t_end - window_s / 2)
    return spec.fire.hrr_peak_kw * min(1.0, t_mid / tp) ** 2


def hrr_tolerance(spec: ScenarioSpec, t_end: float) -> float:
    """10% on the plateau, 20% while still growing (ramp interpolation + ignition lag)."""
    return HRR_TOL if t_end >= spec.fire.t_peak() else 2 * HRR_TOL


def read_hrr_peak(csv_path: Path, window_s: float = HRR_WINDOW_S) -> float:
    """Peak of the moving-average HRR (kW)."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    names = [c.strip() for c in rows[1]]
    j = names.index("HRR")
    t, h = [], []
    for r in rows[2:]:
        if len(r) == len(names):
            t.append(float(r[0])); h.append(float(r[j]))
    return max(_smooth(h, t, window_s)) if h else 0.0
