"""Three validation layers for a generated case.

1. structural  : the ScenarioSpec itself (pydantic) + geometry sanity
2. engineering : warnings on mesh resolution, HRRPUA, exhaust sizing
3. solver      : FDS setup-only run (T_END=0) must exit 0 with no ERROR
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fds.namelist import Record, dump, parse
from fds.spec import ScenarioSpec
from fds.writer import auto_cell_size, char_diameter, snap_burner


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _planar(xb) -> bool:
    return sum(1 for a, b in ((0, 1), (2, 3), (4, 5)) if xb[a] == xb[b]) == 1


def _on_boundary(xb, d) -> bool:
    for a, b, lo, hi in ((0, 1, d[0], d[1]), (2, 3, d[2], d[3]), (4, 5, d[4], d[5])):
        if xb[a] == xb[b] and xb[a] in (lo, hi):
            return True
    return False


def structural(spec: ScenarioSpec) -> Report:
    r = Report()
    d = spec.building.domain.xb
    for op in spec.building.openings:
        if not _planar(op.xb):
            r.errors.append(f"opening {op.id}: XB must be a plane (one coordinate pair equal)")
        elif not _on_boundary(op.xb, d):
            r.errors.append(f"opening {op.id}: must lie on the domain boundary")
    for e in spec.smoke_control.exhaust:
        if not _planar(e.xb):
            r.errors.append(f"exhaust {e.id}: XB must be a plane")
    fx = spec.fire.location_xb
    if fx[4] != fx[5]:
        r.errors.append("fire.location_xb must be a horizontal plane (z0 == z1)")
    if spec.fire.area_m2 <= 0:
        r.errors.append("fire footprint has zero area")
    for o in spec.building.obstructions:
        x0, x1, y0, y1, z0, z1 = o.xb
        if (x0 < fx[1] and fx[0] < x1 and y0 < fx[3] and fx[2] < y1 and z0 < fx[4] + 0.1 and fx[4] < z1):
            r.errors.append(f"fire footprint intersects obstruction {o.id}")
    if spec.fire.fuel.soot_yield <= 0:
        r.errors.append("soot_yield must be > 0 or VISIBILITY is meaningless")
    return r


def engineering(spec: ScenarioSpec) -> Report:
    r = Report()
    dstar = char_diameter(spec.fire.hrr_peak_kw)
    cell = spec.mesh.cell_size if spec.mesh.cell_size != "auto" else auto_cell_size(spec.fire.hrr_peak_kw)
    ratio = dstar / cell
    if not 4 <= ratio <= 16:
        r.warnings.append(f"D*/dx = {ratio:.1f} (D*={dstar:.2f} m, dx={cell:.2f} m); FDS guide recommends 4-16")
    hrrpua = spec.fire.hrr_peak_kw / spec.fire.area_m2
    if not 250 <= hrrpua <= 2500:
        r.warnings.append(f"HRRPUA = {hrrpua:.0f} kW/m2 is outside the usual 250-2500 range")
    bx = snap_burner(spec, cell)
    snapped = (bx[1] - bx[0]) * (bx[3] - bx[2])
    if abs(snapped - spec.fire.area_m2) > 0.5 * spec.fire.area_m2:
        r.warnings.append(f"burner footprint snapped from {spec.fire.area_m2:.2f} to {snapped:.2f} m2 on a {cell:.2f} m grid; HRRPUA adjusted to keep total HRR")
    d = spec.building.domain
    cells = 1
    for e, lo in ((d.x[1] - d.x[0], 0), (d.y[1] - d.y[0], 0), (d.z[1] - d.z[0], 0)):
        cells *= max(1, int(round(e / cell)))
    if cells > 4_000_000:
        r.warnings.append(f"about {cells/1e6:.1f} M cells; expect very long run times")
    q_exh = sum(e.volume_flow_m3s for e in spec.smoke_control.exhaust)
    if q_exh and spec.smoke_control.exhaust and not spec.building.openings:
        r.warnings.append("mechanical exhaust without any opening for make-up air")
    if q_exh and not (0.005 * spec.fire.hrr_peak_kw <= q_exh <= 0.2 * spec.fire.hrr_peak_kw):
        r.warnings.append(f"exhaust {q_exh:.0f} m3/s vs {spec.fire.hrr_peak_kw:.0f} kW: unusual ratio (typical 0.005-0.2 m3/s per kW)")
    return r


def solver(fds_text: str, chid: str, fds_bin: str = "fds", timeout_s: int = 300) -> Report:
    """Run FDS with T_END=0 in a temp dir; errors = ERROR lines or non-zero exit."""
    r = Report()
    recs = parse(fds_text)
    for rec in recs:
        if rec.group == "TIME":
            rec.params["T_END"] = ["0."]
    text = dump(recs)
    tmp = Path(tempfile.mkdtemp(prefix="fdsval_"))
    try:
        (tmp / f"{chid}.fds").write_text(text, encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        proc = subprocess.run([fds_bin, f"{chid}.fds"], cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        out = (tmp / f"{chid}.out").read_text(errors="replace") if (tmp / f"{chid}.out").exists() else ""
        for m in re.finditer(r"^\s*ERROR.*$", out + "\n" + proc.stderr, re.M):
            r.errors.append(m.group(0).strip())
        for m in re.finditer(r"^\s*WARNING.*$", out, re.M):
            r.warnings.append(m.group(0).strip())
        if proc.returncode != 0 and not r.errors:
            r.errors.append(f"fds exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
    except FileNotFoundError:
        r.errors.append(f"{fds_bin} not found on PATH (source FDS6VARS.sh)")
    except subprocess.TimeoutExpired:
        r.errors.append("fds setup run timed out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return r


def validate(spec: ScenarioSpec, fds_text: str, run_solver: bool = True) -> Report:
    r = structural(spec)
    e = engineering(spec)
    r.warnings += e.warnings
    if r.ok and run_solver:
        s = solver(fds_text, spec.chid)
        r.errors += s.errors
        r.warnings += s.warnings
    return r
