"""ScenarioSpec -> FDS input text (module 1).  Pure function, no LLM."""
from __future__ import annotations

import math

from fds.criteria import place_devices
from fds.namelist import Record, dump
from fds.spec import ScenarioSpec

RHO, CP, T_INF, G = 1.2, 1.0, 293.0, 9.81  # kg/m3, kJ/kg/K, K, m/s2


def char_diameter(hrr_kw: float) -> float:
    """Fire characteristic diameter D* (McGrattan et al.)."""
    return (hrr_kw / (RHO * CP * T_INF * math.sqrt(G))) ** 0.4


def auto_cell_size(hrr_kw: float, lo: float = 0.05, hi: float = 0.5) -> float:
    d = char_diameter(hrr_kw) / 10.0
    d = math.floor(d / 0.05) * 0.05
    return max(lo, min(hi, d))


def snap_burner(spec: ScenarioSpec, cell: float) -> list[float]:
    """Burner slab aligned to the mesh: x/y edges on grid planes (>= 1 cell), base on the nearest grid plane, 1 cell thick."""
    d = spec.building.domain
    fx = spec.fire.location_xb

    def down(v, o):
        return o + math.floor((v - o) / cell + 1e-9) * cell

    def up(v, o):
        return o + math.ceil((v - o) / cell - 1e-9) * cell

    x0, x1 = down(fx[0], d.x[0]), up(fx[1], d.x[0])
    y0, y1 = down(fx[2], d.y[0]), up(fx[3], d.y[0])
    if x1 - x0 < cell:
        x1 = x0 + cell
    if y1 - y0 < cell:
        y1 = y0 + cell
    z0 = d.z[0] + round((fx[4] - d.z[0]) / cell) * cell
    z0 = min(max(z0, d.z[0]), d.z[1] - cell)
    return [x0, x1, y0, y1, z0, z0 + cell]


def _f(v: float) -> str:
    return f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.4e}"


def _xb(xb: list[float]) -> list[str]:
    return [_f(v) for v in xb]


def mesh_records(spec: ScenarioSpec) -> tuple[list[Record], float]:
    d = spec.building.domain
    cell = spec.mesh.cell_size if spec.mesh.cell_size != "auto" else auto_cell_size(spec.fire.hrr_peak_kw)
    if spec.mesh.blocks:
        recs = []
        for i, b in enumerate(spec.mesh.blocks):
            ijk = [max(1, int(round((b[2 * a + 1] - b[2 * a]) / cell))) for a in range(3)]
            recs.append(Record("MESH", {"ID": [f"m{i + 1}"], "IJK": [str(v) for v in ijk], "XB": _xb(b)}))
        return recs, cell
    ext = [d.x[1] - d.x[0], d.y[1] - d.y[0], d.z[1] - d.z[0]]
    ijk = [max(1, int(math.ceil(e / cell))) for e in ext]
    total = ijk[0] * ijk[1] * ijk[2]
    n = max(1, int(math.ceil(total / spec.mesh.max_cells_per_mesh)))
    recs = []
    if n == 1:
        recs.append(Record("MESH", {"IJK": [str(v) for v in ijk], "XB": _xb(d.xb)}))
        return recs, cell
    # split along the longest axis into n blocks with equal cell counts
    axis = max(range(3), key=lambda a: ext[a])
    per = [ijk[axis] // n + (1 if i < ijk[axis] % n else 0) for i in range(n)]
    lo = [d.x[0], d.y[0], d.z[0]][axis]
    start = lo
    for i, cnt in enumerate(per):
        xb = list(d.xb)
        xb[2 * axis] = start
        xb[2 * axis + 1] = start + cnt * cell
        ijk_i = list(ijk)
        ijk_i[axis] = cnt
        recs.append(Record("MESH", {"ID": [f"m{i + 1}"], "IJK": [str(v) for v in ijk_i], "XB": _xb(xb)}))
        start = xb[2 * axis + 1]
    return recs, cell


def ramp_records(spec: ScenarioSpec) -> list[Record]:
    f = spec.fire
    if f.hrr_curve:
        pk = f.hrr_peak_kw
        pts = [(float(t), float(q) / pk) for t, q in f.hrr_curve]
        if pts[0][0] > 0:
            pts.insert(0, (0.0, pts[0][1]))
        return [Record("RAMP", {"ID": ["fire_ramp"], "T": [_f(t)], "F": [_f(v)]}) for t, v in pts]
    if f.growth == "constant":
        return []
    tp = f.t_peak()
    pts = [(0.0, 0.0)]
    for k in (0.25, 0.5, 0.75, 1.0):
        pts.append((k * tp, k * k))
    if spec.sim.t_end_s > tp:
        pts.append((spec.sim.t_end_s, 1.0))
    return [Record("RAMP", {"ID": ["fire_ramp"], "T": [_f(t)], "F": [_f(v)]}) for t, v in pts]


def write(spec: ScenarioSpec) -> str:
    recs: list[Record] = []
    recs.append(Record("HEAD", {"CHID": [spec.chid], "TITLE": [spec.title or spec.chid]}))
    meshes, cell = mesh_records(spec)
    recs += meshes
    recs.append(Record("TIME", {"T_END": [_f(spec.sim.t_end_s)]}))
    recs.append(Record("DUMP", {"DT_DEVC": [_f(spec.sim.dt_devc_s)], "DT_SLCF": [_f(spec.sim.dt_slcf_s)], "DT_HRR": ["1."]}))
    recs.append(Record("MISC", {"TMPA": ["20."]}))

    fu = spec.fire.fuel
    recs.append(Record("REAC", {"FUEL": [fu.name], "C": [_f(fu.c)], "H": [_f(fu.h)], "O": [_f(fu.o)], "N": [_f(fu.n)],
                                "SOOT_YIELD": [_f(fu.soot_yield)], "CO_YIELD": [_f(fu.co_yield)],
                                "HEAT_OF_COMBUSTION": [_f(fu.heat_of_combustion)]}))

    # Fire: a one-cell-thick slab snapped to the grid, whose top face burns at HRRPUA = peak / snapped area.
    # (A slab thinner than a cell, or a face between grid planes, is collapsed by FDS and releases no heat.)
    bx = snap_burner(spec, cell)
    area = (bx[1] - bx[0]) * (bx[3] - bx[2])
    hrrpua = spec.fire.hrr_peak_kw / area
    surf = {"ID": ["FIRE"], "HRRPUA": [_f(hrrpua)], "COLOR": ["RED"]}
    if spec.fire.hrr_curve or spec.fire.growth != "constant":
        surf["RAMP_Q"] = ["fire_ramp"]
    recs.append(Record("SURF", surf))
    recs += ramp_records(spec)
    x0, x1, y0, y1 = bx[0], bx[1], bx[2], bx[3]

    for o in spec.building.obstructions:
        recs.append(Record("OBST", {"ID": [o.id], "XB": _xb(o.xb), "SURF_ID": [o.surf_id]}))
        for h in o.holes:
            recs.append(Record("HOLE", {"XB": _xb(h.xb)}))
    for c in spec.smoke_control.curtains:
        recs.append(Record("OBST", {"ID": [c.id], "XB": _xb(c.xb), "SURF_ID": ["INERT"]}))
    # burner last: where a pedestal snapped to the grid overlaps the slab, the last OBST's surfaces take precedence
    recs.append(Record("OBST", {"ID": ["burner"], "XB": _xb(bx), "SURF_IDS": ["FIRE", "INERT", "INERT"]}))
    for op in spec.building.openings:
        recs.append(Record("VENT", {"ID": [op.id], "XB": _xb(op.xb), "SURF_ID": ["OPEN"]}))

    for e in spec.smoke_control.exhaust:
        sid = f"EXH_{e.id}"
        recs.append(Record("SURF", {"ID": [sid], "VOLUME_FLOW": [_f(e.volume_flow_m3s)], "COLOR": ["BLUE"]}))
        vent = {"ID": [e.id], "XB": _xb(e.xb), "SURF_ID": [sid]}
        if e.activation_s > 0:
            tid = f"t_{e.id}"
            recs.append(Record("DEVC", {"ID": [tid], "QUANTITY": ["TIME"], "XYZ": [_f(e.xb[0]), _f(e.xb[2]), _f(e.xb[4])],
                                        "SETPOINT": [_f(e.activation_s)], "INITIAL_STATE": [".FALSE."]}))
            vent["DEVC_ID"] = [tid]
        recs.append(Record("VENT", vent))

    recs += place_devices(spec)

    heights = sorted({round(p.polyline[0][2], 3) for p in spec.escape_paths}) or [1.8]
    for h in heights:
        recs.append(Record("SLCF", {"PBZ": [_f(h)], "QUANTITY": ["VISIBILITY"]}))
        recs.append(Record("SLCF", {"PBZ": [_f(h)], "QUANTITY": ["TEMPERATURE"]}))
    recs.append(Record("SLCF", {"PBY": [_f((y0 + y1) / 2)], "QUANTITY": ["TEMPERATURE"]}))
    recs.append(Record("TAIL", {}))
    header = f"Generated by SmokeAgent writer; cell size {cell:.3f} m; D* = {char_diameter(spec.fire.hrr_peak_kw):.3f} m\n"
    return header + dump(recs)
