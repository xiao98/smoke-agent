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


# Thermal properties from the NIST FDS validation inputs (NBS_Multi-Room, citing NBSIR 88-3752; steel: Drysdale).
# Each entry: (SURF id, MATL/RAMP records, SURF params without ID).
MATERIALS: dict[str, tuple[str, list[Record], dict[str, list[str]]]] = {
    "inert": ("INERT", [], {}),
    "gypsum": ("GYPSUM BOARD",
               [Record("MATL", {"ID": ["GYPSUM"], "CONDUCTIVITY": ["0.17"], "SPECIFIC_HEAT": ["1.09"], "DENSITY": ["930."]})],
               {"MATL_ID": ["GYPSUM"], "COLOR": ["BEIGE"], "THICKNESS": ["0.013"]}),
    "concrete": ("CONCRETE",
                 [Record("MATL", {"ID": ["CONCRETE"], "CONDUCTIVITY": ["1.8"], "SPECIFIC_HEAT": ["1.04"], "DENSITY": ["2280."]})],
                 {"MATL_ID": ["CONCRETE"], "COLOR": ["GRAY 60"], "THICKNESS": ["0.102"]}),
    "ceramic_fiber": ("INSULATION",
                      [Record("MATL", {"ID": ["CERAMIC FIBER"], "CONDUCTIVITY_RAMP": ["k_fiber"], "SPECIFIC_HEAT": ["1.04"],
                                       "DENSITY": ["128."], "EMISSIVITY": ["0.97"]}),
                       Record("RAMP", {"ID": ["k_fiber"], "T": ["20."], "F": ["0.09"]}),
                       Record("RAMP", {"ID": ["k_fiber"], "T": ["300."], "F": ["0.09"]}),
                       Record("RAMP", {"ID": ["k_fiber"], "T": ["600."], "F": ["0.17"]}),
                       Record("RAMP", {"ID": ["k_fiber"], "T": ["900."], "F": ["0.25"]})],
                      {"MATL_ID": ["CERAMIC FIBER"], "COLOR": ["GRAY"], "THICKNESS": ["0.050"]}),
    "calcium_silicate_on_gypsum": ("HALLWAY BOARD",
                                   [Record("MATL", {"ID": ["CALCIUM SILICATE"], "CONDUCTIVITY": ["0.12"], "SPECIFIC_HEAT_RAMP": ["c_cal"],
                                                    "DENSITY": ["720."], "EMISSIVITY": ["0.83"]}),
                                    Record("RAMP", {"ID": ["c_cal"], "T": ["20."], "F": ["1.25"]}),
                                    Record("RAMP", {"ID": ["c_cal"], "T": ["200."], "F": ["1.25"]}),
                                    Record("RAMP", {"ID": ["c_cal"], "T": ["300."], "F": ["1.33"]}),
                                    Record("RAMP", {"ID": ["c_cal"], "T": ["600."], "F": ["1.55"]}),
                                    Record("MATL", {"ID": ["GYPSUM"], "CONDUCTIVITY": ["0.17"], "SPECIFIC_HEAT": ["1.09"], "DENSITY": ["930."]})],
                                   {"MATL_ID(1:2,1)": ["CALCIUM SILICATE", "GYPSUM"], "COLOR": ["WHEAT"], "THICKNESS(1:2)": ["0.013", "0.013"]}),
    "fire_brick": ("FIRE BRICK",
                   [Record("MATL", {"ID": ["FIRE BRICK"], "CONDUCTIVITY_RAMP": ["k_brick"], "SPECIFIC_HEAT": ["1.04"],
                                    "DENSITY": ["750."], "EMISSIVITY": ["0.80"]}),
                    Record("RAMP", {"ID": ["k_brick"], "T": ["20."], "F": ["0.36"]}),
                    Record("RAMP", {"ID": ["k_brick"], "T": ["200."], "F": ["0.36"]}),
                    Record("RAMP", {"ID": ["k_brick"], "T": ["300."], "F": ["0.38"]}),
                    Record("RAMP", {"ID": ["k_brick"], "T": ["600."], "F": ["0.45"]})],
                   {"MATL_ID": ["FIRE BRICK"], "COLOR": ["FIREBRICK"], "THICKNESS": ["0.113"]}),
    "steel": ("STEEL",
              [Record("MATL", {"ID": ["STEEL"], "EMISSIVITY": ["0.95"], "SPECIFIC_HEAT": ["0.46"], "CONDUCTIVITY": ["45.8"], "DENSITY": ["7850."]})],
              {"MATL_ID": ["STEEL"], "COLOR": ["BLACK"], "THICKNESS": ["0.005"]}),
}


def surf_for(material: str) -> str:
    return MATERIALS[material][0]


def material_records(spec: ScenarioSpec) -> list[Record]:
    """MATL/RAMP/SURF records for every material used; the building default SURF carries DEFAULT=.TRUE."""
    default = spec.building.wall_material
    used = [default] + [o.material for o in spec.building.obstructions if o.material] + [m for m in spec.mesh.block_materials if m]
    recs: list[Record] = []
    seen_matl: set[str] = set()
    for m in dict.fromkeys(used):
        sid, matl, surf = MATERIALS[m]
        if not surf:
            continue  # inert
        for r in matl:
            key = r.group + ":" + (r.get("ID") or "") + ":" + (r.get("T") or "")  # MATL once per ID; RAMP once per point
            if key in seen_matl:
                continue
            seen_matl.add(key)
            recs.append(r)
        params = {"ID": [sid]}
        if m == default:
            params["DEFAULT"] = [".TRUE."]
        params.update(surf)
        recs.append(Record("SURF", params))
    return recs


def _faces(b: list[float]) -> list[list[float]]:
    return [[b[0], b[0], b[2], b[3], b[4], b[5]], [b[1], b[1], b[2], b[3], b[4], b[5]],
            [b[0], b[1], b[2], b[2], b[4], b[5]], [b[0], b[1], b[3], b[3], b[4], b[5]],
            [b[0], b[1], b[2], b[3], b[4], b[4]], [b[0], b[1], b[2], b[3], b[5], b[5]]]


def _face_touches(face: list[float], box: list[float]) -> bool:
    """True when the planar face lies on one of the box's planes and shares a positive area with it."""
    k0 = next(k for k in range(3) if face[2 * k] == face[2 * k + 1])
    v = face[2 * k0]
    if abs(box[2 * k0] - v) > 1e-9 and abs(box[2 * k0 + 1] - v) > 1e-9:
        return False
    for k in range(3):
        if k == k0:
            continue
        if min(face[2 * k + 1], box[2 * k + 1]) - max(face[2 * k], box[2 * k]) <= 1e-9:
            return False
    return True


def block_boundary_vents(spec: ScenarioSpec) -> list[Record]:
    """Boundary VENTs giving a mesh block's walls/floor/ceiling their own material (like the NIST room inputs).
    Faces shared with another block (mesh interfaces) or touched by an opening/exhaust are left alone."""
    recs: list[Record] = []
    blocks = spec.mesh.blocks
    planar = [op.xb for op in spec.building.openings] + [e.xb for e in spec.smoke_control.exhaust]
    for i, b in enumerate(blocks):
        m = spec.mesh.block_materials[i] if i < len(spec.mesh.block_materials) else None
        if not m or m == spec.building.wall_material:
            continue
        for k, face in enumerate(_faces(b)):
            if any(_face_touches(face, other) for j, other in enumerate(blocks) if j != i):
                continue
            if any(_face_touches(face, p) for p in planar):
                continue
            recs.append(Record("VENT", {"ID": [f"m{i + 1}_face{k + 1}"], "XB": _xb(face), "SURF_ID": [surf_for(m)]}))
    return recs


def subtract_holes(box: list[float], holes: list[list[float]]) -> list[list[float]]:
    """Boxes left after cutting the holes out of the box (written as separate &OBST, never as &HOLE).
    A &HOLE in an OBST whose face lies on a mesh interface leaves a zero-thickness plate on the neighbouring
    mesh and seals the opening (seen on NBS 100A: fire room sealed, fire starved at 200 s)."""
    pieces = [list(box)]
    for h in holes:
        out = []
        for b in pieces:
            ix = [max(b[0], h[0]), min(b[1], h[1])]
            iy = [max(b[2], h[2]), min(b[3], h[3])]
            iz = [max(b[4], h[4]), min(b[5], h[5])]
            if ix[0] >= ix[1] or iy[0] >= iy[1] or iz[0] >= iz[1]:
                out.append(b)
                continue
            if b[0] < ix[0]:
                out.append([b[0], ix[0], b[2], b[3], b[4], b[5]])
            if ix[1] < b[1]:
                out.append([ix[1], b[1], b[2], b[3], b[4], b[5]])
            if b[2] < iy[0]:
                out.append([ix[0], ix[1], b[2], iy[0], b[4], b[5]])
            if iy[1] < b[3]:
                out.append([ix[0], ix[1], iy[1], b[3], b[4], b[5]])
            if b[4] < iz[0]:
                out.append([ix[0], ix[1], iy[0], iy[1], b[4], iz[0]])
            if iz[1] < b[5]:
                out.append([ix[0], ix[1], iy[0], iy[1], iz[1], b[5]])
        pieces = out
    return pieces


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
    recs += material_records(spec)

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
        pieces = subtract_holes(o.xb, [h.xb for h in o.holes])
        for k, piece in enumerate(pieces):
            pid = o.id if len(pieces) == 1 else f"{o.id}_{k + 1}"
            recs.append(Record("OBST", {"ID": [pid], "XB": _xb(piece), "SURF_ID": [surf_for(o.material or spec.building.wall_material)]}))
    for c in spec.smoke_control.curtains:
        recs.append(Record("OBST", {"ID": [c.id], "XB": _xb(c.xb), "SURF_ID": [surf_for(spec.building.wall_material)]}))
    # burner last: where a pedestal snapped to the grid overlaps the slab, the last OBST's surfaces take precedence
    recs.append(Record("OBST", {"ID": ["burner"], "XB": _xb(bx), "SURF_IDS": ["FIRE", "INERT", "INERT"]}))
    recs += block_boundary_vents(spec)
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
