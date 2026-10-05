import shutil

import pytest

from fds import validate as V
from fds.namelist import parse, records_of
from fds.spec import ScenarioSpec
from fds.writer import auto_cell_size, char_diameter, write

THREE_ROOMS = {
    "chid": "three_rooms_gen",
    "title": "three rooms in a row, fire in room 1",
    "building": {
        "domain": {"x": [0, 12], "y": [0, 4], "z": [0, 2.6]},
        "obstructions": [
            {"id": "wall1", "xb": [3.9, 4.1, 0, 4, 0, 2.6], "holes": [{"xb": [3.8, 4.2, 1.5, 2.5, 0, 2.0]}]},
            {"id": "wall2", "xb": [7.9, 8.1, 0, 4, 0, 2.6], "holes": [{"xb": [7.8, 8.2, 1.5, 2.5, 0, 2.0]}]},
        ],
        "openings": [{"id": "leak", "xb": [12, 12, 1, 3, 0, 0.5]}],
    },
    "fire": {"location_xb": [1.5, 2.5, 1.75, 2.25, 0, 0], "hrr_peak_kw": 500, "growth": "fast"},
    "escape_paths": [{"id": "P1", "polyline": [[6, 2, 1.8], [10, 2, 1.8]], "rset_s": 120}],
    "mesh": {"cell_size": 0.2},
    "sim": {"t_end_s": 300},
}


def test_dstar_and_cell():
    assert 0.7 < char_diameter(500) < 0.8          # ~0.74 m for 500 kW
    assert auto_cell_size(500) == pytest.approx(0.05)
    assert auto_cell_size(5000) == pytest.approx(0.15)


def test_writer_records():
    spec = ScenarioSpec(**THREE_ROOMS)
    text = write(spec)
    recs = parse(text)
    groups = {r.group for r in recs}
    assert {"HEAD", "MESH", "TIME", "REAC", "SURF", "RAMP", "OBST", "VENT", "DEVC", "SLCF", "TAIL"} <= groups
    assert "HOLE" not in groups
    mesh = records_of(recs, "MESH")[0]
    assert mesh.params["IJK"] == ["60", "20", "13"]
    fire = [r for r in records_of(recs, "SURF") if r.get("ID") == "FIRE"][0]
    burner = [r for r in records_of(recs, "OBST") if r.get("ID") == "burner"][0]
    bx = burner.floats("XB")
    # the slab is snapped to the grid; HRRPUA is rescaled so the total HRR is still the requested 500 kW
    assert float(fire.get("HRRPUA")) * (bx[1] - bx[0]) * (bx[3] - bx[2]) == pytest.approx(500.0, rel=1e-3)
    devc = records_of(recs, "DEVC")
    # P1: 4 m path at 2 m spacing -> 3 points x 4 quantities
    assert len(devc) == 12
    assert any(r.get("QUANTITY") == "LAYER HEIGHT" for r in devc)


def test_spec_rejects_outside_domain():
    bad = dict(THREE_ROOMS, fire={"location_xb": [20, 21, 1, 2, 0, 0], "hrr_peak_kw": 500})
    with pytest.raises(ValueError):
        ScenarioSpec(**bad)


def test_structural_and_engineering():
    spec = ScenarioSpec(**THREE_ROOMS)
    assert V.structural(spec).ok
    eng = V.engineering(spec)
    assert any("D*/dx" in w for w in eng.warnings)  # 0.2 m cells are coarse for 500 kW


@pytest.mark.skipif(shutil.which("fds") is None, reason="fds not on PATH")
def test_solver_setup_run():
    spec = ScenarioSpec(**THREE_ROOMS)
    rep = V.validate(spec, write(spec), run_solver=True)
    assert rep.ok, rep.errors


def test_explicit_time_to_peak():
    spec = ScenarioSpec(**dict(THREE_ROOMS, fire={"location_xb": [1.5, 2.5, 1.75, 2.25, 0, 0], "hrr_peak_kw": 1055, "growth": "fast", "time_to_peak_s": 100}))
    assert spec.fire.t_peak() == 100
    recs = parse(write(spec))
    ramps = [r for r in records_of(recs, "RAMP") if r.get("ID") == "fire_ramp"]
    assert any(abs(r.floats("T")[0] - 100) < 1e-6 and r.floats("F")[0] == 1.0 for r in ramps)


def test_hrr_curve():
    fire = {"location_xb": [1.5, 2.5, 1.75, 2.25, 0, 0], "hrr_peak_kw": 1, "growth": "fast",
            "hrr_curve": [[0, 0], [20, 350], [100, 1055], [300, 1055]]}
    spec = ScenarioSpec(**dict(THREE_ROOMS, fire=fire))
    assert spec.fire.hrr_peak_kw == 1055 and spec.fire.t_peak() == 100
    assert abs(spec.fire.hrr_at(60) - (350 + (1055 - 350) * 40 / 80)) < 1e-6
    recs = parse(write(spec))
    ramps = [r for r in records_of(recs, "RAMP") if r.get("ID") == "fire_ramp"]
    assert [r.floats("T")[0] for r in ramps] == [0, 20, 100, 300]
    assert abs(ramps[1].floats("F")[0] - 350 / 1055) < 1e-3


def test_surf_id_coerced():
    b = dict(THREE_ROOMS["building"]); b = {**b, "obstructions": [{"id": "beam", "xb": [0, 9, 2.7, 2.8, 2.2, 2.4], "surf_id": "STEEL BEAM"}]}
    spec = ScenarioSpec(**dict(THREE_ROOMS, building=b))
    assert spec.building.obstructions[0].surf_id == "INERT"


def test_burner_snap():
    from fds.writer import snap_burner
    # NBS-like: 0.3 x 0.3 m burner on a 0.5 m pedestal, 0.2 m grid -> must become a full-cell slab with a FIRE face
    fire = {"location_xb": [10.9, 11.2, 0.0, 0.3, 0.5, 0.5], "hrr_peak_kw": 110, "growth": "constant"}
    b = dict(THREE_ROOMS["building"]); b = {**b, "domain": {"x": [0, 12.2], "y": [0, 5.8], "z": [0, 2.4]}, "obstructions": []}
    spec = ScenarioSpec(**dict(THREE_ROOMS, fire=fire, building=b, mesh={"cell_size": 0.2}))
    bx = snap_burner(spec, 0.2)
    assert bx == pytest.approx([10.8, 11.2, 0.0, 0.4, 0.4, 0.6])
    recs = parse(write(spec))
    burner = [r for r in records_of(recs, "OBST") if r.get("ID") == "burner"][0]
    assert burner.floats("XB") == pytest.approx(bx)
    fire_surf = [r for r in records_of(recs, "SURF") if r.get("ID") == "FIRE"][0]
    assert float(fire_surf.get("HRRPUA")) == pytest.approx(110 / (0.4 * 0.4))


def test_mesh_blocks_and_burner_last():
    b = {**THREE_ROOMS["building"], "domain": {"x": [0, 12.2], "y": [0, 5.8], "z": [0, 2.4]},
         "obstructions": [{"id": "pedestal", "xb": [10.9, 11.2, 0, 0.3, 0, 0.5]}], "openings": []}
    fire = {"location_xb": [10.9, 11.2, 0.0, 0.3, 0.5, 0.5], "hrr_peak_kw": 110, "growth": "constant"}
    mesh = {"cell_size": 0.1, "blocks": [[9.8, 12.2, 0, 3.4, 0, 2.2], [0, 12.2, 3.4, 5.8, 0, 2.4]]}
    spec = ScenarioSpec(**dict(THREE_ROOMS, fire=fire, building=b, mesh=mesh))
    assert V.structural(spec).errors == []
    recs = parse(write(spec))
    meshes = records_of(recs, "MESH")
    assert [m.params["IJK"] for m in meshes] == [["24", "34", "22"], ["122", "24", "24"]]
    obst_ids = [r.get("ID") for r in records_of(recs, "OBST")]
    assert obst_ids[-1] == "burner" and "pedestal" in obst_ids
    bad = ScenarioSpec(**dict(THREE_ROOMS, fire=fire, building=b, mesh={"cell_size": 0.1, "blocks": [[0, 9, 0, 3.4, 0, 2.2]]}))
    assert any("not inside any mesh block" in e for e in V.structural(bad).errors)


def test_subtract_holes():
    from fds.writer import subtract_holes
    wall = [9.8, 12.2, 3.3, 3.4, 0, 2.2]
    door = [11.2, 12.0, 3.3, 3.4, 0, 1.6]
    pieces = subtract_holes(wall, [door])
    assert sorted(pieces) == sorted([[9.8, 11.2, 3.3, 3.4, 0, 2.2], [12.0, 12.2, 3.3, 3.4, 0, 2.2], [11.2, 12.0, 3.3, 3.4, 1.6, 2.2]])
    vol = lambda b: (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])
    assert sum(map(vol, pieces)) == pytest.approx(vol(wall) - vol(door))
    assert subtract_holes(wall, [[0, 1, 0, 1, 0, 1]]) == [wall]
