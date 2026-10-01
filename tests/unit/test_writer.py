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
    assert {"HEAD", "MESH", "TIME", "REAC", "SURF", "RAMP", "OBST", "HOLE", "VENT", "DEVC", "SLCF", "TAIL"} <= groups
    mesh = records_of(recs, "MESH")[0]
    assert mesh.params["IJK"] == ["60", "20", "13"]
    fire = [r for r in records_of(recs, "SURF") if r.get("ID") == "FIRE"][0]
    assert float(fire.get("HRRPUA")) == pytest.approx(1000.0)
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
