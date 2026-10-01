import pathlib

from fds import outparse
from fds.agent import apply_patch

OUT_SAMPLE = """
 Mesh    1
 ERROR: SURF FIRE2 not found
 Time Step:    10, Simulation Time:   0.5 s
"""


def test_outparse_rules():
    errs = outparse.errors(OUT_SAMPLE)
    fps = {e.fingerprint for e in errs}
    assert "surf_undefined" in fps and "generic_error" not in fps
    assert outparse.fingerprint(errs) == outparse.fingerprint(errs)
    assert outparse.to_error_logs(errs)[0].startswith("[surf_undefined]")


def test_outparse_timeout_and_rc():
    errs = outparse.errors("", "", timed_out=True, returncode=-9)
    assert [e.fingerprint for e in errs] == ["timeout"]
    errs = outparse.errors("", "boom", timed_out=False, returncode=3)
    assert [e.fingerprint for e in errs] == ["nonzero_exit"]


def test_apply_patch():
    doc = {"fire": {"hrr_peak_kw": 500, "growth": "fast"}, "escape_paths": [{"id": "P1"}]}
    out = apply_patch(doc, [
        {"op": "replace", "path": "/fire/hrr_peak_kw", "value": 400},
        {"op": "add", "path": "/escape_paths/-", "value": {"id": "P2"}},
        {"op": "remove", "path": "/fire/growth"},
    ])
    assert out["fire"] == {"hrr_peak_kw": 400}
    assert [p["id"] for p in out["escape_paths"]] == ["P1", "P2"]
    assert doc["fire"]["hrr_peak_kw"] == 500  # original untouched


def test_runner_n_meshes():
    from fds.runner import n_meshes
    assert n_meshes("&MESH IJK=1,1,1 XB=0,1,0,1,0,1 /\n&MESH IJK=1,1,1 XB=1,2,0,1,0,1 /") == 2


def test_expected_hrr_peak():
    from fds.criteria import expected_hrr_peak
    from fds.spec import ScenarioSpec
    from test_writer import THREE_ROOMS
    spec = ScenarioSpec(**THREE_ROOMS)           # 500 kW fast: t_peak ~103 s
    assert expected_hrr_peak(spec, 300) == 500
    assert 25 < expected_hrr_peak(spec, 30) < 35  # ((30-5)/103)^2 * 500 ~ 29 kW, mid-window
