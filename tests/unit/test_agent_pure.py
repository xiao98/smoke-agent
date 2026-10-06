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


def test_patch_dicts_roundtrip():
    from fds.agent import PatchOp, ReviewOut, patch_dicts
    out = ReviewOut(analysis="x", patch=[
        PatchOp(op="replace", path="/building/obstructions/5/xb", value_json="[10.9, 11.2, 0.0, 0.3, 0.0, 0.5]"),
        PatchOp(op="add", path="/escape_paths/-", value_json='{"id": "P2", "polyline": [[0,0,1.8],[1,0,1.8]]}'),
        PatchOp(op="remove", path="/fire/growth"),
        PatchOp(op="replace", path="/title", value_json="not json"),
    ])
    d = patch_dicts(out.patch)
    assert d[0]["value"] == [10.9, 11.2, 0.0, 0.3, 0.0, 0.5] and d[1]["value"]["id"] == "P2"
    assert "value" not in d[2] and d[3]["value"] == "not json"
    # strict structured-output schema: every patch field is declared (no free-form dict)
    schema = ReviewOut.model_json_schema()
    assert set(schema["$defs"]["PatchOp"]["properties"]) == {"op", "path", "value_json"}


def _hrr_csv(tmp_path, values):
    p = tmp_path / "x_hrr.csv"
    lines = ["s,kW", "Time,HRR"] + [f"{t},{v}" for t, v in values]
    p.write_text("\n".join(lines) + "\n")
    return p


def test_hrr_check_plateau_burst_and_sealed(tmp_path):
    from fds.criteria import hrr_check
    from fds.spec import ScenarioSpec
    from test_writer import THREE_ROOMS
    fire = {"location_xb": [1.5, 2.5, 1.75, 2.25, 0, 0], "hrr_peak_kw": 110, "growth": "constant",
            "hrr_curve": [[0, 0], [1, 110], [900, 110], [901, 0]]}
    spec = ScenarioSpec(**dict(THREE_ROOMS, fire=fire, sim={"t_end_s": 950}))
    # NBS 100O-like: plateau at 110, burst to 348 around 860 s, fire dies at 895 s (before ramp-off)
    vals = [(t, 110.0 if t < 859 else 348.0 if t < 869 else 110.0 if t < 895 else 0.0) for t in range(951)]
    hc = hrr_check(spec, _hrr_csv(tmp_path, vals))
    assert hc["ok"] and hc["statistic"] == "plateau_median" and abs(hc["sim"] - 110) < 1 and hc["starved_frac"] < 0.05
    # sealed room: fire starves at 200 s
    vals = [(t, 110.0 if t < 200 else 0.0) for t in range(951)]
    hc = hrr_check(spec, _hrr_csv(tmp_path, vals))
    assert not hc["ok"] and hc["starved_frac"] > 0.5
    # half the burner face lost to the grid: median 56 kW
    vals = [(t, 56.0) for t in range(951)]
    assert not hrr_check(spec, _hrr_csv(tmp_path, vals))["ok"]
    # still growing at t_end: smoothed-peak statistic
    spec2 = ScenarioSpec(**dict(THREE_ROOMS, sim={"t_end_s": 30}))
    vals = [(t, 500 * min(1, t / 103) ** 2) for t in range(31)]
    hc = hrr_check(spec2, _hrr_csv(tmp_path, vals))
    assert hc["statistic"] == "smoothed_peak" and hc["ok"]
