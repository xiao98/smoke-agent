import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "firebench"))
from judge import load_nist_uncertainty, nist_band  # noqa: E402


def test_nist_band_matches_table_16_1():
    tab = load_nist_uncertainty()
    q = tab["quantities"]["Ceiling Jet Temperature"]
    assert (q["sigma_e"], q["sigma_m"], q["bias"]) == (0.07, 0.13, 1.08)
    # ceiling jet: sigma = sqrt(0.07^2 + 0.13^2) = 0.148; 2-sigma band around bias 1.08 = x0.80 .. x1.45
    b = nist_band(100.0, 108.0, "Ceiling Jet Temperature", tab)
    assert b["ok"] and b["z"] == pytest.approx(0.0, abs=1e-9)
    assert nist_band(100.0, 140.0, "Ceiling Jet Temperature", tab)["ok"]        # +40 %: inside (bias 1.08 x e^{0.295})
    assert not nist_band(100.0, 150.0, "Ceiling Jet Temperature", tab)["ok"]    # +50 %: outside
    assert nist_band(100.0, 82.0, "Ceiling Jet Temperature", tab)["ok"]         # -18 %: inside
    assert not nist_band(100.0, 75.0, "Ceiling Jet Temperature", tab)["ok"]     # -25 %: outside
    # HGL depth is reported as a negative drop (min - initial): same sign is fine, opposite sign fails
    assert nist_band(-1.5, -1.6, "HGL Depth", tab)["ok"]
    assert not nist_band(-1.5, 0.3, "HGL Depth", tab)["ok"]
    # unknown quantity -> caller falls back to the absolute tolerance
    assert nist_band(1.0, 1.0, "Sprinkler Actuations", tab) is None
