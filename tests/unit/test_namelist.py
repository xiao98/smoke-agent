import glob
import os
import pathlib

import pytest

from fds.namelist import Record, dump, parse, records_of

VAL = pathlib.Path.home() / "work" / "fds" / "Validation"
FILES = sorted(glob.glob(str(VAL / "*" / "FDS_Input_Files" / "*.fds")))


def test_basic_record():
    recs = parse("&HEAD CHID='a_b', TITLE='slash / inside, comma' /\n&TIME T_END=30. /")
    assert recs[0].group == "HEAD"
    assert recs[0].get("CHID") == "a_b"
    assert recs[0].get("TITLE") == "slash / inside, comma"
    assert recs[1].floats("T_END") == [30.0]


def test_arrays_and_logicals():
    recs = parse("&MESH IJK=10,10,10, XB=-0.3,0.7,-0.4,0.6,0.0,1.0 /\n&MISC HUMIDITY=40., TMPA=20. /\n&DEVC ID='x', QUANTITY='TEMPERATURE', XYZ=1,2,3, STATISTICS_END=.TRUE. /")
    m = recs[0]
    assert m.floats("XB") == [-0.3, 0.7, -0.4, 0.6, 0.0, 1.0]
    assert m.params["IJK"] == ["10", "10", "10"]
    assert recs[2].get("STATISTICS_END") == ".TRUE."


def test_indexed_keys_and_multiline():
    txt = """&SURF ID='wall',
      MATL_ID(1,1)='gyp', MATL_ID(2,1)='ins',
      THICKNESS(1:2)=0.01,0.05 /"""
    r = parse(txt)[0]
    assert r.params["MATL_ID(1,1)"] == ["gyp"]
    assert r.params["THICKNESS(1:2)"] == ["0.01", "0.05"]


def test_dump_roundtrip_small():
    recs = parse("&HEAD CHID='x', TITLE=\"it's\" /\n&TAIL /")
    again = parse(dump(recs))
    assert [(r.group, r.params) for r in again] == [(r.group, r.params) for r in recs]


@pytest.mark.skipif(not FILES, reason="validation files not present")
@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(p) for p in FILES])
def test_roundtrip_validation_files(path):
    text = pathlib.Path(path).read_text(errors="replace")
    recs = parse(text)
    assert records_of(recs, "HEAD"), path
    assert records_of(recs, "TAIL"), path
    again = parse(dump(recs))
    assert [(r.group, r.params) for r in again] == [(r.group, r.params) for r in recs]
