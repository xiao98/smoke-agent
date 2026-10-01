import pathlib
from types import SimpleNamespace

import pytest

import solver_target as st
from fds import target as ft


def cfg(**kw):
    base = dict(solver_target="", openfoam_target="", openfoam_fork="foundation",
                database_path="", embedding_model="Qwen/Qwen3-Embedding-0.6B")
    base.update(kw)
    return SimpleNamespace(**base)


def test_empty_routes_to_openfoam():
    c = cfg()
    assert st.runtime_target(c) == "foundation-v10"
    assert not st.is_fds(c)
    assert st.database_path(c).name == "foundation-v10"


def test_fds_target():
    c = cfg(solver_target="fds-6.11", database_path="/tmp/dbx")
    assert st.is_fds(c)
    assert st.runtime_target(c) == "fds-6.11"
    assert st.database_path(c) == pathlib.Path("/tmp/dbx/fds-6.11")


def test_bad_target_rejected():
    with pytest.raises(ValueError):
        st.normalise_solver_target("fds-9")


def test_require_corpus_missing(tmp_path):
    c = cfg(solver_target="fds-6.11", database_path=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        st.require_corpus(c)


def test_require_corpus_manifest_only(tmp_path):
    c = cfg(solver_target="fds-6.11", database_path=str(tmp_path))
    ft.write_manifest(tmp_path / "fds-6.11" / "raw", source="t", n_cases=0)
    with pytest.raises(FileNotFoundError) as e:
        st.require_corpus(c)
    assert "raw files" in str(e.value)
