"""The ``fds-6.11`` solver target: corpus layout and validation.

The corpus deliberately reuses Foam-Agent's raw file names and FAISS index
names so that ``utils.retrieve_faiss`` and the planner's retrieval logic work
without modification.  Only the manifest differs: it declares
``solver_target`` instead of an OpenFOAM version.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TARGET = "fds-6.11"
FDS_VERSION = "6.11"
MANIFEST_NAME = "foamagent_target.json"

# Same names as openfoam_target so the retrieval layer is untouched.
REQUIRED_RAW_FILES = frozenset(
    {
        "openfoam_case_stats.json",
        "openfoam_commands.txt",
        "openfoam_command_help.txt",
        "openfoam_allrun_scripts.txt",
        "openfoam_tutorials_structure.txt",
        "openfoam_tutorials_details.txt",
    }
)
REQUIRED_FAISS_INDICES = frozenset(
    {
        "openfoam_command_help",
        "openfoam_allrun_scripts",
        "openfoam_tutorials_structure",
        "openfoam_tutorials_details",
    }
)

# What the planner should know about this solver (fed into prompts).
DESCRIPTION = {
    "solver": "fds",
    "case_domain": "fire",
    "case_solver": "fds",
    "input": "a single ASCII namelist file <chid>.fds",
    "run": "mpiexec -n <n_mesh> fds <chid>.fds",
    "outputs": ["<chid>.out", "<chid>_devc.csv", "<chid>_hrr.csv", "<chid>.smv"],
    "namelist_groups": [
        "HEAD", "MESH", "TIME", "MISC", "DUMP", "REAC", "SURF", "RAMP", "OBST",
        "HOLE", "VENT", "DEVC", "PROP", "CTRL", "SLCF", "BNDF", "MATL", "TAIL",
    ],
}


def database_path(config: Any) -> Path:
    configured_path = getattr(config, "database_path", None)
    if not configured_path:
        configured_path = Path(__file__).resolve().parent.parent.parent / "database"
    return Path(configured_path).expanduser().resolve() / TARGET


def model_dir_name(config: Any) -> str:
    model = str(getattr(config, "embedding_model", "") or "Qwen/Qwen3-Embedding-0.6B")
    return model.replace("/", "_").replace(":", "_")


def write_manifest(raw_dir: Path, *, source: str, n_cases: int) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "solver_target": TARGET,
                "solver": "fds",
                "fds_version": FDS_VERSION,
                "source": source,
                "n_cases": n_cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def require_corpus(config: Any) -> None:
    db = database_path(config)
    raw_dir = db / "raw"
    manifest_path = raw_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{TARGET} corpus manifest missing at {manifest_path}. Build it with: "
            "python database/script/build_fds_corpus.py --fds_repo <path to firemodels/fds> "
            "then run the four database/script/faiss_*.py scripts with "
            f"--database_path {db}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("solver_target") != TARGET:
        raise ValueError(f"{manifest_path} declares {manifest.get('solver_target')!r}, not {TARGET!r}.")
    missing_raw = sorted(f for f in REQUIRED_RAW_FILES if not (raw_dir / f).is_file())
    index_root = db / "faiss" / model_dir_name(config)
    missing_idx = sorted(
        i
        for i in REQUIRED_FAISS_INDICES
        if not (index_root / i / "index.faiss").is_file() or not (index_root / i / "index.pkl").is_file()
    )
    if missing_raw or missing_idx:
        parts = []
        if missing_raw:
            parts.append("raw files: " + ", ".join(missing_raw))
        if missing_idx:
            parts.append(f"FAISS indices under {index_root}: " + ", ".join(missing_idx))
        raise FileNotFoundError(f"{TARGET} corpus is incomplete (" + "; ".join(parts) + ").")
