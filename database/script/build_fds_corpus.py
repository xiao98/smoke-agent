#!/usr/bin/env python
"""Build the raw corpus for the ``fds-6.11`` target from the firemodels/fds repo.

Writes, under ``<database>/fds-6.11/raw/``, files with the same names and tag
layout Foam-Agent expects for OpenFOAM, so the existing ``faiss_*.py`` scripts
and the planner's retrieval code work unchanged:

    openfoam_tutorials_details.txt    <case_begin><index>..</index><directory_structure>..</directory_structure><tutorials>..</tutorials></case_end>
    openfoam_tutorials_structure.txt  <case_begin><index>..</index><directory_structure>..</directory_structure></case_end>
    openfoam_allrun_scripts.txt       <case_begin><index>..</index><directory_structure>..</directory_structure><allrun_script>..</allrun_script></case_end>
    openfoam_command_help.txt         <command_begin><command>..</command><help_text>..</help_text></command_end>
    openfoam_commands.txt             one command per line
    openfoam_case_stats.json          counts
    foamagent_target.json             manifest (solver_target = fds-6.11)

Usage:
    python database/script/build_fds_corpus.py --fds_repo ~/work/fds [--database_path database]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))

from fds.namelist import parse, records_of  # noqa: E402
from fds.target import TARGET, write_manifest  # noqa: E402

# Validation directories relevant to building smoke movement, with a category label.
WHITELIST: dict[str, str] = {
    "ATF_Corridors": "corridor",
    "Lattimer_Corridor_Ceiling": "corridor",
    "NBS_Multi-Room": "multi-room",
    "NIST_NRC": "compartment",
    "NIST_NRC_Corner_Effects": "compartment",
    "NIST_NRC_Transient_Combustibles": "compartment",
    "NIST_RSE_1994": "compartment",
    "NIST_RSE_2007": "compartment",
    "NIST_FSE_2008": "compartment",
    "Steckler_Compartment": "compartment",
    "Utiskul_Compartment": "compartment",
    "LLNL_Enclosure": "compartment",
    "FM_SNL": "compartment",
    "PRISME": "compartment",
    "VTT": "hall",
    "USN_Hangars": "hall",
    "FM_FPRF_Datacenter": "datacenter",
    "UL_NIJ_Houses": "house",
    "NIST_Smoke_Alarms": "house",
    "NRCC_Smoke_Tower": "stairwell",
    "Bryant_Doorway": "doorway",
    "UL_NIST_Vents": "vent",
    "NIST_Vent_Study": "vent",
    "Restivo_Experiment": "room",
    "Vettori_Flat_Ceiling": "ceiling-jet",
    "Vettori_Sloped_Ceiling": "ceiling-jet",
    "Harrison_Spill_Plumes": "atrium",
    "Beyler_Hood": "hood",
    "UL_NFPRF": "warehouse",
    "FAA_Cargo_Compartments": "cargo",
    "WTC": "office",
    "Arup_Tunnel": "tunnel",
    "Memorial_Tunnel": "tunnel",
    "CSTB_Tunnel": "tunnel",
    "FHWA_Tunnel": "tunnel",
    "SWJTU_Tunnels": "tunnel",
    "Wu_Bakar_Tunnels": "tunnel",
    "McCaffrey_Plume": "plume",
    "Heskestad_Flame_Height": "plume",
}

COMMANDS = [
    ("fds <chid>.fds", "Run FDS on one mesh in the current directory. Writes <chid>.out (progress, warnings, ERROR lines), <chid>.smv, <chid>_devc.csv, <chid>_hrr.csv and slice/boundary files. Source FDS6VARS.sh first."),
    ("mpiexec -n <N> fds <chid>.fds", "Run FDS with N MPI processes; N must equal the number of &MESH records (or a divisor with MPI_PROCESS assignments). Set OMP_NUM_THREADS=1 for pure MPI runs."),
    ("fds <chid>.fds with &TIME T_END=0 /", "Setup-only run: reads the input, builds meshes and obstructions, writes <chid>.smv and stops. Use it as a fast syntax and geometry check before a real run."),
    ("smokeview -runscript <chid>", "Run Smokeview unattended with the script <chid>.ssf (RENDERDIR, LOADSLICE <label> <dir> <coord>, SETTIMEVAL, RENDERONCE, UNLOADALL). Headless: xvfb-run -a -s '-screen 0 1280x800x24' smokeview -runscript <chid>. Slice labels come from <chid>.smv, e.g. 'SOOT VISIBILITY', 'TEMPERATURE'."),
    ("fds2ascii", "Convert slice/boundary/plot3d binary output to ASCII text (interactive prompts)."),
]


def rel_structure(path: Path) -> str:
    return f"{path.name}\n"


def summarize(records) -> str:
    counts = Counter(r.group for r in records)
    return ", ".join(f"{g}:{n}" for g, n in sorted(counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fds_repo", required=True, help="path to a checkout of firemodels/fds")
    ap.add_argument("--database_path", default=str(HERE.parent), help="database root (default: ../ from this script)")
    ap.add_argument("--max_files_per_dir", type=int, default=12, help="cap per validation directory to keep the index balanced")
    args = ap.parse_args()

    repo = Path(args.fds_repo).expanduser().resolve()
    val = repo / "Validation"
    raw = Path(args.database_path).expanduser().resolve() / TARGET / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    details, structure, allrun = [], [], []
    param_examples: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    group_counts: Counter = Counter()
    n_cases = 0
    per_dir: Counter = Counter()

    for dirname, category in WHITELIST.items():
        d = val / dirname / "FDS_Input_Files"
        if not d.is_dir():
            print(f"skip (missing): {dirname}")
            continue
        files = sorted(d.glob("*.fds"))[: args.max_files_per_dir]
        for f in files:
            text = f.read_text(errors="replace")
            recs = parse(text)
            head = records_of(recs, "HEAD")
            chid = head[0].get("CHID", f.stem) if head else f.stem
            title = head[0].get("TITLE", "") if head else ""
            index = (
                "<index>\n"
                f"case name: {chid}\n"
                "case domain: fire\n"
                f"case category: {category}\n"
                "case solver: fds\n"
                f"experiment: {dirname.replace('_', ' ')}\n"
                f"title: {title}\n"
                f"records: {summarize(recs)}\n"
                "</index>\n"
            )
            dir_struct = f"<directory_structure>\n{f.name}\n</directory_structure>\n"
            details.append(f"<case_begin>\n{index}{dir_struct}<tutorials>\n{text.strip()}\n</tutorials>\n</case_end>\n")
            structure.append(f"<case_begin>\n{index}{dir_struct}</case_end>\n")
            n_mesh = len(records_of(recs, "MESH"))
            script = f"mpiexec -n {max(1, n_mesh)} fds {f.name}"
            allrun.append(f"<case_begin>\n{index}{dir_struct}<allrun_script>\n{script}\n</allrun_script>\n</case_end>\n")
            for r in recs:
                group_counts[r.group] += 1
                for k, vs in r.params.items():
                    param_examples[r.group][k][",".join(vs)[:60]] += 1
            n_cases += 1
            per_dir[dirname] += 1

    (raw / "openfoam_tutorials_details.txt").write_text("".join(details), encoding="utf-8")
    (raw / "openfoam_tutorials_structure.txt").write_text("".join(structure), encoding="utf-8")
    (raw / "openfoam_allrun_scripts.txt").write_text("".join(allrun), encoding="utf-8")

    # Command help: run commands + one entry per namelist group with observed parameters.
    help_entries = []
    for cmd, txt in COMMANDS:
        help_entries.append(f"<command_begin>\n<command>\n{cmd}\n</command>\n<help_text>\n{txt}\n</help_text>\n</command_end>\n")
    for group, params in sorted(param_examples.items()):
        lines = [f"&{group} namelist record. Seen in {group_counts[group]} records of the validation corpus. Parameters observed (with the most common example values):"]
        for k, examples in sorted(params.items(), key=lambda kv: -sum(kv[1].values())):
            top = ", ".join(f"{v}" for v, _ in examples.most_common(3))
            lines.append(f"  {k}: e.g. {top}")
        help_entries.append(f"<command_begin>\n<command>\n&{group}\n</command>\n<help_text>\n" + "\n".join(lines) + "\n</help_text>\n</command_end>\n")
    (raw / "openfoam_command_help.txt").write_text("".join(help_entries), encoding="utf-8")
    (raw / "openfoam_commands.txt").write_text("\n".join(c for c, _ in COMMANDS) + "\n" + "\n".join(f"&{g}" for g in sorted(group_counts)) + "\n", encoding="utf-8")
    (raw / "openfoam_case_stats.json").write_text(
        json.dumps({"n_cases": n_cases, "per_directory": per_dir, "namelist_groups": group_counts}, indent=2), encoding="utf-8"
    )
    write_manifest(raw, source=f"firemodels/fds Validation ({len(per_dir)} dirs)", n_cases=n_cases)
    print(f"wrote {n_cases} cases from {len(per_dir)} directories to {raw}")
    print("groups:", dict(group_counts.most_common(12)))


if __name__ == "__main__":
    main()
