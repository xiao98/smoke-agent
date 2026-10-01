"""Run FDS cases (module 2).  Local subprocess now; Slurm later."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fds.namelist import parse, records_of

_STEP_RE = re.compile(r"Time Step:\s*(\d+),\s*Simulation Time:\s*([0-9.Ee+-]+)")


@dataclass
class RunResult:
    chid: str
    case_dir: Path
    returncode: int
    wall_s: float
    timed_out: bool
    out_tail: str = ""
    stdout_tail: str = ""

    @property
    def out_path(self) -> Path:
        return self.case_dir / f"{self.chid}.out"

    def to_meta(self) -> dict:
        return {"chid": self.chid, "returncode": self.returncode, "wall_s": round(self.wall_s, 1), "timed_out": self.timed_out}


def n_meshes(fds_text: str) -> int:
    return max(1, len(records_of(parse(fds_text), "MESH")))


def _progress_writer(case_dir: Path, chid: str, stop: threading.Event, every_s: float = 10.0) -> None:
    stdout = case_dir / f"{chid}.stdout"
    while not stop.wait(every_s):
        _write_progress(case_dir, chid, stdout)
    _write_progress(case_dir, chid, stdout)


def _write_progress(case_dir: Path, chid: str, stdout: Path) -> None:
    try:
        text = stdout.read_text(errors="replace")[-20000:] if stdout.exists() else ""
        m = None
        for m in _STEP_RE.finditer(text):
            pass
        prog = {"chid": chid, "step": int(m.group(1)) if m else 0, "sim_time_s": float(m.group(2)) if m else 0.0,
                "updated": time.time()}
        (case_dir / "progress.json").write_text(json.dumps(prog))
    except OSError:
        pass


def run(case_dir: Path, chid: str, nproc: int | None = None, timeout_s: int = 3600,
        omp_threads: int = 1, fds_bin: str = "fds", mpiexec_bin: str = "mpiexec") -> RunResult:
    case_dir = Path(case_dir)
    fds_file = case_dir / f"{chid}.fds"
    if nproc is None:
        nproc = n_meshes(fds_file.read_text(errors="replace"))
    nproc = max(1, min(nproc, os.cpu_count() or 1))
    env = dict(os.environ, OMP_NUM_THREADS=str(omp_threads))
    cmd = [mpiexec_bin, "-n", str(nproc), fds_bin, fds_file.name]
    stop = threading.Event()
    th = threading.Thread(target=_progress_writer, args=(case_dir, chid, stop), daemon=True)
    th.start()
    t0 = time.time()
    timed_out = False
    with open(case_dir / f"{chid}.stdout", "w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=case_dir, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
        except FileNotFoundError as exc:
            rc = 127
            fh.write(f"launch failed: {exc}\n")
    stop.set()
    th.join(timeout=2)
    wall = time.time() - t0
    out_tail = (case_dir / f"{chid}.out").read_text(errors="replace")[-8000:] if (case_dir / f"{chid}.out").exists() else ""
    stdout_tail = (case_dir / f"{chid}.stdout").read_text(errors="replace")[-4000:]
    res = RunResult(chid, case_dir, rc, wall, timed_out, out_tail, stdout_tail)
    (case_dir / "meta.json").write_text(json.dumps(res.to_meta(), indent=2))
    return res


def run_many(jobs: list[tuple[Path, str]], max_parallel: int = 2, **kw) -> list[RunResult]:
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        return list(ex.map(lambda j: run(j[0], j[1], **kw), jobs))
