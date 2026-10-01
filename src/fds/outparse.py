"""Parse FDS .out / stdout into structured errors with stable fingerprints."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# (fingerprint, regex, typical cause, suggested fix)
RULES: list[tuple[str, str, str, str]] = [
    ("mesh_overlap", r"ERROR.*MESH.*overlap", "two &MESH blocks overlap", "recompute mesh split so blocks touch but do not overlap"),
    ("mesh_misaligned", r"ERROR.*MESH.*(align|abut|coinci)", "mesh boundaries not aligned", "make neighbouring MESH XB faces coincide on cell boundaries"),
    ("surf_undefined", r"ERROR.*SURF.*(not found|undefined|does not exist)", "a SURF_ID is referenced but no &SURF with that ID exists", "add the missing &SURF or fix the ID"),
    ("ramp_undefined", r"ERROR.*RAMP.*(not found|undefined|does not exist)", "RAMP_Q references a missing &RAMP", "add &RAMP lines with that ID"),
    ("devc_undefined", r"ERROR.*DEVC.*(not found|undefined|does not exist)", "DEVC_ID references a missing device", "add the &DEVC used as a trigger"),
    ("spec_undefined", r"ERROR.*SPEC.*(not found|undefined|does not exist)", "SPEC_ID not available (no &REAC defined?)", "define &REAC so combustion species exist"),
    ("obst_outside", r"WARNING.*OBST.*outside", "obstruction outside the computational domain", "shrink the obstruction or enlarge the domain"),
    ("vent_outside", r"(ERROR|WARNING).*VENT.*(outside|not on|does not lie)", "vent not on a mesh boundary or obstruction face", "move the VENT onto a boundary plane"),
    ("instability", r"Numerical Instability", "numerical instability (coarse mesh, huge HRRPUA, bad vent velocity)", "refine the mesh one step or enlarge the fire area"),
    ("cfl_tiny_dt", r"Time Step:\s*\d+,\s*Simulation Time:[^\n]*\n(?:.*\n){0,3}.*DT\s*=\s*[0-9.]+E-0[5-9]", "time step collapsed", "same as instability"),
    ("namelist_syntax", r"ERROR.*(namelist|Namelist|NAMELIST|input file)", "namelist syntax error", "check quotes, commas and the closing slash of the reported record"),
    ("generic_error", r"^\s*ERROR[:\s].*$", "unclassified FDS error", "read the error line"),
]


@dataclass
class FdsError:
    fingerprint: str
    message: str
    cause: str
    fix: str


def errors(out_text: str, stdout_text: str = "", timed_out: bool = False, returncode: int = 0) -> list[FdsError]:
    text = out_text + "\n" + stdout_text
    found: list[FdsError] = []
    seen: set[str] = set()
    for fp, rx, cause, fix in RULES:
        for m in re.finditer(rx, text, re.M | re.I):
            msg = m.group(0).strip()[:300]
            key = fp + "|" + msg
            if key in seen:
                continue
            seen.add(key)
            found.append(FdsError(fp, msg, cause, fix))
    if timed_out:
        found.append(FdsError("timeout", "run exceeded the time limit", "mesh too fine or T_END too long for the time budget", "ask the user; do not change physics automatically"))
    if returncode not in (0, None) and not found:
        found.append(FdsError("nonzero_exit", f"fds exited with code {returncode}", "crash without an ERROR line", "inspect stdout tail"))
    # generic_error only when nothing more specific matched the same line
    specific = {e.message for e in found if e.fingerprint != "generic_error"}
    return [e for e in found if not (e.fingerprint == "generic_error" and e.message in specific)]


def fingerprint(errs: list[FdsError]) -> str:
    key = "|".join(sorted({e.fingerprint for e in errs}))
    return hashlib.sha1(key.encode()).hexdigest()[:12] if key else ""


def to_error_logs(errs: list[FdsError]) -> list[str]:
    """Shape expected by Foam-Agent's reviewer: a list of strings."""
    return [f"[{e.fingerprint}] {e.message}\n  cause: {e.cause}\n  fix: {e.fix}" for e in errs]
