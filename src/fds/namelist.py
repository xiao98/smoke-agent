"""FDS namelist reader/writer.

An FDS input file is free text in which records of the form

    &GROUP KEY=value, KEY2=v1,v2,v3, ... /

are embedded. Everything outside ``&...`` ``/`` pairs is ignored by FDS.
A ``/`` inside a quoted string does not end the record.

This module keeps values as raw strings (quotes stripped for strings) so that
``dump(parse(text))`` is semantically identical to ``text`` for FDS, without
pretending to know every FDS parameter type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

__all__ = ["Record", "parse", "dump", "records_of"]

_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=")


@dataclass
class Record:
    group: str
    params: dict[str, list[str]] = field(default_factory=dict)

    def get(self, key: str, default=None):
        v = self.params.get(key.upper())
        if v is None:
            return default
        return v[0] if len(v) == 1 else v

    def floats(self, key: str) -> list[float]:
        return [float(x) for x in self.params.get(key.upper(), [])]


def _find_records(text: str) -> list[tuple[str, str]]:
    """Return (group, body) pairs. Handles '/' inside quotes."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        amp = text.find("&", i)
        if amp < 0:
            break
        m = re.match(r"&\s*([A-Za-z_][A-Za-z0-9_]*)", text[amp:])
        if not m:
            i = amp + 1
            continue
        group = m.group(1).upper()
        j = amp + m.end()
        in_quote: str | None = None
        start_body = j
        while j < n:
            c = text[j]
            if in_quote:
                if c == in_quote:
                    in_quote = None
            elif c in ("'", '"'):
                in_quote = c
            elif c == "/":
                break
            j += 1
        out.append((group, text[start_body:j]))
        i = j + 1
    return out


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts, buf, in_quote = [], [], None
    for c in s:
        if in_quote:
            buf.append(c)
            if c == in_quote:
                in_quote = None
        elif c in ("'", '"'):
            in_quote = c
            buf.append(c)
        elif c == sep:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    parts.append("".join(buf))
    return parts


def _parse_body(body: str) -> dict[str, list[str]]:
    # locate KEY= positions outside quotes
    keys: list[tuple[int, int, str]] = []  # (start, end_of_eq, key)
    in_quote = None
    i = 0
    while i < len(body):
        c = body[i]
        if in_quote:
            if c == in_quote:
                in_quote = None
            i += 1
            continue
        if c in ("'", '"'):
            in_quote = c
            i += 1
            continue
        m = _KEY_RE.match(body, i)
        if m and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] in "_.")):
            keys.append((m.start(), m.end(), m.group(1).upper()))
            i = m.end()
            continue
        i += 1
    params: dict[str, list[str]] = {}
    for idx, (_, vstart, key) in enumerate(keys):
        vend = keys[idx + 1][0] if idx + 1 < len(keys) else len(body)
        raw = body[vstart:vend]
        vals = []
        for tok in _split_top_level(raw):
            tok = tok.strip()
            if not tok:
                continue
            # space-separated values without commas (rare but legal)
            if not (tok.startswith("'") or tok.startswith('"')) and " " in tok:
                vals.extend(t for t in tok.split() if t)
            else:
                vals.append(tok)
        vals = [_unquote(v) for v in vals]
        params.setdefault(key, []).extend(vals)
    return params


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        q = v[0]
        return v[1:-1].replace(q + q, q)
    return v


def parse(text: str) -> list[Record]:
    return [Record(g, _parse_body(b)) for g, b in _find_records(text)]


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")
_LOGICAL = {".TRUE.", ".FALSE.", "T", "F"}


def _fmt(v: str) -> str:
    if _NUM_RE.match(v) or v.upper() in _LOGICAL:
        return v
    return "'" + v.replace("'", "''") + "'"


def dump(records: list[Record]) -> str:
    lines = []
    for r in records:
        if not r.params:
            lines.append(f"&{r.group} /")
            continue
        parts = [f"{k}={','.join(_fmt(v) for v in vs)}" for k, vs in r.params.items()]
        lines.append(f"&{r.group} " + ", ".join(parts) + " /")
    return "\n".join(lines) + "\n"


def records_of(records: list[Record], group: str) -> list[Record]:
    g = group.upper()
    return [r for r in records if r.group == g]
