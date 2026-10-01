"""ScenarioSpec and CriteriaResult: the two data contracts of SmokeAgent.

Units everywhere: m, s, kW, °C.  ``XB`` is the FDS six-tuple
``x0, x1, y0, y1, z0, z1``.  The LLM fills a ScenarioSpec; everything after
that is deterministic code (see fds/writer.py, fds/criteria.py).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

XB = list[float]


def _check_xb(xb: XB) -> XB:
    if len(xb) != 6:
        raise ValueError("XB needs 6 numbers: x0,x1,y0,y1,z0,z1")
    for lo, hi in ((0, 1), (2, 3), (4, 5)):
        if xb[hi] < xb[lo]:
            raise ValueError(f"XB lower bound exceeds upper bound: {xb}")
    return xb


class Hole(BaseModel):
    xb: XB
    _v = field_validator("xb")(_check_xb)


class Obstruction(BaseModel):
    id: str
    xb: XB
    surf_id: str = "INERT"
    holes: list[Hole] = Field(default_factory=list)
    _v = field_validator("xb")(_check_xb)


class Opening(BaseModel):
    """A permanently open boundary (door/window to outside) -> &VENT SURF_ID='OPEN'."""

    id: str
    xb: XB
    type: Literal["open"] = "open"
    _v = field_validator("xb")(_check_xb)


class Domain(BaseModel):
    x: list[float]
    y: list[float]
    z: list[float]

    @model_validator(mode="after")
    def _ranges(self):
        for name, r in (("x", self.x), ("y", self.y), ("z", self.z)):
            if len(r) != 2 or r[1] <= r[0]:
                raise ValueError(f"domain.{name} must be [min, max] with max > min")
        return self

    @property
    def xb(self) -> XB:
        return [self.x[0], self.x[1], self.y[0], self.y[1], self.z[0], self.z[1]]


class Building(BaseModel):
    domain: Domain
    obstructions: list[Obstruction] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)


class Fuel(BaseModel):
    name: str = "POLYURETHANE"
    soot_yield: float = Field(0.10, gt=0, lt=0.5)
    co_yield: float = Field(0.04, ge=0, lt=0.5)
    heat_of_combustion: float = Field(25000.0, gt=1000)  # kJ/kg
    # Simple C/H/O/N formula for &REAC (polyurethane-like default)
    c: float = 1.0
    h: float = 1.75
    o: float = 0.25
    n: float = 0.065


Growth = Literal["slow", "medium", "fast", "ultrafast", "constant"]

# NFPA 72 t-squared growth coefficients alpha [kW/s^2]
ALPHA = {"slow": 0.00293, "medium": 0.01172, "fast": 0.0469, "ultrafast": 0.1876}


class Fire(BaseModel):
    location_xb: XB  # burner footprint (z0 == z1 on a floor/obstruction top)
    hrr_peak_kw: float = Field(..., gt=0)
    growth: Growth = "fast"
    # Explicit time to reach the peak (s). Overrides the NFPA alpha of `growth`;
    # use it when the source states "peak reached at N s".
    time_to_peak_s: Optional[float] = Field(None, gt=0)
    fuel: Fuel = Field(default_factory=Fuel)
    _v = field_validator("location_xb")(_check_xb)

    @property
    def area_m2(self) -> float:
        x0, x1, y0, y1, _, _ = self.location_xb
        return (x1 - x0) * (y1 - y0)

    def t_peak(self) -> float:
        if self.growth == "constant":
            return 0.0
        if self.time_to_peak_s:
            return float(self.time_to_peak_s)
        return (self.hrr_peak_kw / ALPHA[self.growth]) ** 0.5


class Exhaust(BaseModel):
    id: str
    xb: XB  # a planar face (one of the pairs equal) on a boundary or obstruction
    volume_flow_m3s: float = Field(..., gt=0)
    activation_s: float = Field(0.0, ge=0)
    _v = field_validator("xb")(_check_xb)


class Makeup(BaseModel):
    id: str
    opening: str  # id of an Opening


class Curtain(BaseModel):
    id: str
    xb: XB
    _v = field_validator("xb")(_check_xb)


class SmokeControl(BaseModel):
    exhaust: list[Exhaust] = Field(default_factory=list)
    makeup: list[Makeup] = Field(default_factory=list)
    curtains: list[Curtain] = Field(default_factory=list)


class EscapePath(BaseModel):
    id: str
    polyline: list[list[float]]  # [[x, y, z], ...] z = evaluation height
    rset_s: Optional[float] = Field(None, gt=0)

    @field_validator("polyline")
    @classmethod
    def _pts(cls, v):
        if len(v) < 2 or any(len(p) != 3 for p in v):
            raise ValueError("polyline needs >= 2 points of [x, y, z]")
        return v


class Mesh(BaseModel):
    cell_size: float | Literal["auto"] = "auto"
    max_cells_per_mesh: int = 1_000_000


class Sim(BaseModel):
    t_end_s: float = Field(600.0, gt=0)
    dt_devc_s: float = 1.0
    dt_slcf_s: float = 5.0


class ScenarioSpec(BaseModel):
    chid: str = Field(..., pattern=r"^[A-Za-z0-9_\-]{1,40}$")
    title: str = ""
    building: Building
    fire: Fire
    smoke_control: SmokeControl = Field(default_factory=SmokeControl)
    escape_paths: list[EscapePath] = Field(default_factory=list)
    mesh: Mesh = Field(default_factory=Mesh)
    sim: Sim = Field(default_factory=Sim)
    thresholds_profile: str = "fr_default"

    @model_validator(mode="after")
    def _inside_domain(self):
        d = self.building.domain.xb

        def inside(xb: XB) -> bool:
            return (d[0] <= xb[0] and xb[1] <= d[1] and d[2] <= xb[2] and xb[3] <= d[3]
                    and d[4] <= xb[4] and xb[5] <= d[5])

        for o in self.building.obstructions:
            if not inside(o.xb):
                raise ValueError(f"obstruction {o.id} outside domain")
        if not inside(self.fire.location_xb):
            raise ValueError("fire outside domain")
        for p in self.escape_paths:
            for x, y, z in p.polyline:
                if not (d[0] <= x <= d[1] and d[2] <= y <= d[3] and d[4] <= z <= d[5]):
                    raise ValueError(f"escape path {p.id} point outside domain")
        ids = {m.opening for m in self.smoke_control.makeup}
        known = {o.id for o in self.building.openings}
        if ids - known:
            raise ValueError(f"makeup refers to unknown opening(s): {sorted(ids - known)}")
        return self


# ----------------------------------------------------------------------------
# CriteriaResult
# ----------------------------------------------------------------------------

class QuantityExceed(BaseModel):
    first_exceed_s: Optional[float]  # None = never exceeded within t_end
    worst_point: Optional[list[float]] = None
    worst_value: Optional[float] = None


class PathResult(BaseModel):
    id: str
    rset_s: Optional[float]
    aset_s: float
    censored: bool  # True when ASET == t_end because nothing was exceeded
    margin: Optional[float]  # aset / rset
    passed: Optional[bool]
    quantities: dict[str, QuantityExceed]


class HrrCheck(BaseModel):
    peak_kw_set: float
    peak_kw_sim: float
    ok: bool


class CriteriaResult(BaseModel):
    chid: str
    thresholds_profile: str
    t_end_s: float
    paths: list[PathResult]
    hrr_check: HrrCheck
