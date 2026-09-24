"""Unit registry, conversion, and unit-confusion diagnosis.

Every unit maps to (dimension, factor, offset) such that
``base_value = value * factor + offset``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

HARTREE_EV = 27.211386245988
HARTREE_KCALMOL = 627.5094740631
HARTREE_KJMOL = 2625.4996394799
HARTREE_CM1 = 219474.6313632
HARTREE_J = 4.3597447222071e-18
BOHR_ANGSTROM = 0.529177210903
KCAL_KJ = 4.184
EV_KCALMOL = HARTREE_KCALMOL / HARTREE_EV
EV_KJMOL = HARTREE_KJMOL / HARTREE_EV


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class UnitDef:
    dimension: str
    factor: float
    offset: float = 0.0


_CASE_SENSITIVE: dict[str, UnitDef] = {
    "MHz": UnitDef("frequency", 1e6),
    "mHz": UnitDef("frequency", 1e-3),
    "mV": UnitDef("voltage", 1e-3),
    "MV": UnitDef("voltage", 1e6),
    "ms": UnitDef("time", 1e-3),
    "Å": UnitDef("length", 1.0),
    "A": UnitDef("length", 1.0),  # agents frequently write "A" for angstrom
    "K": UnitDef("temperature", 1.0),
}

_UNITS: dict[str, UnitDef] = {}


def _reg(dimension: str, factor: float, *names: str, offset: float = 0.0) -> None:
    for n in names:
        _UNITS[n] = UnitDef(dimension, factor, offset)


# energy (base: hartree, per particle)
_reg("energy", 1.0, "hartree", "hartrees", "eh", "ha", "e_h")
_reg("energy", 1.0 / HARTREE_EV, "ev", "electronvolt", "electronvolts")
_reg("energy", 1e-3 / HARTREE_EV, "mev")
_reg("energy", 1.0 / HARTREE_KCALMOL, "kcal/mol", "kcalmol-1", "kcal/mole", "kcalmol^-1")
_reg("energy", 1.0 / HARTREE_KJMOL, "kj/mol", "kjmol-1", "kj/mole", "kjmol^-1")
_reg("energy", 1.0 / HARTREE_CM1, "cm-1", "cm^-1", "1/cm", "wavenumber", "wavenumbers")
_reg("energy", 1.0 / HARTREE_J, "j", "joule", "joules")
_reg("energy", 0.5, "ry", "rydberg", "rydbergs")
# length (base: angstrom)
_reg("length", 1.0, "angstrom", "angstroms", "ang", "å", "a")
_reg("length", 10.0, "nm", "nanometer", "nanometers")
_reg("length", 0.01, "pm", "picometer", "picometers")
_reg("length", BOHR_ANGSTROM, "bohr", "bohrs", "a0")
_reg("length", 1e10, "m", "meter", "meters", "metre", "metres")
# molar mass (base: g/mol)
_reg("molar_mass", 1.0, "g/mol", "gmol-1", "gmol^-1", "da", "dalton", "daltons", "amu", "u", "g/mole")
_reg("molar_mass", 1000.0, "kda", "kg/mol", "kilodalton", "kilodaltons")
# density (base: g/cm^3)
_reg("density", 1.0, "g/cm3", "g/cm^3", "g/cc", "g/ml", "gcm-3", "gcm^-3")
_reg("density", 1e-3, "kg/m3", "kg/m^3", "g/l", "kgm-3", "kgm^-3")
# time (base: s)
_reg("time", 1.0, "s", "sec", "second", "seconds")
_reg("time", 1e-3, "millisecond", "milliseconds")
_reg("time", 1e-6, "us", "µs", "μs", "microsecond", "microseconds")
_reg("time", 1e-9, "ns", "nanosecond", "nanoseconds")
_reg("time", 1e-12, "ps", "picosecond", "picoseconds")
_reg("time", 1e-15, "fs", "femtosecond", "femtoseconds")
# voltage (base: V)
_reg("voltage", 1.0, "v", "volt", "volts")
_reg("voltage", 1e3, "kv", "kilovolt", "kilovolts")
# frequency (base: Hz)
_reg("frequency", 1.0, "hz", "hertz")
_reg("frequency", 1e3, "khz", "kilohertz")
_reg("frequency", 1e6, "mhz", "megahertz")
_reg("frequency", 1e9, "ghz", "gigahertz")
# area (base: angstrom^2)
_reg("area", 1.0, "å^2", "a^2", "angstrom^2", "å2", "a2", "angstroms^2", "squareangstroms")
_reg("area", 100.0, "nm^2", "nm2")
# temperature (base: K)
_reg("temperature", 1.0, "k", "kelvin")
_reg("temperature", 1.0, "°c", "degc", "celsius", "ºc", offset=273.15)
# dimensionless
_reg(
    "dimensionless",
    1.0,
    "",
    "dimensionless",
    "unitless",
    "none",
    "-",
    "count",
    "atoms",
    "residues",
    "rings",
    "bonds",
    "centers",
    "stereocenters",
    "donors",
    "acceptors",
    "u/u_lid",
    "u/ulid",
    "normalized",
    "fraction",
    "gwp",
    "co2e",
    "co2-eq",
    "plddt",
    "score",
    "probability",
    "id",
)

_TRANSLATE = str.maketrans({"²": "^2", "³": "^3", "⁻": "^-", "¹": "1", "−": "-", "·": "", "⋅": ""})


def normalize_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    u = unit.strip().translate(_TRANSLATE)
    u = re.sub(r"\s*/\s*", "/", u)
    return u


def lookup(unit: str | None) -> UnitDef:
    raw = normalize_unit(unit)
    if raw in _CASE_SENSITIVE:
        return _CASE_SENSITIVE[raw]
    key = raw.lower()
    if key in _UNITS:
        return _UNITS[key]
    squashed = key.replace(" ", "")
    if squashed in _UNITS:
        return _UNITS[squashed]
    raise UnitError(f"unknown unit: {unit!r}")


def is_known(unit: str | None) -> bool:
    try:
        lookup(unit)
        return True
    except UnitError:
        return False


def dimension(unit: str | None) -> str:
    return lookup(unit).dimension


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float:
    src, dst = lookup(from_unit), lookup(to_unit)
    if src.dimension != dst.dimension:
        raise UnitError(f"cannot convert {from_unit!r} ({src.dimension}) to {to_unit!r} ({dst.dimension})")
    base = value * src.factor + src.offset
    return (base - dst.offset) / dst.factor


# Ratios (got/expected after conversion) that betray a specific, common mistake.
_CONFUSION_RATIOS: list[tuple[float, str]] = [
    (HARTREE_KCALMOL, "hartree <-> kcal/mol mix-up (x627.5)"),
    (HARTREE_KJMOL, "hartree <-> kJ/mol mix-up (x2625.5)"),
    (HARTREE_EV, "hartree <-> eV mix-up (x27.21)"),
    (KCAL_KJ, "kcal <-> kJ mix-up (x4.184)"),
    (EV_KCALMOL, "eV <-> kcal/mol mix-up (x23.06)"),
    (EV_KJMOL, "eV <-> kJ/mol mix-up (x96.49)"),
    (1.0 / BOHR_ANGSTROM, "bohr <-> angstrom mix-up (x1.890)"),
    (10.0, "factor-of-10 slip (e.g. angstrom vs nm)"),
    (1000.0, "SI-prefix slip (x1000, e.g. kJ vs J, ms vs s)"),
    (2.0 * math.pi, "2*pi slip (angular vs ordinary frequency)"),
    (2.0, "factor-of-2 slip (e.g. hartree vs rydberg, radius vs diameter)"),
]


def diagnose_ratio(got: float, expected: float, rel: float = 0.02) -> str | None:
    """Return a human-readable guess at the unit/sign mistake, or None."""
    if expected == 0 or got == 0 or not math.isfinite(got):
        return None
    ratio = got / expected
    if abs(ratio + 1.0) <= rel:
        return "sign flip"
    sign = "sign flip + " if ratio < 0 else ""
    r = abs(ratio)
    for factor, label in _CONFUSION_RATIOS:
        if abs(r / factor - 1.0) <= rel:
            return f"{sign}{label}, reported too large"
        if abs(r * factor - 1.0) <= rel:
            return f"{sign}{label}, reported too small"
    return None
