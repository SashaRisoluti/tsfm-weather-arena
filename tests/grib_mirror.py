"""Synthetic open-data mirrors: small GRIB files with their index files."""

import json
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np

# A 0.25 degree patch stored north to south: 50 N .. 40 N, 0 E .. 20 E.
NJ, NI = 41, 81
STEPS = (6, 12)
MEMBERS = 3  # control plus two perturbed members

COMPLETE = datetime(2026, 6, 1, 0)  # tp and cp, model version 2, first mirror
FALLBACK = datetime(2026, 6, 1, 12)  # on the second mirror only
VERSION_1 = datetime(2026, 5, 11, 12)  # tp only, model version 1
INCOMPLETE = datetime(2026, 6, 2, 0)  # index lacks one member of tp
NOWHERE = datetime(2026, 6, 3, 0)


def field(param: str, member: int, step: int) -> np.ndarray:
    """Values in mm that encode parameter, member, step and row from the north."""
    base = {"tp": 0.0, "cp": 1000.0, "other": 5000.0}[param] + 100.0 * member + step
    rows = np.arange(NJ, dtype="float64")[:, None]
    return np.broadcast_to(base + 0.01 * rows, (NJ, NI)).copy()


def message(param: str, member: int, step: int, init: datetime, version: int) -> bytes:
    # The reader picks messages by index entry, so a decoy only needs to be valid.
    # Parameter ids of the accumulations in kg m-2, as in the real files.
    param_id = {"tp": 228228, "cp": 228143}.get(param, 228228)
    keys = {
        "Ni": NI,
        "Nj": NJ,
        "latitudeOfFirstGridPointInDegrees": 50.0,
        "latitudeOfLastGridPointInDegrees": 40.0,
        "longitudeOfFirstGridPointInDegrees": 0.0,
        "longitudeOfLastGridPointInDegrees": 20.0,
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
        "productDefinitionTemplateNumber": 11,
        "paramId": param_id,
        "perturbationNumber": member,
        "generatingProcessIdentifier": version,
        "dataDate": int(f"{init:%Y%m%d}"),
        "dataTime": init.hour * 100,
        "stepRange": f"0-{step}",
        "bitsPerValue": 24,
    }
    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        for key, value in keys.items():
            eccodes.codes_set(handle, key, value)
        eccodes.codes_set_values(handle, field(param, member, step).ravel())
        return bytes(eccodes.codes_get_message(handle))
    finally:
        eccodes.codes_release(handle)


def _run(
    mirror: Path,
    init: datetime,
    params: tuple[str, ...],
    version: int,
    drop: tuple[str, int, int] | None = None,
) -> None:
    folder = mirror / f"{init:%Y%m%d}" / f"{init:%H}z" / "aifs-ens" / "0p25" / "enfo"
    folder.mkdir(parents=True)
    for step in STEPS:
        for kind, members in (("cf", [0]), ("pf", list(range(1, MEMBERS)))):
            stem = f"{init:%Y%m%d%H}0000-{step}h-enfo-{kind}"
            blob, lines = b"", []
            for member in members:
                for param in ("other", *params):
                    raw = message(param, member, step, init, version)
                    entry = {
                        "domain": "g",
                        "date": f"{init:%Y%m%d}",
                        "time": f"{init:%H}00",
                        "expver": "0001",
                        "class": "ai",
                        "type": kind,
                        "stream": "enfo",
                        "levtype": "sfc",
                        "step": str(step),
                        "param": param,
                        "model": "aifs-ens",
                        "_offset": len(blob),
                        "_length": len(raw),
                    }
                    if kind == "pf":
                        entry["number"] = str(member)
                    blob += raw
                    if (param, member, step) != drop:
                        lines.append(json.dumps(entry))
            (folder / f"{stem}.grib2").write_bytes(blob)
            (folder / f"{stem}.index").write_text("\n".join(lines) + "\n")


def build_mirrors(root: Path) -> tuple[Path, Path]:
    first, second = root / "first", root / "second"
    _run(first, COMPLETE, ("tp", "cp"), version=2)
    _run(first, VERSION_1, ("tp",), version=1)
    _run(first, INCOMPLETE, ("tp", "cp"), version=2, drop=("tp", 2, 12))
    _run(second, FALLBACK, ("tp", "cp"), version=2)
    return first, second
