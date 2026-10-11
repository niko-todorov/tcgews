"""
common/basins.py
================
Basin definitions shared across the whole project — the models in `nc/` and the
EWS in `ews/` must agree on domain, grid, and lead times, so this lives in
`common/` rather than being owned by either package.

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Forecast lead times (hours): 0h .. 48h in 6h steps -> 9 leads.
LEAD_HOURS: List[int] = [0, 6, 12, 18, 24, 30, 36, 42, 48]

EARTH_KM_PER_DEG = 111.32  # mean km per degree of latitude


def disc_area_km2(radius_km: float) -> float:
    """Area (km²) of a disc of the given radius.

    The minimum-disturbance-area floor is best thought of physically: the
    smallest pre-genesis disturbance footprint worth issuing a watch on. A disc
    of radius R has area pi*R^2, so expressing the floor as a radius makes the
    number meteorologically meaningful and grid-independent (km², not cells).
    """
    return math.pi * radius_km * radius_km


def radius_km_from_area_km2(area_km2: float) -> float:
    """Inverse of disc_area_km2: the disc radius (km) for a given area.

    Lets the dashboard read any min-area value back as a physical radius.
    """
    return math.sqrt(max(float(area_km2), 0.0) / math.pi)


@dataclass
class BasinConfig:
    """One basin's domain, grid, alert parameters, and data-folder name.

    Set lon/lat bounds and (nx, ny) to the EXACT training domain so the
    dashboard grid lines up cell-for-cell with the model output. `folder` is the
    directory name under data\\ (change it if the dirs aren't NACSGM / WPMM).

    The minimum-disturbance-area floor is defined per basin from a physical
    radius, ``min_disturbance_radius_km``: the smallest disturbance footprint
    worth warning on. ``min_area_km2`` is derived from it in __post_init__
    (rounded to the nearest 1,000 km² for a tidy value), so the floor is a
    principled, grid-independent scale rather than a hand-set number. Passing
    ``min_area_km2`` directly is still supported — the nominal radius is then
    back-filled so the UI can always show a physical equivalent. The floor is
    deliberately per-basin, not per-storm: it is a fixed operational scale, and
    the per-run "exactly one alarm" behaviour is handled by the alert threshold
    (see ews_core.auto_threshold_one_watch), not by moving this floor.
    """
    key: str
    name: str
    folder: str             # directory under data\  e.g. data\NACSGM\ews
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nx: int
    ny: int
    theta: float            # per-basin alert threshold
    min_area_km2: float = 0.0            # components smaller than this are noise
    min_disturbance_radius_km: float = 0.0  # physical basis for the area floor
    hotspots: list = field(default_factory=list)  # (lon, lat, weight), synthetic only

    def __post_init__(self) -> None:
        # A physical radius is the source of truth when given: derive the area
        # floor from it (rounded to the nearest 1,000 km²). Otherwise back-fill
        # the nominal radius from an explicitly-set area so the dashboard can
        # always display a physical equivalent.
        if self.min_disturbance_radius_km and self.min_disturbance_radius_km > 0:
            self.min_area_km2 = float(
                round(disc_area_km2(self.min_disturbance_radius_km) / 1000.0) * 1000)
        elif self.min_area_km2 and self.min_area_km2 > 0:
            self.min_disturbance_radius_km = radius_km_from_area_km2(self.min_area_km2)
        else:
            raise ValueError(
                f"{self.key}: set either min_disturbance_radius_km or min_area_km2")

    def lon_coords(self) -> np.ndarray:
        return np.linspace(self.lon_min, self.lon_max, self.nx, dtype="float32")

    def lat_coords(self) -> np.ndarray:
        # Descending (north -> south), matching typical ERA5 latitude ordering.
        return np.linspace(self.lat_max, self.lat_min, self.ny, dtype="float32")


BASINS = {
    "NACSGM": BasinConfig(
        key="NACSGM", name="North Atlantic / Caribbean / Gulf", folder="NACSGM",
        # 0..360 longitude; full training box, used as the inference canvas.
        # lon 263..305 (-97..-55), lat 11..32  ->  W=169, H=85  (14,365 cells).
        lon_min=263.0, lon_max=305.0, lat_min=11.0, lat_max=32.0,
        nx=169, ny=85,       # 169*85 = 14,365 cells
        theta=0.012,         # calibrated: OOS max-F1 (P0.92/R0.97). 0.019 -> P0.98/R0.90.
        # Min disturbance footprint: a disc of radius ~90 km -> 25,000 km²
        # (~35 cells at 0.25°). Atlantic easterly-wave / MDR genesis scale.
        min_disturbance_radius_km=90.0,
        hotspots=[(275.0, 15.0, 1.0), (270.0, 24.0, 0.7), (290.0, 12.5, 0.55)],
    ),
    "WPMM": BasinConfig(
        key="WPMM", name="Western Pacific", folder="WPMM",
        lon_min=105.0, lon_max=180.0, lat_min=5.0, lat_max=45.0,
        nx=301, ny=161,      # 301*161 = 48,461 cells (matches the WPMM grid)
        # PROVISIONAL, grid-scaled from NACSGM pending F1 calibration.
        # WPMM uses the SAME ocean-cell spatial softmax as NACSGM, so its peak
        # values are on the tiny softmax scale (NOT 0..1). With ~3-4x more ocean
        # cells and more diffuse genesis, the peak sits BELOW NACSGM's, so
        # theta < 0.012. 0.004 ~= 0.012 * (NACSGM_ocean / WPMM_ocean).
        # Replace with the OOS max-F1 value from calibrate_theta.py --basin WPMM
        # once the WP all-years/LOYO peak-probability outputs exist.
        theta=0.004,
        # Larger footprint than the Atlantic: monsoon-trough / TUTT / MJO-forced
        # genesis is more diffuse, so a disc of radius ~113 km -> 40,000 km²
        # (~57 cells at 0.25°).
        min_disturbance_radius_km=113.0,
        hotspots=[(135.0, 15.0, 1.0), (150.0, 20.0, 0.68), (122.0, 10.0, 0.5)],
    ),
}
