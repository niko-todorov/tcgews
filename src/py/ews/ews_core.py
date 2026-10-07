"""
ews/ews_core.py
===============
EWS output contract + alerting. Imports basin config from `common` and writes
NetCDF outputs to data\\{basin}\\ews\\.

Format split (unchanged elsewhere):
    ERA5 input  -> NetCDF      (read by the preprocessing)
    ML features -> CSV         (the ncio.py pipeline in nc\\, untouched)
    EWS output  -> NetCDF      (defined here; consumed by dashboard.py)

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage

from basins import BASINS, BasinConfig, LEAD_HOURS, EARTH_KM_PER_DEG
from paths import ews_output_dir
from provenance import stamp_provenance


# --------------------------------------------------------------------------- #
#  NetCDF output schema (writer)
# --------------------------------------------------------------------------- #
def build_dataset(
    prob: np.ndarray,
    land_mask: np.ndarray,
    basin: BasinConfig,
    analysis_time: dt.datetime,
    model_id: str = "TCGUNetConvLSTM",
) -> xr.Dataset:
    """CF-style Dataset for one run. prob: (n_lead, ny, nx); land_mask: (ny, nx)."""
    prob = np.asarray(prob, dtype="float32")
    expected = (len(LEAD_HOURS), basin.ny, basin.nx)
    if prob.shape != expected:
        raise ValueError(f"prob shape {prob.shape} != expected {expected} for {basin.key}")

    return xr.Dataset(
        data_vars=dict(
            prob=(("lead", "lat", "lon"), prob,
                  {"long_name": "tropical cyclogenesis probability", "units": "1",
                   "valid_range": np.array([0.0, 1.0], "float32")}),
            land_mask=(("lat", "lon"), np.asarray(land_mask, "int8"),
                       {"long_name": "land-sea mask",
                        "flag_values": np.array([0, 1], "int8"),
                        "flag_meanings": "ocean land"}),
        ),
        coords=dict(
            lead=("lead", np.asarray(LEAD_HOURS, "int16"),
                  {"long_name": "forecast lead time", "units": "1",
                   "description": "forecast lead time in hours (0..48)"}),
            lat=("lat", basin.lat_coords(),
                 {"long_name": "latitude", "units": "degrees_north"}),
            lon=("lon", basin.lon_coords(),
                 {"long_name": "longitude", "units": "degrees_east"}),
        ),
        attrs=dict(
            Conventions="CF-1.8",
            title=f"TCG early warning output — {basin.name}",
            basin=basin.key, basin_name=basin.name,
            analysis_time=analysis_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            alert_threshold=float(basin.theta),
            min_area_km2=float(basin.min_area_km2),
            model=model_id,
            created=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )


def output_filename(basin_key: str, analysis_time: dt.datetime,
                    label: Optional[str] = None) -> str:
    tag = f"_{label}" if label else ""
    return f"{basin_key}{tag}_{analysis_time:%Y%m%dT%H}Z.nc"


def write_ews_netcdf(
    prob: np.ndarray,
    land_mask: np.ndarray,
    basin: BasinConfig,
    analysis_time: dt.datetime,
    out_dir: Optional[Path] = None,
    model_id: str = "TCGUNetConvLSTM",
    era5_stream: str = "unknown",
    label: Optional[str] = None,
    truth: Optional[tuple] = None,
) -> Path:
    """Write one run to data\\{basin}\\ews\\ (or an explicit out_dir).

    Hook onto the end of the inference:
        write_ews_netcdf(preds, land_mask, BASINS["NACSGM"], analysis_time)
    """
    out_dir = Path(out_dir) if out_dir is not None else ews_output_dir(basin.folder, create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(prob, land_mask, basin, analysis_time, model_id)
    stamp_provenance(ds, analysis_time, era5_stream)   # record ERA5 vs ERA5T + lag
    if truth is not None:                              # ground-truth genesis for QA
        ds.attrs["truth_lat"] = float(truth[0])
        ds.attrs["truth_lon"] = float(truth[1])
    path = out_dir / output_filename(basin.key, analysis_time, label=label)
    comp = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, encoding={"prob": comp, "land_mask": comp})
    return path


# --------------------------------------------------------------------------- #
#  Reader + synthetic fallback
# --------------------------------------------------------------------------- #
def load_latest(basin: BasinConfig, out_dir: Optional[Path] = None) -> Tuple[xr.Dataset, bool]:
    """Return (dataset, is_synthetic). Synthetic fallback if no file exists."""
    out_dir = Path(out_dir) if out_dir is not None else ews_output_dir(basin.folder)
    files = sorted(out_dir.glob(f"{basin.key}_*.nc")) if out_dir.exists() else []
    if files:
        latest = max(files, key=lambda p: p.stat().st_mtime)
        # Eager .load() reads all data into memory and lets the file handle close
        # cleanly. Returning a lazy dataset instead makes Streamlit reruns hit
        # 'NetCDF: Not a valid ID' when a later read touches a GC-closed handle.
        with xr.open_dataset(latest, decode_timedelta=False) as _ds:
            return _ds.load(), False
    return synthetic_dataset(basin), True


def synthetic_dataset(basin: BasinConfig, analysis_time: Optional[dt.datetime] = None) -> xr.Dataset:
    """Plausible fake output for the no-file demo: Gaussian genesis blobs that
    broaden and fade with lead time. Peaks are pinned to the model's softmax
    regime (a few × theta) so the demo alerts the way real output would, rather
    than saturating against the calibrated per-cell threshold."""
    if analysis_time is None:
        analysis_time = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    lon, lat = basin.lon_coords(), basin.lat_coords()
    LON, LAT = np.meshgrid(lon, lat)
    coslat = np.cos(np.deg2rad(LAT))

    # Peak genesis probability, tied to THIS basin's calibrated threshold so the
    # demo alerts sensibly whatever the model's output scale: early leads sit a few
    # × theta (alerts fire), the last lead dips below theta (alerts fade). Works for
    # NACSGM's tiny softmax values (theta~0.01) and WPMM's 0..1 values (theta~0.6).
    peak0    = min(max(4.0 * basin.theta, 0.02), 0.95)
    peak_end = 0.8 * basin.theta
    prob = np.zeros((len(LEAD_HOURS), basin.ny, basin.nx), dtype="float32")
    for li, L in enumerate(LEAD_HOURS):
        sig  = 2.1 + (6.6 - 2.1) * (L / 48.0)          # spatial spread grows with lead
        peak = peak0 - (peak0 - peak_end) * (L / 48.0)  # peak decays with lead
        f = np.zeros((basin.ny, basin.nx), dtype="float32")
        for hlon, hlat, w in basin.hotspots:
            dx, dy = (LON - hlon) * coslat, LAT - hlat
            f += w * np.exp(-(dx * dx + dy * dy) / (2 * sig * sig))
        f = f / max(float(f.max()), 1e-9)               # normalise blob shape to peak 1
        prob[li] = np.clip(f * peak + 5e-4, 0.0, 1.0)   # scale to target peak + faint floor

    land = np.zeros((basin.ny, basin.nx), dtype="int8")
    return build_dataset(prob, land, basin, analysis_time, model_id="synthetic")


# --------------------------------------------------------------------------- #
#  Alert detection (connected components -> multiple simultaneous disturbances)
# --------------------------------------------------------------------------- #
@dataclass
class Watch:
    peak_prob: float
    peak_lat: float
    peak_lon: float
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    n_cells: int


def _cell_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    dlat = abs(float(np.mean(np.diff(lat))))
    dlon = abs(float(np.mean(np.diff(lon))))
    col = (dlat * EARTH_KM_PER_DEG) * (dlon * EARTH_KM_PER_DEG * np.cos(np.deg2rad(lat))[:, None])
    return np.broadcast_to(col, (len(lat), len(lon))).astype("float64")


def detect_watches(ds: xr.Dataset, lead_hours: int, theta: float, min_area_km2: float) -> List[Watch]:
    """Label connected ocean components above theta at one lead; one Watch each."""
    prob = ds["prob"].sel(lead=lead_hours).values.astype("float32")
    lat, lon = ds["lat"].values, ds["lon"].values
    ocean = (ds["land_mask"].values == 0) if "land_mask" in ds else np.ones_like(prob, bool)

    mask = (prob >= theta) & ocean
    if not mask.any():
        return []

    cell_area = _cell_area_km2(lat, lon)
    labels, n = ndimage.label(mask)
    watches: List[Watch] = []
    for lab in range(1, n + 1):
        sel = labels == lab
        area = float(cell_area[sel].sum())
        if area < min_area_km2:
            continue
        pj, pi = np.unravel_index(np.argmax(np.where(sel, prob, -1)), prob.shape)
        cj, ci = ndimage.center_of_mass(sel)
        watches.append(Watch(
            peak_prob=float(prob[pj, pi]),
            peak_lat=float(lat[pj]), peak_lon=float(lon[pi]),
            centroid_lat=float(np.interp(cj, np.arange(len(lat)), lat)),
            centroid_lon=float(np.interp(ci, np.arange(len(lon)), lon)),
            area_km2=area, n_cells=int(sel.sum()),
        ))
    watches.sort(key=lambda x: x.peak_prob, reverse=True)
    return watches


def peak_curve(ds: xr.Dataset) -> Tuple[List[int], List[float]]:
    """Max probability at each lead — the degradation curve for the sparkline."""
    ocean = (ds["land_mask"].values == 0) if "land_mask" in ds else None
    peaks = []
    for L in LEAD_HOURS:
        p = ds["prob"].sel(lead=L).values
        peaks.append(float(p[ocean].max() if ocean is not None else p.max()))
    return list(LEAD_HOURS), peaks
