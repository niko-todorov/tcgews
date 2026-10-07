"""
8. ews/TCG-EWS.py
====================
Real-time 2 Basins ERA5 Fetch + ConvLSTM TCG Early Warning System
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"

Input channels (8, ordered to match training pipeline from ncio.py):
  idx  var    source-fn      ERA5 field                    level
  ─────────────────────────────────────────────────────────────────
   0   sst    ncs2csv()      sea_surface_temperature        surface
   1   vwsh   ncvwsh2csv()   |wind(850)| − |wind(200)|      derived
   2   d      ncp2csv()      divergence                     850 hPa
   3   cape   ncs2csv()      convective_available_…         surface
   4   r      ncp2csv()      relative_humidity              850 hPa
   5   msl    ncs2csv()      mean_sea_level_pressure        surface
   6   vo     ncp2csv()      vorticity                      850 hPa
   7   pi     pi2csv()       potential intensity (tcpyPI)   derived

Lead times (hours before genesis):  0, 6, 12, 18, 24, 30, 36, 42, 48
                                     ←────── 9 lead-time steps ──────→

Workflow:
  1. Resolve latest available ERA5 analysis hour (~5-day lag).
  2. fetch_ncp()     → pressure-level NetCDF  (vo, d, r, u850, u200, v850, v200,
                                               T(all levels), q(all levels) for PI)
  3. fetch_ncs()     → single-level NetCDF    (sst, msl, cape)
  4. ncvwsh2csv()    → compute wind-shear field from u/v at 850 & 200 hPa
  5. pi2csv()        → compute potential-intensity field via tcpyPI
  6. build_tensor()  → assemble (1, T=9, C=8, H, W) normalised tensor
  7. run_inference() → 9 probability heatmaps, one per lead time
  8. save_netcdf() + plot_ews_panel() → NetCDF archive + PNG panel

Dependencies:
  pip install cdsapi xarray numpy torch matplotlib cartopy netCDF4
  pip install tcpyPI   (or install from https://github.com/dgilford/tcpyPI)

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import logging
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cdsapi
import numpy as np
from scipy.ndimage import label as _cc_label, distance_transform_edt as _edt
import torch
import torch.nn as nn
import xarray as xr

# --- path bootstrap: find the src/py dir (the one holding basins.py) and put it
#     on sys.path, so basins / ews / provenance resolve no matter the launch cwd.
for _up in Path(__file__).resolve().parents:
    if (_up / "basins.py").exists():
        if str(_up) not in sys.path:
            sys.path.insert(0, str(_up))
        break

from provenance import detect_era5_stream, stamp_provenance
from basins import BASINS
from ews import ews_core
from paths import final_dir, ews_output_dir, ews_cache_dir

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TCG-EWS")

# ─────────────────────────────────────────────────────────────────────────────
# tcpyPI — optional; required only for PI channel
# ─────────────────────────────────────────────────────────────────────────────
_TCPYPI_PATHS = [
    os.path.abspath(r"D:\GitHub\causal\tcpyPI\src"),   # Windows dev machine
    os.path.abspath("./tcpyPI/src"),                    # relative fallback
]
for _p in _TCPYPI_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from tcpyPI import pi as _tcpypi_pi
    TCPYPI_AVAILABLE = True
    log.info("tcpyPI loaded successfully")
except ImportError:
    TCPYPI_AVAILABLE = False
    log.warning(
        "tcpyPI not found — PI channel will be filled with NaN. "
        "Install from https://github.com/dgilford/tcpyPI"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# NACSGM domain — must match training-time extraction exactly
# Longitudes follow the 0–360° convention used by ncio.py / IBTrACS pipeline.
# CDS API requires −180/180; conversion is handled in _cds_area() below.
#   263.0° E  →  −97.0°   (western boundary, ~Gulf of Mexico)
#   305.0° E  →  −55.0°   (eastern boundary, ~Lesser Antilles)
# Domain is no longer hardcoded here — it comes from the shared basin registry
# (basins.py) via the active basin selected with --basin. See _set_active_basin().
GRID_RES = 0.25   # degrees — ERA5 native resolution

# ── Channel definitions (must match training pipeline order) ──────────────────
# Channels delivered by ncp2csv  (pressure-level)
NCP_VARS = [
    # (era5_short_name,  pressure_hPa,  output_alias)
    ("vo",  850, "vo"),    # relative vorticity 850 hPa
    ("d",   850, "d"),     # divergence          850 hPa
    ("r",   850, "r"),     # relative humidity   850 hPa
    # u/v at both shear levels — needed by ncvwsh2csv and pi2csv
    ("u",   850, "u850"),
    ("v",   850, "v850"),
    ("u",   200, "u200"),
    ("v",   200, "v200"),
    # temperature & specific humidity at ALL levels for tcpyPI
    ("t",   None, "t"),    # full column — all levels
    ("q",   None, "q"),    # full column — all levels
]

# Channels delivered by ncs2csv  (single-level)
NCS_VARS = [
    # (era5_name,                               output_alias)
    ("sea_surface_temperature",                 "sst"),
    ("mean_sea_level_pressure",                 "msl"),
    ("convective_available_potential_energy",   "cape"),
]

# Pressure levels used for PI column integration (full troposphere)
PI_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 825,
     800, 775, 750, 700, 650, 600, 550, 500,
     450, 400, 350, 300, 250, 225, 200, 175,
     150, 125, 100,
]

# ── Per-basin channel order (the model reads channels POSITIONALLY) ───────────
# The 8 fields fed to the model, in the EXACT order each basin's checkpoint was
# trained on. sst/msl/cape are single-level; vo/d/r are 850 hPa; vwsh and pi are
# derived (ncvwsh2csv, pi2csv). If a basin was trained in a different order, fix
# it HERE — a wrong order runs without error but yields silently wrong output.
CHANNELS_BY_BASIN = {
    # NACSGM checkpoint's training order (matches TCG-multilead-KL-NACSGM.py L206 / ncio-negatives.py MODEL_VARIABLES; channels filled by name).
    "NACSGM": ["sst", "msl", "cape", "r", "vo", "d", "vwsh", "pi"],
    #"NACSGM": ["sst", "vwsh", "d", "cape", "r", "msl", "vo", "pi"],
    
    "WPMM":   ["sst", "msl", "cape", "r", "vo", "d", "vwsh", "pi"],
}

# ── Active basin: set once by run_ews() from --basin. Drives domain, channels,
#    extremes, checkpoint, and output tags. basins.py is the single source of
#    truth for the domain box and grid.
ACTIVE_BASIN    = "NACSGM"
DOMAIN          = {}                                  # filled by _set_active_basin()
CHANNEL_ORDER   = CHANNELS_BY_BASIN[ACTIVE_BASIN]
N_CHANNELS      = len(CHANNEL_ORDER)


def _set_active_basin(basin_key: str) -> None:
    """Point every basin-dependent global at `basin_key` (from the basins.py registry)."""
    global ACTIVE_BASIN, DOMAIN, CHANNEL_ORDER, N_CHANNELS, EXTREMES_PATH, MODEL_CKPT_PATH, OUTPUT_DIR, ERA5_CACHE_DIR
    if basin_key not in BASINS:
        raise ValueError(f"Unknown basin '{basin_key}'. Known: {list(BASINS)}")
    if basin_key not in CHANNELS_BY_BASIN:
        raise ValueError(f"No channel order for '{basin_key}' in CHANNELS_BY_BASIN.")
    b = BASINS[basin_key]
    ACTIVE_BASIN  = basin_key
    DOMAIN        = dict(lat_north=b.lat_max, lat_south=b.lat_min,      # 0..360 convention,
                         lon_west=b.lon_min,  lon_east=b.lon_max)       # matching basins.py
    CHANNEL_ORDER = CHANNELS_BY_BASIN[basin_key]
    N_CHANNELS    = len(CHANNEL_ORDER)
    EXTREMES_PATH   = final_dir(b.folder) / "extremes.csv"
    MODEL_CKPT_PATH = final_dir(b.folder) / f"convlstm_{basin_key.lower()}_best.pt"
    OUTPUT_DIR      = ews_output_dir(b.folder, create=True)
    ERA5_CACHE_DIR  = ews_cache_dir(b.folder, create=True)

# Lead times in hours — 9 steps at 6 h cadence
LEAD_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]
N_LEADS    = len(LEAD_HOURS)      # 9
MAX_LAG_H  = max(LEAD_HOURS)      # 48

# Min-max normalisation extremes + trained checkpoint are per-basin, managed by
# _set_active_basin(). Initialise to the NACSGM defaults; run_ews() overrides.
EXTREMES_PATH   = final_dir("NACSGM") / "extremes.csv"
MODEL_CKPT_PATH = final_dir("NACSGM") / "convlstm_nacsgm_best.pt"
_set_active_basin("NACSGM")

# Output + cache directories
OUTPUT_DIR     = ews_output_dir("NACSGM", create=False)
ERA5_CACHE_DIR = ews_cache_dir("NACSGM", create=False)




# ─────────────────────────────────────────────────────────────────────────────
# ── NORMALISATION HELPERS ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def load_extremes(extremes_path=None):
    """
    Load per-variable min/max extremes from the training-set CSV produced
    by ncio.py.  Returns dict: { variable_name: (min, max) }.

    Variables in file: sst, msl, cape, r, d, vo, vwsh, pi  (all 8 channels).
    PI (channel 7) bounds are read from the 'pi' row — no separate file needed.
    """
    import pandas as pd
    path = Path(extremes_path) if extremes_path else EXTREMES_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    extremes = {
        str(row["variable"]).strip(): (float(row["min"]), float(row["max"]))
        for _, row in df.iterrows()
    }
    log.info(
        "Extremes loaded from %s: %s", path,
        {k: (f"{v[0]:.4g}", f"{v[1]:.4g}") for k, v in extremes.items()},
    )
    return extremes


def _minmax_channel(arr, vmin, vmax, clip=True):
    """
    Min-max normalise arr to [0, 1] using training-set extremes.
    Values outside [vmin, vmax] are clipped first so out-of-distribution
    real-time inputs never push the tensor outside [0, 1].
    """
    denom = vmax - vmin
    if denom == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    if clip:
        arr = np.clip(arr, vmin, vmax)
    return ((arr - vmin) / denom).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ── ERA5 FETCH HELPERS ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _latest_era5_time() -> datetime:
    """
    Return the most recent ERA5 analysis hour that is (almost certainly)
    available on CDS.  ERA5 is published with ~5-day lag; ERA5T (preliminary
    back-extension) with ~2-day lag.  We conservatively use 5 days and snap
    to the nearest 6-hourly synoptic hour (00 / 06 / 12 / 18 UTC).
    """
    now = datetime.now(tz=timezone.utc)
    lag = now - timedelta(days=5)
    synoptic_hour = (lag.hour // 6) * 6
    t0 = lag.replace(hour=synoptic_hour, minute=0, second=0, microsecond=0)
    log.info("Latest estimated ERA5 analysis time: %s UTC",
             t0.strftime("%Y-%m-%d %H:%M"))
    return t0


def _to_std_lon(lon_360: float) -> float:
    """Convert a 0-360 degree longitude to the -180/180 convention CDS API expects."""
    return lon_360 - 360.0 if lon_360 > 180.0 else lon_360


def _cds_area() -> list:
    """
    CDS-API area list: [North, West, South, East] in -180/180 convention.
    NACSGM longitudes are stored in 0-360 and converted here.
    """
    return [
        DOMAIN["lat_north"],
        _to_std_lon(DOMAIN["lon_west"]),   # 263.0 -> -97.0
        DOMAIN["lat_south"],
        _to_std_lon(DOMAIN["lon_east"]),   # 305.0 -> -55.0
    ]


def _timestamps_for_window(t0: datetime) -> list[datetime]:
    """
    Return the 9 UTC timestamps needed: t0-48h … t0, in ascending order.
    These are the 9 model input steps (oldest → newest).
    """
    return sorted([t0 - timedelta(hours=h) for h in LEAD_HOURS])


def _group_by_ymd(timestamps: list[datetime]) -> dict:
    """Group timestamps by (year, month, day) → list of 'HH:00' strings."""
    groups: dict = defaultdict(list)
    for ts in timestamps:
        groups[(ts.year, ts.month, ts.day)].append(f"{ts.hour:02d}:00")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# ── fetch_ncp  (mirrors ncp2csv — pressure-level download) ───────────────────
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ncp(
    t0: datetime,
    cache_dir: Path = ERA5_CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """
    Download ERA5 pressure-level fields for the NACSGM domain, covering the
    9-timestamp window [t0-48h … t0].

    Variables downloaded:
      • vo  (vorticity)           → needed at 850 hPa
      • d   (divergence)          → needed at 850 hPa
      • r   (relative_humidity)   → needed at 850 hPa
      • u, v (wind components)    → needed at 850 & 200 hPa for vwsh
      • t, q (temp, spec. humid.) → needed at ALL PI_LEVELS for tcpyPI

    Returns path to the cached pressure-level NetCDF.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag     = t0.strftime("%Y%m%d_%H")
    nc_path = cache_dir / f"ncp_{tag}.nc"

    if nc_path.exists() and not overwrite:
        log.info("ncp cache hit: %s", nc_path)
        return nc_path

    timestamps = _timestamps_for_window(t0)
    by_ymd     = _group_by_ymd(timestamps)

    all_dates = [f"{y}-{m:02d}-{d:02d}" for (y, m, d) in by_ymd]
    all_times = sorted({t for tlist in by_ymd.values() for t in tlist})
    all_levels = sorted(
        {850, 200}                          # vo@850 / d@850 / r@850 / u,v@850,200
        | set(PI_LEVELS)                    # full column for tcpyPI
    )

    pl_variables = [
        "vorticity",
        "divergence",
        "relative_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
    ]

    log.info(
        "fetch_ncp  | dates=%s  times=%s  levels=%s  vars=%s",
        all_dates, all_times, all_levels, pl_variables,
    )

    c = cdsapi.Client(quiet=False)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable":       pl_variables,
            "pressure_level": [str(lv) for lv in all_levels],
            "year":   sorted({str(y)     for (y, _, _) in by_ymd}),
            "month":  sorted({f"{m:02d}" for (_, m, _) in by_ymd}),
            "day":    sorted({f"{d:02d}" for (_, _, d) in by_ymd}),
            "time":   all_times,
            "area":   _cds_area(),
            "grid":   [GRID_RES, GRID_RES],
            "format": "netcdf",
        },
        str(nc_path),
    )
    log.info("ncp saved → %s", nc_path)
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# ── fetch_ncs  (mirrors ncs2csv — single-level download) ─────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ncs(
    t0: datetime,
    cache_dir: Path = ERA5_CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """
    Download ERA5 single-level fields for the NACSGM domain.

    Variables downloaded:
      • sea_surface_temperature                (sst)
      • mean_sea_level_pressure                (msl)
      • convective_available_potential_energy  (cape)

    Returns path to the cached single-level NetCDF.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag     = t0.strftime("%Y%m%d_%H")
    nc_path = cache_dir / f"ncs_{tag}.nc"

    if nc_path.exists() and not overwrite:
        log.info("ncs cache hit: %s", nc_path)
        return nc_path

    timestamps = _timestamps_for_window(t0)
    by_ymd     = _group_by_ymd(timestamps)

    all_dates = [f"{y}-{m:02d}-{d:02d}" for (y, m, d) in by_ymd]
    all_times = sorted({t for tlist in by_ymd.values() for t in tlist})

    sl_variables = [
        "sea_surface_temperature",
        "mean_sea_level_pressure",
        "convective_available_potential_energy",
    ]

    log.info(
        "fetch_ncs  | dates=%s  times=%s  vars=%s",
        all_dates, all_times, sl_variables,
    )

    c = cdsapi.Client(quiet=False)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable":  sl_variables,
            "year":   sorted({str(y)     for (y, _, _) in by_ymd}),
            "month":  sorted({f"{m:02d}" for (_, m, _) in by_ymd}),
            "day":    sorted({f"{d:02d}" for (_, _, d) in by_ymd}),
            "time":   all_times,
            "area":   _cds_area(),
            "grid":   [GRID_RES, GRID_RES],
            "format": "netcdf",
        },
        str(nc_path),
    )
    log.info("ncs saved → %s", nc_path)
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# ── ncvwsh2csv  (mirrors ncvwsh2csv — wind shear computation) ─────────────────
# ─────────────────────────────────────────────────────────────────────────────

def ncvwsh2csv(
    ds_plev: xr.Dataset,
    times_sel: list,
) -> np.ndarray:
    """
    Compute the vertical wind shear magnitude between 850 and 200 hPa,
    replicating the ncvwsh2csv() pipeline from ncio.py.

    vwsh = sqrt((u850 - u200)² + (v850 - v200)²)   [m s⁻¹]

    Parameters
    ----------
    ds_plev : xr.Dataset
        Pressure-level dataset from fetch_ncp(), already time-selected.
    times_sel : list
        List of np.datetime64 timestamps (length T=9).

    Returns
    -------
    vwsh : np.ndarray  shape (T, H, W)  in m s⁻¹
    """
    T  = len(times_sel)
    # Infer grid size from dataset
    lat_dim = "latitude" if "latitude" in ds_plev.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds_plev.dims else "lon"
    H = ds_plev.dims[lat_dim]
    W = ds_plev.dims[lon_dim]

    vwsh = np.zeros((T, H, W), dtype=np.float32)
    for ti, ts in enumerate(times_sel):
        u850 = ds_plev["u"].sel(time=ts, level=850).values.astype(np.float32)
        v850 = ds_plev["v"].sel(time=ts, level=850).values.astype(np.float32)
        u200 = ds_plev["u"].sel(time=ts, level=200).values.astype(np.float32)
        v200 = ds_plev["v"].sel(time=ts, level=200).values.astype(np.float32)
        vwsh[ti] = np.sqrt((u850 - u200) ** 2 + (v850 - v200) ** 2)

    return vwsh   # (T, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ── pi2csv  (mirrors pi2csv — potential intensity via tcpyPI) ─────────────────
# ─────────────────────────────────────────────────────────────────────────────

def pi2csv(
    ds_plev: xr.Dataset,
    ds_sfc:  xr.Dataset,
    times_sel: list,
) -> np.ndarray:
    """
    Compute potential intensity (PI) for every ocean grid point and timestamp,
    replicating the pi2csv() pipeline from ncio.py using tcpyPI.

    tcpyPI signature:
        pi(sst_C, msl_hPa, levels_hPa, T_C_profile, q_kg_kg_profile,
           CKCD=0.9, ascent_flag=0, diss_flag=1, ptop=50, miss_handle=1)
        → (vmax, pmin, ifl, t0, otl)

    Parameters
    ----------
    ds_plev   : pressure-level dataset (t, q at PI_LEVELS; time-selected)
    ds_sfc    : single-level  dataset (sst, msl; time-selected)
    times_sel : list of np.datetime64 timestamps (length T=9)

    Returns
    -------
    pi_arr : np.ndarray  shape (T, H, W)  in m s⁻¹  (NaN over land / failures)
    """
    lat_dim = "latitude" if "latitude" in ds_plev.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds_plev.dims else "lon"
    H = ds_plev.dims[lat_dim]
    W = ds_plev.dims[lon_dim]
    T = len(times_sel)

    pi_arr = np.full((T, H, W), np.nan, dtype=np.float32)

    if not TCPYPI_AVAILABLE:
        log.warning("tcpyPI unavailable — PI channel set to NaN")
        return pi_arr

    # PI pressure levels — EXACTLY the 21 levels ncio.py used for training PI,
    # ordered surface -> top (descending pressure), intersected with what's here.
    NCIO_PI_LEVELS = {1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                      550, 500, 450, 400, 350, 300, 250, 200, 150, 100}
    avail_levels = sorted(
        [int(lv) for lv in ds_plev["level"].values if int(lv) in NCIO_PI_LEVELS],
        reverse=True,   # surface (high P) -> top (low P)
    )
    levels_pi = np.asarray(avail_levels, dtype=np.float64)

    log.info("pi2csv  | computing PI for T=%d  H=%d  W=%d  n_levels=%d",
             T, H, W, len(levels_pi))
    successful = 0

    # Per-grid-point PI, replicating ncio.py's pi2csv exactly (tcpyPI.pi is
    # jitted for scalar SST/MSL + 1-D profiles, so it must be called point-wise).
    for ti, ts in enumerate(times_sel):
        sst_all = ds_sfc["sst"].sel(time=ts).values.astype(np.float64)          # (H,W) K
        msl_all = ds_sfc["msp" if "msp" in ds_sfc else "msl"].sel(time=ts).values.astype(np.float64)  # (H,W) Pa
        t_all = np.stack([ds_plev["t"].sel(time=ts, level=int(lv)).values
                          for lv in avail_levels], axis=0).astype(np.float64)    # (n_lev,H,W) K
        q_all = np.stack([ds_plev["q"].sel(time=ts, level=int(lv)).values
                          for lv in avail_levels], axis=0).astype(np.float64)    # (n_lev,H,W) kg/kg

        for iy in range(H):
            for ix in range(W):
                sst = sst_all[iy, ix]
                if sst > 200.0:
                    sst = sst - 273.15                       # K -> C
                msl = msl_all[iy, ix]
                if msl > 10000.0:
                    msl = msl / 100.0                        # Pa -> hPa
                if not np.isfinite(sst) or not np.isfinite(msl) or sst < 5.0:
                    continue                                 # land / cold: leave NaN

                t_prof = t_all[:, iy, ix]
                q_prof = q_all[:, iy, ix]
                if np.all(np.isnan(t_prof)) or np.all(np.isnan(q_prof)):
                    continue
                if np.nanmax(t_prof) > 200.0:
                    t_prof = t_prof - 273.15                 # K -> C

                # specific humidity -> mixing ratio in g/kg (tcpyPI's expected R)
                q_c  = np.clip(q_prof, 0.0, 0.999)
                mixr = (q_c / (1.0 - q_c)) * 1000.0

                try:
                    VMAX, PMIN, IFL, TO, LNB = _tcpypi_pi(
                        sst, msl, levels_pi, t_prof, mixr,
                        CKCD=0.9, ascent_flag=0, diss_flag=1,
                        V_reduc=0.8, miss_handle=1,
                    )
                    if IFL == 1 and np.isfinite(VMAX):
                        pi_arr[ti, iy, ix] = np.float32(VMAX)  # m/s
                        successful += 1
                except Exception:
                    pass

    log.info("pi2csv  | converged grid-points: %d / %d", successful, T * H * W)
    return pi_arr   # (T, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ── PREPROCESSING  ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def build_input_tensor(
    ncp_path: Path,
    ncs_path: Path,
    t0: datetime,
    extremes:  Optional[dict] = None,
) -> tuple[torch.Tensor, np.ndarray, str]:
    """
    Assemble the (1, T=9, C=8, H, W) model input tensor from the two ERA5
    NetCDF files, computing vwsh and pi inline, then applying min-max
    normalisation using the training-set extremes from extremes.csv.

    Channel order: sst(0)  vwsh(1)  d(2)  cape(3)  r(4)  msl(5)  vo(6)  pi(7)

    Normalisation
    -------------
    All 8 channels (including pi):
        (x - min) / (max - min)  using values from extremes.csv

    Returns
    -------
    X         : torch.Tensor  (1, 9, 8, H, W)  normalised to [0, 1]
    land_mask : np.ndarray    (H, W)  bool — True = ocean
    era5_stream : str  'ERA5' | 'ERA5T' | 'mixed' — input data provenance
    """
    # ── Select the 9 timestamps (oldest → newest) ──
    times = sorted([t0 - timedelta(hours=h) for h in LEAD_HOURS])
    times_np = [np.datetime64(ts.replace(tzinfo=None)) for ts in times]

    # ── Open datasets ──
    ds_plev = xr.open_dataset(ncp_path)
    ds_sfc  = xr.open_dataset(ncs_path)

    # ── Provenance: read expver BEFORE _normalise_cds drops it below ──
    _streams = {detect_era5_stream(ds_plev), detect_era5_stream(ds_sfc)}
    era5_stream = ("ERA5T" if _streams == {"ERA5T"}
                   else "ERA5" if _streams == {"ERA5"}
                   else "mixed")   # two files disagree, or a straddling window
    log.info("ERA5 stream: %s  (per-file: %s)", era5_stream, _streams)

    # New CDS (ecmwf.datastores) NetCDFs name the time axis 'valid_time' and may
    # carry singleton 'number'/'expver' dims — normalise to the legacy layout so
    # the rest of this function (which selects on 'time') is unchanged.
    def _normalise_cds(ds):
        if "valid_time" in ds.variables:
            ds = ds.rename({"valid_time": "time"})
        if "pressure_level" in ds.variables:
            ds = ds.rename({"pressure_level": "level"})
        for _extra in ("number", "expver"):
            if _extra in ds.dims:
                ds = ds.squeeze(_extra, drop=True)
            if _extra in ds.coords:
                ds = ds.drop_vars(_extra, errors="ignore")
        return ds
    ds_plev = _normalise_cds(ds_plev)
    ds_sfc  = _normalise_cds(ds_sfc)

    # Normalise SST variable name (ERA5 sometimes uses 'sst' or full name)
    for _old in ["sea_surface_temperature"]:
        if _old in ds_sfc:
            ds_sfc = ds_sfc.rename({_old: "sst"})
    for _old in ["mean_sea_level_pressure"]:
        if _old in ds_sfc:
            ds_sfc = ds_sfc.rename({_old: "msl"})
    for _old in ["convective_available_potential_energy"]:
        if _old in ds_sfc:
            ds_sfc = ds_sfc.rename({_old: "cape"})

    # Select only the 9 needed timestamps
    ds_plev = ds_plev.sel(time=times_np)
    ds_sfc  = ds_sfc.sel(time=times_np)

    lat_dim = "latitude" if "latitude" in ds_plev.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds_plev.dims else "lon"
    H = ds_plev.dims[lat_dim]
    W = ds_plev.dims[lon_dim]
    T = len(times)
    log.info("build_input_tensor | H=%d  W=%d  T=%d  C=%d", H, W, T, N_CHANNELS)

    # ── Land mask: ocean where SST is finite ──
    sst_t0    = ds_sfc["sst"].isel(time=-1).values
    land_mask = np.isfinite(sst_t0)   # True = ocean

    # ── Compute derived fields ──
    vwsh_arr = ncvwsh2csv(ds_plev, times_np)       # (T, H, W)
    pi_arr   = pi2csv(ds_plev, ds_sfc, times_np)   # (T, H, W)

    # ── Assemble per-timestep channel stacks ──
    # ── Assemble per-timestep channel stacks in the ACTIVE basin's order ──
    # Build each field by name, then stack in CHANNEL_ORDER — so NACSGM and WPMM
    # (which have different orders) each get the exact layout their checkpoint
    # was trained on. Data order and the normalisation loop below stay in sync.
    frames = []
    for ti in range(T):
        sst_  = ds_sfc["sst"].isel(time=ti).values.astype(np.float32)
        # ERA5/IFS SST is kelvin; training extremes (11.01–33.53) are °C. Convert,
        # matching pi2csv and ncio.py, else min-max saturates all ocean to 1.0.
        sst_  = np.where(sst_ > 200.0, sst_ - 273.15, sst_)
        chan = {
            "sst":  sst_,
            "msl":  ds_sfc["msl"].isel(time=ti).values.astype(np.float32),
            "cape": ds_sfc["cape"].isel(time=ti).values.astype(np.float32),
            "r":    ds_plev["r"].sel(time=times_np[ti],  level=850).values.astype(np.float32),
            "vo":   ds_plev["vo"].sel(time=times_np[ti], level=850).values.astype(np.float32),
            "d":    ds_plev["d"].sel(time=times_np[ti],  level=850).values.astype(np.float32),
            "vwsh": vwsh_arr[ti],
            "pi":   pi_arr[ti],
        }
        frame = np.stack([chan[name] for name in CHANNEL_ORDER], axis=0)   # (8, H, W)
        frames.append(frame)

    X_raw = np.stack(frames, axis=0).astype(np.float32)   # (T, C, H, W) = (9, 8, H, W)

    # ── Mask land cells with NaN before normalisation ──
    # (zeroed out after scaling; masking first prevents land from
    #  contaminating the clipping range)
    X_raw[:, :, ~land_mask] = np.nan

    # ── Load extremes if not already provided ──
    if extremes is None:
        log.warning(
            "extremes not provided — loading from default path %s",
            EXTREMES_PATH,
        )
        extremes = load_extremes()

    # ── Min-max normalise all 8 channels using extremes.csv ──
    for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
        if ch_name not in extremes:
            raise KeyError(
                f"Channel '{ch_name}' (idx {ch_idx}) not found in "
                f"extremes.csv. Available: {list(extremes.keys())}"
            )
        vmin, vmax = extremes[ch_name]
        X_raw[:, ch_idx] = _minmax_channel(X_raw[:, ch_idx], vmin, vmax)
        log.debug("  %-5s  ch%d  min=%.6g  max=%.6g",
                  ch_name, ch_idx, vmin, vmax)

    log.info(
        "Min-max normalisation applied for all 8 channels using extremes.csv"
    )

    # ── Zero land cells; replace any residual NaNs ──
    X_raw[:, :, ~land_mask] = 0.0
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # ── Add batch dimension → (1, T, C, H, W) ──
    X = torch.tensor(X_raw).unsqueeze(0)   # (1, 9, 8, H, W)
    return X, land_mask, era5_stream


# ─────────────────────────────────────────────────────────────────────────────
# ── MODEL ARCHITECTURE  ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class _ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — orthogonal init + forget-gate bias=1 + cell clamp.
    Ported verbatim from the training script (TCG-multilead-KL-WP.py) so the
    state_dict key names (clstm*.conv.*) match the trained checkpoints."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=pad, bias=True,
        )
        nn.init.orthogonal_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)
        self.conv.bias.data[hidden_dim:2 * hidden_dim].fill_(1.0)  # forget gate

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = torch.clamp(f * c + i * g, -10.0, 10.0)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class TemporalTCG(nn.Module):
    """ConvLSTM + temporal-attention genesis model — MUST match the trained
    checkpoints (keys: embed / clstm1 / clstm2 / t_attn / refine). Ported verbatim
    from the training script (TCG-multilead-KL-WP.py).

    Input : (B, T, C, H, W)
    Output: (B, 1, H, W) raw logits (NO activation). The ocean-masked spatial
    softmax that turns logits into a probability map lives in run_inference(),
    matching the training loss (log_softmax over ocean cells).
    """

    def __init__(self, input_channels: int = N_CHANNELS, base_filters: int = 32):
        super().__init__()
        bf = base_filters
        self.embed = nn.Sequential(
            nn.Conv2d(input_channels, bf, kernel_size=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
        )
        self.clstm1 = _ConvLSTMCell(bf, bf,     kernel_size=3)
        self.clstm2 = _ConvLSTMCell(bf, bf * 2, kernel_size=3)
        self.t_attn = nn.Sequential(
            nn.Conv2d(bf * 2, 1, kernel_size=1, bias=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf * 2, bf,     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf,     1,      kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        device = x.device

        emb = [self.embed(x[:, t]) for t in range(T)]

        h1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        c1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        out1 = []
        for t in range(T):
            h1, c1 = self.clstm1(emb[t], h1, c1)
            out1.append(h1)

        h2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        c2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        out2 = []
        for t in range(T):
            h2, c2 = self.clstm2(out1[t], h2, c2)
            out2.append(h2)

        seq = torch.stack(out2, dim=1)                                  # (B, T, bf*2, H, W)
        scores = torch.stack([self.t_attn(seq[:, t]) for t in range(T)],
                             dim=1)                                     # (B, T, 1, H, W)
        weights = torch.softmax(scores, dim=1)                          # softmax over T
        attended = (seq * weights).sum(dim=1)                           # (B, bf*2, H, W)
        return self.refine(attended)                                    # (B, 1, H, W) logits


# ─────────────────────────────────────────────────────────────────────────────
# ── INFERENCE  ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def basin_taper(land_mask: np.ndarray, taper_cells: int = 5) -> np.ndarray:
    """Boundary-artifact suppressor (see Appendix A). 0 over land and the
    disconnected Eastern-Pacific wedge; linear ramp 0->1 within taper_cells of
    any coast or domain edge; 1 in open Atlantic/Caribbean/Gulf water."""
    lab, _   = _cc_label(land_mask)
    sizes    = np.bincount(lab.ravel()); sizes[0] = 0        # 0 = land/background
    atlantic = lab == int(sizes.argmax())                    # largest ocean body
    padded   = np.pad(atlantic, 1, constant_values=False)    # domain edge = boundary
    dist     = _edt(padded)[1:-1, 1:-1]                       # cells to nearest boundary
    return (np.clip(dist / float(taper_cells), 0.0, 1.0) * atlantic).astype(np.float32)

def _nearest_ocean_fill(X, ocean):
    """No-retrain coastal-sliver fix: replace every land cell of each (time,
    channel) field with its nearest ocean-cell value, removing the ocean->0
    discontinuity the convolutions amplify into shoreline ridges. Trained weights
    unchanged. X:(1,T,C,H,W) tensor; ocean:(H,W) bool True=ocean."""
    ocean = np.asarray(ocean, dtype=bool)
    if ocean.all():
        return X
    iy, ix = _edt(~ocean, return_distances=False, return_indices=True)
    iy = torch.as_tensor(iy, device=X.device, dtype=torch.long)
    ix = torch.as_tensor(ix, device=X.device, dtype=torch.long)
    return X[..., iy, ix]


def _forward_halo(model, X_lead, halo):
    """No-retrain domain-edge fix ('inference halo'): replicate-pad the spatial
    dims by `halo` cells, run the model, then crop back. The zero-padding edge
    artifact forms in the discarded halo instead of on the S/E/N/W walls."""
    if halo <= 0:
        return model(X_lead)[0, 0]
    b, t, c, H, W = X_lead.shape
    xp = torch.nn.functional.pad(
        X_lead.reshape(b * t, c, H, W), (halo, halo, halo, halo), mode='replicate'
    ).reshape(b, t, c, H + 2 * halo, W + 2 * halo)
    out = model(xp)[0, 0]
    return out[halo:halo + H, halo:halo + W]


def robust_peak(prob, mode='component', rel_thresh=0.30):
    """Sliver-robust point estimate of the genesis cell (row, col).
      'raw'       - global argmax (legacy; can land on a thin boundary ridge).
      'component' - peak of the connected blob carrying the most probability mass
                    (thin ridges have high peak but little mass, so are ignored).
      'centroid'  - probability-weighted centroid of that same blob."""
    prob = np.asarray(prob, dtype=float)
    if mode == 'raw' or prob.max() <= 0:
        return tuple(int(v) for v in np.unravel_index(prob.argmax(), prob.shape))
    lab, n = _cc_label(prob > rel_thresh * prob.max())
    if n == 0:
        return tuple(int(v) for v in np.unravel_index(prob.argmax(), prob.shape))
    masses = np.bincount(lab.ravel(), weights=prob.ravel())
    comp = lab == int(np.argmax(masses[1:])) + 1
    if mode == 'centroid':
        ys, xs = np.nonzero(comp); w = prob[comp]
        return (int(round((ys * w).sum() / w.sum())), int(round((xs * w).sum() / w.sum())))
    pm = np.where(comp, prob, -np.inf)
    return tuple(int(v) for v in np.unravel_index(pm.argmax(), pm.shape))


def _out_suffix(taper=False, ocean_fill=False, halo=0, peak_mode='raw'):
    s = '_taper' if taper else ''
    if ocean_fill: s += '_fill'
    if halo: s += f'_halo{halo}'
    if peak_mode and peak_mode != 'raw': s += f'_{peak_mode}'
    return s


def run_inference(
    X: torch.Tensor,       # (1, T=9, C=8, H, W)
    model: "TemporalTCG",
    device: torch.device,
    land_mask: np.ndarray,   # (H, W) True = ocean
    apply_taper: bool = True,
    ocean_fill: bool = False,
    halo: int = 0,
    peak_mode: str = 'raw',
) -> dict[int, np.ndarray]:
    """
    Run TemporalTCG for each of the 9 lead times and return per-lead probability
    heatmaps. The model emits raw logits; the probability map is a spatial softmax
    over OCEAN cells only — matching the training loss (log_softmax over ocean).

    For lead time τ hours, the model receives the T consecutive snapshots whose
    last element is t0 − τ. The oldest T=9 frames are padded at the front for
    short windows.

    Returns
    -------
    preds : dict  { lead_hours → np.ndarray (H, W) probability map }
    """
    model.eval()
    X = X.to(device)
    if ocean_fill:
        X = _nearest_ocean_fill(X, land_mask)
        log.info("Nearest-ocean fill ON — land cells filled from nearest ocean (coastal-sliver mitigation)")
    ocean = torch.as_tensor(land_mask, device=device, dtype=torch.bool)  # True = ocean
    neg_inf = torch.tensor(float("-inf"), device=device)
    preds: dict[int, np.ndarray] = {}
    # --taper ON: boundary-artifact suppression (Appendix A). OFF (no flag):
    # all-ones taper => log_taper=0 and valid=ocean, i.e. the raw pre-taper
    # field (incl. the Eastern-Pacific wedge) for a faithful 'before' panel.
    taper_np = (basin_taper(land_mask, taper_cells=5) if apply_taper
                else np.ones_like(land_mask, dtype=np.float32))
    taper = torch.as_tensor(taper_np, device=device, dtype=torch.float32)
    log_taper = torch.log(taper.clamp_min(1e-6))
    valid = ocean & (taper > 0)  # tapered: drops the Eastern-Pacific wedge
    log.info("Boundary taper %s — %d/%d ocean cells retained",
             "ON (Appendix A)" if apply_taper else "OFF (raw 'before')",
             int(valid.sum().item()), int(ocean.sum().item()))
    if halo > 0:
        log.info("Inference halo ON — replicate-padded %d cells (S/E/N/W edge artifact cropped)", halo)
    with torch.no_grad():
        for lead_h in LEAD_HOURS:
            n_skip = lead_h // 6   # trailing steps to drop (most-recent removed)

            if n_skip == 0:
                X_lead = X                              # (1, 9, 8, H, W)
            else:
                X_lead = X[:, :-n_skip, :, :, :]       # drop the n_skip newest

            # Pad front with oldest frame if shorter than T_required=9
            T_req   = 9
            T_avail = X_lead.shape[1]
            if T_avail < T_req:
                pad = X_lead[:, :1].expand(-1, T_req - T_avail, -1, -1, -1)
                X_lead = torch.cat([pad, X_lead], dim=1)

            logits = _forward_halo(model, X_lead, halo)   # (H, W) raw logits (halo-cropped if halo>0)
            logits = logits + log_taper  # damp boundary/coast; drop Pacific wedge
            masked = torch.where(valid, logits, neg_inf)
            prob   = torch.softmax(masked.reshape(-1), dim=0).reshape(masked.shape)
            prob   = torch.nan_to_num(prob, nan=0.0)
            preds[lead_h] = prob.cpu().numpy()

            peak_idx = np.unravel_index(preds[lead_h].argmax(), preds[lead_h].shape)
            rpk = robust_peak(preds[lead_h], mode=peak_mode)
            log.info(
                "Lead %2dh → peak %.5f  raw-argmax %s  robust[%s] %s",
                lead_h, preds[lead_h].max(), peak_idx, peak_mode, tuple(rpk),
            )

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# ── VISUALISATION  ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_ews_panel(
    preds:     dict[int, np.ndarray],
    land_mask: np.ndarray,
    t0:        datetime,
    out_dir:   Path = OUTPUT_DIR,
    source:    str = "era5",
    truth:     Optional[tuple] = None,
    taper:     bool = True,
    peak_mode: str = 'raw',
    ocean_fill: bool = False,
    halo:      int = 0,
) -> Path:
    """
    3×3 panel of TCG probability heatmaps — one cell per lead time —
    plotted over the NACSGM domain with coastlines and a shared colour scale.

    Colours use a power-normalised (γ=0.4) yellow–orange–red ramp so that
    low-probability signals are still visible against the background.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        log.warning("cartopy not found — plotting without map projection")

    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = preds[0].shape
    lats = np.linspace(DOMAIN["lat_north"], DOMAIN["lat_south"], H)
    # ERA5 is downloaded in -180/180 convention; use _to_std_lon() for display
    # so that Cartopy set_extent and pcolormesh agree.
    lon_w_std = _to_std_lon(DOMAIN["lon_west"])    # 263 -> -97
    lon_e_std = _to_std_lon(DOMAIN["lon_east"])    # 305 -> -55
    lons = np.linspace(lon_w_std, lon_e_std, W)

    vmax = max(float(arr.max()) for arr in preds.values())
    vmax = max(vmax, 1e-6)
    norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax)
    cmap = plt.cm.YlOrRd

    proj     = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig_kw   = dict(figsize=(18, 12), constrained_layout=True)
    sub_kw   = {"projection": proj} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(3, 3, subplot_kw=sub_kw, **fig_kw)

    for idx, lead_h in enumerate(LEAD_HOURS):
        ax   = axes[idx // 3, idx % 3]
        prob = preds[lead_h].copy().astype(float)
        prob[~land_mask] = np.nan

        valid_time = t0 + timedelta(hours=lead_h)

        if HAS_CARTOPY:
            im = ax.pcolormesh(lons, lats, prob,
                               cmap=cmap, norm=norm,
                               transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
            ax.add_feature(cfeature.LAND,      facecolor="#d0d0d0", zorder=0)
            ax.set_extent(
                [lon_w_std, lon_e_std,
                 DOMAIN["lat_south"], DOMAIN["lat_north"]],
                crs=ccrs.PlateCarree(),
            )
            ax.gridlines(draw_labels=(idx == 6), linewidth=0.3, alpha=0.5)
        else:
            im = ax.pcolormesh(lons, lats, prob, cmap=cmap, norm=norm)
            ax.set_xlim(lon_w_std, lon_e_std)
            ax.set_ylim(DOMAIN["lat_south"], DOMAIN["lat_north"])

        # Ground-truth genesis location — semi-transparent circle on t=0 panel (QA).
        if truth is not None and lead_h == 0:
            t_lat, t_lon = truth
            t_lon = _to_std_lon(t_lon)
            _tf = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
            ax.scatter([t_lon], [t_lat], s=700, facecolors="none",
                       edgecolors=(0.0, 0.149, 0.561, 0.5), linewidths=2.0, zorder=6, **_tf)
            # ax.text(t_lon, t_lat, "  obs", color="#00268f", fontsize=8,
            #         va="center", ha="left", zorder=7, **_tf)

        # Predicted genesis — point estimate under the chosen argmax mode.
        _pr, _pc = robust_peak(preds[lead_h], mode=peak_mode)
        _pf = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        ax.scatter([lons[_pc]], [lats[_pr]], s=110, marker="+",
                   color="#111111", linewidths=1.6, zorder=7, **_pf)

        ax.set_title(
            f"+{lead_h:2d} h  ({valid_time.strftime('%b %d  %H:00 UTC')})",
            fontsize=10, fontweight="bold",
        )

    fig.colorbar(im, ax=axes, orientation="vertical",
                 fraction=0.02, pad=0.02,
                 label=f"TCG probability (softmax over {ACTIVE_BASIN})")

    analysis_str = t0.strftime("%Y-%m-%d %H:00 UTC")
    src_label = "IFS Open Data forecast" if source == "ifs-opendata" else "ERA5 analysis"
    fig.suptitle(
        f"{ACTIVE_BASIN} TCG Early Warning System\n"
        f"{src_label}: {analysis_str}  |  "
        f"Channels: {', '.join(CHANNEL_ORDER)}  |  Lead times: 0–48 h",
        fontsize=12, fontweight="bold",
    )
    _tsuf = _out_suffix(taper, ocean_fill, halo, peak_mode)
    png_path = out_dir / f"TCG-{ACTIVE_BASIN}-EWS_{source}_{t0.strftime('%Y%m%d_%H')}Z{_tsuf}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt; _plt.close(fig)
    log.info("EWS panel saved → %s", png_path)
    return png_path


def save_netcdf(
    preds:     dict[int, np.ndarray],
    land_mask: np.ndarray,
    t0:        datetime,
    out_dir:   Path = OUTPUT_DIR,
    era5_stream: str = "unknown",
    source:    str = "era5",
    taper:     bool = True,
    ocean_fill: bool = False,
    halo:      int = 0,
    peak_mode: str = 'raw',
) -> Path:
    """
    Archive all 9 lead-time probability maps into a single NetCDF,
    preserving analysis time, valid time, and channel metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W        = preds[0].shape
    lats        = np.linspace(DOMAIN["lat_north"], DOMAIN["lat_south"], H)
    # Longitude coordinates stored in -180/180 (matching CDS download convention)
    lons        = np.linspace(_to_std_lon(DOMAIN["lon_west"]),   # -97.0
                               _to_std_lon(DOMAIN["lon_east"]),  # -60.0
                               W)
    prob_stack  = np.stack([preds[lh] for lh in LEAD_HOURS], axis=0)   # (9, H, W)
    prob_stack[:, ~land_mask] = np.nan

    valid_times = np.array(
        [np.datetime64((t0 + timedelta(hours=lh)).isoformat())
         for lh in LEAD_HOURS]
    )

    ds_out = xr.Dataset(
        {
            "tcg_probability": xr.DataArray(
                prob_stack,
                dims=["lead_time", "latitude", "longitude"],
                coords={
                    "lead_time":  (["lead_time"], np.array(LEAD_HOURS),
                                   {"units": "hours", "long_name": "Forecast lead time"}),
                    "valid_time": (["lead_time"], valid_times),
                    "latitude":   (["latitude"],  lats, {"units": "degrees_north"}),
                    "longitude":  (["longitude"], lons, {"units": "degrees_east",
                                                         "convention": "-180/180"}),
                },
                attrs={
                    "long_name": (
                        "Predicted TCG probability — softmax distribution "
                        "over NACSGM basin grid points"
                    ),
                    "units":     "1",
                    "channels":  ", ".join(CHANNEL_ORDER),
                },
            )
        },
        attrs={
            "title":         f"{ACTIVE_BASIN} TCG Early Warning System",
            "analysis_time": t0.isoformat(),
            "model":         "TemporalTCG (ConvLSTM + temporal attention)",
            "input_source":  ("ECMWF IFS Open Data" if source == "ifs-opendata"
                              else "ERA5 reanalysis (CDS API)"),
            "lead_times_h":  str(LEAD_HOURS),
            "channels":      ", ".join(CHANNEL_ORDER),
            "created":       datetime.now(tz=timezone.utc).isoformat(),
        },
    )

    stamp_provenance(ds_out, t0, era5_stream)   # ERA5 vs ERA5T + lag, self-describing

    _tsuf = _out_suffix(taper, ocean_fill, halo, peak_mode)
    nc_path = out_dir / f"TCG-{ACTIVE_BASIN}-EWS_{source}_{t0.strftime('%Y%m%d_%H')}Z{_tsuf}.nc"
    ds_out.to_netcdf(nc_path)
    log.info("EWS NetCDF saved → %s", nc_path)
    return nc_path


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN PIPELINE  ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run_ews(
    analysis_time:   Optional[datetime] = None,
    model_ckpt:      Optional[Path] = None,
    extremes_path:   Optional[Path] = None,
    overwrite_cache: bool = False,
    source:          str = "era5",
    basin:           str = "NACSGM",
    truth:           Optional[tuple] = None,
    apply_taper:     bool = True,
    ocean_fill:      bool = False,
    halo:            int = 0,
    peak_mode:       str = 'raw',
) -> tuple[Path, Path]:
    """
    Full TCG-NACSGM early warning pipeline.

    Steps
    -----
    1.  Resolve latest available ERA5 analysis hour.
    2.  fetch_ncp()         — pressure-level ERA5 NetCDF (vo, d, r, u, v, t, q)
    3.  fetch_ncs()         — single-level  ERA5 NetCDF  (sst, msl, cape)
    4.  build_input_tensor()
          → ncvwsh2csv()   — wind-shear field  (T, H, W)
          → pi2csv()       — potential-intensity field (T, H, W)
          → assemble (1, 9, 8, H, W) normalised tensor
    5.  run_inference()     — 9 probability heatmaps
    6.  save_netcdf()       — NetCDF archive
    7.  plot_ews_panel()    — 3×3 PNG panel

    Parameters
    ----------
    analysis_time : datetime, optional
        Override analysis time (UTC). Default: latest available ERA5 hour.
    model_ckpt : Path
        Trained ConvLSTM checkpoint (.pt).
    extremes_path : Path
        Path to extremes.csv from ncio.py training run
        (default: <repo>/data/NACSGM/final/extremes.csv).
        Must include a 'pi' row with min/max bounds (all 8 channels).
    overwrite_cache : bool
        Force re-download of ERA5 even if cached NetCDFs exist.

    Returns
    -------
    (png_path, nc_path)
    """
    # ── Step 0: activate basin (sets domain, channels, extremes, ckpt, tags) ──
    _set_active_basin(basin)
    if model_ckpt is None:
        model_ckpt = MODEL_CKPT_PATH
    if extremes_path is None:
        extremes_path = EXTREMES_PATH

    # ── Step 1: resolve analysis time + input source ──
    if source == "ifs-opendata":
        from ews_opendata import fetch_ifs_opendata, latest_ifs_analysis
        t0 = analysis_time or latest_ifs_analysis()      # ~now − a few hours
    else:
        t0 = analysis_time or _latest_era5_time()        # ~D−5

    log.info("════════════════════════════════════════════════════")
    log.info("  %s TCG Early Warning System", basin)
    log.info("  Basin         : %s  (%s)", basin, BASINS[basin].name)
    log.info("  Input source  : %s", source)
    log.info("  Analysis time : %s UTC", t0.strftime("%Y-%m-%d %H:%M"))
    log.info("  Lead times    : %s h", LEAD_HOURS)
    log.info("  Channels (%d) : %s", N_CHANNELS, CHANNEL_ORDER)
    log.info("════════════════════════════════════════════════════")

    # ── Step 2–3: fetch inputs (ERA5T reanalysis or IFS Open Data) ──
    if source == "ifs-opendata":
        ncp_path, ncs_path = fetch_ifs_opendata(t0, cache_dir=ERA5_CACHE_DIR,
                                                basin_key=basin,
                                                overwrite=overwrite_cache)
    else:
        ncp_path = fetch_ncp(t0, cache_dir=ERA5_CACHE_DIR, overwrite=overwrite_cache)
        ncs_path = fetch_ncs(t0, cache_dir=ERA5_CACHE_DIR, overwrite=overwrite_cache)

    # ── Step 4: Load min-max extremes from extremes.csv (all 8 channels) ──
    if not extremes_path.exists():
        raise FileNotFoundError(
            f"extremes.csv not found at {extremes_path}.\n"
            "Make sure the path points to the file produced by ncio.py "
            "during training dataset assembly, extended with a 'pi' row."
        )
    extremes = load_extremes(extremes_path)

    # ── Step 4 cont.: Build input tensor (vwsh + pi computed here) ──
    X, land_mask, era5_stream = build_input_tensor(ncp_path, ncs_path, t0, extremes)
    if source == "ifs-opendata":
        era5_stream = "IFS-OpenData"        # provenance label; IFS has no expver
    log.info("Input tensor shape: %s", tuple(X.shape))   # (1, 9, 8, H, W)

    # ── Step 5: Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Inference device: %s", device)

    if model_ckpt.exists():
        ckpt  = torch.load(model_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        bf    = int(state["embed.0.weight"].shape[0])   # infer base_filters from checkpoint
        model = TemporalTCG(input_channels=N_CHANNELS, base_filters=bf).to(device)
        model.load_state_dict(state)
        log.info("Checkpoint loaded from %s (base_filters=%d)", model_ckpt, bf)
    else:
        model = TemporalTCG(input_channels=N_CHANNELS, base_filters=32).to(device)
        log.warning(
            "Checkpoint not found at %s — running with random weights. "
            "Outputs are meaningless until a trained checkpoint is loaded.",
            model_ckpt,
        )

    # ── Step 6: Inference ──
    preds = run_inference(X, model, device, land_mask, apply_taper=apply_taper, ocean_fill=ocean_fill, halo=halo, peak_mode=peak_mode)

    # ── Step 7: Save outputs ──
    nc_path  = save_netcdf(preds,   land_mask, t0, era5_stream=era5_stream, source=source, out_dir=OUTPUT_DIR, taper=apply_taper, ocean_fill=ocean_fill, halo=halo, peak_mode=peak_mode)
    png_path = plot_ews_panel(preds, land_mask, t0, source=source, out_dir=OUTPUT_DIR, truth=truth, taper=apply_taper, peak_mode=peak_mode, ocean_fill=ocean_fill, halo=halo)

    # ── Step 7b: also write the dashboard-format file via the shared EWS writer,
    #     so dashboard.py (which reads data/NACSGM/ews/NACSGM_*.nc) picks it up. ──
    prob_stack   = np.stack([preds[lh] for lh in LEAD_HOURS], axis=0).astype("float32")
    land_for_ews = (~land_mask).astype("int8")     # ews_core convention: 1=land, 0=ocean
    basin_cfg    = BASINS[ACTIVE_BASIN]
    if prob_stack.shape[1:] != (basin_cfg.ny, basin_cfg.nx):
        raise ValueError(
            "NACSGM grid mismatch between the model output and basins.py.\n"
            f"  model output (ny, nx) = {prob_stack.shape[1:]}\n"
            f"  basins.py    (ny, nx) = ({basin_cfg.ny}, {basin_cfg.nx})\n"
            f"Set nx={prob_stack.shape[2]}, ny={prob_stack.shape[1]} in basins.py."
        )
    dash_path = ews_core.write_ews_netcdf(
        prob_stack, land_for_ews, basin_cfg, t0, era5_stream=era5_stream, label=source, truth=truth
    )

    log.info("════ EWS run complete ════")
    log.info("PNG       → %s", png_path)
    log.info("NC        → %s", nc_path)
    log.info("Dashboard → %s", dash_path)
    return png_path, nc_path


# ─────────────────────────────────────────────────────────────────────────────
# ── CLI  ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "TCG-NACSGM-EWS — real-time ERA5 fetch + ConvLSTM inference "
            "for tropical cyclogenesis early warning over the NACSGM basin.\n\n"
            "Channels (8): sst, vwsh, d, cape, r, msl, vo, pi\n"
            "Lead times  : 0, 6, 12, 18, 24, 30, 36, 42, 48 h"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--time", type=str, default=None,
        help=(
            "Analysis time in ISO format, e.g. '2024-09-15T12:00:00'. "
            "Defaults to the latest available ERA5 hour (~5 days ago)."
        ),
    )
    parser.add_argument(
        "--basin", choices=sorted(BASINS.keys()), default="NACSGM",
        help="Basin to run (from basins.py). Selects domain, channels, extremes, "
             "checkpoint, and output tags. Default: NACSGM.",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to the trained ConvLSTM checkpoint (.pt). "
             "Default: <repo>/data/<basin>/final/convlstm_<basin>_best.pt",
    )
    parser.add_argument(
        "--extremes", type=Path, default=None,
        help="Path to extremes.csv from the training run. "
             "Default: <repo>/data/<basin>/final/extremes.csv. "
             "Must include a 'pi' row with min/max bounds.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Force re-download of ERA5 even if cached NetCDFs exist.",
    )
    parser.add_argument(
        "--source", choices=["era5", "ifs-opendata"], default="era5",
        help=("Input data source. 'era5' = ERA5T reanalysis (~5-day lag, matches "
              "training). 'ifs-opendata' = ECMWF IFS Open Data (near-real-time, "
              "genuine 0–48 h lead; see ews_opendata.py for the sst/pi caveats)."),
    )
    parser.add_argument(
        "--truth", type=str, default=None,
        help="Ground-truth genesis 'LAT,LON' (e.g. '15.0,-78.0') to mark as a "
             "semi-transparent circle on the t=0 panel for visual QA. "
             "LON may be -180..180 or 0..360.",
    )
    parser.add_argument(
        "--taper", action="store_true",
        help="Apply boundary-artifact suppression (Appendix A) before the ocean "
             "softmax and suffix outputs with '_taper'. Omit for the raw pre-taper "
             "'before' field (original filename, no suffix).",
    )
    parser.add_argument("--ocean-fill", action="store_true",
        help="Fill land cells from nearest ocean before inference (coastal-sliver fix).")
    parser.add_argument("--halo", type=int, default=0,
        help="Inference halo: replicate-pad N cells and crop (S/E/N/W edge-sliver fix). Try 6.")
    parser.add_argument("--argmax", choices=["raw", "component", "centroid"], default="raw",
        help="Sliver-robust point estimate for the predicted-genesis marker.")
    args = parser.parse_args()

    t0 = None
    if args.time:
        t0 = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)

    truth = None
    if args.truth:
        try:
            _lat, _lon = (float(x) for x in args.truth.split(","))
            truth = (_lat, _lon)
        except ValueError:
            parser.error("--truth must be 'LAT,LON', e.g. '15.0,-78.0'")

    run_ews(
        analysis_time   = t0,
        model_ckpt      = args.model,
        extremes_path   = args.extremes,
        overwrite_cache = args.overwrite,
        source          = args.source,
        basin           = args.basin,
        truth           = truth,
        apply_taper     = args.taper,
        ocean_fill      = args.ocean_fill,
        halo            = args.halo,
        peak_mode       = args.argmax,
    )
