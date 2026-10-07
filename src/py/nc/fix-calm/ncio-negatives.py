"""
4. ERA5 Negative Sample Downloader
================================
Downloads ERA5 atmospheric data for non-developing tropical disturbances
identified by hurdat2_nondeveloping_parser.py.

Downloads the SAME variables, levels, and domain as the existing positive
storm pipeline in ncio.py, producing one CSV per disturbance matching
the format of the existing NACSGM storm CSVs.

Workflow:
  1. Reads ../../../data/era5_negative_requests_nacsgm.csv (from hurdat2 parser)
  2. Groups requests by storm_id
  3. For each storm: downloads pressure-level + single-level ERA5 data
  4. Computes derived variables (VWSH, PI) to match positive samples
  5. Saves per-storm CSVs matching the ../../../data/NACSGM/final/NACSGM.csv format
     with origin=-1 (distinguishing from origin=0 spatial context
     and origin=1 genesis locations in positive samples)

Usage:
  python ncio-negatives.py [--requests ../../../data/era5_negative_requests.csv]
                                     [--output-dir ../../../data/NACSGM/negatives]
                                     [--dry-run]

# Tier 1 only (HURDAT2 non-developing TDs):
python ncio-negatives.py --skip-calm

# Tier 1 + Tier 3 (random non-genesis snapshots/year):
python ncio-negatives.py

# Tier 3 only (random non-genesis snapshots/year):
python ncio-negatives.py --skip-tier1

# Dry run first to see what would be downloaded:
python ncio-negatives.py --dry-run

Requires:
  - cdsapi (pip install cdsapi)
  - xarray, netCDF4, numpy, pandas
  - tcpyPI (optional, for potential intensity)
  - Valid CDS API key in ~/.cdsapirc

Author: Nikolay Todorov
For: Tropical Cyclogenesis Prediction Model (Dissertation)
"""

import os
import sys
import argparse
import time as time_module
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Add tcpyPI to path (adjust for the machine)
TCPYPI_PATH = os.path.abspath(r'../../tcpyPI/src')
if TCPYPI_PATH not in sys.path:
    sys.path.insert(0, TCPYPI_PATH)

try:
    from tcpyPI import pi

    TCPYPI_AVAILABLE = True
    print("✓ tcpyPI available for PI calculations")
except ImportError:
    TCPYPI_AVAILABLE = False
    print("⚠ tcpyPI not found — PI will be set to NaN")

# =============================================================================
# CONFIGURATION — match the existing ncio.py / NACS pipeline
# =============================================================================

# Basin domain — NACSGM combined MBR (Caribbean Sea + Gulf of Mexico)
# 0–360 convention (matches TCG-NACS.py GRID_CONFIG)
DOMAIN = {
    "north": 30.0,
    "south": 11.0,
    "west": 263.0,
    "east": 300.0,
}

# CDS API uses ±180 convention; convert west/east if > 180
DOMAIN_CDS = {
    "north": DOMAIN["north"],
    "south": DOMAIN["south"],
    "west": DOMAIN["west"] - 360 if DOMAIN["west"] > 180 else DOMAIN["west"],  # -97
    "east": DOMAIN["east"] - 360 if DOMAIN["east"] > 180 else DOMAIN["east"],  # -60
}

# Grid resolution
GRID_RES = 0.25  # degrees

# Pressure levels for profile variables (match the ncio.py)
PRESSURE_LEVELS = [
    '1000', '975', '950', '925', '900', '875', '850', '825',
    '800', '775', '750', '700', '650', '600', '550', '500',
    '450', '400', '350', '300', '250', '225', '200', '175',
    '150', '125', '100', '70', '50', '30', '20', '10', '7',
    '5', '3', '2', '1'
]

# Pressure-level variables to download
PL_VARIABLES = [
    'temperature',
    'specific_humidity',
    'relative_humidity',
    'u_component_of_wind',
    'v_component_of_wind',
    'vorticity',
    'divergence',
]

# Single-level variables to download
SL_VARIABLES = [
    'sea_surface_temperature',
    'mean_sea_level_pressure',
    'convective_available_potential_energy',
]

# the model's 8 input channels (must match NACS.csv column names)
MODEL_VARIABLES = ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']

# Min-max normalization extremes — loaded at runtime from extremes.csv.
# The file is written by the ERA5 positive-sample pipeline and reflects
# the full 1980-2025 NACSGM domain; it updates automatically as new
# storm years are added.
EXTREMES_PATH = "../../../data/NACSGM/final/extremes.csv"


def load_extremes(path: str = EXTREMES_PATH) -> dict:
    """
    Load per-variable (min, max) from extremes.csv.

    Expected format:
        Variable,min,max
        sst,23.75,35.02
        ...

    Raises FileNotFoundError with a clear message if the file is absent
    so the user knows to run the positive-sample pipeline first.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"extremes.csv not found at {path}\n"
            "Run the ERA5 positive-sample pipeline to generate it, or set\n"
            "EXTREMES_PATH to the correct location before running this script."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    extremes = {
        row['variable'].strip(): (float(row['min']), float(row['max']))
        for _, row in df.iterrows()
    }
    print(f"✓ Loaded normalization extremes from {path} ({len(extremes)} variables)")
    return extremes


# Loaded once at module level; used by normalize() throughout.
EXTREMES = load_extremes()

# =============================================================================
# CALM-PERIOD SAMPLING (easy negatives from TC-free dates)
# =============================================================================
OFF_SEASON_MONTHS = [12, 1, 2, 3, 4]
IN_SEASON_MONTHS  = [5, 6, 7, 8, 9, 10, 11]

def load_ibtracs_active_dates(ibtracs_path, buffer_days=7):
    """Build set of dates when any TC was active in/near NACSGM, ±buffer."""
    print(f"  Reading IBTrACS: {ibtracs_path}")
    df = pd.read_csv(ibtracs_path, low_memory=False, skiprows=[1])

    # Standardize columns
    col_map = {c: c.strip().upper() for c in df.columns}
    df = df.rename(columns=col_map)

    # Find time column
    time_col = next(c for c in ['ISO_TIME', 'TIME', 'DATETIME'] if c in df.columns)
    df['time'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['time'])

    lat_col = 'LAT' if 'LAT' in df.columns else 'LATITUDE'
    lon_col = 'LON' if 'LON' in df.columns else 'LONGITUDE'
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])

    lon = df[lon_col].values.copy()
    lon[lon < 0] += 360
    df['lon_360'] = lon

    MARGIN = 5.0
    basin_mask = (
        (df[lat_col] >= DOMAIN['south'] - MARGIN) &
        (df[lat_col] <= DOMAIN['north'] + MARGIN) &
        (df['lon_360'] >= DOMAIN['west'] - MARGIN) &
        (df['lon_360'] <= DOMAIN['east'] + MARGIN)
    )
    active_dates = set(df[basin_mask]['time'].dt.date.unique())
    print(f"    Active TC dates in/near NACSGM: {len(active_dates)}")

    excluded = set()
    for d in active_dates:
        for offset in range(-buffer_days, buffer_days + 1):
            excluded.add(d + timedelta(days=offset))
    print(f"    Excluded dates (±{buffer_days}d buffer): {len(excluded)}")
    return excluded


def generate_calm_requests(ibtracs_path, target_count=180,
                           buffer_days=7, seed=42,
                           year_start=1980, year_end=None):
    # ERA5 reanalysis lags ~5 days behind real time; clamp to last complete year
    # so random samples never land on a date that doesn't exist yet.
    if year_end is None:
        from datetime import date as _date
        _today = _date.today()
        # If we're in Jan, the prior year may still be incomplete — use year-2 to be safe
        year_end = _today.year - 1 if _today.month > 3 else _today.year - 2
        print(f"  year_end auto-set to {year_end} (safe ERA5 cutoff)")
    """
    Sample calm-period dates and return a DataFrame matching the
    era5_negative_requests.csv format so process_storm() can handle them.
    """
    from datetime import date as date_cls
    rng = np.random.RandomState(seed)

    excluded = load_ibtracs_active_dates(ibtracs_path, buffer_days)

    # Build candidate pool of calm 6-hourly slots
    off_candidates, in_candidates = [], []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            if month == 12:
                n_days = (datetime(year + 1, 1, 1) - datetime(year, month, 1)).days
            else:
                n_days = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

            for day in range(1, n_days + 1):
                d = date_cls(year, month, day)
                if d in excluded:
                    continue
                for hour in [0, 6, 12, 18]:
                    entry = {
                        'year': year, 'month': month, 'date': d,
                        'datetime': datetime(year, month, day, hour),
                    }
                    if month in OFF_SEASON_MONTHS:
                        off_candidates.append(entry)
                    else:
                        in_candidates.append(entry)

    off_df = pd.DataFrame(off_candidates)
    in_df  = pd.DataFrame(in_candidates)
    print(f"  Calm slots — off-season: {len(off_df):,}, in-season: {len(in_df):,}")

    # Sample: ~55% off-season, ~45% in-season
    off_target = int(target_count * 0.55)

    # 1 per (year, month) from off-season
    off_sampled = off_df.groupby(['year', 'month']).apply(
        lambda g: g.sample(n=1, random_state=rng), include_groups=False).reset_index(drop=True)
    if len(off_sampled) > off_target:
        off_sampled = off_sampled.sample(n=off_target, random_state=rng)

    # Fill remainder from in-season
    in_target = target_count - len(off_sampled)
    in_pool = in_df.groupby(['year', 'month']).apply(
        lambda g: g.sample(n=min(3, len(g)), random_state=rng), include_groups=False).reset_index(drop=True)
    if len(in_pool) > in_target:
        in_sampled = in_pool.sample(n=in_target, random_state=rng)
    else:
        in_sampled = in_pool

    sampled = pd.concat([off_sampled, in_sampled], ignore_index=True)
    sampled = sampled.sort_values('datetime').reset_index(drop=True)
    print(f"  Sampled {len(sampled)} calm-period dates")

    # Format as request DataFrame matching era5_negative_requests.csv
    rows = []
    for _, row in sampled.iterrows():
        dt = row['datetime']
        rows.append({
            'storm_id':    f"CLM_{dt.strftime('%Y%m%d%H')}",
            # ID format: CLM_{YYYY}{MMDD}{HH} → year at [4:8], matches
            # HURDAT2's AL052011 slice used by TCG-NACS.py LOYO split
            'year':        row['datetime'].year,
            'datetime':    dt.strftime('%Y-%m-%d %H:%M:%S'),
            'center_lat':  np.nan,
            'center_lon':  np.nan,
            'status':      'NONE',
            'max_wind_kt': 0,
            'era5_north':  DOMAIN['north'],
            'era5_south':  DOMAIN['south'],
            'era5_west':   DOMAIN_CDS['west'],
            'era5_east':   DOMAIN_CDS['east'],
        })

    return pd.DataFrame(rows)

def normalize(value, var_name):
    """Min-max normalize a value to [0, 1] using pre-computed extremes."""
    if np.isnan(value):
        return np.nan
    lo, hi = EXTREMES[var_name]
    return (value - lo) / (hi - lo)

# Wind shear levels (850-200 hPa)
SHEAR_LEVEL_LOW = 850
SHEAR_LEVEL_HIGH = 200

# Temp directory for intermediate NetCDF files
TEMP_DIR = "../../../data/NACSGM/era5_temp"

# CDS API retry settings
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds between retries


# =============================================================================
# ERA5 DOWNLOAD FUNCTIONS
# =============================================================================

def init_cds_client() -> cdsapi.Client:
    """Initialize CDS API client."""
    try:
        client = cdsapi.Client()
        print("✓ CDS API client initialized")
        return client
    except Exception as e:
        print(f"✗ Failed to initialize CDS API client: {e}")
        print("  Make sure ~/.cdsapirc is configured properly.")
        print("  See: https://cds.climate.copernicus.eu/how-to-api")
        raise


def download_pressure_levels(
        client: cdsapi.Client,
        output_file: str,
        year: str,
        month: str,
        days: List[str],
        times: List[str],
) -> bool:
    """
    Download ERA5 pressure-level data for the full basin domain.

    Groups multiple days/times into a single request to minimize
    API calls (CDS prefers fewer, larger requests).
    """
    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': PL_VARIABLES,
        'pressure_level': PRESSURE_LEVELS,
        'year': year,
        'month': month,
        'day': days,
        'time': times,
        'area': [
            DOMAIN_CDS["north"],
            DOMAIN_CDS["west"],
            DOMAIN_CDS["south"],
            DOMAIN_CDS["east"],
        ],
    }

    for attempt in range(MAX_RETRIES):
        try:
            print(f"    Downloading pressure levels: {year}-{month} "
                  f"({len(days)} days, {len(times)} times)...")
            client.retrieve('reanalysis-era5-pressure-levels', request, output_file)
            print(f"    ✓ Saved to {output_file}")
            return True
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time_module.sleep(RETRY_DELAY)

    return False


def download_single_levels(
        client: cdsapi.Client,
        output_file: str,
        year: str,
        month: str,
        days: List[str],
        times: List[str],
) -> bool:
    """
    Download ERA5 single-level (surface) data for the full basin domain.
    """
    request = {
        'product_type': 'reanalysis',
        'data_format': 'netcdf',
        'variable': SL_VARIABLES,
        'year': year,
        'month': month,
        'day': days,
        'time': times,
        'area': [
            DOMAIN_CDS["north"],
            DOMAIN_CDS["west"],
            DOMAIN_CDS["south"],
            DOMAIN_CDS["east"],
        ],
    }

    for attempt in range(MAX_RETRIES):
        try:
            print(f"    Downloading single levels: {year}-{month} "
                  f"({len(days)} days, {len(times)} times)...")
            client.retrieve('reanalysis-era5-single-levels', request, output_file)
            print(f"    ✓ Saved to {output_file}")
            return True
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time_module.sleep(RETRY_DELAY)

    return False


# =============================================================================
# VARIABLE COMPUTATION — match the ncio.py derived variables
# =============================================================================

def compute_wind_shear(ds_pl: xr.Dataset, time_idx: int) -> np.ndarray:
    """
    Compute 850-200 hPa vertical wind shear magnitude.
    Returns (lat, lon) array in m/s.
    """
    try:
        level_dim = 'level' if 'level' in ds_pl.dims else 'pressure_level'
        time_dim = 'valid_time' if 'valid_time' in ds_pl.dims else 'time'

        u_low = ds_pl['u'].sel({level_dim: SHEAR_LEVEL_LOW}, method='nearest')
        v_low = ds_pl['v'].sel({level_dim: SHEAR_LEVEL_LOW}, method='nearest')
        u_high = ds_pl['u'].sel({level_dim: SHEAR_LEVEL_HIGH}, method='nearest')
        v_high = ds_pl['v'].sel({level_dim: SHEAR_LEVEL_HIGH}, method='nearest')

        if time_dim in u_low.dims:
            u_low = u_low.isel({time_dim: time_idx})
            v_low = v_low.isel({time_dim: time_idx})
            u_high = u_high.isel({time_dim: time_idx})
            v_high = v_high.isel({time_dim: time_idx})

        shear = np.sqrt((u_high - u_low) ** 2 + (v_high - v_low) ** 2)
        return shear.values

    except Exception as e:
        print(f"      ⚠ Wind shear computation failed: {e}")
        return None

def compute_potential_intensity(
        sst_grid: np.ndarray,
        msl_grid: np.ndarray,
        t_profile: np.ndarray,
        q_profile: np.ndarray,
        levels: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
) -> np.ndarray:
    """
    Compute potential intensity using tcpyPI for all ocean grid points.

    Parameters
    ----------
    sst_grid : (lat, lon) SST in Kelvin
    msl_grid : (lat, lon) MSL pressure in Pa
    t_profile : (level, lat, lon) temperature in K
    q_profile : (level, lat, lon) specific humidity in kg/kg
    levels : (level,) pressure levels in hPa

    Returns
    -------
    pi_grid : (lat, lon) potential intensity in m/s
    """
    if not TCPYPI_AVAILABLE:
        return np.full((len(lats), len(lons)), np.nan)

    pi_grid = np.full((len(lats), len(lons)), np.nan)

    # Convert units for tcpyPI
    sst_C = sst_grid - 273.15  # K -> °C
    msl_hPa = msl_grid / 100.0  # Pa -> hPa

    for i in range(len(lats)):
        for j in range(len(lons)):
            try:
                sst_val = float(sst_C[i, j])
                msl_val = float(msl_hPa[i, j])

                # Skip land points (SST is NaN or very low)
                if np.isnan(sst_val) or sst_val < 0:
                    continue

                t_col = t_profile[:, i, j] - 273.15  # K -> °C
                q_col = q_profile[:, i, j] * 1000.0  # kg/kg -> g/kg

                # Compute mixing ratio from specific humidity
                # r = q / (1 - q), then convert to g/kg
                q_kgkg = q_profile[:, i, j]
                r_col = (q_kgkg / (1 - q_kgkg)) * 1000.0  # g/kg

                # tcpyPI expects arrays sorted from surface to top
                # ERA5 levels are typically high->low pressure, so reverse if needed
                if levels[0] < levels[-1]:
                    t_col = t_col[::-1]
                    r_col = r_col[::-1]
                    lvls = levels[::-1]
                else:
                    lvls = levels

                result = pi.pi(
                    TEFR=sst_val,
                    MSL=msl_val,
                    P=lvls,
                    TC=t_col,
                    R=r_col,
                    TEFR_is_SST=1,
                    ascent_flag=0,
                    dession_flag=0,
                    V_reduc=0.8,
                    ptop=50,
                    miss_handle=1,
                )

                # result = (VMAX, PMIN, IFL, TO, LNB)
                vmax = result[0]
                if not np.isnan(vmax) and vmax >= 0:
                    pi_grid[i, j] = vmax

            except Exception:
                continue

    return pi_grid


# =============================================================================
# GRID EXTRACTION — produce CSV matching the NACSGM.csv format
# =============================================================================

def extract_grid_to_rows(
        ds_pl: xr.Dataset,
        ds_sl: xr.Dataset,
        time_sel,
        storm_id: str,
        is_origin: bool = False,
) -> List[Dict]:
    """
    Extract full basin grid at one timestamp into list of row dicts
    matching the existing NACSGM.csv format:

    Columns: ID, time, latitude, longitude, sst, msl, cape, r, vo, d, vwsh, pi, origin
    """
    level_dim = 'level' if 'level' in ds_pl.dims else 'pressure_level'

    # --- Select time slice ---
    # --- Resolve dimension names (CDS API v2 uses 'valid_time', older uses 'time') ---
    time_dim = 'valid_time' if 'valid_time' in ds_sl.dims else 'time'
    level_dim = 'level' if 'level' in ds_pl.dims else 'pressure_level'

    # Drop expver dimension if present (ERA5/ERA5T overlap)
    if 'expver' in ds_sl.dims:
        ds_sl = ds_sl.sel(expver=1)
    if 'expver' in ds_pl.dims:
        ds_pl = ds_pl.sel(expver=1)

    # Find nearest time index explicitly
    nc_times = pd.to_datetime(ds_sl[time_dim].values)
    time_target = pd.Timestamp(time_sel)
    time_idx = np.argmin(np.abs(nc_times - time_target))

    def sel_time(ds, var):
        _time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        da = ds[var]
        if _time_dim in da.dims:
            da = da.isel({_time_dim: time_idx})
        return da

    # --- Surface variables ---
    # SST (K in ERA5)
    sst_key = 'sst' if 'sst' in ds_sl else 'skt' if 'skt' in ds_sl else None
    if sst_key:
        sst_da = sel_time(ds_sl, sst_key)
        sst_vals = sst_da.values  # Keep in K for PI calc
    else:
        print("      ⚠ No SST variable found")
        return []

    # MSL (Pa in ERA5)
    msl_key = 'msl' if 'msl' in ds_sl else 'sp' if 'sp' in ds_sl else None
    if msl_key:
        msl_da = sel_time(ds_sl, msl_key)
        msl_vals = msl_da.values  # Keep in Pa for PI calc
    else:
        print("      ⚠ No MSL variable found")
        return []

    # CAPE (J/kg)
    cape_key = 'cape' if 'cape' in ds_sl else None
    if cape_key:
        cape_da = sel_time(ds_sl, cape_key)
        cape_vals = cape_da.values
    else:
        cape_vals = np.full_like(sst_vals, np.nan)

    # --- Pressure-level variables at 850 hPa (representative level) ---
    # Relative humidity at 850 hPa
    r_da = sel_time(ds_pl, 'r').sel({level_dim: 850}, method='nearest')
    r_vals = r_da.values

    # Vorticity at 850 hPa
    vo_da = sel_time(ds_pl, 'vo').sel({level_dim: 850}, method='nearest')
    vo_vals = vo_da.values

    # Divergence at 850 hPa
    d_da = sel_time(ds_pl, 'd').sel({level_dim: 850}, method='nearest')
    d_vals = d_da.values

    # --- Derived variables ---
    # Wind shear (850-200 hPa)
    vwsh_vals = compute_wind_shear(ds_pl, time_idx)
    if vwsh_vals is None:
        vwsh_vals = np.full_like(sst_vals, np.nan)

    # Potential intensity
    lats = ds_sl.latitude.values if 'latitude' in ds_sl.dims else ds_sl.lat.values
    lons = ds_sl.longitude.values if 'longitude' in ds_sl.dims else ds_sl.lon.values
    levels = ds_pl[level_dim].values.astype(float)

    # No need to compute PI for negative storms
    # t_prof = sel_time(ds_pl, 't').values if 't' in ds_pl else sel_time(ds_pl, 'temperature').values
    # q_prof = sel_time(ds_pl, 'q').values if 'q' in ds_pl else sel_time(ds_pl, 'specific_humidity').values
    # pi_vals = compute_potential_intensity(sst_vals, msl_vals, t_prof, q_prof, levels, lats, lons)

    # --- Convert units to match the normalization ---
    # SST: K -> °C (the min/max: 23.75 – 35.02, so definitely °C)
    sst_out = sst_vals - 273.15

    # MSL: keep in Pa (the min/max: 95314 – 102352, so Pa)
    msl_out = msl_vals

    # CAPE: J/kg (the min/max: 0 – 6202, matches J/kg)
    cape_out = cape_vals

    # RH: % (the min/max: 4.16 – 100, so %)
    r_out = r_vals

    # VO: s^-1 (the min/max: -0.000387 – 0.002305, matches ERA5 units)
    vo_out = vo_vals

    # D: s^-1 (the min/max: -0.000586 – 0.000303, matches ERA5 units)
    d_out = d_vals

    # VWSH: m/s (the min/max: 0.017 – 59.68, matches m/s)
    vwsh_out = vwsh_vals

    # PI: m/s
    pi_out = 0

    # --- Force all grids to 2D (lat, lon) ---
    # ERA5 older data can have extra dims like 'expver'
    sst_out = np.squeeze(sst_out)
    msl_out = np.squeeze(msl_out)
    cape_out = np.squeeze(cape_out)
    r_out = np.squeeze(r_out)
    vo_out = np.squeeze(vo_out)
    d_out = np.squeeze(d_out)
    vwsh_out = np.squeeze(vwsh_out)
    pi_out = np.squeeze(pi_out)
    print(f"      DEBUG shapes: sst={sst_out.shape} msl={msl_out.shape} r={r_out.shape}")


    # --- Build rows ---
    time_str = pd.Timestamp(time_sel).strftime('%Y-%m-%d %H:%M:%S') \
        if not isinstance(time_sel, str) else time_sel

    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Convert longitude to match the convention (0-360 vs -180/180)
            lon_out = float(lon)
            if lon_out < 0:
                lon_out += 360  # Convert to 0-360 if the data uses that

            rows.append({
                'ID': storm_id,
                'time': time_str,
                'latitude': round(float(lat), 2),
                'longitude': round(lon_out, 2),
                'sst': normalize(float(sst_out[i, j]), 'sst'),
                'msl': normalize(float(msl_out[i, j]), 'msl'),
                'cape': normalize(float(cape_out[i, j]), 'cape'),
                'r': normalize(float(r_out[i, j]), 'r'),
                'vo': normalize(float(vo_out[i, j]), 'vo'),
                'd': normalize(float(d_out[i, j]), 'd'),
                'vwsh': normalize(float(vwsh_out[i, j]), 'vwsh'),
                'pi': 0.0,  # PI not computed for negatives
                'origin': -1,
            })
    return rows


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def group_requests_by_storm(requests_csv: str) -> Dict[str, pd.DataFrame]:
    """
    Read the ERA5 request CSV and group by storm_id.
    Each storm gets its own set of timestamps to download.
    """
    df = pd.read_csv(requests_csv, parse_dates=['datetime'])

    storms = {}
    for storm_id, group in df.groupby('storm_id'):
        storms[storm_id] = group.sort_values('datetime').reset_index(drop=True)

    print(f"Loaded {len(df)} requests for {len(storms)} storms")
    return storms


def group_by_yearmonth(storm_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group a storm's requests by year-month for efficient CDS API batching.
    CDS works best with one request per year-month.
    """
    storm_df = storm_df.copy()
    storm_df['ym'] = storm_df['datetime'].dt.strftime('%Y-%m')
    return {ym: grp for ym, grp in storm_df.groupby('ym')}


def process_storm(
        client: cdsapi.Client,
        storm_id: str,
        storm_df: pd.DataFrame,
        output_dir: str,
        temp_dir: str,
        dry_run: bool = False,
) -> Optional[str]:
    """
    Process one non-developing storm:
      1. Download ERA5 data for all its timestamps (batched by month)
      2. Extract grid variables at each timestamp
      3. Save combined CSV matching NACSGM.csv format

    Returns path to output CSV, or None if failed.
    """
    year = storm_df['year'].iloc[0]
    output_file = os.path.join(output_dir, f"{storm_id}.csv")

    # Skip if already processed
    if os.path.exists(output_file):
        print(f"  ✓ {storm_id} already processed, skipping")
        return output_file

    print(f"\n  Processing {storm_id} ({len(storm_df)} timestamps, year {year})")

    if dry_run:
        print(f"    [DRY RUN] Would download {len(storm_df)} timestamps")
        return None

    all_rows = []

    # Group timestamps by year-month for efficient downloading
    ym_groups = group_by_yearmonth(storm_df)

    for ym, ym_df in ym_groups.items():
        year_str, month_str = ym.split('-')

        # Unique days and times for this month
        days = sorted(ym_df['datetime'].dt.strftime('%d').unique().tolist())
        times = sorted(ym_df['datetime'].dt.strftime('%H:%M').unique().tolist())

        # File paths for this month's data
        pl_file = os.path.join(temp_dir, f"{storm_id}_{ym}_pl.nc")
        sl_file = os.path.join(temp_dir, f"{storm_id}_{ym}_sl.nc")

        # Download pressure levels
        if not os.path.exists(pl_file):
            success = download_pressure_levels(
                client, pl_file, year_str, month_str, days, times
            )
            if not success:
                print(f"    ✗ Failed to download PL data for {ym}")
                continue

        # Download single levels
        if not os.path.exists(sl_file):
            success = download_single_levels(
                client, sl_file, year_str, month_str, days, times
            )
            if not success:
                print(f"    ✗ Failed to download SL data for {ym}")
                continue

        # Open datasets
        try:
            ds_pl = xr.open_dataset(pl_file, engine='netcdf4')
            ds_sl = xr.open_dataset(sl_file, engine='netcdf4')
        except Exception as e:
            print(f"    ✗ Failed to open NetCDF files for {ym}: {e}")
            continue

        # Extract grid for each timestamp
        for _, row in ym_df.iterrows():
            time_sel = row['datetime']
            print(f"    Extracting {time_sel.strftime('%Y-%m-%d %H:%M')}...")

            try:
                grid_rows = extract_grid_to_rows(
                    ds_pl, ds_sl, time_sel,
                    storm_id=storm_id,
                    is_origin=False,  # non-developing → never origin
                )
                all_rows.extend(grid_rows)
                print(f"      ✓ {len(grid_rows)} grid points")
            except Exception as e:
                import traceback
                print(f"      ✗ Extraction failed: {e}")
                traceback.print_exc()

        ds_pl.close()
        ds_sl.close()

    # Save combined CSV
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(output_file, index=False)
        n_times = df_out['time'].nunique()
        n_points = len(df_out)
        print(f"  ✓ Saved {output_file}: {n_points} rows ({n_times} timesteps)")
        return output_file
    else:
        print(f"  ✗ No data extracted for {storm_id}")
        return None


def merge_negatives_with_positives(
        negative_dir: str,
        positive_csv: str,
        output_csv: str,
) -> pd.DataFrame:
    """
    Merge all negative storm CSVs with the existing positive NACSGM.csv
    to create a combined dataset ready for model training.

    Labeling convention:
      origin =  1  → genesis location (positive sample)
      origin =  0  → non-genesis grid point in a developing storm
      origin = -1  → all grid points in a non-developing storm (negative)

    This three-value scheme lets the dataset class distinguish negative
    samples from spatial context within positive samples, avoiding the
    class-imbalance problem of lumping both as origin=0.
    """
    print(f"\n{'=' * 60}")
    print("MERGING POSITIVES AND NEGATIVES")
    print(f"{'=' * 60}")

    # Load positives
    print(f"Loading positives from {positive_csv}...")
    df_pos = pd.read_csv(positive_csv, parse_dates=['time'])
    n_pos_storms = df_pos['ID'].nunique()
    n_pos_origins = (df_pos['origin'] == 1).sum()
    print(f"  {n_pos_storms} developing storms, {n_pos_origins} genesis points")

    # Load all negative CSVs
    neg_files = sorted(Path(negative_dir).glob("*.csv"))
    print(f"\nLoading {len(neg_files)} negative storm files...")

    neg_dfs = []
    for f in neg_files:
        try:
            df = pd.read_csv(f, parse_dates=['time'])
            neg_dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Skipping {f.name}: {e}")

    if neg_dfs:
        df_neg = pd.concat(neg_dfs, ignore_index=True)
        n_neg_storms = df_neg['ID'].nunique()
        print(f"  {n_neg_storms} non-developing storms, "
              f"{len(df_neg)} total rows, "
              f"origin=-1 count: {(df_neg['origin'] == -1).sum()} (should = total rows)")
    else:
        print("  ⚠ No negative files loaded!")
        return df_pos

    # Merge
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    df_combined = df_combined.sort_values(['ID', 'time']).reset_index(drop=True)

    df_combined.to_csv(output_csv, index=False)

    n_total = df_combined['ID'].nunique()
    print(f"\nCombined dataset: {output_csv}")
    print(f"  Total storms: {n_total} ({n_pos_storms} pos + {n_neg_storms} neg)")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Ratio: {n_neg_storms / n_pos_storms:.1f}:1 neg:pos")

    return df_combined


# =============================================================================
# TIER 3: RANDOM NON-GENESIS SEASON SNAPSHOTS
# =============================================================================

def generate_random_negatives_csv(
        positive_csv: str,
        output_requests_csv: str,
        n_per_year: int = 10,
        exclusion_days: int = 5,
        year_start: int = 1980,
        year_end: int = 2024,
        seed: int = 42,
) -> pd.DataFrame:
    """
    Generate ERA5 download requests for random non-genesis hurricane
    season snapshots (Tier 3 negatives).

    These supplement the HURDAT2 non-developing TDs (Tier 1) with
    "easy negatives" from times when no tropical activity existed.

    Parameters
    ----------
    positive_csv : str
        Path to the existing NACSGM.csv (to find genesis dates to exclude)
    n_per_year : int
        Number of random snapshots per year
    exclusion_days : int
        Exclude dates within ±N days of any genesis event
    """
    np.random.seed(seed)

    # Load genesis dates from positives
    df_pos = pd.read_csv(positive_csv, parse_dates=['time'])
    genesis_rows = df_pos[df_pos['origin'] == 1]
    genesis_dates = genesis_rows.groupby('ID')['time'].first().values
    genesis_dates = pd.to_datetime(genesis_dates)

    print(f"Generating random negative requests...")
    print(f"  Excluding ±{exclusion_days} days around {len(genesis_dates)} genesis events")

    requests = []
    storm_counter = 0

    for year in range(year_start, year_end + 1):
        # Hurricane season: June 1 – November 30
        season_start = pd.Timestamp(f'{year}-06-01')
        season_end = pd.Timestamp(f'{year}-11-30')

        # All synoptic times during the season
        all_times = pd.date_range(season_start, season_end, freq='6h')

        # Exclude windows around genesis events
        year_genesis = genesis_dates[
            (genesis_dates >= season_start) & (genesis_dates <= season_end)
            ]

        valid = np.ones(len(all_times), dtype=bool)
        for gd in year_genesis:
            valid &= np.abs((all_times - gd).total_seconds()) > exclusion_days * 86400

        valid_times = all_times[valid]

        if len(valid_times) == 0:
            continue

        # Sample
        n_sample = min(n_per_year, len(valid_times))
        chosen = np.random.choice(valid_times, size=n_sample, replace=False)

        for t in sorted(chosen):
            storm_counter += 1
            storm_id = f"RND_{year}{storm_counter:04d}"
            # ID format: RND_{YYYY}{NNNN} → year at [4:8], matches
            # HURDAT2's AL052011 slice used by TCG-NACS.py LOYO split
            t_ts = pd.Timestamp(t)  # ensure pandas Timestamp for strftime

            requests.append({
                'storm_id': storm_id,
                'year': year,
                'datetime': t_ts.strftime('%Y-%m-%d %H:%M:%S'),
                'center_lat': np.nan,  # no center — full basin
                'center_lon': np.nan,
                'status': 'NONE',
                'max_wind_kt': 0,
                'era5_north': DOMAIN["north"],
                'era5_south': DOMAIN["south"],
                'era5_west': DOMAIN_CDS["west"],
                'era5_east': DOMAIN_CDS["east"],
            })

    df_requests = pd.DataFrame(requests)
    df_requests.to_csv(output_requests_csv, index=False)

    n_years = year_end - year_start + 1
    print(f"  Generated {len(requests)} random negative requests "
          f"(~{len(requests) / n_years:.0f}/year) over {n_years} years")
    print(f"  Saved to: {output_requests_csv}")

    return df_requests

def group_requests_by_storm_df(df):
    """Group a request DataFrame by storm_id (in-memory version)."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return {sid: grp.sort_values('datetime').reset_index(drop=True)
            for sid, grp in df.groupby('storm_id')}


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 negative samples (non-developing + calm-period)"
    )
    parser.add_argument(
        '--requests',
        default='../../../data/era5_negative_requests_nacsgm.csv',
        help='Tier 1: HURDAT2 non-developing disturbance requests CSV'
    )
    parser.add_argument(
        '--output-dir',
        default='../../../data/NACSGM/negatives',
        help='Directory to save per-storm CSV files'
    )
    parser.add_argument(
        '--temp-dir',
        default=TEMP_DIR,
        help='Directory for intermediate NetCDF files'
    )
    parser.add_argument(
        '--positive-csv',
        default='../../../data/NACSGM/final/NACSGM.csv',
        help='Positive samples CSV (for merging)'
    )
    parser.add_argument(
        '--combined-csv',
        default='../../../data/NACSGM/final/NACSGM_combined.csv',
        help='Output path for merged pos+neg CSV'
    )
    parser.add_argument(
        '--ibtracs',
        default='../../../data/1979-ibtracs.since1980.list.v04r01.csv',
        help='Full IBTrACS CSV for calm-period exclusion calendar'
    )
    parser.add_argument(
        '--calm-count', type=int, default=180,
        help='Number of calm-period snapshots to sample (default: 180)'
    )
    parser.add_argument(
        '--buffer-days', type=int, default=7,
        help='Exclusion buffer around active TC dates (default: 7)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for calm-period sampling (default: 42)'
    )
    parser.add_argument(
        '--skip-tier1', action='store_true',
        help='Skip Tier 1 (non-developing disturbances)'
    )
    parser.add_argument(
        '--skip-calm', action='store_true',
        help='Skip calm-period snapshots'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be downloaded without actually downloading'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    print("=" * 60)
    print("ERA5 NEGATIVE SAMPLE DOWNLOADER")
    print("=" * 60)
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Tier 1:      {'SKIP' if args.skip_tier1 else args.requests}")
    print(f"  Calm period: {'SKIP' if args.skip_calm else f'{args.calm_count} snapshots'}")
    print(f"  Dry run:     {args.dry_run}")

    storms = {}

    # --- Tier 1: Non-developing disturbances from HURDAT2 ---
    if not args.skip_tier1:
        if os.path.exists(args.requests):
            print(f"\n--- Tier 1: Non-developing disturbances ---")
            tier1_storms = group_requests_by_storm(args.requests)
            storms.update(tier1_storms)
            print(f"  Loaded {len(tier1_storms)} non-developing systems")
        else:
            print(f"\n⚠ Tier 1 requests not found: {args.requests}")
            print("  Run HURDAT2-parser.py first, or use --skip-tier1")

    # --- Tier 3: Calm-period basin snapshots ---
    if not args.skip_calm:
        print(f"\n--- Tier 3: Calm-period snapshots ---")
        calm_df = generate_calm_requests(
            ibtracs_path=args.ibtracs,
            target_count=args.calm_count,
            buffer_days=args.buffer_days,
            seed=args.seed,
        )
        # Save for reference
        calm_csv = os.path.join(os.path.dirname(args.output_dir), '../1979-calm-NA-CSGM.csv')
        calm_df.to_csv(calm_csv, index=False)
        print(f"  Saved request list to {calm_csv}")

        calm_storms = group_requests_by_storm_df(calm_df)
        storms.update(calm_storms)
        print(f"  Added {len(calm_storms)} calm-period snapshots")

    if not storms:
        print("\n✗ No storms to process. Check inputs.")
        return

    print(f"\n--- Processing {len(storms)} total negatives ---")

    if not args.dry_run:
        client = init_cds_client()
    else:
        client = None

    results = {}
    for i, (storm_id, storm_df) in enumerate(storms.items()):
        print(f"\n[{i+1}/{len(storms)}] ", end="")
        output = process_storm(
            client, storm_id, storm_df,
            args.output_dir, args.temp_dir,
            dry_run=args.dry_run,
        )
        results[storm_id] = output

    # --- Summary ---
    n_success = sum(1 for v in results.values() if v is not None)
    n_fail = sum(1 for v in results.values() if v is None)

    print(f"\n{'=' * 60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Successful: {n_success}/{len(storms)}")
    print(f"  Failed:     {n_fail}/{len(storms)}")

    # --- Merge with positives ---
    if not args.dry_run and n_success > 0 and os.path.exists(args.positive_csv):
        merge_negatives_with_positives(
            negative_dir=args.output_dir,
            positive_csv=args.positive_csv,
            output_csv=args.combined_csv,
        )

    print("\nDone!")

if __name__ == "__main__":
    main()