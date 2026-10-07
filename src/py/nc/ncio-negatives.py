"""
4. ERA5 Negative Sample Downloader
==================================
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
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

import os
import sys
import glob
import argparse
import time as time_module
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Add tcpyPI to path (adjust for machine)
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
# CONFIGURATION — match existing ncio.py / NACS pipeline
# =============================================================================

# Basin domain — NACSGM combined MBR (Caribbean Sea + Gulf of Mexico)
# 0–360 convention (matches TCG-NACS.py GRID_CONFIG)
DOMAIN = {
    "north": 32.0,
    "south": 11.0,
    "west": 263.0,
    "east": 305.0,
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

# NOTE: No normalization is applied here.
# Values are written in physical units matching the positive sample pipeline
# (ncio.py / NACSGM.csv).  Normalization is applied at training time using
# extremes.csv, exactly as for positive storms.

# =============================================================================
# PI IMPUTATION HELPERS  (shared with nci.py)
# =============================================================================

# Monthly climatological PI for the NACSGM basin (Caribbean Sea + Gulf of Mexico).
# Used as last-resort fallback when both the exact PI cell and its 5×5
# neighbourhood are all NaN (e.g. isolated coastal cells, cold-season edges).
# Units: m/s
# PI fallback = 0.0 for all months.
# Land/coastal cells where tcpyPI returns NaN have no thermodynamic support
# for genesis. Setting PI=0 makes this explicit:
#   - TCG-NACSGM.py  (min-max): 0 → 0.0 (clear minimum, suppresses genesis)
#   - TCG-multilead  (z-score): 0 → ~2σ below mean (strong negative signal)
# This prevents imputed values from creating spurious high-PI regions that
# bias predictions toward coastal/boundary cells.
PI_CLIMO_NACSGM = {
    1:  0.0,   #42.0,   # January
    2:  0.0,   #40.0,   # February
    3:  0.0,   #42.0,   # March
    4:  0.0,   #46.0,   # April
    5:  0.0,   #52.0,   # May
    6:  0.0,   #58.0,   # June
    7:  0.0,   #63.0,   # July
    8:  0.0,   #67.0,   # August
    9:  0.0,   #68.0,   # September  — peak season
    10: 0.0,   #63.0,   # October
    11: 0.0,   #55.0,   # November
    12: 0.0,   #47.0,   # December
}


def impute_pi_grid(pi_grid, month, half_window=2):
    """
    Impute NaN cells in a (H, W) PI grid using:
      1. 5×5 spatial mean of valid neighbours
      2. Monthly NACSGM climatology where the neighbourhood is also all-NaN

    Parameters
    ----------
    pi_grid     : (H, W) float32 array — NaN on land / tcpyPI failures
    month       : int 1-12
    half_window : int — half-size of neighbourhood (default 2 → 5×5)

    Returns
    -------
    imputed   : (H, W) float32 — no NaN values remaining
    n_spatial : int — cells filled by spatial mean
    n_climo   : int — cells filled by climatology
    """
    out      = pi_grid.copy()
    nan_mask = ~np.isfinite(out)
    if not nan_mask.any():
        return out, 0, 0

    nrows, ncols = out.shape
    climo_val    = float(PI_CLIMO_NACSGM.get(month, 60.0))
    n_spatial = 0
    n_climo   = 0

    for r, c in zip(*np.where(nan_mask)):
        r0, r1 = max(0, r - half_window), min(nrows, r + half_window + 1)
        c0, c1 = max(0, c - half_window), min(ncols, c + half_window + 1)
        valid = out[r0:r1, c0:c1]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 0:
            out[r, c] = float(valid.mean())
            n_spatial += 1
        else:
            out[r, c] = climo_val
            n_climo   += 1

    return out, n_spatial, n_climo

# =============================================================================
# NC-SOURCED NEGATIVE STORMS
# =============================================================================
# These 10 storms have NC files already present under data/NACSGM/<var>/
# (sourced from HURDAT2, not IBTrACS) but must be treated as negatives
# (origin = -1).  They are read directly from disk — no CDS download needed.
NC_NEGATIVE_STEMS = [
    '2010090506-53-GASTON',
    '2013052815-22-BARBARA',
    '2018070621-40-BERYL',
    '2020070312-38-FAY',
    '2022081718-61-UNNAMED',
    '2023102112-79-UNNAMED',
    '2025062500-38-BARRY',
    '2025070106-40-CHANTAL',
    '2025092109-80-IMELDA',
    '2025101712-93-MELISSA',
]

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


def compute_positive_distribution(positive_csv, year_start=1980, year_end=None):
    """
    Read the positive samples and return the per-(year, month) GENESIS
    distribution — one count per storm, taken at its origin=1 genesis row.

    Returns:
        dist     : dict {(year, month): n_positive_genesis_events}
        total    : int total positive genesis events (== number of pos storms)
        by_month : dict {month: n}   (marginal, for reporting)
        by_year  : dict {year: n}    (marginal, for reporting)

    Counting only origin=1 rows means each storm is counted once at its
    genesis time/place — exactly the event negatives should balance against,
    not the 9x-inflated lead-time context rows.
    """
    df = pd.read_csv(positive_csv, usecols=['ID', 'time', 'origin'])
    gen = df[df['origin'] == 1].copy()
    if len(gen) == 0:
        raise ValueError(
            f"No origin==1 genesis rows in {positive_csv}. Run the origin "
            "repair in ncio.py before balancing negatives.")
    gen['time'] = pd.to_datetime(gen['time'])
    gen['year'] = gen['time'].dt.year
    gen['month'] = gen['time'].dt.month
    gen = gen.drop_duplicates(subset='ID')  # one genesis per storm

    if year_end is not None:
        gen = gen[(gen['year'] >= year_start) & (gen['year'] <= year_end)]

    dist = gen.groupby(['year', 'month']).size().to_dict()
    by_month = gen.groupby('month').size().to_dict()
    by_year = gen.groupby('year').size().to_dict()
    return dist, int(len(gen)), by_month, by_year


def generate_calm_requests(ibtracs_path, target_count=180,
                           buffer_days=7, seed=42,
                           year_start=1980, year_end=None,
                           off_season_frac=0.00,
                           target_dist=None, already_have=None):
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

    # Peak-season months (Aug-Oct) use a tighter 2-day buffer; all other
    # months use the full buffer_days (default 7). Peak season is so active
    # that wide exclusions leave almost no valid calm slots, starving exactly
    # the (year, month) cells where positives cluster. A 2-day buffer keeps
    # samples clear of active-TC days while opening enough calm dates to match
    # the positive distribution. PEAK_BUFFER_MONTHS is the set affected.
    PEAK_BUFFER_MONTHS = {8, 9, 10}
    PEAK_BUFFER_DAYS = 2
    excluded_wide = load_ibtracs_active_dates(ibtracs_path, buffer_days)
    excluded_peak = load_ibtracs_active_dates(ibtracs_path, PEAK_BUFFER_DAYS)

    # Build candidate pool of calm 6-hourly slots
    off_candidates, in_candidates = [], []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            if month == 12:
                n_days = (datetime(year + 1, 1, 1) - datetime(year, month, 1)).days
            else:
                n_days = (datetime(year, month + 1, 1) - datetime(year, month, 1)).days

            # Peak months (Aug-Oct) use the tighter 2-day buffer; others 7-day.
            excluded = excluded_peak if month in PEAK_BUFFER_MONTHS else excluded_wide

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

    all_df = pd.concat([off_df, in_df], ignore_index=True)

    # ── Distribution-matched sampling ─────────────────────────────────────────
    # If a target per-(year, month) distribution is supplied (from the positive
    # genesis histogram), fill each cell to match the positives, minus whatever
    # the other tiers (Tier 0 + Tier 1) already contribute in that cell. This
    # makes the TOTAL negative set mirror the positives' monthly+annual shape,
    # scaled toward 1:1. No hardcoded month weights.
    if target_dist is not None:
        already = already_have or {}
        # candidate slots indexed by (year, month)
        by_cell = {k: g for k, g in all_df.groupby(['year', 'month'])}
        parts = []
        deficit_total = 0
        unmet = []
        for (yr, mo), need in sorted(target_dist.items()):
            have = already.get((yr, mo), 0)
            want = max(0, need - have)        # calm fills the remaining gap
            if want == 0:
                continue
            deficit_total += want
            pool = by_cell.get((yr, mo))
            if pool is None or len(pool) == 0:
                unmet.append((yr, mo, want))   # no calm slot available here
                continue
            take = min(want, len(pool))
            parts.append(pool.sample(n=take, random_state=rng))
            if take < want:
                unmet.append((yr, mo, want - take))
        sampled = (pd.concat(parts, ignore_index=True)
                   if parts else all_df.iloc[0:0].copy())
        sampled = sampled.sort_values('datetime').reset_index(drop=True)
        print(f"  Distribution-matched calm fill: requested {deficit_total} "
              f"cells-worth, sampled {len(sampled)}")
        if unmet:
            short = sum(u[2] for u in unmet)
            print(f"  ⚠ {short} calm slots unmet across {len(unmet)} "
                  f"(year,month) cells (active-TC exclusions left no calm dates). "
                  f"First few: {unmet[:5]}")
    else:
        # ── Legacy fallback (no target distribution) ──────────────────────────
        # Sample: off_season_frac off-season (default 0.0), remainder in-season,
        # weighted by a fixed approximation of the positive monthly distribution.
        off_target = int(target_count * off_season_frac)
        off_sampled = off_df.groupby(['year', 'month']).apply(
            lambda g: g.sample(n=1, random_state=rng),
            include_groups=False).reset_index(drop=True)
        if len(off_sampled) > off_target:
            off_sampled = off_sampled.sample(n=off_target, random_state=rng)

        MONTH_WEIGHTS = {
            4: 1, 5: 4, 6: 8, 7: 8, 8: 30, 9: 180, 10: 5, 11: 10, 12: 2
        }
        in_target = target_count - len(off_sampled)
        total_weight = sum(MONTH_WEIGHTS.get(m, 0) for m in IN_SEASON_MONTHS)
        month_quota  = {
            m: max(1, round(in_target * MONTH_WEIGHTS.get(m, 0) / total_weight))
            for m in IN_SEASON_MONTHS
        }
        in_parts = []
        for m, quota in month_quota.items():
            m_df = in_df[in_df['month'] == m]
            if len(m_df) == 0:
                continue
            m_pool = m_df.groupby('year').apply(
                lambda g: g.sample(n=1, random_state=rng), include_groups=False
            ).reset_index(drop=True)
            if len(m_pool) > quota:
                m_pool = m_pool.sample(n=quota, random_state=rng)
            in_parts.append(m_pool)
        in_sampled = pd.concat(in_parts, ignore_index=True) if in_parts else pd.DataFrame()

        sampled = pd.concat([off_sampled, in_sampled], ignore_index=True)
        sampled = sampled.sort_values('datetime').reset_index(drop=True)
        print(f"  Sampled {len(sampled)} calm-period dates (legacy weighting)")

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
# VARIABLE COMPUTATION — match ncio.py derived variables
# =============================================================================

def compute_wind_shear(ds_pl: xr.Dataset, time_idx: int) -> np.ndarray:
    """
    Compute 850-200 hPa vertical wind shear magnitude in m/s.

    Mirrors ncio.py ncvwsh2csv() exactly:
      vwsh = sqrt((u200 - u850)^2 + (v200 - v850)^2)

    No normalisation — raw m/s values returned to match the positive
    pipeline output from nci.py.
    """
    try:
        # Resolve dimension names — CDS API v2 uses 'valid_time' / 'pressure_level'
        level_dim = 'pressure_level' if 'pressure_level' in ds_pl.dims else 'level'
        time_dim  = 'valid_time'     if 'valid_time'     in ds_pl.dims else 'time'

        # Drop expver if present (ERA5/ERA5T overlap artefact) — mirrors ncio.py
        _ds = ds_pl
        for drop_var in ['number', 'expver']:
            if drop_var in _ds:
                _ds = _ds.drop_vars([drop_var])

        # Resolve u/v variable names (CDS may use short or long names)
        u_name = ('u' if 'u' in _ds else
                  'u_component_of_wind' if 'u_component_of_wind' in _ds else None)
        v_name = ('v' if 'v' in _ds else
                  'v_component_of_wind' if 'v_component_of_wind' in _ds else None)

        if u_name is None or v_name is None:
            print(f"      ⚠ Wind shear: u/v not found in dataset vars: "
                  f"{list(_ds.data_vars)}")
            return None

        # Select 850 and 200 hPa — matches ncio.py ncvwsh2csv()
        u850 = _ds[u_name].sel({level_dim: 850}, method='nearest')
        v850 = _ds[v_name].sel({level_dim: 850}, method='nearest')
        u200 = _ds[u_name].sel({level_dim: 200}, method='nearest')
        v200 = _ds[v_name].sel({level_dim: 200}, method='nearest')

        # Select time slice if time dimension present
        if time_dim in u850.dims:
            u850 = u850.isel({time_dim: time_idx})
            v850 = v850.isel({time_dim: time_idx})
            u200 = u200.isel({time_dim: time_idx})
            v200 = v200.isel({time_dim: time_idx})

        # Wind shear magnitude — identical to ncio.py
        vwsh = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        return vwsh.values

    except Exception as e:
        print(f"      ⚠ Wind shear computation failed: {type(e).__name__}: {e}")
        return None

def compute_potential_intensity(
        sst_grid: np.ndarray,
        msl_grid: np.ndarray,
        t_profile: np.ndarray,
        q_profile: np.ndarray,
        levels: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        sst_already_celsius: bool = False,
) -> np.ndarray:
    """
    Compute potential intensity using tcpyPI for all ocean grid points.

    Parameters
    ----------
    sst_grid  : (lat, lon) SST — Kelvin if sst_already_celsius=False (default),
                °C if sst_already_celsius=True
    msl_grid  : (lat, lon) MSL pressure in Pa
    t_profile : (level, lat, lon) temperature in K
    q_profile : (level, lat, lon) specific humidity in kg/kg
    levels    : (level,) pressure levels in hPa
    sst_already_celsius : set True if sst_grid is already in °C

    Returns
    -------
    pi_grid : (lat, lon) potential intensity in m/s  (NaN on land/failure)
    """
    if not TCPYPI_AVAILABLE:
        print("      ⚠  tcpyPI not available — PI will be imputed to 0")
        return np.full((len(lats), len(lons)), np.nan)

    pi_grid = np.full((len(lats), len(lons)), np.nan)

    # Convert units for tcpyPI
    # SST: tcpyPI expects °C
    sst_C   = sst_grid if sst_already_celsius else sst_grid - 273.15
    msl_hPa = msl_grid / 100.0   # Pa → hPa

    n_computed = 0
    n_skipped  = 0
    first_error = None   # capture first failure for diagnosis

    for i in range(len(lats)):
        for j in range(len(lons)):
            try:
                sst_val = float(sst_C[i, j])
                msl_val = float(msl_hPa[i, j])

                # Skip land points — ERA5 SST is NaN over land
                if not np.isfinite(sst_val):
                    n_skipped += 1
                    continue

                # Skip genuinely cold water (no TC genesis below ~22°C)
                if sst_val < 10.0:
                    n_skipped += 1
                    continue

                t_col   = t_profile[:, i, j] - 273.15   # K → °C
                q_kgkg  = q_profile[:, i, j]
                r_col   = (q_kgkg / (1.0 - q_kgkg)) * 1000.0   # mixing ratio g/kg

                # tcpyPI expects profiles sorted surface → top (decreasing pressure)
                # ERA5 pressure levels are typically 1000→1 hPa (already decreasing)
                # Reverse only if ascending (1→1000 hPa)
                if levels[0] < levels[-1]:
                    t_col  = t_col[::-1]
                    r_col  = r_col[::-1]
                    lvls   = levels[::-1]
                else:
                    lvls = levels

                result = pi(
                    SSTC=sst_val,
                    MSL=msl_val,
                    P=lvls,
                    TC=t_col,
                    R=r_col,
                    CKCD=0.9,
                    ascent_flag=0,
                    diss_flag=1,
                    V_reduc=0.8,
                    ptop=50,
                    miss_handle=1,
                )

                # result = (VMAX, PMIN, IFL, TO, LNB)
                vmax = result[0]
                if np.isfinite(vmax) and vmax >= 0:
                    pi_grid[i, j] = vmax
                    n_computed += 1

            except Exception as e:
                if first_error is None:
                    first_error = (i, j, float(sst_C[i, j]) if np.isfinite(sst_C[i, j]) else None,
                                   type(e).__name__, str(e))
                continue

    # Diagnostic — guard against all-NaN grid
    valid = pi_grid[np.isfinite(pi_grid)]
    if len(valid):
        print(f"      PI: {n_computed} computed, {n_skipped} land/cold skipped"
              f" — mean={valid.mean():.1f} m/s  max={valid.max():.1f} m/s")
    else:
        print(f"      PI: 0 values computed, {n_skipped} skipped — all NaN")
        if first_error:
            i, j, sst_v, etype, emsg = first_error
            print(f"      First failure at grid [{i},{j}]  sst={sst_v}°C"
                  f"  {etype}: {emsg}")
        else:
            print(f"      No exceptions caught — check TCPYPI_AVAILABLE={TCPYPI_AVAILABLE}"
                  f" and that tcpyPI is importable in this environment")
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
    # ERA5 uses -273.15 as a fill sentinel for land/missing cells — replace with NaN
    # CAPE (J/kg) - read directly from ERA5; xarray decodes _FillValue to NaN
    cape_key = 'cape' if 'cape' in ds_sl else None
    if cape_key:
        cape_da = sel_time(ds_sl, cape_key)
        cape_vals = cape_da.values.astype(np.float32)
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

    # Potential intensity — compute using tcpyPI; impute NaN with spatial mean
    # then monthly climatology (last resort).
    # NOTE: sst_vals is still in Kelvin here (conversion to °C happens below)
    t_prof = sel_time(ds_pl, 't').values if 't' in ds_pl else \
             sel_time(ds_pl, 'temperature').values
    q_prof = sel_time(ds_pl, 'q').values if 'q' in ds_pl else \
             sel_time(ds_pl, 'specific_humidity').values
    pi_vals = compute_potential_intensity(
        sst_vals, msl_vals, t_prof, q_prof, levels, lats, lons,
        sst_already_celsius=False,   # sst_vals is in Kelvin at this point
    )
    # Impute: 5×5 spatial mean → monthly climatology
    _month = pd.Timestamp(time_sel).month
    pi_vals, n_sp, n_cl = impute_pi_grid(pi_vals, _month)
    if n_sp or n_cl:
        print(f"      PI imputation: {n_sp} spatial, {n_cl} climatology")

    # --- Output values in physical units (no normalization) ---
    # Matches positive storm pipeline (ncio.py / NACSGM.csv):
    #   sst  : °C  (converted from K)
    #   msl  : Pa
    #   cape : J/kg
    #   r    : %
    #   vo   : s⁻¹
    #   d    : s⁻¹
    #   vwsh : m/s
    #   pi   : m/s  (0 where undefined — land / cold-edge cells)
    sst_out  = sst_vals - 273.15   # K → °C
    msl_out  = msl_vals            # Pa
    cape_out = cape_vals           # J/kg
    r_out    = r_vals              # %
    vo_out   = vo_vals             # s⁻¹
    d_out    = d_vals              # s⁻¹
    vwsh_out = vwsh_vals           # m/s
    pi_out   = pi_vals             # m/s  (NaN already imputed to 0 above)

    # --- Force all grids to 2D (lat, lon) ---
    # ERA5 older data can have extra dims like 'expver'
    sst_out  = np.squeeze(sst_out)
    msl_out  = np.squeeze(msl_out)
    cape_out = np.squeeze(cape_out)
    r_out    = np.squeeze(r_out)
    vo_out   = np.squeeze(vo_out)
    d_out    = np.squeeze(d_out)
    vwsh_out = np.squeeze(vwsh_out)
    pi_out   = np.squeeze(pi_out)

    # --- Build rows (inner join on SST — drops land points) ---
    time_str = pd.Timestamp(time_sel).strftime('%Y-%m-%d %H:%M:%S') \
        if not isinstance(time_sel, str) else time_sel

    rows = []
    n_land = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            sst_val = float(sst_out[i, j])

            # Inner join on SST: skip land cells (SST is NaN over land in ERA5)
            if not np.isfinite(sst_val):
                n_land += 1
                continue

            lon_out = float(lon)
            if lon_out < 0:
                lon_out += 360   # convert to 0-360 convention

            rows.append({
                'ID':        storm_id,
                'time':      time_str,
                'latitude':  round(float(lat), 2),
                'longitude': round(lon_out, 2),
                'sst':       sst_val,
                'msl':       float(msl_out[i, j]),
                'cape':      float(cape_out[i, j]),
                'r':         float(r_out[i, j]),
                'vo':        float(vo_out[i, j]),
                'd':         float(d_out[i, j]),
                'vwsh':      float(vwsh_out[i, j]),
                'pi':        float(pi_out[i, j]),
                'origin':    -1,
            })
    if n_land:
        print(f"      Land filter: dropped {n_land} NaN-SST cells, "
              f"{len(rows)} ocean rows kept")
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


# =============================================================================
# ABSENT STORM TRACK ROW INJECTION
# =============================================================================
# Storms whose NC files exist on disk but have no rows in NACSGM_combined.csv.
# Their 9 track rows (origin=0) are injected here so nci.py can extract their
# ERA5 variables on the next run. _apply_genesis_fixes in nci.py sets origin=1.
# Coordinates sourced from IBTrACS (lon converted to 0-360).
_INJECT_STORMS = [
    ('1994242N20267',    '1994-08-29 12:00', 20.50, 267.00),  # 71-UNNAMED
    ('1994268N16276',    '1994-09-24 12:00', 16.00, 275.50),  # 89-UNNAMED
    ('1994272N20274',    '1994-09-29 06:00', 20.50, 274.00),  # 92-UNNAMED
    ('2005236N23285',    '2005-08-26 06:00', 25.40, 278.70),  # 61-KATRINA
    ('1983209N12333',    '1983-08-02 06:00', 18.30, 298.20),  # 39-UNNAMED
    # ── Removed (malformed SIDs / fail filter) ────────────────────────────────
    # The three below carried coordinate-injected SIDs and a domain-ENTRY point
    # mislabelled as genesis. Verified against canonical IBTrACS:
    #   KATIA 2012... -> true SID 2011240N10341, genesis 9.5N/-19.0W (Cape Verde),
    #       NATURE=DS, OUT OF DOMAIN  -> DROP
    #   PATTY 2012... -> true SID 2012285N26288, genesis 25.5N/-72.5W,
    #       NATURE=DS  -> DROP (DS not in TS/MX/SS filter)
    #   NICOLE 2016.. -> true SID 2016278N23300, genesis 23.2N/-59.8W,
    #       NATURE=TS, in-domain  -> ADD as a proper positive via the IBTrACS
    #       origins set + ncio.py (NOT injected here, to get the clean SID and
    #       true genesis rather than the malformed domain-entry version).
]

# NC filename stems for timestamp lookup (sst/ used as proxy)
_INJECT_PATTERNS = {
    '1994242N20267':    '-71-UNNAMED',
    '1994268N16276':    '-89-UNNAMED',
    '1994272N20274':    '-92-UNNAMED',
    '2005236N23285':    '-61-KATRINA',
    '2012285N02542876': '-16-PATTY',
    '2011248N02302994': '-12-KATIA',
    '1983209N12333':    '-39-UNNAMED',
    '2016278N02352997': '-15-NICOLE',
}

import re as _re2
_STEM_RE2 = _re2.compile(r'^(\d{10})-(\d+)-([A-Za-z][A-Za-z-]*)$')


def _get_nc_timestamps_inject(nc_dir, pattern, genesis_ts, window_days=10):
    """Return sorted list of (timestamp, number) from NC filenames."""
    sst_dir = Path(nc_dir) / 'sst'
    results = []
    seen_nums = set()
    for f in sst_dir.glob(f'*{pattern}.nc'):
        m = _STEM_RE2.match(f.stem)
        if not m:
            continue
        dt_str = m.group(1)
        try:
            dt = pd.Timestamp(year=int(dt_str[0:4]), month=int(dt_str[4:6]),
                              day=int(dt_str[6:8]),  hour=int(dt_str[8:10]))
        except ValueError:
            continue
        if abs((dt - genesis_ts).total_seconds()) <= window_days * 86400:
            results.append((dt, int(m.group(2))))
            seen_nums.add(int(m.group(2)))
    if len(seen_nums) > 1:
        best = min(seen_nums, key=lambda n: min(
            abs((dt - genesis_ts).total_seconds())
            for dt, num in results if num == n))
        results = [(dt, num) for dt, num in results if num == best]
    return sorted(results)


def _inject_absent_storms(df_pos, nc_dir='../../../data/NACSGM'):
    """
    Append 9 track rows (origin=0) for each absent storm into df_pos
    if not already present. nci.py's _apply_genesis_fixes sets origin=1.
    """
    present  = set(df_pos['ID'].astype(str).unique())
    new_rows = []

    for sid, genesis_time_str, lat, lon in _INJECT_STORMS:
        if sid in present:
            continue
        pattern    = _INJECT_PATTERNS[sid]
        genesis_ts = pd.Timestamp(genesis_time_str)
        ts_list    = _get_nc_timestamps_inject(nc_dir, pattern, genesis_ts)
        if not ts_list:
            print(f"  inject: WARNING — no NC files for {sid} (*{pattern}.nc)")
            continue
        for ts, _ in ts_list:
            # Snap to nearest 0.25° grid point
            _lat = max(11.0, min(30.0, round(round(lat / 0.25) * 0.25, 2)))
            _lon = max(263.0, min(300.0, round(round(lon / 0.25) * 0.25, 2)))
            new_rows.append({
                'ID':        sid,
                'time':      ts,
                'origin':    0,
                'latitude':  _lat,
                'longitude': _lon,
            })
        print(f"  inject: {sid}  {len(ts_list)} rows "
              f"({ts_list[0][0].strftime('%Y-%m-%d')} → "
              f"{ts_list[-1][0].strftime('%Y-%m-%d')})")

    if new_rows:
        inject_df = pd.DataFrame(new_rows)
        df_pos    = pd.concat([df_pos, inject_df], ignore_index=True)
        print(f"  inject: {len(new_rows)} rows added for "
              f"{inject_df['ID'].nunique()} storm(s)")
    else:
        print(f"  inject: all storms already present — nothing to add")

    return df_pos


def merge_negatives_with_positives(
        negative_dir: str,
        positive_csv: str,
        output_csv: str,
        keep_context_only: bool = False,
        inject_absent: bool = False,
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

    # ── Deduplicate genesis labels ────────────────────────────────────────────
    # A storm must have exactly ONE genesis cell (origin=1). Earlier timestamp-
    # snapping fixes (sub-hourly genesis re-snapped to whole hours, e.g. DORIAN
    # 2019, HARVEY 2017) can leave a stale pre-snap origin=1 row beside the
    # corrected one, giving a storm 2 genesis cells. Keep the earliest genesis
    # row per storm and demote extras to origin=0 so the model sees one genesis.
    dup = df_pos.loc[df_pos['origin'] == 1].groupby('ID').size()
    dup_ids = dup[dup > 1].index.tolist()
    if dup_ids:
        print(f"\nDeduplicating genesis for {len(dup_ids)} storm(s) "
              f"with >1 origin=1 row...")
        for sid in dup_ids:
            rows = df_pos[(df_pos['ID'] == sid) & (df_pos['origin'] == 1)]
            keep_idx = rows.sort_values('time').index[0]
            demote = [i for i in rows.index if i != keep_idx]
            df_pos.loc[demote, 'origin'] = 0
            print(f"  {sid}: kept 1 genesis, demoted {len(demote)} extra -> origin=0")

    # ── Keep only positive storms that have a labelled genesis (origin=1) ──────
    # The positive set is DEFINED as storms with a genesis event. Storms present
    # only as origin=0 context (no genesis) are residue — filtered-out systems
    # or foreign per-storm CSVs that share the NACSGM/ directory and got stacked
    # into NACSGM.csv. They are neither positives (no genesis to predict) nor
    # negatives (origin=-1), so they are dropped here. Set keep_context_only=True
    # to retain them (not recommended).
    if not keep_context_only:
        gen_ids = set(df_pos.loc[df_pos['origin'] == 1, 'ID'].astype(str))
        before_storms = df_pos['ID'].nunique()
        df_pos = df_pos[df_pos['ID'].astype(str).isin(gen_ids)].copy()
        dropped = before_storms - df_pos['ID'].nunique()
        print(f"  Genesis-only filter: kept {df_pos['ID'].nunique()} storms "
              f"with a genesis label, dropped {dropped} context-only storm(s)")

    # Inject track rows for absent storms (legacy; off by default now that the
    # genesis-only filter defines the positive set). Enable with inject_absent.
    if inject_absent:
        print("\nInjecting absent storm track rows...")
        df_pos = _inject_absent_storms(df_pos)

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

    # ── Drop physically invalid non-genesis rows ─────────────────────────
    # SST=0   → land cell sentinel (ERA5)
    # MSL=0   → physically impossible, extraction error
    # CAPE=-273.15 → ERA5 land/missing fill value
    _CAPE_FILL_V = -273.15
    _CAPE_TOL_V  =   0.01

    def _drop_bad_rows(df, label):
        is_genesis = df['origin'] == 1
        bad = (
            (df['sst'].astype(float)  == 0.0) |
            (df['msl'].astype(float)  == 0.0) |
            (df['cape'].astype(float).sub(_CAPE_FILL_V).abs() < _CAPE_TOL_V)
        ) & ~is_genesis   # never drop genesis rows
        n_bad = bad.sum()
        if n_bad:
            df = df[~bad].copy()
            print(f"  {label}: dropped {n_bad:,} invalid rows "
                  f"(SST=0 / MSL=0 / CAPE=-273.15) — genesis rows preserved")
        else:
            print(f"  {label}: no invalid rows found ✓")
        return df

    print("\nFiltering invalid rows...")
    df_pos = _drop_bad_rows(df_pos, 'positives')
    df_neg = _drop_bad_rows(df_neg, 'negatives')

    # ── Normalize longitude convention to [0, 360) ────────────────────────────
    # Canonical convention for this dataset is 0-360 (matches the grid bounds
    # 263-305, the PI cache, and the training loader's internal frame). The
    # per-storm variable files are written in -180/+180 (domain _west=-97,
    # _east=-55), while the calm/Tier-0 builders use 0-360 — so a single
    # longitude column must be unified or the same meridian is encoded two ways.
    # Map any negative longitude to 0-360 via (lon % 360). Idempotent —
    # values already in [0,360) are unchanged.
    def _to_360(df, label):
        n_neg = int((df['longitude'] < 0).sum())
        if n_neg:
            df = df.copy()
            df.loc[df['longitude'] < 0, 'longitude'] = (
                df.loc[df['longitude'] < 0, 'longitude'] % 360.0).round(4)
            print(f"  {label}: converted {n_neg:,} rows from -180/+180 to 0-360")
        else:
            print(f"  {label}: longitude already in 0-360 ✓")
        return df

    print("\nNormalizing longitude convention to [0, 360)...")
    df_pos = _to_360(df_pos, 'positives')
    df_neg = _to_360(df_neg, 'negatives')

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


def generate_tier0_requests(stems):
    """
    Build single-timestamp ERA5 download requests for the Tier-0 reclassified
    storms, in the SAME request format the calm/Tier-1 path uses, so they go
    through process_storm() and land in the identical grid + -180/+180 frame
    as every other negative (rather than the legacy 0-360 NC-read path).

    Each stem encodes the genesis date/hour: YYYYMMDDHH-<num>-<NAME>. We take
    that single timestamp as the negative snapshot (these systems did not
    qualify as in-domain genesis, so the timestamp itself is the negative).
    """
    rows = []
    for stem in stems:
        m = _FNAME_RE.match(stem)
        if not m:
            print(f"  ⚠ Tier0: cannot parse stem {stem} — skipping")
            continue
        dt_str = m.group(1)
        dt = datetime(int(dt_str[0:4]), int(dt_str[4:6]),
                      int(dt_str[6:8]), int(dt_str[8:10]))
        rows.append({
            'storm_id':   stem,              # keep full stem as ID
            'year':       dt.year,
            'datetime':   dt.strftime('%Y-%m-%d %H:%M:%S'),
            'center_lat': np.nan,
            'center_lon': np.nan,
            'status':     'NONE',
            'max_wind_kt': 0,
            'era5_north': DOMAIN['north'],
            'era5_south': DOMAIN['south'],
            'era5_west':  DOMAIN_CDS['west'],
            'era5_east':  DOMAIN_CDS['east'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# NC-SOURCED NEGATIVE PROCESSING  (read from disk, no CDS download)
# =============================================================================

import re as _re
_FNAME_RE = _re.compile(r'^(\d{10})-(\d+)-([A-Za-z][A-Za-z-]*)$')
_SST_K_THRESHOLD = 200.0
_COORD_NAMES = {
    'latitude', 'longitude', 'lat', 'lon', 'time', 'valid_time',
    'number', 'expver', 'step', 'level', 'realization',
}

NC_VARS = ['sst', 'msl', 'cape', 'r', 'vo', 'd']   # from .nc files (direct read)
# vwsh is computed from u/v in its NC file — handled separately
# pi comes from .csv files (same as nci.py)


def _compute_vwsh_from_nc(path: Path,
                           n_lat: int = 77, n_lon: int = 149) -> Optional[np.ndarray]:
    """
    Compute 850-200 hPa wind shear from a vwsh NC file that contains u/v
    at multiple pressure levels (as downloaded by ncio.py ncvwsh2csv).

    Mirrors ncio.py exactly:
        vwsh = sqrt((u200 - u850)^2 + (v200 - v850)^2)

    Returns (77, 149) float32 array in m/s, or None on failure.
    """
    try:
        ds = xr.open_dataset(str(path), engine='netcdf4')

        # Drop expver/number if present — mirrors ncio.py
        for drop_var in ['number', 'expver']:
            if drop_var in ds:
                ds = ds.drop_vars([drop_var])

        # Rename valid_time → time if needed
        if 'valid_time' in ds.dims and 'time' not in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        level_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'

        # Resolve u/v variable names
        u_name = ('u' if 'u' in ds else
                  'u_component_of_wind' if 'u_component_of_wind' in ds else None)
        v_name = ('v' if 'v' in ds else
                  'v_component_of_wind' if 'v_component_of_wind' in ds else None)

        if u_name is None or v_name is None:
            print(f"\n      ⚠  vwsh NC has no u/v vars — found: {list(ds.data_vars)}")
            ds.close()
            return None

        # Select 850 and 200 hPa — same levels as ncio.py ncvwsh2csv
        u850 = ds[u_name].sel({level_dim: 850}, method='nearest').squeeze()
        v850 = ds[v_name].sel({level_dim: 850}, method='nearest').squeeze()
        u200 = ds[u_name].sel({level_dim: 200}, method='nearest').squeeze()
        v200 = ds[v_name].sel({level_dim: 200}, method='nearest').squeeze()

        # Wind shear magnitude — identical formula to ncio.py
        vwsh = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        data = vwsh.values.astype(np.float32).squeeze()
        ds.close()

        if data.ndim == 1 and data.shape[0] == n_lat * n_lon:
            data = data.reshape(n_lat, n_lon)
        if data.ndim != 2:
            return None

        n_valid = int(np.isfinite(data).sum())
        print(f"\n      vwsh: computed from u/v 850-200 hPa  "
              f"mean={np.nanmean(data):.2f} m/s  n_valid={n_valid}")
        return data

    except Exception as e:
        print(f"\n      ⚠  vwsh computation from NC failed: {type(e).__name__}: {e}")
        return None


def _read_nc_grid(path: Path, var_hint: str) -> Optional[np.ndarray]:
    """Read a single-timestep NC file → (77, 149) float32 grid, or None."""
    try:
        ds = xr.open_dataset(str(path), engine='netcdf4')
        candidates = [v for v in ds.data_vars if v.lower() not in _COORD_NAMES]
        dvar = (var_hint if var_hint in candidates else
                var_hint.lower() if var_hint.lower() in candidates else
                candidates[0] if candidates else None)
        if dvar is None:
            ds.close()
            return None
        data = ds[dvar].values.astype(np.float32).squeeze()
        ds.close()
        if data.ndim == 1 and data.shape[0] == 77 * 149:
            data = data.reshape(77, 149)
        if data.ndim != 2:
            return None
        if np.nanmax(data) > _SST_K_THRESHOLD:
            data = data - 273.15   # K → °C
        return data
    except Exception as e:
        print(f"      ⚠  could not read {path.name}: {e}")
        return None


def _read_pi_csv(path: Path,
                 n_lat: int = 77, n_lon: int = 149) -> np.ndarray:
    """
    Read a PI CSV into a (77, 149) float32 grid aligned to the NACSGM grid.

    Returns a (n_lat, n_lon) array with NaN on land cells and missing values.
    Uses nearest-grid-cell snapping (0.25° tolerance) to avoid floating-point
    key-mismatch issues that occur with dict lookups.
    """
    try:
        df = pd.read_csv(str(path), dtype={'latitude': float,
                                           'longitude': float, 'pi': float})
        grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

        # Build index maps from the NACSGM grid
        lats_grid = np.round(np.linspace(30.0, 11.0, n_lat), 2)   # decreasing
        lons_grid = np.round(np.linspace(263.0, 300.0, n_lon), 2)  # increasing

        for r in df.itertuples():
            if not pd.notna(r.pi):
                continue
            # Snap to nearest grid row/col
            lat_r = round(float(r.latitude), 2)
            lon_r = round(float(r.longitude), 2)
            row = int(np.argmin(np.abs(lats_grid - lat_r)))
            col = int(np.argmin(np.abs(lons_grid - lon_r)))
            grid[row, col] = float(r.pi)

        n_valid = int(np.isfinite(grid).sum())
        return grid, n_valid

    except Exception as e:
        print(f"      ⚠  could not read PI CSV {path.name}: {e}")
        return np.full((n_lat, n_lon), np.nan, dtype=np.float32), 0


def process_nc_negatives(
        nc_dir: Path,
        output_dir: str,
        stems: List[str],
        dry_run: bool = False,
) -> Dict[str, Optional[str]]:
    """
    Convert the NC-sourced misclassified storms (already on disk) to
    negative-sample CSVs with origin = -1.

    For each stem in `stems`:
      - Reads  <nc_dir>/<var>/<stem>.nc  for each of the 7 NC variables
      - Reads  <nc_dir>/pi/<stem>.csv    for PI
      - Drops land points (inner join on SST: rows where SST is NaN)
      - Imputes missing PI with 0
      - Writes  <output_dir>/<stem>.csv  with origin = -1

    Returns {stem: output_path_or_None}
    """
    results = {}

    # Pre-build lat/lon grids (77×149, decreasing lat)
    lats = np.round(np.linspace(30.0, 11.0, 77), 2)
    lons = np.round(np.linspace(263.0, 300.0, 149), 2)

    print(f"\n--- Tier 0: NC-sourced misclassified negatives ({len(stems)}) ---")

    for stem in stems:
        out_path = os.path.join(output_dir, f"{stem}.csv")

        # If already processed, check whether PI is populated — reprocess if blank
        if os.path.exists(out_path):
            try:
                _existing = pd.read_csv(out_path)
                pi_blank   = ('pi'   not in _existing.columns or
                              _existing['pi'].isna().all() or
                              (_existing['pi'] == 0).all())
                vwsh_blank = ('vwsh' not in _existing.columns or
                              _existing['vwsh'].isna().all() or
                              (_existing['vwsh'] == 0).all())
                if not pi_blank and not vwsh_blank:
                    print(f"  ✓ {stem} already processed, skipping")
                    results[stem] = out_path
                    continue
                else:
                    reasons = []
                    if pi_blank:   reasons.append('PI blank')
                    if vwsh_blank: reasons.append('vwsh blank')
                    print(f"  ↺ {stem} exists but {', '.join(reasons)} — reprocessing")
                    os.remove(out_path)
            except Exception:
                pass   # unreadable CSV → fall through and reprocess

        # Parse datetime and storm name from stem
        m = _FNAME_RE.match(stem)
        if not m:
            print(f"  ✗ {stem}: cannot parse filename — skipping")
            results[stem] = None
            continue
        dt_str, storm_name = m.group(1), m.group(3).upper()
        try:
            dt = pd.Timestamp(
                year=int(dt_str[0:4]), month=int(dt_str[4:6]),
                day=int(dt_str[6:8]),  hour=int(dt_str[8:10]),
            )
        except ValueError:
            print(f"  ✗ {stem}: cannot parse datetime — skipping")
            results[stem] = None
            continue

        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Processing {stem}  ({time_str}) ...", end='', flush=True)

        if dry_run:
            print("  [DRY RUN]")
            results[stem] = None
            continue

        # ── Read NC grids ──────────────────────────────────────────────────
        grids = {}
        any_missing = False
        for var in NC_VARS:
            nc_path = nc_dir / var / f"{stem}.nc"
            if not nc_path.exists():
                print(f"\n      ⚠  {var}/{stem}.nc not found")
                any_missing = True
                grids[var] = None
            else:
                grids[var] = _read_nc_grid(nc_path, var)
                if grids[var] is None:
                    any_missing = True

        # ── Compute vwsh from u/v in vwsh NC file ─────────────────────────
        vwsh_nc_path = nc_dir / 'vwsh' / f"{stem}.nc"
        if vwsh_nc_path.exists():
            grids['vwsh'] = _compute_vwsh_from_nc(vwsh_nc_path)
        else:
            print(f"\n      ⚠  vwsh/{stem}.nc not found — vwsh will be NaN")
            grids['vwsh'] = None

        # ── Read PI CSV → 2D grid, then impute NaN ────────────────────────
        pi_path = nc_dir / 'pi' / f"{stem}.csv"
        if pi_path.exists():
            pi_grid, n_pi = _read_pi_csv(pi_path)
            print(f"\n      PI CSV: {n_pi} ocean values loaded from {pi_path.name}")
        else:
            pi_grid = np.full((77, 149), np.nan, dtype=np.float32)
            n_pi = 0
            print(f"\n      ⚠  pi/{stem}.csv not found — "
                  f"using spatial+climatological imputation")

        # Impute NaN cells: 5×5 spatial mean → monthly climatology
        pi_grid, n_sp, n_cl = impute_pi_grid(pi_grid, dt.month)
        if n_sp or n_cl:
            print(f"      PI imputation: {n_sp} spatial mean, "
                  f"{n_cl} climatology (month={dt.month})")

        # ── Build rows (inner join on SST = ocean cells only) ─────────────
        sst_grid = grids.get('sst')
        if sst_grid is None:
            print(f"\n  ✗ {stem}: SST grid unavailable — cannot inner-join, skipping")
            results[stem] = None
            continue

        rows = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                sst_val = float(sst_grid[i, j])
                if not np.isfinite(sst_val):
                    continue   # land point — skip (inner join on SST)

                # PI already fully imputed — no cell should be NaN at this point
                pi_val = float(pi_grid[i, j])

                row = {
                    'ID':        stem,
                    'time':      time_str,
                    'latitude':  round(float(lat), 2),
                    'longitude': round(float(lon), 2),
                    'sst':       sst_val,
                    'msl':       float(grids['msl'][i, j])  if grids.get('msl')  is not None else np.nan,
                    'cape':      float(grids['cape'][i, j]) if grids.get('cape') is not None else np.nan,
                    'r':         float(grids['r'][i, j])    if grids.get('r')    is not None else np.nan,
                    'vo':        float(grids['vo'][i, j])   if grids.get('vo')   is not None else np.nan,
                    'd':         float(grids['d'][i, j])    if grids.get('d')    is not None else np.nan,
                    'vwsh':      float(grids['vwsh'][i, j]) if grids.get('vwsh') is not None else np.nan,
                    'pi':        pi_val,
                    'origin':    -1,
                }
                rows.append(row)

        if not rows:
            print(f"\n  ✗ {stem}: no ocean rows extracted — skipping")
            results[stem] = None
            continue

        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, index=False)
        print(f"  ✓  {len(rows)} ocean rows → {out_path}")
        results[stem] = out_path

    n_ok   = sum(1 for v in results.values() if v is not None)
    n_fail = len(results) - n_ok
    print(f"\n  NC negatives: {n_ok} ok, {n_fail} failed")
    return results


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 negative samples (non-developing + calm-period)"
    )
    parser.add_argument(
        '--requests',
        default='../../../data/1979-negative-NA-CSGM.csv',
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
        default='../../../data/1979-NA-CSGM-ibtracs.since1980.list.v04r01.csv',
        help='Full IBTrACS CSV for calm-period exclusion calendar'
    )
    parser.add_argument(
        '--calm-count', type=int, default=400,
        help='Number of calm-period snapshots to sample (default: 400)'
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
    parser.add_argument(
        '--nc-dir',
        default='../../../data/NACSGM',
        help='Root dir containing cape/, sst/, pi/, … subdirs for NC-sourced negatives'
    )
    parser.add_argument(
        '--skip-nc-negatives', action='store_true',
        help='Skip processing of the 10 NC-sourced misclassified negatives'
    )
    parser.add_argument(
        '--skip-merge', action='store_true',
        help='Skip merging negatives with positives into NACSGM_combined.csv'
    )
    parser.add_argument(
        '--merge-only', action='store_true',
        help='Skip all downloads; only merge existing negatives with positives'
    )
    parser.add_argument(
        '--off-season-frac', type=float, default=0.00,
        help='Fraction of Tier3 calm storms from off-season (default: 0.00 = on-season only)'
    )
    parser.add_argument(
        '--match-positives', action='store_true',
        help='Make the TOTAL negative set match the positive per-(year,month) '
             'genesis distribution. Tier 3 calm fills each cell to the positive '
             'count (scaled by --neg-pos-ratio) minus what Tier 0+1 supply. '
             'Overrides --calm-count for Tier 3 sizing.'
    )
    parser.add_argument(
        '--neg-pos-ratio', type=float, default=1.0,
        help='Target negative:positive ratio when --match-positives is set '
             '(default: 1.0 = balanced 1:1).'
    )
    parser.add_argument(
        '--tier0-from-disk', action='store_true',
        help='Use the legacy disk-read path for Tier-0 (reads pre-existing '
             '.nc files via process_nc_negatives). Default is to DOWNLOAD the '
             '10 Tier-0 storms fresh from CDS like Tier 1/3, since their .nc '
             'source files are not present.'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # ── Optional: positive-matched target distribution ────────────────────────
    target_dist = None
    pos_total = None
    if args.match_positives:
        target_dist, pos_total, by_month, by_year = compute_positive_distribution(
            args.positive_csv)
        # scale by requested neg:pos ratio
        if args.neg_pos_ratio != 1.0:
            target_dist = {k: int(round(v * args.neg_pos_ratio))
                           for k, v in target_dist.items()}
        scaled_total = sum(target_dist.values())
        print(f"  Positive genesis events: {pos_total} "
              f"(target negatives ≈ {scaled_total} at "
              f"{args.neg_pos_ratio:.2f}:1)")
        _mo_str = ', '.join(f'{m}:{by_month.get(m,0)}' for m in range(1, 13)
                            if by_month.get(m, 0))
        print(f"  Positive monthly genesis: {_mo_str}")

    print("=" * 60)
    print("ERA5 NEGATIVE SAMPLE DOWNLOADER")
    print("=" * 60)
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Tier 0 (NC neg):  {'SKIP' if args.skip_nc_negatives or args.merge_only else f'{len(NC_NEGATIVE_STEMS)} storms from {args.nc_dir}'}")
    print(f"  Tier 1:           {'SKIP' if args.skip_tier1 or args.merge_only else args.requests}")
    print(f"  Calm period:      {'SKIP' if args.skip_calm or args.merge_only else f'{args.calm_count} snapshots'}")
    print(f"  Merge:            {'SKIP' if args.skip_merge else args.combined_csv}")
    print(f"  Dry run:          {args.dry_run}")

    if not args.merge_only:
        storms = {}

        # --- Tier 0: Reclassified false-positive storms ---
        # These 10 systems were initially in the positive set but determined
        # not to be in-domain genesis events. Their source .nc files do not
        # exist on this machine, so they are DOWNLOADED fresh via the same CDS
        # path as Tier 1/3 (single genesis-timestamp snapshot each), landing in
        # the identical grid + -180/+180 frame. The legacy disk-read path
        # (process_nc_negatives) is used only if --tier0-from-disk is passed.
        if not args.skip_nc_negatives:
            if args.tier0_from_disk:
                process_nc_negatives(
                    nc_dir=Path(args.nc_dir),
                    output_dir=args.output_dir,
                    stems=NC_NEGATIVE_STEMS,
                    dry_run=args.dry_run,
                )
            else:
                print(f"\n--- Tier 0: Reclassified false positives "
                      f"({len(NC_NEGATIVE_STEMS)}, CDS download) ---")
                tier0_df = generate_tier0_requests(NC_NEGATIVE_STEMS)
                tier0_storms = group_requests_by_storm_df(tier0_df)
                storms.update(tier0_storms)
                print(f"  Queued {len(tier0_storms)} Tier-0 storms for download")

        # --- Tier 1: Non-developing disturbances from HURDAT2 ---
        if not args.skip_tier1:
            if os.path.exists(args.requests):
                print(f"\n--- Tier 1: Non-developing disturbances ---")
                tier1_storms = group_requests_by_storm(args.requests)
                # Cap off-season Tier1 to respect --off-season-frac
                _on_ids, _off_ids = [], []
                for _sid, _df in tier1_storms.items():
                    _mo = pd.to_datetime(_df['datetime'].iloc[0]).month
                    if _mo in OFF_SEASON_MONTHS:
                        _off_ids.append(_sid)
                    else:
                        _on_ids.append(_sid)
                _off_budget = int((len(tier1_storms) + args.calm_count)
                                  * args.off_season_frac)
                if len(_off_ids) > _off_budget:
                    import random as _rand
                    _rand.seed(args.seed)
                    _off_ids = _rand.sample(_off_ids, _off_budget)
                    print(f"  Off-season Tier1 capped to {len(_off_ids)} "
                          f"(targeting {args.off_season_frac*100:.0f}% off-season)")
                _kept = {k: v for k, v in tier1_storms.items()
                         if k in set(_on_ids) | set(_off_ids)}
                storms.update(_kept)
                print(f"  Loaded {len(_kept)} non-developing systems "
                      f"({len(_on_ids)} on-season, {len(_off_ids)} off-season)")
            else:
                print(f"\n⚠ Tier 1 requests not found: {args.requests}")
                print("  Run HURDAT2-parser.py first, or use --skip-tier1")

        # --- Tier 3: Calm-period basin snapshots ---
        if not args.skip_calm:
            print(f"\n--- Tier 3: Calm-period snapshots ---")

            # When matching positives, count what Tier 0 + Tier 1 already
            # contribute per (year, month) so calm only fills the remaining
            # deficit in each cell. Tier 0 NC negatives live on disk; Tier 1
            # systems are in `storms` at this point.
            already_have = None
            if args.match_positives:
                already_have = {}
                # Tier 0 (CDS mode) + Tier 1 are in `storms` now — count them.
                for _sid, _sdf in storms.items():
                    _dt0 = pd.to_datetime(_sdf['datetime'].iloc[0])
                    _key = (_dt0.year, _dt0.month)
                    already_have[_key] = already_have.get(_key, 0) + 1
                # Only in legacy --tier0-from-disk mode is Tier 0 written to
                # disk (not in `storms`); count those files to avoid missing
                # them. In the default CDS mode they're already in `storms`
                # above, so skip the glob to prevent double-counting.
                if args.tier0_from_disk:
                    for _f in glob.glob(os.path.join(args.output_dir, '*.csv')):
                        try:
                            _t = pd.read_csv(_f, usecols=['time'])['time']
                            _d = pd.to_datetime(_t.iloc[0])
                            _key = (_d.year, _d.month)
                            already_have[_key] = already_have.get(_key, 0) + 1
                        except Exception:
                            pass

            calm_df = generate_calm_requests(
                ibtracs_path=args.ibtracs,
                target_count=args.calm_count,
                buffer_days=args.buffer_days,
                seed=args.seed,
                off_season_frac=args.off_season_frac,
                target_dist=target_dist,
                already_have=already_have,
            )
            calm_csv = os.path.join(os.path.dirname(args.output_dir), '../1979-calm-NA-CSGM.csv')
            calm_df.to_csv(calm_csv, index=False)
            print(f"  Saved request list to {calm_csv}")
            calm_storms = group_requests_by_storm_df(calm_df)
            storms.update(calm_storms)
            print(f"  Added {len(calm_storms)} calm-period snapshots")

        if storms:
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

            n_success = sum(1 for v in results.values() if v is not None)
            n_fail    = sum(1 for v in results.values() if v is None)
            print(f"\n{'=' * 60}")
            print(f"DOWNLOAD COMPLETE")
            print(f"{'=' * 60}")
            print(f"  Successful: {n_success}/{len(storms)}")
            print(f"  Failed:     {n_fail}/{len(storms)}")
        else:
            print("\n  No download storms to process.")

    # --- Merge with positives ---
    if not args.skip_merge and not args.dry_run:
        if os.path.exists(args.positive_csv):
            merge_negatives_with_positives(
                negative_dir=args.output_dir,
                positive_csv=args.positive_csv,
                output_csv=args.combined_csv,
            )
        else:
            print(f"\n⚠  Positive CSV not found: {args.positive_csv}")
            print("   Run nci.py first to generate NACSGM.csv, then re-run with --merge-only")

    print("\nDone!")

if __name__ == "__main__":
    main()