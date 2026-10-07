"""
3. nci_wp.py
============
Primary positives builder for WP-MM basin.

For each positive storm in IBTrACS WP (NATURE in {TS, SS, MX}, TRACK_TYPE='main'),
this script reads the per-storm NetCDF files that ncio.py downloaded and
assembles a per-cell x per-timestep CSV with raw (un-normalised) variable
values, in the format required by the training scripts.

This is the WP equivalent of the original NACSGM ncio.py role -- it produces
WPMM.csv (positives only, raw variables, full grid). ncio-neg-wp.py then
extends it with negatives to produce WPMM_combined.csv.

NC file structure (confirmed from inspection)
---------------------------------------------
  Filename   : {YYYYMMDDHH}-{season_number}-{storm_name}.nc
               e.g.  2024101900-16-OSCAR.nc

  Directories: ../../../data/WPMM/<variable>/<YYYYMMDDHH>-<NN>-<NAME>.nc
               where <variable> in {sst, msl, cape, r, vo, d, vwsh, pi}

  Single-level vars (sst, msl, cape):
               dims (1, 161, 301), data variable named same as subdir

  Pressure-level vars (r, vo, d):
               dims (1, 1, 161, 301), single pressure level baked in

  Wind shear (vwsh):
               dims (1, 2, 161, 301), data variables 'u' and 'v'
               at two pressure levels (200 + 850 hPa)
               Computed:  vwsh = sqrt((u_top - u_bot)^2 + (v_top - v_bot)^2)

  Potential intensity (pi):
               Two files per timestep: {stem}_pl.nc + {stem}_sl.nc
               Contain inputs only (t, q profiles + SST/MSL); PI computation
               itself is non-trivial. For v1 we leave PI as NaN, matching
               the OLD WPMM.csv.bak behavior. PI computation can be added
               later via tcpyPI without changing the output schema.

Output: data/WPMM/final/WPMM.csv
  Columns: ID, latitude, longitude, time, sst, vwsh, d, cape, r, msl, vo, pi, origin
  One row per (storm, timestep, lat-cell, lon-cell) where SST is non-NaN
  (land cells dropped). origin=1 only at genesis cell on t=0; origin=0
  elsewhere.

Usage:
  python nci_wp.py
  python nci_wp.py --dry-run                     # process 1 storm only, sanity check
  python nci_wp.py --restart                     # delete existing WPMM.csv first
  python nci_wp.py --limit 5                     # process first 5 storms only
  python nci_wp.py --ibtracs path/to/ibtracs.csv

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time as time_module
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG
# =============================================================================
_DATA_ROOT     = Path('../../../data/WPMM')
_FINAL_ROOT    = _DATA_ROOT / 'final'
_OUT_CSV       = _FINAL_ROOT / 'WPMM.csv'
_IBTRACS_CSV   = Path('../../../data/1979-WP-MM-origins-ibtracs.since1980.list.v04r01.csv')

# Basin grid (5-45N, 105-180E, 0.25 deg)
DOMAIN = {'south': 5.0, 'north': 45.0, 'west': 105.0, 'east': 180.0}
GRID_RES = 0.25
N_LAT = int(round((DOMAIN['north'] - DOMAIN['south']) / GRID_RES)) + 1   # 161
N_LON = int(round((DOMAIN['east']  - DOMAIN['west'])  / GRID_RES)) + 1   # 301

# Lat/lon arrays in the same order ncio.py wrote them: lat descending, lon ascending
LATS_DESC = np.round(np.linspace(DOMAIN['north'], DOMAIN['south'], N_LAT), 2)
LONS_ASC  = np.round(np.linspace(DOMAIN['west'],  DOMAIN['east'],  N_LON), 2)

# Timestep offsets in hours from genesis (t=0 is genesis; negatives are PRE-genesis)
TIMESTEP_OFFSETS_H = [0, -6, -12, -18, -24, -30, -36, -42, -48]

# IBTrACS classification
POSITIVE_NATURES   = {'TS', 'SS', 'MX'}
ALLOWED_TRACK_TYPE = 'main'

# Output columns in the same order as the OLD WPMM.csv.bak
OUTPUT_COLS = ['ID', 'latitude', 'longitude', 'time',
               'sst', 'vwsh', 'd', 'cape', 'r', 'msl', 'vo', 'pi', 'origin']

# Per-variable NC config (vwsh and pi are special-cased below)
VAR_CONFIG = {
    'sst':  {'subdir': 'sst',  'nc_var': 'sst',  'kelvin_to_c': True},
    'msl':  {'subdir': 'msl',  'nc_var': 'msl',  'kelvin_to_c': False},
    'cape': {'subdir': 'cape', 'nc_var': 'cape', 'kelvin_to_c': False},
    'r':    {'subdir': 'r',    'nc_var': 'r',    'kelvin_to_c': False},
    'vo':   {'subdir': 'vo',   'nc_var': 'vo',   'kelvin_to_c': False},
    'd':    {'subdir': 'd',    'nc_var': 'd',    'kelvin_to_c': False},
}

# Stream-write batch size (storms per CSV flush)
WRITE_BATCH = 5

# Land-snap fallback: if IBTrACS-reported genesis snaps to a land cell (SST=NaN),
# search outward for the nearest ocean cell within this radius (degrees).
# 1.0 deg = 4 grid cells at 0.25 deg resolution.
LAND_SNAP_MAX_RADIUS_DEG = 1.0

# Sentinel for SST in Kelvin (above this threshold, convert to Celsius)
SST_KELVIN_THRESHOLD = 200.0


# =============================================================================
# IBTRACS LOADING
# =============================================================================

def load_ibtracs_positives(ibtracs_path: Path) -> pd.DataFrame:
    """Load IBTrACS WP, filter to positives only."""
    if not ibtracs_path.exists():
        raise FileNotFoundError(f'IBTrACS CSV not found: {ibtracs_path}')

    df = pd.read_csv(ibtracs_path, low_memory=False)
    print(f'  Loaded {len(df):,} rows from {ibtracs_path.name}')

    rename_map = {'SID': 'ID', 'NAME': 'name', 'SEASON': 'year',
                  'LAT': 'latitude', 'LON': 'longitude', 'ISO_TIME': 'time'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    needed = {'ID', 'name', 'NATURE', 'TRACK_TYPE', 'time', 'latitude', 'longitude'}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f'IBTrACS missing required columns: {missing}')

    df['name']       = df['name'].astype(str).str.strip().str.upper()
    df['NATURE']     = df['NATURE'].astype(str).str.strip()
    df['TRACK_TYPE'] = df['TRACK_TYPE'].astype(str).str.strip()
    df['time']       = pd.to_datetime(df['time'], errors='coerce')

    n0 = len(df)
    df = df[(df['NATURE'].isin(POSITIVE_NATURES)) &
            (df['TRACK_TYPE'] == ALLOWED_TRACK_TYPE)]
    print(f'  Filtered to NATURE in {sorted(POSITIVE_NATURES)} '
          f"AND TRACK_TYPE='{ALLOWED_TRACK_TYPE}': {n0} -> {len(df)}")

    if 'NUMBER' in df.columns:
        df['NUMBER'] = pd.to_numeric(df['NUMBER'], errors='coerce').astype('Int64')
    else:
        df['NUMBER'] = pd.NA
        print('  WARNING: IBTrACS lacks NUMBER column; NC file matching may fail.')

    return df


# =============================================================================
# NC FILENAME / READING
# =============================================================================

_FNAME_RE = re.compile(r'^(\d{10})-(\d+)-([A-Za-z][A-Za-z0-9_-]*)\.(?:nc|csv)$')

def fname_safe_name(name: str) -> str:
    """Convert an IBTrACS NAME to its on-disk filename form.

    IBTrACS uses ':' to indicate name changes mid-track (e.g., 'TESS:VAL',
    'KEN:LOLA'). Filenames replace ':' with '-' since colons aren't valid
    on Windows filesystems. Also uppercases for consistency with ncio.py.
    """
    return name.upper().replace(':', '-')


def build_nc_stem(dt: pd.Timestamp, season_num: int, name: str) -> str:
    """Build the NC filename stem matching ncio.py's output convention."""
    return f'{dt.strftime("%Y%m%d%H")}-{int(season_num)}-{fname_safe_name(name)}'


def find_nc_for_timestep(var_subdir: str, season_num: int, name: str,
                         target_dt: pd.Timestamp,
                         tolerance_h: int = 0) -> Optional[Path]:
    """
    Locate an NC file by storm NUMBER+NAME, requiring a timestamp prefix
    within `tolerance_h` hours of `target_dt`. Each NC file is named for
    its specific timestep, so tolerance=0 is the strict (and expected) case.

    Returns the closest matching Path, or None.
    """
    var_dir = _DATA_ROOT / var_subdir
    if not var_dir.is_dir():
        return None
    pattern = f'*-{int(season_num)}-{fname_safe_name(name)}.nc'
    candidates = list(var_dir.glob(pattern))
    if not candidates:
        return None

    best = None
    best_delta = pd.Timedelta(hours=tolerance_h + 1)
    target_ns  = target_dt.value
    for c in candidates:
        m = _FNAME_RE.match(c.name)
        if not m:
            continue
        try:
            file_dt = pd.to_datetime(m.group(1), format='%Y%m%d%H')
        except Exception:
            continue
        delta = abs(pd.Timedelta(file_dt.value - target_ns))
        if delta <= pd.Timedelta(hours=tolerance_h) and delta < best_delta:
            best       = c
            best_delta = delta
    return best


def open_var_nc(season_num: int, name: str, target_dt: pd.Timestamp,
                var_subdir: str) -> Optional[xr.Dataset]:
    """Find + open the NC file for one variable + storm-timestep, or None."""
    p = find_nc_for_timestep(var_subdir, season_num, name, target_dt,
                             tolerance_h=0)
    if p is None:
        return None
    try:
        return xr.open_dataset(p)
    except Exception as e:
        print(f'    WARN open {p.name}: {e}')
        return None


def read_grid(ds: xr.Dataset, var_name: str) -> Optional[np.ndarray]:
    """Read a 2D (lat, lon) grid from the NC, squeezing singleton dims."""
    if var_name not in ds.data_vars:
        return None
    da = ds[var_name].squeeze(drop=True)
    arr = da.values.astype(np.float32)
    if arr.shape != (N_LAT, N_LON):
        if arr.shape == (N_LON, N_LAT):
            arr = arr.T
        else:
            print(f'    WARN unexpected shape {arr.shape} for {var_name}')
            return None
    return arr


def compute_vwsh_from_uv(season_num: int, name: str,
                          target_dt: pd.Timestamp) -> Optional[np.ndarray]:
    """
    Compute vertical wind shear magnitude from u/v at two pressure levels.
    Returns a (N_LAT, N_LON) array, or None if the NC file is missing.
    """
    p = find_nc_for_timestep('vwsh', season_num, name, target_dt,
                             tolerance_h=0)
    if p is None:
        return None
    try:
        ds = xr.open_dataset(p)
    except Exception as e:
        print(f'    WARN open {p.name}: {e}')
        return None

    if 'u' not in ds.data_vars or 'v' not in ds.data_vars:
        ds.close()
        return None

    u = ds['u'].squeeze(drop=True).values.astype(np.float32)
    v = ds['v'].squeeze(drop=True).values.astype(np.float32)
    ds.close()

    if u.ndim != 3 or u.shape[0] != 2 or u.shape[-2:] != (N_LAT, N_LON):
        return None

    # Shear MAGNITUDE is identical regardless of pressure level order
    du = u[1] - u[0]
    dv = v[1] - v[0]
    return np.sqrt(du * du + dv * dv).astype(np.float32)


def find_pi_csv_for_storm(season_num: int, name: str,
                           genesis_dt: pd.Timestamp,
                           tolerance_h: int = 48) -> Optional[Path]:
    """
    Locate the per-storm PI CSV (one CSV per storm with all 9 timesteps).
    The file is named with a timestamp prefix near genesis-48h, but we
    glob by NUMBER+NAME and accept any candidate within tolerance_h hours
    of (genesis - 48h). Default tolerance 48h handles any naming drift.
    """
    pi_dir = _DATA_ROOT / 'pi'
    if not pi_dir.is_dir():
        return None
    pattern    = f'*-{int(season_num)}-{fname_safe_name(name)}.csv'
    candidates = list(pi_dir.glob(pattern))
    if not candidates:
        return None

    # Ideal anchor is genesis - 48h (earliest of the 9 timesteps)
    target = genesis_dt + pd.Timedelta(hours=TIMESTEP_OFFSETS_H[-1])

    best = None
    best_delta = pd.Timedelta(hours=tolerance_h + 1)
    for c in candidates:
        m = _FNAME_RE.match(c.name)
        if not m:
            continue
        try:
            file_dt = pd.to_datetime(m.group(1), format='%Y%m%d%H')
        except Exception:
            continue
        delta = abs(pd.Timedelta(file_dt.value - target.value))
        if delta <= pd.Timedelta(hours=tolerance_h) and delta < best_delta:
            best       = c
            best_delta = delta
    return best


def load_pi_grids_for_storm(season_num: int, name: str,
                             genesis_dt: pd.Timestamp,
                             timesteps: List[pd.Timestamp]
                             ) -> Dict[pd.Timestamp, Optional[np.ndarray]]:
    """
    Open the per-storm PI CSV once, return {timestep: (N_LAT, N_LON) array}.

    The PI CSV has columns [latitude, longitude, time, pi] with 'time' as
    'YYYYMMDD HH:MM' strings and one row per (lat, lon, time). We pivot
    each timestep into a 2D grid aligned to the basin lat/lon mesh.

    Vectorized via groupby + reindex; no per-row Python loops.
    """
    csv_path = find_pi_csv_for_storm(season_num, name, genesis_dt)
    out: Dict[pd.Timestamp, Optional[np.ndarray]] = {ts: None for ts in timesteps}
    if csv_path is None:
        return out

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'    WARN read {csv_path.name}: {e}')
        return out

    if not {'latitude', 'longitude', 'time', 'pi'}.issubset(df.columns):
        return out

    # Parse time column once (format 'YYYYMMDD HH:MM' per Q2 diagnostic)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H:%M', errors='coerce')

    # Snap lat/lon to grid resolution to avoid float-precision misses
    df['lat_snap'] = (df['latitude']  / GRID_RES).round() * GRID_RES
    df['lon_snap'] = (df['longitude'] / GRID_RES).round() * GRID_RES

    # Build a target index covering the full basin grid (lat desc, lon asc)
    full_index = pd.MultiIndex.from_product(
        [LATS_DESC, LONS_ASC], names=['lat_snap', 'lon_snap'])

    # For each requested timestep, slice + pivot vectorized
    for ts in timesteps:
        sub = df[df['time'] == ts]
        if len(sub) == 0:
            continue
        # Average duplicates (defensive; should be 1 row per cell normally)
        gridded = (sub.groupby(['lat_snap', 'lon_snap'])['pi'].mean()
                      .reindex(full_index)
                      .values
                      .reshape(N_LAT, N_LON)
                      .astype(np.float32))
        out[ts] = gridded
    return out



def read_storm_timestep_grids(season_num: int, name: str,
                               target_dt: pd.Timestamp,
                               pi_grids: Optional[Dict[pd.Timestamp, Optional[np.ndarray]]] = None
                               ) -> Dict[str, Optional[np.ndarray]]:
    """For one storm-timestep, read all 8 variables aligned to the basin grid.

    pi_grids is the per-storm dict produced by load_pi_grids_for_storm()
    and reused across all 9 timesteps of a storm. If None, PI is left as NaN.
    """
    grids: Dict[str, Optional[np.ndarray]] = {}

    for var, cfg in VAR_CONFIG.items():
        ds = open_var_nc(season_num, name, target_dt, cfg['subdir'])
        if ds is None:
            grids[var] = None
            continue
        arr = read_grid(ds, cfg['nc_var'])
        ds.close()
        if arr is not None and cfg['kelvin_to_c']:
            if np.nanmax(arr) > SST_KELVIN_THRESHOLD:
                arr = arr - 273.15
        grids[var] = arr

    grids['vwsh'] = compute_vwsh_from_uv(season_num, name, target_dt)
    grids['pi']   = pi_grids.get(target_dt) if pi_grids is not None else None

    return grids


# =============================================================================
# PER-STORM ASSEMBLY
# =============================================================================

def nearest_grid_idx(lat: float, lon: float) -> Optional[Tuple[int, int]]:
    """Snap (lat, lon) to nearest grid cell. Returns (row, col) or None."""
    lat_r = round(round(lat / GRID_RES) * GRID_RES, 2)
    lon_r = round(round(lon / GRID_RES) * GRID_RES, 2)
    try:
        row = np.where(LATS_DESC == lat_r)[0][0]
        col = np.where(LONS_ASC  == lon_r)[0][0]
        return int(row), int(col)
    except IndexError:
        return None


def build_storm_dataframe(storm_id: str, name: str, season_num: int,
                          genesis_time: pd.Timestamp,
                          genesis_lat: float, genesis_lon: float,
                          missing_log: dict) -> Optional[pd.DataFrame]:
    """For one positive storm, build the per-cell x per-timestep DataFrame."""
    timesteps = [genesis_time + pd.Timedelta(hours=h) for h in TIMESTEP_OFFSETS_H]

    genesis_idx = nearest_grid_idx(genesis_lat, genesis_lon)
    if genesis_idx is None:
        missing_log.setdefault('out_of_grid', []).append((storm_id, genesis_lat, genesis_lon))
        return None

    # lat/lon meshgrid: lats descend (row 0 = north), lons ascend
    lat_grid, lon_grid = np.meshgrid(LATS_DESC, LONS_ASC, indexing='ij')
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    # Open the per-storm PI CSV ONCE and extract grids for all 9 timesteps.
    # PI is then served from this in-memory dict during the per-timestep loop.
    pi_grids = load_pi_grids_for_storm(season_num, name, genesis_time, timesteps)

    timestep_dfs = []
    for ts in timesteps:
        grids = read_storm_timestep_grids(season_num, name, ts, pi_grids=pi_grids)

        sst = grids.get('sst')
        if sst is None:
            missing_log.setdefault('no_sst', []).append((storm_id,
                                           build_nc_stem(ts, season_num, name)))
            continue

        df = pd.DataFrame({
            'ID':        storm_id,
            'latitude':  lat_flat,
            'longitude': lon_flat,
            'time':      ts,
        })

        for var in ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']:
            arr = grids.get(var)
            if arr is None:
                df[var] = np.nan
            else:
                df[var] = arr.ravel()

        df['origin'] = np.int8(0)
        if ts == genesis_time:
            grow, gcol = genesis_idx
            target_lat = float(LATS_DESC[grow])
            target_lon = float(LONS_ASC[gcol])
            mask = ((np.abs(df['latitude']  - target_lat) < 1e-6) &
                    (np.abs(df['longitude'] - target_lon) < 1e-6))
            df.loc[mask, 'origin'] = np.int8(1)

        # Drop land cells (SST is NaN). Matches OLD WPMM.csv.bak behavior.
        df = df[df['sst'].notna()].reset_index(drop=True)

        # Land-snap fallback: if this is the genesis timestep AND the genesis
        # cell was dropped as land, snap origin=1 to the nearest surviving
        # ocean cell within LAND_SNAP_MAX_RADIUS_DEG. Methodology note for
        # dissertation: "Storms whose IBTrACS-reported genesis snapped to a
        # land cell were assigned origin=1 at the nearest ocean cell within
        # 1 degree. If no such cell existed, the storm was excluded."
        if ts == genesis_time and (df['origin'] == 1).sum() == 0:
            target_lat = float(LATS_DESC[genesis_idx[0]])
            target_lon = float(LONS_ASC[genesis_idx[1]])
            dist = np.sqrt((df['latitude']  - target_lat) ** 2 +
                           (df['longitude'] - target_lon) ** 2)
            within = dist <= LAND_SNAP_MAX_RADIUS_DEG
            if within.any():
                # Pick the closest cell
                idx = dist[within].idxmin()
                actual_lat = df.loc[idx, 'latitude']
                actual_lon = df.loc[idx, 'longitude']
                offset_deg = float(dist.loc[idx])
                df.loc[idx, 'origin'] = np.int8(1)
                missing_log.setdefault('land_snap', []).append(
                    (storm_id, target_lat, target_lon,
                     actual_lat, actual_lon, offset_deg))
            else:
                # No ocean cell within radius of 1 deg drop the storm entirely
                missing_log.setdefault('land_snap_failed', []).append(
                    (storm_id, target_lat, target_lon))
                return None

        timestep_dfs.append(df)

    if not timestep_dfs:
        missing_log.setdefault('no_timesteps', []).append(storm_id)
        return None

    out = pd.concat(timestep_dfs, ignore_index=True)
    return out[OUTPUT_COLS]


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def get_genesis_row_per_storm(ibtracs: pd.DataFrame) -> pd.DataFrame:
    """One row per storm at its genesis (earliest ISO_TIME)."""
    g = ibtracs.dropna(subset=['time']).sort_values(['ID', 'time'])
    g = g.drop_duplicates(subset='ID', keep='first').reset_index(drop=True)
    return g


def get_existing_storm_ids(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    print(f'  Reading existing IDs from {out_csv} for resume...')
    ids = set()
    for chunk in pd.read_csv(out_csv, usecols=['ID'], chunksize=500_000):
        ids.update(chunk['ID'].unique())
    print(f'  Resume: {len(ids)} storms already in output')
    return ids


def write_batch(rows: List[pd.DataFrame], out_csv: Path,
                first_write: bool) -> int:
    if not rows:
        return 0
    out_df = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mode   = 'w' if first_write else 'a'
    header = first_write
    out_df.to_csv(out_csv, mode=mode, header=header, index=False)
    return len(out_df)


def sanity_check(out_csv: Path) -> None:
    print()
    print('=' * 70)
    print('SANITY CHECK')
    print('=' * 70)
    if not out_csv.exists():
        print('  Output CSV does not exist.')
        return

    n_rows = 0
    n_origin1 = 0
    var_cols  = ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']
    var_min   = {v:  np.inf for v in var_cols}
    var_max   = {v: -np.inf for v in var_cols}
    var_nan   = {v: 0 for v in var_cols}
    storms = set()

    for chunk in pd.read_csv(out_csv, chunksize=500_000):
        n_rows    += len(chunk)
        n_origin1 += int((chunk['origin'] == 1).sum())
        storms.update(chunk['ID'].unique())
        for v in var_cols:
            if v not in chunk.columns:
                continue
            x = chunk[v].astype(float)
            var_nan[v] += int(x.isna().sum())
            valid = x.dropna()
            if len(valid):
                var_min[v] = min(var_min[v], float(valid.min()))
                var_max[v] = max(var_max[v], float(valid.max()))

    print(f'  Total rows:               {n_rows:,}')
    print(f'  Unique storms:            {len(storms):,}')
    print(f'  origin=1 rows:            {n_origin1:,}  '
          f'(should equal #storms = {len(storms)})')
    print()
    print(f'  {"variable":<8s} {"min":>14s}  {"max":>14s}  {"NaN":>14s}')
    print(f'  {"-"*8} {"-"*14}  {"-"*14}  {"-"*14}')
    for v in var_cols:
        mn = f'{var_min[v]:.4g}' if var_min[v] != np.inf else 'all NaN'
        mx = f'{var_max[v]:.4g}' if var_max[v] != -np.inf else 'all NaN'
        print(f'  {v:<8s} {mn:>14s}  {mx:>14s}  {var_nan[v]:>14,}')


def main():
    global _DATA_ROOT, _FINAL_ROOT, _OUT_CSV, _IBTRACS_CSV

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ibtracs',   type=Path, default=_IBTRACS_CSV)
    parser.add_argument('--data-root', type=Path, default=_DATA_ROOT)
    parser.add_argument('--out',       type=Path, default=_OUT_CSV)
    parser.add_argument('--restart',   action='store_true',
                        help='Delete existing output CSV first')
    parser.add_argument('--limit',     type=int, default=0,
                        help='Process at most N storms (0 = unlimited)')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Process 1 storm only, do not write CSV')
    args = parser.parse_args()

    _DATA_ROOT   = args.data_root.resolve()
    _FINAL_ROOT  = _DATA_ROOT / 'final'
    _OUT_CSV     = args.out.resolve()
    _IBTRACS_CSV = args.ibtracs.resolve()

    print('=' * 70)
    print('nci_wp.py -- primary positives builder for WP-MM')
    print('=' * 70)
    print(f'  data_root:   {_DATA_ROOT}')
    print(f'  ibtracs:     {_IBTRACS_CSV}')
    print(f'  output:      {_OUT_CSV}')
    print(f'  grid:        {N_LAT} x {N_LON} '
          f'(lat {DOMAIN["south"]}-{DOMAIN["north"]}, '
          f'lon {DOMAIN["west"]}-{DOMAIN["east"]})')
    print(f'  timesteps:   {len(TIMESTEP_OFFSETS_H)} per storm '
          f'({TIMESTEP_OFFSETS_H[0]} to {TIMESTEP_OFFSETS_H[-1]} h)')
    print()

    if args.restart and _OUT_CSV.exists() and not args.dry_run:
        print(f'  --restart: removing existing {_OUT_CSV}')
        _OUT_CSV.unlink()
        print()

    print('Loading IBTrACS positives...')
    ibtracs = load_ibtracs_positives(_IBTRACS_CSV)
    genesis = get_genesis_row_per_storm(ibtracs)
    print(f'  {len(genesis)} positive storms after dedup (one row per storm)')
    print()

    already_done = set() if args.dry_run else get_existing_storm_ids(_OUT_CSV)

    todo = genesis[~genesis['ID'].isin(already_done)].reset_index(drop=True)
    if args.limit > 0:
        todo = todo.head(args.limit)
    if args.dry_run:
        todo = todo.head(1)
    print(f'  Storms to process this run: {len(todo)}')
    print()

    print('=' * 70)
    print('PROCESSING')
    print('=' * 70)

    missing_log: dict = {'out_of_grid': [], 'no_sst': [], 'no_timesteps': [],
                         'land_snap': [], 'land_snap_failed': []}
    batch: List[pd.DataFrame] = []
    first_write = (not _OUT_CSV.exists())
    n_done = 0
    n_rows_written = 0
    t_start = time_module.time()

    for i, row in todo.iterrows():
        storm_id = str(row['ID'])
        name     = str(row['name'])
        season   = int(row['NUMBER']) if pd.notna(row['NUMBER']) else 0
        gtime    = row['time']
        glat     = float(row['latitude'])
        glon     = float(row['longitude'])
        if glon < 0:
            glon += 360.0   # 0-360 convention to match WP basin

        df_storm = build_storm_dataframe(storm_id, name, season,
                                         gtime, glat, glon, missing_log)

        elapsed = time_module.time() - t_start
        rate    = (i + 1) / elapsed if elapsed > 0 else 0
        eta_min = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0

        if df_storm is None:
            print(f'  [{i+1}/{len(todo)}] {storm_id} {name} -- SKIPPED (no data)  '
                  f'rate={rate:.2f}/s eta={eta_min:.1f}m')
            n_done += 1
            continue

        print(f'  [{i+1}/{len(todo)}] {storm_id} {name} -- {len(df_storm):,} rows  '
              f'rate={rate:.2f}/s eta={eta_min:.1f}m')

        batch.append(df_storm)
        n_done += 1

        if not args.dry_run and len(batch) >= WRITE_BATCH:
            n = write_batch(batch, _OUT_CSV, first_write)
            n_rows_written += n
            first_write = False
            batch = []

    if batch and not args.dry_run:
        n = write_batch(batch, _OUT_CSV, first_write)
        n_rows_written += n

    print()
    print('=' * 70)
    print('PROCESSING COMPLETE')
    print('=' * 70)
    print(f'  Storms processed:   {n_done}')
    print(f'  Rows written:       {n_rows_written:,}')
    print(f'  Elapsed:            {(time_module.time() - t_start)/60:.1f} min')
    if missing_log['out_of_grid']:
        print(f'  Out-of-grid genesis:  {len(missing_log["out_of_grid"])}')
    if missing_log['no_sst']:
        print(f'  Missing SST NCs:      {len(missing_log["no_sst"])} timesteps')
    if missing_log['no_timesteps']:
        print(f'  Storms with 0 valid timesteps: {len(missing_log["no_timesteps"])}')
    if missing_log['land_snap']:
        print(f'  Land-snapped genesis: {len(missing_log["land_snap"])} '
              f'(snapped to nearest ocean cell within {LAND_SNAP_MAX_RADIUS_DEG} deg)')
        # Save snap manifest for the methods section
        snap_df = pd.DataFrame(missing_log['land_snap'],
                               columns=['SID', 'reported_lat', 'reported_lon',
                                        'snapped_lat', 'snapped_lon', 'offset_deg'])
        snap_path = _OUT_CSV.parent / 'WPMM_land_snap_manifest.csv'
        snap_df.to_csv(snap_path, index=False)
        print(f'    Manifest saved: {snap_path}')
        for r in missing_log['land_snap']:
            print(f'    {r[0]}: ({r[1]}, {r[2]}) -> ({r[3]}, {r[4]}) '
                  f'offset={r[5]:.3f} deg')
    if missing_log['land_snap_failed']:
        print(f'  Land-snap FAILED (storm dropped): '
              f'{len(missing_log["land_snap_failed"])}')
        for r in missing_log['land_snap_failed']:
            print(f'    {r[0]}: ({r[1]}, {r[2]}) - no ocean cell within '
                  f'{LAND_SNAP_MAX_RADIUS_DEG} deg')

    if not args.dry_run:
        sanity_check(_OUT_CSV)

    print()
    print('NEXT STEP:')
    print('  python ncio-neg-wp.py    # adds DS/NR negatives -> WPMM_combined.csv')

    return 0


if __name__ == '__main__':
    sys.exit(main())
