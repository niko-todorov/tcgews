"""
4. ncio-neg-wp.py
=================
Negatives extender for WP-MM basin.

Reads WPMM.csv (positives, produced by nci_wp.py), then for each negative
storm in IBTrACS WP (NATURE in {DS, NR}, TRACK_TYPE='main') it builds the
same per-cell x per-timestep grid using the existing per-storm NCs, with
origin=-1 throughout. The result is appended to produce WPMM_combined.csv.

This mirrors the original NACSGM workflow (ncio.py -> NACSGM.csv;
ncio-negatives.py -> NACSGM_combined.csv) but does not download from CDS --
all NCs already exist on disk from the earlier ncio.py run.

Reuses functions from nci_wp.py for NC reading and per-storm DataFrame
construction, with origin set to -1 across all rows for negative storms.

Output: data/WPMM/final/WPMM_combined.csv
  Columns: ID, latitude, longitude, time, sst, vwsh, d, cape, r, msl, vo, pi, origin
  origin=1 only at genesis cell of positives; origin=-1 for all cells of negatives;
  origin=0 for context cells of positives.

Usage:
  python ncio-neg-wp.py
  python ncio-neg-wp.py --dry-run                  # process 1 negative storm only
  python ncio-neg-wp.py --restart                  # delete existing WPMM_combined.csv first
  python ncio-neg-wp.py --limit 5                  # first 5 negative storms only
  python ncio-neg-wp.py --positives-csv path/to/WPMM.csv

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time as time_module
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Reuse the heavy lifting from nci_wp.py
from nci_wp import (
    DOMAIN, GRID_RES, N_LAT, N_LON, LATS_DESC, LONS_ASC,
    TIMESTEP_OFFSETS_H, ALLOWED_TRACK_TYPE,
    OUTPUT_COLS, WRITE_BATCH,
    build_storm_dataframe, write_batch, sanity_check,
)


# =============================================================================
# CONFIG
# =============================================================================
_DATA_ROOT      = Path('../../../data/WPMM')
_FINAL_ROOT     = _DATA_ROOT / 'final'
_POSITIVES_CSV  = _FINAL_ROOT / 'WPMM.csv'
_OUT_CSV        = _FINAL_ROOT / 'WPMM_combined.csv'
_IBTRACS_CSV    = Path('../../../data/1979-WP-MM-origins-ibtracs.since1980.list.v04r01.csv')

NEGATIVE_NATURES = {'DS', 'NR'}


# =============================================================================
# IBTRACS LOADING (negatives variant)
# =============================================================================

def load_ibtracs_negatives(ibtracs_path: Path) -> pd.DataFrame:
    """Load IBTrACS WP, filter to negatives only (NATURE in {DS, NR}, main)."""
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
    df = df[(df['NATURE'].isin(NEGATIVE_NATURES)) &
            (df['TRACK_TYPE'] == ALLOWED_TRACK_TYPE)]
    print(f'  Filtered to NATURE in {sorted(NEGATIVE_NATURES)} '
          f"AND TRACK_TYPE='{ALLOWED_TRACK_TYPE}': {n0} -> {len(df)}")

    if 'NUMBER' in df.columns:
        df['NUMBER'] = pd.to_numeric(df['NUMBER'], errors='coerce').astype('Int64')
    else:
        df['NUMBER'] = pd.NA
        print('  WARNING: IBTrACS lacks NUMBER column; NC file matching may fail.')

    return df


def get_first_track_row_per_storm(ibtracs: pd.DataFrame) -> pd.DataFrame:
    """One row per storm at its earliest ISO_TIME (the 'genesis' for negatives)."""
    g = ibtracs.dropna(subset=['time']).sort_values(['ID', 'time'])
    g = g.drop_duplicates(subset='ID', keep='first').reset_index(drop=True)
    return g


# =============================================================================
# NEGATIVE-SPECIFIC ASSEMBLY
# =============================================================================

def build_negative_storm_dataframe(storm_id: str, name: str, season_num: int,
                                    first_time: pd.Timestamp,
                                    first_lat: float, first_lon: float,
                                    missing_log: dict) -> Optional[pd.DataFrame]:
    """
    For a negative storm, build the per-cell x per-timestep DataFrame, then
    overwrite the origin column to -1 throughout.

    Reuses build_storm_dataframe from nci_wp.py (which sets origin=1 at the
    genesis cell of t=0 for positives) and then re-stamps every row to -1.
    """
    df = build_storm_dataframe(storm_id, name, season_num,
                               first_time, first_lat, first_lon,
                               missing_log)
    if df is None:
        return None
    df['origin'] = np.int8(-1)
    return df


# =============================================================================
# MAIN
# =============================================================================

def get_negative_storm_ids_in_combined(out_csv: Path,
                                        positives_ids: set) -> set:
    """
    Find which storm IDs in the (already-existing) combined CSV correspond
    to NEGATIVES that have already been processed. Used for resume.

    A negative is identified as: in combined, but NOT in positives_ids.
    """
    if not out_csv.exists():
        return set()
    print(f'  Reading existing IDs from {out_csv} for resume...')
    all_ids = set()
    for chunk in pd.read_csv(out_csv, usecols=['ID'], chunksize=500_000):
        all_ids.update(chunk['ID'].unique())
    neg_done = all_ids - positives_ids
    print(f'  Resume: {len(neg_done)} negatives already in combined CSV')
    return neg_done


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ibtracs',         type=Path, default=_IBTRACS_CSV)
    parser.add_argument('--data-root',       type=Path, default=_DATA_ROOT)
    parser.add_argument('--positives-csv',   type=Path, default=_POSITIVES_CSV,
                        help='WPMM.csv (positives) input -- copied to start the combined output')
    parser.add_argument('--out',             type=Path, default=_OUT_CSV)
    parser.add_argument('--restart',         action='store_true',
                        help='Delete existing combined CSV and re-copy positives')
    parser.add_argument('--limit',           type=int, default=0,
                        help='Process at most N negative storms (0 = unlimited)')
    parser.add_argument('--dry-run',         action='store_true',
                        help='Process 1 negative storm only, do not write')
    args = parser.parse_args()

    pos_csv  = args.positives_csv.resolve()
    out_csv  = args.out.resolve()
    ibtracs  = args.ibtracs.resolve()

    print('=' * 70)
    print('ncio-neg-wp.py -- negatives extender for WP-MM')
    print('=' * 70)
    print(f'  data_root:         {args.data_root.resolve()}')
    print(f'  ibtracs:           {ibtracs}')
    print(f'  positives input:   {pos_csv}')
    print(f'  combined output:   {out_csv}')
    print(f'  grid:              {N_LAT} x {N_LON}')
    print(f'  timesteps:         {len(TIMESTEP_OFFSETS_H)} per storm '
          f'({TIMESTEP_OFFSETS_H[0]} to {TIMESTEP_OFFSETS_H[-1]} h)')
    print()

    if not pos_csv.exists():
        print(f'ERROR: positives CSV not found: {pos_csv}')
        print('       Run nci_wp.py first to produce WPMM.csv.')
        return 1

    # --restart: drop combined and start fresh
    if args.restart and out_csv.exists() and not args.dry_run:
        print(f'  --restart: removing existing {out_csv}')
        out_csv.unlink()
        print()

    # If combined doesn't exist yet, seed it from the positives CSV.
    # This is an O(file size) copy -- WPMM.csv is large but it's just a disk copy.
    if not out_csv.exists() and not args.dry_run:
        print(f'  Seeding combined CSV by copying {pos_csv.name} -> {out_csv.name}')
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        t_copy = time_module.time()
        shutil.copy2(pos_csv, out_csv)
        sz_mb = out_csv.stat().st_size / 1e6
        print(f'  Copied {sz_mb:.1f} MB in {time_module.time()-t_copy:.1f}s')
        print()

    # PHASE 1: identify positives + negatives
    print('Loading IBTrACS negatives...')
    ibtracs_neg = load_ibtracs_negatives(ibtracs)
    first_per_storm = get_first_track_row_per_storm(ibtracs_neg)
    print(f'  {len(first_per_storm)} negative storms after dedup')
    print()

    # Resume support: which negatives are already in the combined CSV?
    print('Identifying positives already in combined CSV (for resume)...')
    if pos_csv.exists():
        positives_ids = set()
        for chunk in pd.read_csv(pos_csv, usecols=['ID'], chunksize=500_000):
            positives_ids.update(chunk['ID'].unique())
        print(f'  Positives in {pos_csv.name}: {len(positives_ids)}')
    else:
        positives_ids = set()
    print()

    already_done_neg = (set() if args.dry_run
                        else get_negative_storm_ids_in_combined(out_csv, positives_ids))

    todo = first_per_storm[~first_per_storm['ID'].isin(already_done_neg)].reset_index(drop=True)
    if args.limit > 0:
        todo = todo.head(args.limit)
    if args.dry_run:
        todo = todo.head(1)
    print(f'  Negative storms to process this run: {len(todo)}')
    print()

    # PHASE 2: process loop
    print('=' * 70)
    print('PROCESSING NEGATIVES')
    print('=' * 70)

    missing_log: dict = {'out_of_grid': [], 'no_sst': [], 'no_timesteps': [],
                         'land_snap': [], 'land_snap_failed': []}
    batch: List[pd.DataFrame] = []
    n_done = 0
    n_rows_written = 0
    t_start = time_module.time()

    # We always APPEND to the combined CSV (positives are already there)
    first_write = (not out_csv.exists())

    for i, row in todo.iterrows():
        storm_id = str(row['ID'])
        name     = str(row['name'])
        season   = int(row['NUMBER']) if pd.notna(row['NUMBER']) else 0
        ftime    = row['time']
        flat     = float(row['latitude'])
        flon     = float(row['longitude'])
        if flon < 0:
            flon += 360.0

        df_storm = build_negative_storm_dataframe(storm_id, name, season,
                                                   ftime, flat, flon, missing_log)

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
            n = write_batch(batch, out_csv, first_write)
            n_rows_written += n
            first_write = False
            batch = []

    if batch and not args.dry_run:
        n = write_batch(batch, out_csv, first_write)
        n_rows_written += n

    print()
    print('=' * 70)
    print('PROCESSING COMPLETE')
    print('=' * 70)
    print(f'  Negative storms processed:  {n_done}')
    print(f'  Rows appended:              {n_rows_written:,}')
    print(f'  Elapsed:                    {(time_module.time() - t_start)/60:.1f} min')
    if missing_log['out_of_grid']:
        print(f'  Out-of-grid first track:    {len(missing_log["out_of_grid"])}')
    if missing_log['no_sst']:
        print(f'  Missing SST NCs:            {len(missing_log["no_sst"])} timesteps')
    if missing_log['no_timesteps']:
        print(f'  Storms with 0 valid timesteps: {len(missing_log["no_timesteps"])}')
    if missing_log['land_snap']:
        # Note: for negatives, land_snap snapping is logged but origin gets
        # overwritten to -1 throughout anyway (the snap label is irrelevant
        # since the whole storm is a negative sample).
        print(f'  Land-snapped first track:   {len(missing_log["land_snap"])} '
              f'(informational only -- origin=-1 stamped on all rows)')
    if missing_log['land_snap_failed']:
        print(f'  Land-snap FAILED (storm dropped): '
              f'{len(missing_log["land_snap_failed"])}')
        for r in missing_log['land_snap_failed']:
            print(f'    {r[0]}: ({r[1]}, {r[2]}) - no ocean cell within '
                  f'1.0 deg')

    if not args.dry_run:
        print()
        print('Final sanity check on combined CSV:')
        sanity_check(out_csv)

    print()
    print('NEXT STEPS:')
    print('  1. Verify WPMM_combined.csv structure:')
    print('       - origin distribution should show 1 (positives), 0 (positive context),')
    print('         and -1 (negatives)')
    print('       - Variable ranges should be raw physical values (sst in C, msl in Pa, etc.)')
    print('  2. Generate WP extremes.csv if not already done')
    print('  3. Run boundary audit on WP combined CSV')
    print('  4. Train multilead-KL on WPMM_combined.csv')

    return 0


if __name__ == '__main__':
    sys.exit(main())
