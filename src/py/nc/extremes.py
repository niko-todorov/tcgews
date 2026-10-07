"""
5. extremes.py
==============
Recomputes extremes.csv from the FINAL combined dataset
(../../../data/NACSGM/final/NACSGM_combined.csv) rather than by scanning the
per-variable NC files on disk.

Why compute from the combined CSV instead of the NC files?
  - The CSV is exactly what the model trains on: same rows, same longitude
    convention (-180/+180), same Tier-0/1/3 negatives, same genesis-only
    positive filtering, same invalid-row drops. Extremes computed here are
    therefore guaranteed consistent with the normalization the model applies.
  - Scanning NC files can drift from the CSV: it may include quarantined or
    duplicate per-storm files, apply different unit handling, or miss the
    merge-time filtering — producing min/max that don't match the training set.

Variables: sst, msl, cape, r, vo, d, vwsh, pi  (whichever are present as columns)

Output: ../../../data/NACSGM/final/extremes.csv
  variable,min,max

By default, extremes are computed over ALL rows in the combined file
(origin = 1, 0, and -1) — i.e. every grid point the model can see, positive
AND negative. This is the correct basis for min-max normalization: if extremes
were computed over positives only, a negative sample could contain a value
outside [min, max] and normalize outside [0, 1]. Use --positives-only to
restrict to positive-storm rows (origin in {0, 1}) if that matches the
training-time normalization instead.

Usage
-----
  python extremes.py
  python extremes.py --csv path/to/NACSGM_combined.csv
  python extremes.py --positives-only     # extremes over origin in {0,1} only
  python extremes.py --dry-run            # compute but don't write
  python extremes.py --chunksize 2000000  # tune memory vs speed

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Default paths (relative to src/py/nc/)
_COMBINED_CSV = Path('../../../data/NACSGM/final/NACSGM_combined.csv')
_OUT_CSV      = Path('../../../data/NACSGM/final/extremes.csv')

# Variables to compute extremes for, in output order.
VARIABLES = ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']

# SST stored in Kelvin if values exceed this; convert to Celsius for extremes.
# (Detected per-file from the data itself — not assumed.)
SST_KELVIN_THRESHOLD = 200.0

# ERA5 land/missing sentinels that must NOT pollute the physical min/max.
# These mirror the invalid-row filtering done at merge time:
#   SST = 0 over land, MSL = 0 missing, CAPE = -273.15 sentinel.
# We drop these per-variable when accumulating extremes.
_SENTINELS = {
    'sst':  [0.0],
    'msl':  [0.0],
    'cape': [-273.15],
}


def _clean_series(col_name, values):
    """
    Return finite, physical values for a variable: drop NaN/inf and known
    sentinels. For SST, convert Kelvin->Celsius if the column is in Kelvin.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return v

    # Drop sentinels FIRST, while they still hold their stored values
    # (e.g. SST land sentinel = 0.0). Doing this before any unit conversion
    # avoids a 0.0 sentinel turning into -273.15 after a Kelvin->Celsius
    # subtraction and then evading the sentinel filter.
    for s in _SENTINELS.get(col_name, []):
        v = v[v != s]
    if len(v) == 0:
        return v

    # SST unit normalization: detect Kelvin (max > threshold) and convert.
    # Done AFTER sentinel removal so only real temperatures are shifted.
    if col_name == 'sst' and np.nanmax(v) > SST_KELVIN_THRESHOLD:
        v = v - 273.15

    return v


def compute_extremes(csv_path, variables, positives_only=False,
                     chunksize=2_000_000):
    """
    Stream the combined CSV in chunks and accumulate per-variable min/max.
    Streaming keeps memory bounded for the ~40M-row combined file.

    Returns: dict var -> [min, max], plus counts dict var -> n_valid_values.
    """
    # Determine which columns actually exist (read just the header).
    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)
    present = [v for v in variables if v in cols]
    missing = [v for v in variables if v not in cols]
    if missing:
        print(f"  NOTE: columns not in CSV (skipped): {', '.join(missing)}")
    if 'origin' not in cols and positives_only:
        raise ValueError("--positives-only requires an 'origin' column.")

    usecols = present + (['origin'] if 'origin' in cols else [])

    vmin = {v: float('inf') for v in present}
    vmax = {v: -float('inf') for v in present}
    vcount = {v: 0 for v in present}

    n_rows = 0
    n_chunks = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize,
                             low_memory=False):
        n_chunks += 1
        if positives_only and 'origin' in chunk.columns:
            chunk = chunk[chunk['origin'].isin([0, 1])]
        n_rows += len(chunk)
        for v in present:
            vals = _clean_series(v, chunk[v].values)
            if len(vals):
                vmin[v] = min(vmin[v], float(vals.min()))
                vmax[v] = max(vmax[v], float(vals.max()))
                vcount[v] += int(len(vals))
        print(f"  scanned {n_rows:,} rows ({n_chunks} chunks)...", end='\r')

    print(f"  scanned {n_rows:,} rows ({n_chunks} chunks).            ")

    extremes = {}
    for v in present:
        if vcount[v] > 0 and np.isfinite(vmin[v]) and np.isfinite(vmax[v]):
            extremes[v] = [vmin[v], vmax[v]]
        else:
            print(f"  WARNING: no valid values for '{v}' — not written")
    return extremes, vcount


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=_COMBINED_CSV,
                        help='Combined dataset CSV (default: '
                             'NACSGM_combined.csv)')
    parser.add_argument('--out', type=Path, default=_OUT_CSV,
                        help='Output extremes.csv path')
    parser.add_argument('--positives-only', action='store_true',
                        help='Compute extremes over positive-storm rows only '
                             '(origin in {0,1}). Default: all rows incl. '
                             'negatives (origin=-1).')
    parser.add_argument('--chunksize', type=int, default=2_000_000,
                        help='Rows per chunk when streaming (memory/speed).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute but do not write extremes.csv')
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: combined CSV not found: {args.csv}")
        return

    scope = 'positives only (origin in {0,1})' if args.positives_only \
            else 'ALL rows (positives + negatives)'
    print(f"Computing extremes from {args.csv}")
    print(f"Scope: {scope}\n")

    extremes, counts = compute_extremes(
        args.csv, VARIABLES,
        positives_only=args.positives_only,
        chunksize=args.chunksize,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXTREMES SUMMARY")
    print(f"{'='*64}")
    print(f"  {'variable':<10}  {'min':>15}  {'max':>15}  {'n_valid':>14}")
    print(f"  {'-'*10}  {'-'*15}  {'-'*15}  {'-'*14}")
    for v in VARIABLES:
        if v in extremes:
            lo, hi = extremes[v]
            print(f"  {v:<10}  {lo:>15.8g}  {hi:>15.8g}  {counts[v]:>14,}")
        else:
            print(f"  {v:<10}  {'NOT COMPUTED':>15}")

    # ── Compare with existing extremes.csv ────────────────────────────────────
    if args.out.exists():
        print(f"\n{'='*64}")
        print("COMPARISON WITH EXISTING extremes.csv")
        print(f"{'='*64}")
        old = pd.read_csv(args.out, low_memory=False)
        old.columns = [c.strip().lower() for c in old.columns]
        old_map = {str(row['variable']).strip().lower(): row
                   for _, row in old.iterrows()
                   if 'variable' in old.columns}
        print(f"  {'var':<8}  {'old_min':>12}  {'new_min':>12}  "
              f"{'old_max':>12}  {'new_max':>12}  note")
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for v in VARIABLES:
            if v not in extremes:
                continue
            nlo, nhi = extremes[v]
            if v in old_map:
                try:
                    olo = float(old_map[v]['min'])
                    ohi = float(old_map[v]['max'])
                except (KeyError, ValueError):
                    olo = ohi = float('nan')
            else:
                olo = ohi = float('nan')
            changed = (not np.isfinite(olo)) or \
                      abs(nlo - olo) > 1e-6 or abs(nhi - ohi) > 1e-6
            note = 'CHANGED' if changed else 'same'
            print(f"  {v:<8}  {olo:>12.6g}  {nlo:>12.6g}  "
                  f"{ohi:>12.6g}  {nhi:>12.6g}  {note}")

    # ── Write ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n  DRY RUN — {args.out} not written.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'variable': v, 'min': extremes[v][0], 'max': extremes[v][1]}
            for v in VARIABLES if v in extremes]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n  Written -> {args.out}")
    print(f"  {len(rows)} variables saved")


if __name__ == '__main__':
    main()
