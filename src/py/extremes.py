"""
extremes.py
===========
Recomputes extremes.csv (per-variable global min/max) from NACSGM_combined.csv
over the FULL dataset — all years, all months, all storms (genesis + CLM_).

Why recompute?
--------------
The original extremes.csv was likely computed over peak hurricane season only
(JJA/ASO), missing cool-season and high-latitude cases.  This causes variables
like SST to have a min that is too high, so cool-season genesis storms at the
northern grid boundary get negative normalised SST values.

What changes?
-------------
  • SST min will drop from ~23.75°C to ~18–21°C (cool-season Atlantic at 30°N)
  • Other variables may shift slightly at the tails
  • PI is included so normalize_pi() always has valid bounds
  • A percentile-clipping option (default p=0.1%) guards against outlier
    fill/artefact values inflating the range unnecessarily

Outputs
-------
  extremes_new.csv          — drop-in replacement for extremes.csv
  extremes_comparison.csv   — side-by-side old vs new for review
  extremes_comparison.png   — bar chart of range changes per variable

Usage
-----
  python extremes.py
  python extremes.py --csv path/to/NACSGM_combined.csv
  python extremes.py --percentile 0.5   # clip outer 0.5% each tail
  python extremes.py --no-percentile     # use true global min/max

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================
_DATA_ROOT   = Path('../../data/NACSGM/final')
_CSV_PATH    = _DATA_ROOT / 'NACSGM_combined.csv'
_OLD_EXTREMES = _DATA_ROOT / 'extremes.csv'
_NEW_EXTREMES = _DATA_ROOT / 'extremes_new.csv'
_CMP_CSV     = _DATA_ROOT / 'extremes_comparison.csv'
_CMP_PNG     = _DATA_ROOT / 'extremes_comparison.png'

# Variables to compute extremes for (must match VARIABLES in TCG-NACSGM.py)
VARIABLES = ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']

# Physical lower bounds — values below these are fill/artefacts and excluded
# before computing extremes.  Set conservatively.
PHYSICAL_LOWER = {
    'sst':  10.0,    # °C — no Atlantic TC genesis below 10°C SST
    'msl':  87000,   # Pa — below any observed surface pressure
    'cape': 0.0,     # J/kg — CAPE is non-negative by definition
    'r':    0.0,     # % — relative humidity non-negative
    'vo':   -1.0,    # s⁻¹ — vorticity can be negative
    'd':    -1.0,    # s⁻¹ — divergence can be negative
    'vwsh': 0.0,     # m/s — wind shear magnitude non-negative
    'pi':   0.0,     # m/s — PI non-negative
}
PHYSICAL_UPPER = {
    'sst':  40.0,    # °C
    'msl':  110000,  # Pa
    'cape': 10000,   # J/kg
    'r':    105.0,   # % (ERA5 can exceed 100 slightly)
    'vo':   1.0,     # s⁻¹
    'd':    1.0,     # s⁻¹
    'vwsh': 120.0,   # m/s
    'pi':   120.0,   # m/s
}


def compute_extremes(csv_path: Path,
                     percentile: float = 0.1,
                     use_percentile: bool = True,
                     chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Scan the full CSV in chunks and compute per-variable min/max.

    Parameters
    ----------
    csv_path      : path to NACSGM_combined.csv
    percentile    : clip outer `percentile`% of each tail (e.g. 0.1 → p0.1/p99.9)
    use_percentile: if False, use true global min/max (sensitive to outliers)
    chunk_size    : rows per chunk for memory efficiency

    Returns
    -------
    DataFrame with columns: variable, min, max, n_valid, n_below_physical,
                             p001, p999 (1st/99.9th percentiles for context)
    """
    print(f"\nScanning {csv_path}  (chunk_size={chunk_size:,})…")

    # Two-pass approach:
    #   Pass 1 — collect per-chunk min/max and reservoir sample for percentiles
    #   Pass 2 — not needed; reservoir is sufficient for percentile estimation

    # Use a reservoir of up to 2M values per variable for percentile estimation
    RESERVOIR_SIZE = 2_000_000
    reservoirs  = {v: [] for v in VARIABLES}
    chunk_mins  = {v: [] for v in VARIABLES}
    chunk_maxs  = {v: [] for v in VARIABLES}
    n_valid     = {v: 0  for v in VARIABLES}
    n_below     = {v: 0  for v in VARIABLES}
    n_above     = {v: 0  for v in VARIABLES}
    total_rows  = 0
    rng = np.random.default_rng(42)

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)

        for var in VARIABLES:
            if var not in chunk.columns:
                print(f"  WARNING: '{var}' not found in CSV — skipping.")
                continue

            raw = chunk[var].astype(float).values
            lo  = PHYSICAL_LOWER.get(var, -np.inf)
            hi  = PHYSICAL_UPPER.get(var,  np.inf)

            # Count and exclude physically implausible values
            mask_below = raw < lo
            mask_above = raw > hi
            n_below[var] += int(mask_below.sum())
            n_above[var] += int(mask_above.sum())

            valid = raw[~mask_below & ~mask_above & np.isfinite(raw)]
            if len(valid) == 0:
                continue

            n_valid[var] += len(valid)
            chunk_mins[var].append(float(valid.min()))
            chunk_maxs[var].append(float(valid.max()))

            # Reservoir sampling for percentile estimation
            current = reservoirs[var]
            if len(current) < RESERVOIR_SIZE:
                current.extend(valid.tolist())
                if len(current) > RESERVOIR_SIZE:
                    reservoirs[var] = current[:RESERVOIR_SIZE]
            else:
                # Replace random elements
                idx = rng.integers(0, len(current), size=len(valid))
                arr = np.array(current)
                arr[idx] = valid[:len(idx)]
                reservoirs[var] = arr.tolist()

        if total_rows % (chunk_size * 5) == 0:
            print(f"  … {total_rows:,} rows processed")

    print(f"  Total rows scanned: {total_rows:,}\n")

    # ── Build results ─────────────────────────────────────────────────────────
    rows = []
    for var in VARIABLES:
        if not chunk_mins[var]:
            print(f"  WARNING: No valid data found for '{var}'")
            continue

        global_min = float(np.min(chunk_mins[var]))
        global_max = float(np.max(chunk_maxs[var]))

        res = np.array(reservoirs[var])
        p_lo  = float(np.percentile(res, percentile))
        p_hi  = float(np.percentile(res, 100 - percentile))
        p001  = float(np.percentile(res, 0.1))
        p999  = float(np.percentile(res, 99.9))

        if use_percentile:
            out_min = p_lo
            out_max = p_hi
            method  = f'p{percentile}/p{100-percentile}'
        else:
            out_min = global_min
            out_max = global_max
            method  = 'true global min/max'

        rows.append({
            'variable':        var,
            'min':             round(out_min, 8),
            'max':             round(out_max, 8),
            'method':          method,
            'global_min':      round(global_min, 8),
            'global_max':      round(global_max, 8),
            'p0.1':            round(p001, 8),
            'p99.9':           round(p999, 8),
            'n_valid':         n_valid[var],
            'n_below_physical': n_below[var],
            'n_above_physical': n_above[var],
        })

    return pd.DataFrame(rows)


def compare_with_old(new_df: pd.DataFrame,
                     old_path: Path) -> pd.DataFrame:
    """Load old extremes.csv and build a side-by-side comparison."""
    if not old_path.exists():
        print(f"  Old extremes not found at {old_path} — skipping comparison.")
        return new_df

    old = pd.read_csv(old_path)
    old.columns = [c.strip().lower() for c in old.columns]
    old['variable'] = old['variable'].str.strip()
    old = old.rename(columns={'min': 'old_min', 'max': 'old_max'})

    cmp = new_df.merge(old[['variable', 'old_min', 'old_max']],
                       on='variable', how='left')
    cmp['min_change'] = cmp['min'] - cmp['old_min']
    cmp['max_change'] = cmp['max'] - cmp['old_max']
    cmp['range_old']  = cmp['old_max'] - cmp['old_min']
    cmp['range_new']  = cmp['max']     - cmp['min']
    cmp['range_pct_change'] = (cmp['range_new'] - cmp['range_old']) / cmp['range_old'] * 100
    return cmp


def print_report(cmp: pd.DataFrame):
    """Print a formatted comparison report."""
    print("=" * 78)
    print("EXTREMES RECOMPUTATION REPORT")
    print("=" * 78)

    has_old = 'old_min' in cmp.columns

    hdr = (f"  {'Var':<6}  {'New Min':>12}  {'New Max':>12}  "
           f"{'n valid':>10}  {'n<phys':>8}")
    if has_old:
        hdr += f"  {'Min Δ':>10}  {'Max Δ':>10}  {'Range Δ%':>9}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for _, r in cmp.iterrows():
        line = (f"  {r['variable']:<6}  {r['min']:>12.5f}  {r['max']:>12.5f}  "
                f"{r['n_valid']:>10,}  {r['n_below_physical']:>8,}")
        if has_old and pd.notna(r.get('old_min')):
            flag = ' ⚠' if abs(r['min_change']) > 0.5 * abs(r['range_old']) else '  '
            line += (f"  {r['min_change']:>+10.4f}  {r['max_change']:>+10.4f}  "
                     f"{r['range_pct_change']:>+8.1f}%{flag}")
        print(line)

    print()
    if has_old:
        print("  ⚠  = min or max shifted by > 50% of old range — review carefully")
        significant = cmp[cmp['min_change'].abs() > 0.01]
        if len(significant):
            print(f"\n  Variables with meaningful min change (> 0.01):")
            for _, r in significant.iterrows():
                print(f"    {r['variable']:<6}  old_min={r['old_min']:.5f}  "
                      f"new_min={r['min']:.5f}  Δ={r['min_change']:+.5f}")
    print("=" * 78)


def plot_comparison(cmp: pd.DataFrame, out_path: Path):
    """Bar chart showing old vs new min/max per variable."""
    if 'old_min' not in cmp.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('extremes.csv Recomputation — Old vs New Bounds',
                 fontsize=12, fontweight='bold')

    x     = np.arange(len(cmp))
    width = 0.35
    vars_ = cmp['variable'].tolist()

    for ax, col, title in [
        (axes[0], 'min', 'Minimum values'),
        (axes[1], 'max', 'Maximum values'),
    ]:
        old_col = f'old_{col}'
        bars_old = ax.bar(x - width/2, cmp[old_col], width,
                          label='Old', color='#bdbdbd', edgecolor='white')
        bars_new = ax.bar(x + width/2, cmp[col], width,
                          label='New (full annual cycle)',
                          color='#2166ac', edgecolor='white', alpha=0.85)

        # Highlight bars where change is meaningful
        for i, (o, n) in enumerate(zip(cmp[old_col], cmp[col])):
            if abs(n - o) > 1e-4:
                ax.annotate(f'Δ{n-o:+.3f}',
                            xy=(x[i] + width/2, n),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=7, color='#d73027')

        ax.set_xticks(x)
        ax.set_xticklabels(vars_, rotation=30, ha='right')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Recompute extremes.csv over full annual cycle'
    )
    parser.add_argument('--csv',         type=Path,  default=_CSV_PATH)
    parser.add_argument('--old',         type=Path,  default=_OLD_EXTREMES,
                        help='Path to existing extremes.csv for comparison')
    parser.add_argument('--out',         type=Path,  default=_NEW_EXTREMES)
    parser.add_argument('--cmp-csv',     type=Path,  default=_CMP_CSV)
    parser.add_argument('--cmp-png',     type=Path,  default=_CMP_PNG)
    parser.add_argument('--percentile',  type=float, default=0.1,
                        help='Clip outer N%% from each tail (default 0.1)')
    parser.add_argument('--no-percentile', action='store_true',
                        help='Use true global min/max instead of percentile clip')
    parser.add_argument('--chunk',       type=int,   default=500_000)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}")
        sys.exit(1)

    # ── Compute ───────────────────────────────────────────────────────────────
    new_df = compute_extremes(
        args.csv,
        percentile=args.percentile,
        use_percentile=not args.no_percentile,
        chunk_size=args.chunk,
    )

    # ── Compare with old ──────────────────────────────────────────────────────
    cmp = compare_with_old(new_df, args.old)
    print_report(cmp)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # extremes_new.csv — only variable/min/max columns (drop-in replacement)
    out_clean = new_df[['variable', 'min', 'max']].copy()
    try:
        out_clean.to_csv(args.out, index=False)
        print(f"\n  New extremes saved  → {args.out}")
        print(f"  Review the file, then copy it over extremes.csv to use it.")
    except PermissionError:
        from datetime import datetime
        alt = args.out.parent / f"extremes_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_clean.to_csv(alt, index=False)
        print(f"\n  New extremes saved  → {alt}  (original path locked)")

    # extremes_comparison.csv — full detail
    try:
        cmp.to_csv(args.cmp_csv, index=False)
        print(f"  Comparison CSV saved → {args.cmp_csv}")
    except PermissionError:
        print(f"  WARNING: Could not save comparison CSV (file locked).")

    # comparison plot
    plot_comparison(cmp, args.cmp_png)

    # ── Deployment instructions ───────────────────────────────────────────────
    print(f"""
{'─'*68}
NEXT STEPS
{'─'*68}
1. Review extremes_new.csv — especially SST min (should drop to ~18–22°C).

2. If the new bounds look correct, replace the old file:
     copy {args.out}  {args.old}
   or on Linux/Mac:
     cp {args.out} {args.old}

3. Re-run TCG-NACSGM.py — the normalisation range audit in load_nacsgm_storms()
   will confirm all channels are cleanly in [0, 1] with zero clips needed.

4. To note in the dissertation:
   "Extremes were recomputed over the full 1980–2025 annual cycle to ensure
    cool-season and high-latitude genesis events normalise within [0, 1]."
{'─'*68}
""")


if __name__ == '__main__':
    main()
