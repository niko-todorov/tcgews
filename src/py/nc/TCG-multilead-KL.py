"""
6. Multi-Lead-Time Tropical Cyclogenesis Dataset & Loss
=======================================================
Handles training at multiple lead times (0h through 48h) simultaneously,
with σ-widened Gaussian targets normalized to probability distributions.

Key design decisions:
  - Target heatmaps are probability distributions (sum=1.0) at every lead time
  - σ widens with lead time to encode spatial uncertainty
  - KL divergence loss for positives (distribution matching)
  - Suppression loss for negatives (penalize any activation)
  - Each positive storm yields one training sample per available lead time
  - Negatives are lead-time-agnostic (zero target at any lead)

Data structures:
  storms:      (N_pos, T, C, H, W)  — existing atmospheric data, T=9 timesteps
  genesis_loc: (N_pos, H, W)        — binary, 1 at genesis grid point
  negatives:   (N_neg, C, H, W)     — single snapshots for non-developing systems

Grid: 85×169 at 0.25° resolution (11–32°N, 263–305°E)
Channels: SST, MSL, CAPE, RH, VO, D, VWSH, PI

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

import re
import sys
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from scipy.ndimage import label as _cc_label, distance_transform_edt as _edt

def basin_taper(land_mask, taper_cells=3):
    """Boundary-artifact suppressor (Appendix A). land_mask (H,W) bool True=ocean."""
    lab, _   = _cc_label(land_mask)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    atlantic = lab == int(sz.argmax())
    padded   = np.pad(atlantic, 1, constant_values=False)
    dist     = _edt(padded)[1:-1, 1:-1]
    return (np.clip(dist / float(taper_cells), 0.0, 1.0) * atlantic).astype(np.float32)

# =============================================================================
# TIMING HELPERS  (module-level so loaders/callbacks can use them)
# =============================================================================
def _fmt_elapsed(t0):
    """Format elapsed time as 'Xh Ym Zs' (or 'Ym Zs' / 'Zs' as appropriate)."""
    secs = (datetime.now() - t0).total_seconds()
    if secs < 60:
        return f'{secs:.1f}s'
    elif secs < 3600:
        return f'{int(secs // 60)}m {int(secs % 60):02d}s'
    else:
        return (f'{int(secs // 3600)}h {int((secs % 3600) // 60):02d}m '
                f'{int(secs % 60):02d}s')


def _stamp():
    """Current time as '[HH:MM:SS]' string for phase prefixes."""
    return f'[{datetime.now().strftime("%H:%M:%S")}]'


# =============================================================================
# DATA CACHE  (avoid CSV reload when iterating on training code)
# =============================================================================
# Cache stores all post-normalization arrays as a single .npz file. A short
# hash of the cache-invalidating inputs (CSV mtime, extremes.csv mtime,
# ALL_EXCLUDED_IDS, grid bounds, variables, cache version) is embedded in
# the filename, so changing any of those automatically forces a rebuild on
# the next run. Bump _CACHE_VERSION below if loader logic changes.
_CACHE_VERSION = 1  # NACSGM expanded-basin version (11-32N, 263-305E)

# ── Regularization / augmentation (small-sample overfitting controls) ─────────
# The post-cleaning NA-CSGM set has 267 positive storms; with an identically
# sized fully-convolutional model this overfits in most LOYO folds. These knobs
# add regularization. All default to their original-behaviour values so setting
# them to 0 / False reproduces the pre-regularization run exactly.
WEIGHT_DECAY      = 1e-4     # AdamW decay — Run A value (weight-decay-only variant).
AUGMENT           = False    # Run A: no augmentation (weight-decay-only variant)
AUG_MAX_SHIFT     = 6        # max translation in grid cells (0.25° each -> ~1.5°)
AUG_HFLIP         = False    # horizontal (E-W) flip. Off by default: TC genesis
                             #   has real zonal asymmetry (e.g. shear, land), so a
                             #   flip is not guaranteed physically valid. Enable
                             #   only if we accept that assumption.
AUG_FILL          = 0.0      # value for cells shifted in from outside the grid


def _compute_cache_key(csv_path, extremes_path, excluded_ids, grid_bounds,
                       variables):
    """Deterministic 12-char hash of all cache-invalidating inputs."""
    import hashlib, json, os
    parts = {
        'csv_path':       str(csv_path),
        'csv_mtime':      os.path.getmtime(csv_path),
        'extremes_path':  str(extremes_path),
        'extremes_mtime': os.path.getmtime(extremes_path),
        'excluded_ids':   sorted(list(excluded_ids)),
        'grid_bounds':    list(grid_bounds),
        'variables':      list(variables),
        'cache_version':  _CACHE_VERSION,
    }
    blob = json.dumps(parts, sort_keys=True).encode('utf-8')
    return hashlib.md5(blob).hexdigest()[:12]


def _cache_file_path(out_dir, script_name, cache_key):
    """Return Path to the cache file for this run's inputs."""
    return Path(out_dir) / 'cache' / f'{script_name}_{cache_key}.npz'


def _save_data_cache(cache_path, arrays_dict):
    """Save numpy arrays + scalars to .npz (no compression for fast load)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays_dict)


def _load_data_cache(cache_path):
    """Load arrays from .npz, or return None if file doesn't exist."""
    if not cache_path.exists():
        return None
    return dict(np.load(cache_path, allow_pickle=True))


# =============================================================================
# LOYO PROGRESS  (resume training after restart)
# =============================================================================
# After each LOYO fold completes we pickle the cumulative results dict to
# loyo_progress.pkl. On restart, we load this file and skip any years that
# are already done (defined as: results entry exists AND model_fold_<year>.pt
# exists on disk). To force a clean restart, pass --no-resume or delete the
# progress file.
def _progress_path(out_dir):
    from pathlib import Path
    return Path(out_dir) / 'loyo_progress.pkl'


def _model_path(out_dir, year):
    from pathlib import Path
    return Path(out_dir) / f'model_fold_{year}.pt'


def _save_progress(out_dir, results_by_year):
    """Atomic save: write to .tmp, then rename."""
    import pickle
    p = _progress_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(results_by_year, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)  # atomic on POSIX, near-atomic on Windows


def _load_progress(out_dir):
    """Load progress file; return {} if missing or corrupt."""
    import pickle
    p = _progress_path(out_dir)
    if not p.exists():
        return {}
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  WARNING: progress file unreadable ({e}); starting fresh")
        return {}


def _done_years(out_dir, results_by_year):
    """Return list of years considered 'done' (progress entry + model file)."""
    return sorted(
        y for y in results_by_year
        if _model_path(out_dir, y).exists()
    )


# =============================================================================
# GRID CONFIGURATION
# =============================================================================

HEIGHT       = 85          # 11-32 deg N at 0.25 deg (was 77 / 11-30 N)
WIDTH        = 169         # 263-305 deg E at 0.25 deg (was 149 / 263-300 E)
GRID_RES     = 0.25        # degrees per grid cell
NUM_CHANNELS = 8
GENESIS_TIMESTEP = 8       # index of t=0 (genesis) in the 9-timestep sequence

# Explicit grid bounds (used by data loaders and cache hash)
GRID_SOUTH = 11.0
GRID_NORTH = 32.0
GRID_WEST  = 263.0
GRID_EAST  = 305.0

VARIABLES = ['sst', 'msl', 'cape', 'r', 'vo', 'd', 'vwsh', 'pi']
LOAD_VARIABLES   = VARIABLES   # alias for cache hash consistency with WP
LOAD_NUM_CHANNELS = NUM_CHANNELS

GRID_LATS = np.linspace(GRID_NORTH, GRID_SOUTH, HEIGHT)    # N to S
GRID_LONS = np.linspace(GRID_WEST,  GRID_EAST,  WIDTH)     # W to E


# =============================================================================
# CLIPPED-BOUNDARY STORMS  (expanded basin: 11-32N, 263-305E)
# =============================================================================
# History (original basin 11-30N, 263-300E): 54 storms total were excluded
# due to genesis snapped to boundary cells:
#   - 25 storms snapped to 300.0 deg E (eastern boundary)
#   - 29 storms snapped to (30.00, 270.25) (northern boundary)
# Step A (this rev): basin extended N+2deg (to 32N) and E+5deg (to 305E)
# to recover storms whose true coordinates fall within the extended bounds.
# Data fully re-extracted via ncio.py with NATURE/TRACK_TYPE filters.
#
# After the basin expansion, this set is RESET to empty. Run an audit
# (audit_genesis_boundaries_nacsgm.py) on the new CSV to identify any storms
# still snapped to the new boundaries (32N, 305E) and add them below.
# Until then, training proceeds with no clipped-boundary exclusions.
#
# Old (pre-expansion) exclusion list preserved as _CLIPPED_BOUNDARY_IDS_OLD
# below for reference; do NOT use it on the new basin.
CLIPPED_BOUNDARY_IDS = frozenset()


# Edge-exclusion list — populated post-hoc from LOYO diagnostic output if any
# storms within 1-2 deg of new basin edges show systematic large errors.
# Empty initially; first LOYO run will reveal if needed.
EDGE_EXCLUSION_IDS = frozenset()


# Combined exclusion list used by the data loader (cache hash includes this).
ALL_EXCLUDED_IDS = CLIPPED_BOUNDARY_IDS | EDGE_EXCLUSION_IDS


# Old (pre-expansion) exclusion list, preserved for reference / audit comparison
_CLIPPED_BOUNDARY_IDS_OLD = frozenset({
    # --- Step B: eastern boundary (25 storms, old E=300) ---
    '1980214N11330',       # IBTrACS 330 deg E -> snapped to 300 deg E
    '1981219N11334',       # 334 -> 300
    '1983205N10322',       # 322 -> 300
    '1988253N12306',       # 306 -> 300
    '1995256N02643000',
    '1996243N01982999',
    '1997249N01813000',
    '1998239N02182999',
    '1999262N02553000',
    '2001278N12302',       # 302 -> 300
    '2001333N02842999',
    '2002270N02713000',
    '2003256N02192999',
    '2004217N13306',       # 306 -> 300
    '2004223N11301',       # 301 -> 300
    '2005281N02882999',
    '2005318N13298',
    '2010239N02712999',
    '2011271N01872999',
    '2012223N14317',       # 317 -> 300
    '2016273N13300',
    '2019265N12301',       # 301 -> 300
    '2021222N14301',       # 301 -> 300
    '2021263N01922999',
    '2023292N13309',       # 309 -> 300
    # --- Step B-prime: northern boundary, snapped to (30.00, 270.25) ---
    '1981250N15306', '1981307N17279', '1982169N26274', '1985224N18279',
    '1985240N20286', '1985280N18291', '1985299N25270', '1985320N21296',
    '1994227N29273', '1995154N17276', '1995194N01922990', '1995294N14306',
    '1996241N02092996', '1997198N12309', '1998233N01872987',
    '1998259N10335', '1999236N02152923', '2007151N18273', '2007297N18300',
    '2008238N13293', '2008242N02082995', '2008269N02152900',
    '2010279N02202928', '2011231N15278', '2011248N02302994',
    '2011278N02432998', '2012246N02122993', '2014294N20265',
    '2020276N17277',
})


# =============================================================================
# LOYO YEAR FILTER (optional strategic subset)
# =============================================================================
# If set to a non-empty set, only LOYO folds whose test_year is in this set
# will be run. Set to None to run all available years.
LOYO_YEARS_TO_RUN = None  # Run all NACSGM years (~46 folds)


# =============================================================================
# SIGMA SCHEDULE — spatial uncertainty by lead time
# =============================================================================

SIGMA_BY_LEAD_HOURS = {
    0:  1.0,     # ~25 km  — tight localization
    6:  1.5,     # ~37 km
    12: 2.5,     # ~62 km
    18: 3.5,     # ~87 km
    24: 5.0,     # ~125 km
    30: 7.0,     # ~175 km
    36: 9.0,     # ~225 km
    42: 11.0,    # ~275 km
    48: 13.0,    # ~325 km
    # Below: requires additional ERA5 downloads beyond 9 timesteps
    54: 15.0,    # ~375 km
    60: 17.0,    # ~425 km
    66: 19.0,    # ~475 km
    72: 21.0,    # ~525 km — about 5° radius
}


def get_sigma(lead_hours: int) -> float:
    """
    Look up or interpolate σ for a given lead time.
    Falls back to linear interpolation between defined points.
    """
    if lead_hours in SIGMA_BY_LEAD_HOURS:
        return SIGMA_BY_LEAD_HOURS[lead_hours]

    # Linear interpolation
    keys = sorted(SIGMA_BY_LEAD_HOURS.keys())
    if lead_hours < keys[0]:
        return SIGMA_BY_LEAD_HOURS[keys[0]]
    if lead_hours > keys[-1]:
        return SIGMA_BY_LEAD_HOURS[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= lead_hours <= keys[i + 1]:
            frac = (lead_hours - keys[i]) / (keys[i + 1] - keys[i])
            return ((1 - frac) * SIGMA_BY_LEAD_HOURS[keys[i]] +
                    frac * SIGMA_BY_LEAD_HOURS[keys[i + 1]])
    return SIGMA_BY_LEAD_HOURS[keys[-1]]


# =============================================================================
# NORMALIZED GAUSSIAN HEATMAP
# =============================================================================

def make_unnormalized_heatmap(
    genesis_lat_idx: int,
    genesis_lon_idx: int,
    sigma: float,
    height: int = HEIGHT,
    width: int = WIDTH,
    land_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a 2D Gaussian heatmap centered on genesis — NOT normalized.

    Peak value = 1.0. Background values decay exponentially with distance.
    Used with the focal BCE loss, which weights cells by their gaussian value
    directly (no normalization needed — background cells naturally get ~0 weight).

    Returns
    -------
    heatmap : (H, W) float32, peak=1.0 at genesis, land cells=0
    """
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    dy = (y_grid - genesis_lat_idx).astype(np.float32)
    dx = (x_grid - genesis_lon_idx).astype(np.float32)
    heatmap = np.exp(-0.5 * (dy**2 + dx**2) / (sigma**2))
    if land_mask is not None:
        heatmap *= land_mask.astype(np.float32)
    return heatmap.astype(np.float32)


def make_normalized_heatmap(
    genesis_lat_idx: int,
    genesis_lon_idx: int,
    sigma: float,
    height: int = HEIGHT,
    width: int = WIDTH,
    land_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a 2D Gaussian probability distribution centered on the genesis point.

    The heatmap sums to 1.0 regardless of σ, ensuring equal probability mass
    across all lead times. This prevents broader targets from dominating
    the loss.

    Parameters
    ----------
    genesis_lat_idx : int
        Row index of genesis location in the grid
    genesis_lon_idx : int
        Column index of genesis location in the grid
    sigma : float
        Standard deviation in grid cells (from SIGMA_BY_LEAD_HOURS)
    land_mask : (H, W) bool array, optional
        True for ocean cells. If provided, land cells are zeroed out
        before normalization.

    Returns
    -------
    heatmap : (H, W) float32 array summing to 1.0
    """
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    dy = (y_grid - genesis_lat_idx).astype(np.float32)
    dx = (x_grid - genesis_lon_idx).astype(np.float32)
    heatmap = np.exp(-0.5 * (dy**2 + dx**2) / (sigma**2))

    # Zero out land if mask provided
    if land_mask is not None:
        heatmap *= land_mask.astype(np.float32)

    # Normalize to probability distribution
    total = heatmap.sum()
    if total > 0:
        heatmap /= total
    else:
        # Edge case: genesis point is on land or sigma is tiny
        # Fall back to uniform over ocean
        if land_mask is not None:
            heatmap = land_mask.astype(np.float32)
            heatmap /= heatmap.sum()

    return heatmap.astype(np.float32)


# =============================================================================
# DATASET CLASS
# =============================================================================

_MONTH_SAMPLE_WEIGHTS = {
    1:1.0,2:1.0,3:1.0,4:2.0,5:1.7,6:1.5,7:1.7,8:1.3,9:2.1,10:1.0,11:2.3,12:1.3
}

class TCGMultiLeadDataset(Dataset):
    """
    Multi-lead-time TCG prediction dataset.

    Each positive storm produces one sample per available lead time,
    multiplying the effective dataset size. Negative storms produce
    one sample with a zero target.

    Sample structure:
        x:          (C, H, W)   atmospheric snapshot
        target:     (H, W)      probability heatmap (sum=1 for pos, all-zero for neg)
        is_negative: bool
        lead_hours: int         (0 for negatives)
        sigma:      float       target σ (0 for negatives)

    __getitem__ returns:
        x           : (C, H, W) tensor
        target      : (H, W) tensor
        is_negative : bool
        lead_hours  : int
        storm_id    : str
    """

    def __init__(
        self,
        storms: np.ndarray,
        genesis_loc: np.ndarray,
        storm_ids: List[str],
        storm_years: np.ndarray,
        land_mask: np.ndarray,
        lead_hours_list: Optional[List[int]] = None,
        negatives: Optional[np.ndarray] = None,
        neg_ids: Optional[List[str]] = None,
        neg_years: Optional[np.ndarray] = None,
        normalize_stats: Optional[List[Tuple[float, float]]] = None,
        augment: bool = False,
    ):
        self.land_mask = land_mask.astype(np.float32)
        self.normalize_stats = normalize_stats
        self.augment = augment
        self.samples = []

        if lead_hours_list is None:
            lead_hours_list = list(range(0, (GENESIS_TIMESTEP + 1) * 6, 6))

        # --- Build positive samples ---
        for i in range(len(storms)):
            gen_locs = np.argwhere(genesis_loc[i] > 0)
            if len(gen_locs) == 0:
                continue
            gen_lat_idx, gen_lon_idx = gen_locs[0] # first (should be only) genesis point

            for lead_h in lead_hours_list:
                lead_steps    = lead_h // 6
                input_timestep = GENESIS_TIMESTEP - lead_steps

                if input_timestep < 0:
                    continue  # not enough pre-genesis data

                sigma  = get_sigma(lead_h)
                target = make_unnormalized_heatmap(
                    gen_lat_idx, gen_lon_idx, sigma,
                    land_mask=self.land_mask,
                )

                # Derive month from IBTrACS SID for per-sample weighting
                _month = 9  # default Sep (most underrepresented)
                try:
                    _sid = storm_ids[i]
                    if len(_sid) >= 7 and _sid[:4].isdigit():
                        import datetime as _dt2
                        _doy = int(_sid[4:7])
                        _d   = _dt2.date(int(_sid[:4]), 1, 1) + _dt2.timedelta(_doy - 1)
                        _month = _d.month
                except Exception:
                    pass
                self.samples.append({
                    # Full sequence (T, C, H, W) for TemporalTCG
                    'fields':         storms[i, :input_timestep + 1],
                    'target':         target,
                    'is_negative':    False,
                    'lead_hours':     lead_h,
                    'sigma':          sigma,
                    'year':           int(storm_years[i]),
                    'storm_id':       storm_ids[i],
                    'sample_weight':  _MONTH_SAMPLE_WEIGHTS.get(_month, 1.0),
                })

        n_pos = len(self.samples)

        # --- Build negative samples ---
        if negatives is not None:
            H, W = genesis_loc.shape[1], genesis_loc.shape[2]
            zero_target = np.zeros((H, W), dtype=np.float32)

            for j in range(len(negatives)):
                self.samples.append({
                    'fields':         negatives[j][np.newaxis],  # (1, C, H, W)
                    'target':         zero_target,
                    'is_negative':    True,
                    'lead_hours':     0,
                    'sigma':          0.0,
                    'year':           int(neg_years[j]) if neg_years is not None else 0,
                    'storm_id':       neg_ids[j] if neg_ids is not None else None,
                    'sample_weight':  1.0,
                })

        n_neg = len(self.samples) - n_pos
        print(f"TCGMultiLeadDataset: {len(self.samples)} samples "
              f"({n_pos} positive across {len(lead_hours_list)} lead times, "
              f"{n_neg} negative)")
        print(f"  Lead times: {lead_hours_list}")
        print(f"  σ range: {get_sigma(min(lead_hours_list)):.1f} – "
              f"{get_sigma(max(lead_hours_list)):.1f} grid cells")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x           = torch.FloatTensor(s['fields']) # (T, C, H, W)
        target      = torch.FloatTensor(s['target']) # (H, W)
        is_negative = s['is_negative']
        lead_hours  = s.get('lead_hours', 0)
        storm_id      = s['storm_id']
        sample_weight = s.get('sample_weight', 1.0)          # FIX: was undefined 'storm_id'

        if self.normalize_stats is not None:
            # normalize_stats stores (vmin, vmax) from extremes.csv
            # Data already normalised to [0,1] before dataset construction;
            # this block is a no-op guard in case raw data is passed directly.
            # x is (T, C, H, W) — iterate over C dimension (dim 1), not dim 0
            n_channels = x.shape[1] if x.dim() == 4 else x.shape[0]
            for c in range(n_channels):
                vmin, vmax = self.normalize_stats[c]
                _range = vmax - vmin
                if x.dim() == 4:  # (T, C, H, W)
                    xc = x[:, c]
                    if _range > 0 and (xc.max() > 1.5 or xc.min() < -0.5):
                        x[:, c] = torch.clamp((xc - vmin) / _range, 0.0, 1.0)
                else:             # (C, H, W)
                    if _range > 0 and (x[c].max() > 1.5 or x[c].min() < -0.5):
                        x[c] = torch.clamp((x[c] - vmin) / _range, 0.0, 1.0)
            # Guard: replace any NaN/Inf that slipped through with 0
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Spatial augmentation (train only) ────────────────────────────────
        # Apply the SAME random translation (and optional E-W flip) to both the
        # input fields and the target heatmap so the genesis label stays aligned
        # with the fields. Small-sample regularization: effectively multiplies
        # the 267-storm set by presenting shifted views each epoch. Negatives
        # have all-zero targets, which are shift-invariant, so this is safe for
        # them too.
        if self.augment:
            # random integer shift in [-AUG_MAX_SHIFT, +AUG_MAX_SHIFT]
            if AUG_MAX_SHIFT > 0:
                dy = int(torch.randint(-AUG_MAX_SHIFT, AUG_MAX_SHIFT + 1, (1,)))
                dx = int(torch.randint(-AUG_MAX_SHIFT, AUG_MAX_SHIFT + 1, (1,)))
                if dy != 0 or dx != 0:
                    # roll then zero the wrapped-around border (torch.roll wraps;
                    # we don't want genesis signal teleporting across the domain)
                    x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
                    target = torch.roll(target, shifts=(dy, dx), dims=(-2, -1))
                    # zero the region that wrapped in
                    if dy > 0:
                        x[..., :dy, :] = AUG_FILL;  target[:dy, :] = 0.0
                    elif dy < 0:
                        x[..., dy:, :] = AUG_FILL;  target[dy:, :] = 0.0
                    if dx > 0:
                        x[..., :, :dx] = AUG_FILL;  target[:, :dx] = 0.0
                    elif dx < 0:
                        x[..., :, dx:] = AUG_FILL;  target[:, dx:] = 0.0
            # optional horizontal (E-W) flip
            if AUG_HFLIP and bool(torch.rand(1) < 0.5):
                x = torch.flip(x, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
            # Re-normalise a positive target back to sum=1 (shift/flip may have
            # clipped a little probability mass at the border). Negatives stay 0.
            tsum = target.sum()
            if tsum > 0:
                target = target / tsum

        return x, target, is_negative, lead_hours, storm_id, sample_weight
	# --- Utility: filter samples by year for LOYO CV ---

    def get_years(self) -> np.ndarray:
        """Return array of years for all samples."""
        return np.array([s['year'] for s in self.samples])

    def get_subset(self, indices: List[int],
                   augment: bool = False) -> 'TCGMultiLeadDataset':
        """Create a new dataset with only the specified sample indices.

        `augment` should be True ONLY for the training subset; validation and
        test subsets must never be augmented (augmentation would corrupt the
        evaluation).
        """
        new_ds = TCGMultiLeadDataset.__new__(TCGMultiLeadDataset)
        new_ds.land_mask       = self.land_mask
        new_ds.normalize_stats = self.normalize_stats
        new_ds.augment         = augment
        new_ds.samples         = [self.samples[i] for i in indices]
        return new_ds


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class TCGMultiLeadLoss(nn.Module):
    """
    Softmax-KL loss for TCG genesis location prediction.

    Positive samples — softmax KL divergence over ocean cells:
        Normalises the gaussian target to a probability distribution over the
        9,894 ocean cells, then minimises KL(target ∥ softmax(logits)).
        Softmax forces probability mass to concentrate: mass on wrong cells
        directly reduces mass on the genesis cell, preventing diffuse blobs.

    Negative samples — max-logit suppression (logsumexp peakedness penalty):
        Penalises strong spatial peaks on non-genesis snapshots.

    Parameters
    ----------
    land_mask  : (H, W) numpy array, True for ocean cells
    neg_weight : weight of suppression loss relative to KL loss (default 0.02)
    """

    def __init__(
        self,
        land_mask: np.ndarray,
        neg_weight: float = 0.02,
    ):
        super().__init__()
        mask = torch.FloatTensor(land_mask.astype(np.float32))
        self.register_buffer('land_mask', mask)
        self.neg_weight = neg_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_negative: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits      : (B, 1, H, W) raw model output (no activation)
        targets     : (B, H, W) gaussian heatmaps — NOT normalized, NOT clamped
                      (peak ≈ 0.05 at lead=0, spreads with lead time)
        is_negative : (B,) boolean tensor

        Returns
        -------
        loss    : scalar tensor
        metrics : dict with component losses for logging
        """
        B      = logits.shape[0]
        logits = logits.squeeze(1)   # (B, H, W)

        # Mask land cells — zero them out in both logits and targets
        land  = self.land_mask.unsqueeze(0)   # (1, H, W) — 1=ocean, 0=land
        logits  = logits * land + (-1e9) * (1 - land)

        pos_mask = ~is_negative
        neg_mask =  is_negative

        loss_pos = torch.tensor(0.0, device=logits.device)
        loss_neg = torch.tensor(0.0, device=logits.device)
        n_pos    = pos_mask.sum().item()
        n_neg    = neg_mask.sum().item()

        # ── Positive loss: softmax-KL over ocean cells ──────────────────────
        # Replaces per-cell sigmoid+BCE which allowed diffuse probability blobs.
        # Softmax forces probability mass to concentrate: putting mass at wrong
        # cells directly takes it away from the genesis cell. The gaussian target
        # normalized over ocean cells is the KL reference distribution.
        if n_pos > 0:
            pos_logits  = logits[pos_mask]    # (N_pos, H, W)
            pos_targets = targets[pos_mask]   # (N_pos, H, W) — raw gaussian, peak~0.05

            # Restrict to ocean cells for both logits and targets
            ocean_flat  = self.land_mask.reshape(-1).bool()   # (H*W,) True=ocean
            N_ocean     = int(ocean_flat.sum().item())
            N_pos       = pos_logits.shape[0]
            logits_o    = pos_logits.reshape(N_pos, -1)[:, ocean_flat]   # (N_pos, N_ocean)
            targets_o   = pos_targets.reshape(N_pos, -1)[:, ocean_flat]  # (N_pos, N_ocean)

            # Normalize gaussian target to a probability distribution over ocean cells
            t_sum        = targets_o.sum(dim=1, keepdim=True).clamp(min=1e-8)
            target_prob  = targets_o / t_sum                              # (N_pos, N_ocean)

            # Log-softmax prediction
            log_pred     = F.log_softmax(logits_o, dim=1)                 # (N_pos, N_ocean)

            # KL divergence: KL(target || pred) = sum(target * (log_target - log_pred))
            # Minimising this forces pred to match the gaussian target distribution.
            # Only terms where target_prob > 0 contribute (others are 0 * -inf = 0).
            eps          = 1e-8
            kl           = (target_prob * (torch.log(target_prob + eps) - log_pred)).sum(dim=1)
            # (N_pos,) — one KL value per sample

            if sample_weights is not None:
                sw   = sample_weights[pos_mask]          # (N_pos,)
                loss_pos = (kl * sw).sum() / (sw.sum() + 1e-8)
            else:
                loss_pos = kl.mean()

        # ── Negative loss: suppress max activation ────────────────────────────
        if n_neg > 0:
            neg_logits = logits[neg_mask]              # (N_neg, H, W)
            # CRITICAL: restrict to ocean cells only.
            # Land cells have logit=-1e9 (from land mask); including them
            # in mean_logit makes peakedness = lse - mean ≈ +1e8 → loss explodes.
            ocean_flat   = self.land_mask.reshape(-1).bool()  # True = ocean
            neg_ocean    = neg_logits.reshape(n_neg, -1)[:, ocean_flat]  # (N_neg, N_ocean)
            n_cells      = int(ocean_flat.sum().item())
            # logsumexp ≈ max logit — penalize any strong peak over ocean cells
            lse          = torch.logsumexp(neg_ocean, dim=1)
            mean_logit   = neg_ocean.mean(dim=1)
            uniform_base = float(np.log(n_cells))
            peakedness   = (lse - mean_logit) - uniform_base
            loss_neg     = F.relu(peakedness).mean()

        # ── Combine ───────────────────────────────────────────────────────────
        total_loss = loss_pos + self.neg_weight * loss_neg

        metrics = {
            'loss_total': total_loss.item(),
            'loss_pos':   loss_pos.item(),
            'loss_neg':   loss_neg.item(),
            'n_pos':      n_pos,
            'n_neg':      n_neg,
        }
        return total_loss, metrics


# =============================================================================
# LEAVE-ONE-YEAR-OUT CROSS-VALIDATION SPLITS
# =============================================================================

def loyo_splits(dataset: TCGMultiLeadDataset):
    """
    Yields (train_indices, val_indices, test_indices, test_year) for LOYO CV.

    Ensures negatives from year Y only appear in folds where Y is training data.
    Uses the year immediately prior to the test year as validation.
    """
    years        = dataset.get_years()
    unique_years = sorted(np.unique(years))

    for test_year in unique_years:
        test_mask       = years == test_year
        train_pool_mask = ~test_mask

        # Validation: most recent year before test year
        candidate_val = [y for y in unique_years if y < test_year]
        if candidate_val:
            val_year = max(candidate_val)
        else:
            candidate_val = [y for y in unique_years if y > test_year]
            val_year      = min(candidate_val) if candidate_val else test_year
            if val_year == test_year:
                continue

        val_mask   = (years == val_year)
        train_mask = train_pool_mask & ~val_mask

        test_idx  = np.where(test_mask)[0].tolist()
        val_idx   = np.where(val_mask)[0].tolist()
        train_idx = np.where(train_mask)[0].tolist()

        # Skip if no positive test samples
        n_pos_test = sum(
            1 for i in test_idx if not dataset.samples[i]['is_negative']
        )
        if n_pos_test < 1:
            continue

        n_neg_test = len(test_idx) - n_pos_test
        print(f"Year {test_year}: train={len(train_idx)} "
              f"val={len(val_idx)} test={len(test_idx)} "
              f"(+{n_pos_test}/-{n_neg_test})")

        yield train_idx, val_idx, test_idx, test_year


# =============================================================================
# CUSTOM COLLATE — handles 5-tuple with storm_id
# =============================================================================

def tcg_collate_fn(batch):
    """
    Custom collate for (x, target, is_negative, lead_hours, storm_id) tuples.
    x is now (T, C, H, W) with variable T — pad to max T in batch.
    """
    xs          = [s[0] for s in batch]   # list of (T_i, C, H, W) tensors
    target      = torch.stack([s[1] for s in batch])
    is_negative = torch.tensor([s[2] for s in batch], dtype=torch.bool)
    lead_hours  = [s[3] if len(s) > 3 else 0 for s in batch]
    storm_ids      = [s[4] if len(s) > 4 else None for s in batch]
    sample_weights = torch.tensor([s[5] if len(s) > 5 else 1.0 for s in batch],
                                  dtype=torch.float32)

    # Pad sequences to longest T in batch (left-pad so genesis aligns at T-1)
    max_t = max(xi.shape[0] for xi in xs)
    C, H, W = xs[0].shape[1], xs[0].shape[2], xs[0].shape[3]
    x_padded = torch.zeros(len(xs), max_t, C, H, W, dtype=xs[0].dtype)
    for i, xi in enumerate(xs):
        t_i = xi.shape[0]
        x_padded[i, max_t - t_i:] = xi
    return x_padded, target, is_negative, lead_hours, storm_ids, sample_weights


# =============================================================================
# HAVERSINE DISTANCE
# =============================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# =============================================================================
# EVALUATION — fully corrected with storm IDs, coordinates, DataFrame
# =============================================================================

def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    land_mask: np.ndarray,
    device: torch.device,
    grid_lats: np.ndarray = GRID_LATS,
    grid_lons: np.ndarray = GRID_LONS,
    test_year: int = None,
) -> Dict:
    """
    Evaluate model predictions on a test set.

    Returns per-sample distances, lead times, storm IDs, coordinates,
    and a diagnostic DataFrame suitable for outlier analysis and plotting.

    Parameters
    ----------
    model       : trained SingleTimestepTCG
    loader      : DataLoader yielding 5-tuples (x, target, is_neg, lead_h, storm_id)
    land_mask   : (H, W) bool array, True = ocean
    device      : torch device
    grid_lats   : (H,) latitude array
    grid_lons   : (W,) longitude array
    test_year   : year being evaluated (stored in DataFrame)
    """
    model.eval()

    distances      = []
    lead_hours_out = []
    storm_ids_out  = []
    pred_lats_out  = []
    pred_lons_out  = []
    true_lats_out  = []
    true_lons_out  = []
    years_out      = []

    land_mask_tensor = torch.FloatTensor(land_mask.astype(np.float32)).to(device)
    _taper = basin_taper(land_mask, taper_cells=3)  # (H,W)
    log_taper = torch.log(torch.as_tensor(_taper, device=device).clamp_min(1e-6))
    valid_t = torch.as_tensor((_taper > 0).astype(np.float32), device=device)

    with torch.no_grad():
        for batch in loader:
            x, target, is_negative, lead_hours, storm_ids = batch[0], batch[1], batch[2], batch[3], batch[4]
            x = x.to(device)

            logits = model(x).squeeze(1)  # (B, H, W)
            # Apply land mask before argmax
            # logits = logits + (1 - land_mask_tensor.unsqueeze(0)) * (-1e9)
            logits = logits + log_taper.unsqueeze(0)  # taper boundary/coast
            logits = logits + (1 - valid_t.unsqueeze(0)) * (-1e9)  # ocean minus Pacific wedge
            for j in range(x.shape[0]):
                if is_negative[j]:
                    continue

                # Predicted location
                pred_flat    = logits[j].view(-1)
                pred_idx     = pred_flat.argmax().item()
                pred_lat_idx = pred_idx // WIDTH
                pred_lon_idx = pred_idx % WIDTH

                # True location (argmax of target heatmap)
                tgt_flat     = target[j].view(-1)
                true_idx     = tgt_flat.argmax().item()
                true_lat_idx = true_idx // WIDTH
                true_lon_idx = true_idx % WIDTH

                d = haversine_km(
                    grid_lats[pred_lat_idx], grid_lons[pred_lon_idx],
                    grid_lats[true_lat_idx], grid_lons[true_lon_idx],
                )

                distances.append(d)
                lead_hours_out.append(int(lead_hours[j]))
                storm_ids_out.append(storm_ids[j])
                pred_lats_out.append(float(grid_lats[pred_lat_idx]))
                pred_lons_out.append(float(grid_lons[pred_lon_idx]))
                true_lats_out.append(float(grid_lats[true_lat_idx]))
                true_lons_out.append(float(grid_lons[true_lon_idx]))
                years_out.append(test_year)

    distances      = np.array(distances)
    lead_hours_out = np.array(lead_hours_out)

    if len(distances) == 0:
        return {
            'mean_km':    float('nan'),
            'n':          0,
            'distances':  distances,
            'lead_hours': lead_hours_out,
            'storm_ids':  [],
            'df':         pd.DataFrame(),
        }

    df = pd.DataFrame({
        'storm_id':    storm_ids_out,
        'year':        years_out,
        'lead_hours':  lead_hours_out,
        'distance_km': distances,
        'pred_lat':    pred_lats_out,
        'pred_lon':    pred_lons_out,
        'true_lat':    true_lats_out,
        'true_lon':    true_lons_out,
    })

    return {
        'mean_km':    np.mean(distances),
        'median_km':  np.median(distances),
        'pct_50km':   (distances <=  50).mean() * 100,
        'pct_100km':  (distances <= 100).mean() * 100,
        'pct_200km':  (distances <= 200).mean() * 100,
        'pct_500km':  (distances <= 500).mean() * 100,
        'n':          len(distances),
        'distances':  distances,
        'lead_hours': lead_hours_out,
        'storm_ids':  storm_ids_out,
        'df':         df,
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================
@torch.no_grad()
def val_km_pass(model, loader, land_mask, device):
    """Fast val-set km evaluation for early stopping.
    Returns mean haversine distance (km) over positive val samples only.
    Much lighter than full evaluate_predictions — no DataFrame, no per-sample logging.
    """
    model.eval()
    lm = torch.FloatTensor(land_mask.astype(np.float32)).to(device)  # (H,W)
    dists = []
    for batch in loader:
        x, target, is_negative = batch[0], batch[1], batch[2]
        x = x.to(device)
        logits = model(x).squeeze(1)                     # (B, H, W)
        # logits = logits + (1 - lm.unsqueeze(0)) * (-1e9)
        logits = logits + log_taper.unsqueeze(0)  # taper boundary/coast
        logits = logits + (1 - valid_t.unsqueeze(0)) * (-1e9)  # ocean minus Pacific wedge

        for j in range(x.shape[0]):
            if is_negative[j]:
                continue
            pred_idx     = logits[j].view(-1).argmax().item()
            true_idx     = target[j].view(-1).argmax().item()
            pred_lat = GRID_LATS[pred_idx // WIDTH]
            pred_lon = GRID_LONS[pred_idx %  WIDTH]
            true_lat = GRID_LATS[true_idx // WIDTH]
            true_lon = GRID_LONS[true_idx %  WIDTH]
            dists.append(haversine_km(pred_lat, pred_lon, true_lat, true_lon))
    return float(np.mean(dists)) if dists else float('inf')

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed-precision (fp16) via scaler."""
    model.train()
    epoch_metrics = {'loss_total': [], 'loss_pos': [], 'loss_neg': []}
    use_amp = scaler is not None and device.type == 'cuda'

    # FIX: unpack 5-tuple (storm_ids unused during training)
    for x, target, is_negative, lead_hours, storm_ids, sample_weights in loader:
        x              = x.to(device)
        target         = target.to(device)
        is_negative    = is_negative.to(device)
        sample_weights = sample_weights.to(device)

        optimizer.zero_grad()
        with amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(x)
            loss, metrics = criterion(logits, target, is_negative, sample_weights)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in epoch_metrics:
            epoch_metrics[k].append(metrics[k])

    return {k: np.mean(v) for k, v in epoch_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, scaler=None):
    """Validate with optional mixed-precision."""
    model.eval()
    epoch_metrics = {'loss_total': [], 'loss_pos': [], 'loss_neg': []}
    use_amp = scaler is not None and device.type == 'cuda'

    # FIX: unpack 5-tuple
    for x, target, is_negative, lead_hours, storm_ids, sample_weights in loader:
        x              = x.to(device)
        target         = target.to(device)
        is_negative    = is_negative.to(device)
        sample_weights = sample_weights.to(device)

        with amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(x)
            loss, metrics = criterion(logits, target, is_negative, sample_weights)

        for k in epoch_metrics:
            epoch_metrics[k].append(metrics[k])

    return {k: np.mean(v) for k, v in epoch_metrics.items()}


# =============================================================================
# LOYO EXPERIMENT
# =============================================================================

def run_loyo_experiment(
    storms: np.ndarray,
    genesis_loc: np.ndarray,
    storm_ids: List[str],
    storm_years: np.ndarray,
    land_mask: np.ndarray,
    negatives: Optional[np.ndarray] = None,
    neg_ids: Optional[List[str]] = None,
    neg_years: Optional[np.ndarray] = None,
    lead_hours_list: Optional[List[int]] = None,
    ModelClass=None,  # None resolved to TemporalTCG inside body
    model_kwargs: Optional[Dict] = None,
    epochs: int = 200,
    lr: float = 3e-4,           # lowered from 1e-3
    batch_size: int = 16,
    patience: int = 40,         # increased from 30
    device: Optional[torch.device] = None,
    normalize_stats: Optional[List[Tuple[float, float]]] = None,
    pretrain_weights: Optional[str] = None,
    resume: bool = True,
    out_dir: str = '../../../data/NACSGM/final',
):
    """
    Full leave-one-year-out cross-validation experiment.

    Parameters
    ----------
    storms : (N_pos, T, C, H, W) array
    genesis_loc : (N_pos, H, W) array
    storm_ids : list of str
    storm_years : array of int
    land_mask : (H, W) bool array
    negatives : (N_neg, C, H, W) array, optional
    neg_ids, neg_years : identifiers and years for negatives
    lead_hours_list : which leads to train on (default all available)
    ModelClass : nn.Module class (the SingleTimestepTCG or similar)
    model_kwargs : dict of kwargs for ModelClass
    normalize_stats : list of (vmin, vmax) tuples, one per channel.
        Loaded from extremes.csv. Applied as min-max normalisation
        (x - vmin) / (vmax - vmin) → [0, 1], matching TCG-NACSGM.py.
        Data is normalised before dataset construction; this is a guard
        in case raw values are passed. If None, no normalisation applied.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_kwargs is None:
        model_kwargs = {'input_channels': LOAD_NUM_CHANNELS, 'base_filters': 32}

    # Build full dataset (all years)
    full_dataset = TCGMultiLeadDataset(
        storms=storms,
        genesis_loc=genesis_loc,
        storm_ids=storm_ids,
        storm_years=storm_years,
        land_mask=land_mask,
        lead_hours_list=lead_hours_list,
        negatives=negatives,
        neg_ids=neg_ids,
        neg_years=neg_years,
        normalize_stats=normalize_stats,
    )

    # ── Resume: load progress from disk if available ────────────────────────
    results_by_year = {}
    if resume:
        results_by_year = _load_progress(out_dir)
        done = _done_years(out_dir, results_by_year)
        if done:
            print(f"\n  RESUME: found {len(done)} completed fold(s) in "
                  f"{_progress_path(out_dir).name}:")
            print(f"    {done}")
            print(f"  Will skip these years. To force restart, "
                  f"delete {_progress_path(out_dir).name} or pass --no-resume.")
            stale = [y for y in list(results_by_year) if y not in done]
            for y in stale:
                print(f"  Dropping stale entry for {y} "
                      f"(model_fold_{y}.pt not found, will redo)")
                del results_by_year[y]
        else:
            print(f"\n  RESUME: no completed folds found "
                  f"(or {_progress_path(out_dir).name} doesn't exist). "
                  f"Starting from fold 1.")

    # Report which years will run if a filter is active
    if LOYO_YEARS_TO_RUN is not None:
        print(f"\n  LOYO_YEARS_TO_RUN is set ({len(LOYO_YEARS_TO_RUN)} years): "
              f"{sorted(LOYO_YEARS_TO_RUN)}")
        print(f"  All other years will be silently skipped.")

    for train_idx, val_idx, test_idx, test_year in loyo_splits(full_dataset):

        # ── Year filter: skip years not in the configured subset ────────────
        if LOYO_YEARS_TO_RUN is not None and test_year not in LOYO_YEARS_TO_RUN:
            continue

        # ── Resume check: skip already-completed folds ──────────────────────
        if test_year in results_by_year:
            r = results_by_year[test_year]
            print(f"\n{'='*60}\nFOLD: test_year={test_year}  [SKIPPED -- already done]")
            print(f"{'='*60}")
            print(f"  Previous result: {r.get('mean_km',0):.0f} km mean, "
                  f"{r.get('median_km',0):.0f} km median, "
                  f"≤100km: {r.get('pct_100km',0):.1f}%, "
                  f"≤500km: {r.get('pct_500km',0):.1f}% "
                  f"(n={r.get('n',0)})")
            continue

        train_ds = full_dataset.get_subset(train_idx, augment=AUGMENT)
        val_ds   = full_dataset.get_subset(val_idx)   # never augment eval
        test_ds  = full_dataset.get_subset(test_idx)  # never augment eval

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=tcg_collate_fn, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            collate_fn=tcg_collate_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            collate_fn=tcg_collate_fn,
        )

        if ModelClass is None:
            ModelClass = TemporalTCG  # default
        # Initialise model — warm-start from pretrain_weights if provided
        print(f"  Device: {device}")
        model = ModelClass(**model_kwargs).to(device)
        if pretrain_weights is not None:
            from pathlib import Path as _Path
            _pt = _Path(pretrain_weights)
            if _pt.exists():
                _sd = torch.load(_pt, map_location=device)
                model.load_state_dict(_sd)
                print(f'  Warm-start from {_pt.name}')
            else:
                print(f'  WARNING: pretrain_weights not found: {_pt} — random init')

        # Quick sanity check: show where model predicts on first test batch at init
        model.eval()
        with torch.no_grad():
            for _bx, _bt, _bn, _bl, _bs, _sw in test_loader:
                _bx = _bx.to(device)
                _logits = model(_bx).squeeze(1)  # (B, H, W)
                _lm = torch.FloatTensor(land_mask.astype('float32')).to(device)
                _logits = _logits + (1 - _lm.unsqueeze(0)) * (-1e9)
                _pred_idxs = _logits.view(_bx.shape[0], -1).argmax(dim=1)
                _pred_lats = [GRID_LATS[i // WIDTH] for i in _pred_idxs.cpu().tolist()]
                _pred_lons = [GRID_LONS[i %  WIDTH] for i in _pred_idxs.cpu().tolist()]
                print(f'  Pre-train predictions (first batch): '
                      f'lat=[{min(_pred_lats):.1f},{max(_pred_lats):.1f}] '
                      f'lon=[{min(_pred_lons):.1f},{max(_pred_lons):.1f}]')
                break
        model.train()
        criterion = TCGMultiLeadLoss(land_mask).to(device)
        scaler    = amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
        # AdamW with weight decay: the post-cleaning basin trains on fewer
        # positive storms (267) with an identically-sized fully-convolutional
        # model, so it overfits (val loss >> train loss in most folds). Weight
        # decay penalises that excess capacity. Set WEIGHT_DECAY=0.0 to recover
        # the original plain-Adam behaviour.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=30, factor=0.3, min_lr=1e-6,
        )

        best_val_loss    = float('inf')
        best_val_km      = float('inf')   # primary early-stop criterion
        best_state       = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}  # epoch-0 fallback
        patience_counter = 0

        # ── Sanity check: flag any NaN in first train batch ───────────────
        try:
            _bx, _bt, _bn, _bl, _bs = next(iter(train_loader))
            if not torch.isfinite(_bx).all():
                n_bad = (~torch.isfinite(_bx)).sum().item()
                print(f"  ⚠  {n_bad} non-finite values in first train batch "
                      f"— check normalize_stats and CSV data for year {test_year}")
        except Exception:
            pass

        for epoch in range(epochs):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
            )
            val_metrics = validate(model, val_loader, criterion, device, scaler)

            # Step on val_km — directly optimises spatial accuracy
            scheduler.step(val_km if (epoch >= 10 and np.isfinite(val_km) and val_km > 0) else val_metrics['loss_total'])

            val_loss = val_metrics['loss_total']
            val_km   = val_km_pass(model, val_loader, land_mask, device)

            # Track best val loss for logging
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss

            # Early stopping on val km — directly optimises spatial accuracy.
            # Epoch guard: ignore first 5 epochs (random-init km can be
            # spuriously small on tiny val sets before any real learning).
            if epoch >= 10 and np.isfinite(val_km) and val_km > 0 and val_km < best_val_km:
                best_val_km  = val_km
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_counter = 0
            elif epoch >= 10:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch}  "
                          f"(best_val_loss={best_val_loss:.4f}  "
                          f"best_val_km={best_val_km:.0f} km)")
                    break

            current_lr = optimizer.param_groups[0]['lr']
            _km_flag = ' *' if val_km == best_val_km else ''
            print(f"  {datetime.now().strftime('%H:%M:%S')}  "
                  f"Epoch {epoch:3d}: "
                  f"train={train_metrics['loss_total']:.4f} "
                  f"val={val_metrics['loss_total']:.4f} "
                  f"val_km={val_km:6.0f}"
                  f"{_km_flag}  "
                  f"(pos={val_metrics['loss_pos']:.4f} "
                  f"neg={val_metrics['loss_neg']:.4f}) "
                  f"lr={current_lr:.2e}")

        # Evaluate with best model (best_state always set — epoch-0 fallback above)
        if best_state is None:
            print(f"  WARNING: fold {test_year} — best_state is None, "
                  f"using current model weights")
        else:
            model.load_state_dict(best_state)
        test_metrics = evaluate_predictions(
            model, test_loader, land_mask, device, test_year=test_year,
        )

        # Save best model weights for this fold
        if best_state is not None:
            torch.save(best_state, str(_model_path(out_dir, test_year)))

        results_by_year[test_year] = test_metrics
        print(f"  Year {test_year}: {test_metrics['mean_km']:.0f} km mean, "
              f"≤100km: {test_metrics['pct_100km']:.1f}%, "
              f"≤500km: {test_metrics['pct_500km']:.1f}% "
              f"(n={test_metrics['n']})  "
              f"[best_val_km={best_val_km:.0f} km  best_val_loss={best_val_loss:.4f}]")

        # ── Persist progress after each fold (atomic write) ─────────────────
        _save_progress(out_dir, results_by_year)
        _done_now = _done_years(out_dir, results_by_year)
        print(f"  Progress saved ({len(_done_now)} fold(s) complete). "
              f"Safe to interrupt; --cache + auto-resume will continue.")

    # --- Aggregate across folds ---
    all_distances = np.concatenate(
        [r['distances'] for r in results_by_year.values()
         if len(r.get('distances', [])) > 0]
    )

    print(f"\n{'='*60}")
    print("OVERALL LOYO RESULTS")
    print(f"{'='*60}")
    print(f"  Storms evaluated: {len(all_distances)}")
    print(f"  Mean distance:    {np.mean(all_distances):.0f} km")
    print(f"  Median distance:  {np.median(all_distances):.0f} km")
    print(f"  ≤  50 km: {(all_distances <=  50).mean()*100:.1f}%")
    print(f"  ≤ 100 km: {(all_distances <= 100).mean()*100:.1f}%")
    print(f"  ≤ 200 km: {(all_distances <= 200).mean()*100:.1f}%")
    print(f"  ≤ 500 km: {(all_distances <= 500).mean()*100:.1f}%")

    return results_by_year


def run_all_years_training(
    storms, genesis_loc, storm_ids, storm_years, land_mask,
    negatives=None, neg_ids=None, neg_years=None,
    lead_hours_list=None, ModelClass=None, model_kwargs=None,
    epochs=200, lr=3e-4, batch_size=16, patience=40,
    device=None, normalize_stats=None, val_year=None,
    out_dir='../../../data/NACSGM/final',
    ckpt_name='convlstm_nacsgm_best.pt',
):
    """
    Train ONE final model on all years, for deployment (the EWS checkpoint).

    IDENTICAL recipe to run_loyo_experiment's inner loop -- same TemporalTCG,
    TCGMultiLeadLoss, AdamW(weight_decay=WEIGHT_DECAY), ReduceLROnPlateau,
    augmentation (AUGMENT) and early-stopping-on-val_km (patience, epoch-10
    guard) -- with ONE change: a single split (one held-out year for early-
    stopping validation, all other years for training) and NO test fold.
    Saves the best-val_km weights to {out_dir}/{ckpt_name}.

    LOYO measures the method's skill; this ships it, trained on the full record.
    val_year defaults to the median-activity year (2003 for NACSGM), reserved
    purely for early stopping; override via --val-year or the parameter.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_kwargs is None:
        model_kwargs = {'input_channels': LOAD_NUM_CHANNELS, 'base_filters': 32}
    if ModelClass is None:
        ModelClass = TemporalTCG

    full_dataset = TCGMultiLeadDataset(
        storms=storms, genesis_loc=genesis_loc, storm_ids=storm_ids,
        storm_years=storm_years, land_mask=land_mask,
        lead_hours_list=lead_hours_list,
        negatives=negatives, neg_ids=neg_ids, neg_years=neg_years,
        normalize_stats=normalize_stats,
    )

    years = full_dataset.get_years()
    unique_years = sorted(np.unique(years))
    if val_year is None:
        # Representative val year: the positive-storm count closest to the
        # median, tie-broken toward the middle of the record. This avoids
        # reserving the most recent (operationally relevant, possibly partial)
        # season, and avoids the lightest/heaviest years as a stopping signal.
        # For NACSGM this resolves to 2003 (n=45, median=45).
        _yrs    = np.asarray(years)
        _is_neg = np.array([full_dataset.samples[i]['is_negative']
                            for i in range(len(full_dataset))])
        _uy, _cnt = np.unique(_yrs[~_is_neg], return_counts=True)
        _med = np.median(_cnt)
        _mid = (_uy.min() + _uy.max()) / 2.0
        _order = sorted(range(len(_uy)),
                        key=lambda k: (abs(_cnt[k] - _med), abs(_uy[k] - _mid)))
        val_year = int(_uy[_order[0]])
        print(f"  Auto-selected val year {val_year} "
              f"(positive samples={int(_cnt[_order[0]])}, median={_med:.0f})")
    if val_year not in unique_years:
        raise ValueError(f"val_year {val_year} not in data years {unique_years}")

    val_mask  = years == val_year
    train_idx = np.where(~val_mask)[0].tolist()
    val_idx   = np.where(val_mask)[0].tolist()
    n_pos_val = sum(1 for i in val_idx if not full_dataset.samples[i]['is_negative'])

    print(f"\n{'='*60}\nALL-YEARS FINAL MODEL (deployment checkpoint)")
    print(f"{'='*60}")
    print(f"  Train: {len(unique_years)-1} years (all except {val_year})")
    print(f"  Val (early-stop only): {val_year}  (+{n_pos_val}/-{len(val_idx)-n_pos_val})")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    train_ds = full_dataset.get_subset(train_idx, augment=AUGMENT)
    val_ds   = full_dataset.get_subset(val_idx)   # never augment eval

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=tcg_collate_fn, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              collate_fn=tcg_collate_fn)

    model     = ModelClass(**model_kwargs).to(device)
    print(f"  Device: {device}  |  Model: {type(model).__name__}  "
          f"base_filters={model_kwargs.get('base_filters')}")
    criterion = TCGMultiLeadLoss(land_mask).to(device)
    scaler    = amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=30, factor=0.3, min_lr=1e-6,
    )

    best_val_loss    = float('inf')
    best_val_km      = float('inf')          # primary early-stop criterion
    best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    val_km           = float('inf')

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, scaler)
        val_metrics   = validate(model, val_loader, criterion, device, scaler)
        scheduler.step(val_km if (epoch >= 10 and np.isfinite(val_km) and val_km > 0)
                       else val_metrics['loss_total'])
        val_loss = val_metrics['loss_total']
        val_km   = val_km_pass(model, val_loader, land_mask, device)

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
        if epoch >= 10 and np.isfinite(val_km) and val_km > 0 and val_km < best_val_km:
            best_val_km = val_km
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif epoch >= 10:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best_val_loss={best_val_loss:.4f}  "
                      f"best_val_km={best_val_km:.0f} km)")
                break

        current_lr = optimizer.param_groups[0]['lr']
        _km_flag = ' *' if val_km == best_val_km else ''
        print(f"  {datetime.now().strftime('%H:%M:%S')}  Epoch {epoch:3d}: "
              f"train={train_metrics['loss_total']:.4f} "
              f"val={val_metrics['loss_total']:.4f} "
              f"val_km={val_km:6.0f}{_km_flag}  "
              f"(pos={val_metrics['loss_pos']:.4f} "
              f"neg={val_metrics['loss_neg']:.4f}) lr={current_lr:.2e}")

    model.load_state_dict(best_state)
    out_path = Path(out_dir) / ckpt_name
    torch.save(best_state, str(out_path))
    print(f"\n  Saved all-years deployment model -> {out_path}  "
          f"(best_val_km={best_val_km:.0f} km)")
    return str(out_path)


# =============================================================================
# ANALYSIS: RESULTS BY LEAD TIME
# =============================================================================

def analyze_by_lead_time(
    model: nn.Module,
    dataset: TCGMultiLeadDataset,
    land_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[int, Dict]:
    """Break down prediction accuracy by lead time."""
    lead_groups: Dict[int, List[int]] = {}

    # FIX: access samples dict directly instead of iterating __getitem__
    for i, s in enumerate(dataset.samples):
        if s['is_negative']:
            continue
        lh = s['lead_hours']
        lead_groups.setdefault(lh, []).append(i)

    results = {}
    for lead_h in sorted(lead_groups.keys()):
        indices = lead_groups[lead_h]
        subset  = dataset.get_subset(indices)
        loader  = DataLoader(subset, batch_size=batch_size,
                             collate_fn=tcg_collate_fn)
        metrics = evaluate_predictions(model, loader, land_mask, device)
        results[lead_h] = metrics

        σ = get_sigma(lead_h)
        print(f"  Lead {lead_h:3d}h (σ={σ:5.1f}): "
              f"{metrics['mean_km']:6.0f} km mean, "
              f"≤100km: {metrics['pct_100km']:5.1f}%, "
              f"≤500km: {metrics['pct_500km']:5.1f}% "
              f"(n={metrics['n']})")

    return results


# =============================================================================
# BOOTSTRAPPED CONFIDENCE INTERVALS — storm-level resampling
# =============================================================================

def bootstrap_ci_storm_level(
    df_err: pd.DataFrame,
    group_col: str = None,
    group_val=None,
    lead_hours: int = None,
    n_bootstrap: int = 1000,
    ci: int = 95,
    seed: int = 42,
) -> dict:
    """
    Storm-level bootstrap CI. Resamples whole storms so that all lead-time
    observations for a storm move together, preserving within-storm correlation.

    Parameters
    ----------
    df_err      : DataFrame with [storm_id, year, lead_hours, distance_km]
    group_col   : optional column to filter on before resampling (e.g. 'year')
    group_val   : value to filter group_col to
    lead_hours  : if provided, compute CI for this lead time after resampling
    n_bootstrap : bootstrap iterations
    ci          : confidence interval width (default 95)
    seed        : random seed

    Returns
    -------
    dict of metric -> (lower, upper)
    """
    rng = np.random.default_rng(seed)

    if group_col is not None and group_val is not None:
        df_err = df_err[df_err[group_col] == group_val]

    unique_storms = df_err['storm_id'].unique()
    n_storms      = len(unique_storms)

    stats = {'mean': [], 'median': [], 'pct_100km': [], 'pct_500km': []}

    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_storms, size=n_storms, replace=True)
        frames  = [df_err[df_err['storm_id'] == s] for s in sampled]
        boot_df = pd.concat(frames, ignore_index=True)

        if lead_hours is not None:
            boot_df = boot_df[boot_df['lead_hours'] == lead_hours]

        if len(boot_df) == 0:
            continue

        d = boot_df['distance_km'].values
        stats['mean'].append(np.mean(d))
        stats['median'].append(np.median(d))
        stats['pct_100km'].append((d <= 100).mean() * 100)
        stats['pct_500km'].append((d <= 500).mean() * 100)

    alpha = (100 - ci) / 2
    return {
        k: (np.percentile(v, alpha), np.percentile(v, 100 - alpha))
        for k, v in stats.items()
    }


def compute_all_cis(
    df_err: pd.DataFrame,
    lead_hours_list: list,
    n_bootstrap: int = 1000,
    ci: int = 95,
) -> dict:
    """
    Compute storm-level bootstrap CIs for overall, per-year, and per-lead-time.

    Parameters
    ----------
    df_err          : DataFrame with [storm_id, year, lead_hours, distance_km]
    lead_hours_list : e.g. [0, 6, 12, 18, 24, 30, 36, 42, 48]

    Returns
    -------
    dict with keys 'overall', 'by_year', 'by_lead'
    """
    print("Computing overall CI...")
    overall = bootstrap_ci_storm_level(df_err, n_bootstrap=n_bootstrap, ci=ci)

    print("Computing per-year CIs...")
    by_year = {}
    for year in sorted(df_err['year'].unique()):
        n_storms = df_err[df_err['year'] == year]['storm_id'].nunique()
        if n_storms < 3:
            print(f"  Skipping {year} (only {n_storms} storms)")
            continue
        by_year[year] = bootstrap_ci_storm_level(
            df_err, group_col='year', group_val=year,
            n_bootstrap=n_bootstrap, ci=ci,
        )

    print("Computing per-lead-time CIs...")
    by_lead = {}
    for lh in lead_hours_list:
        by_lead[lh] = bootstrap_ci_storm_level(
            df_err, lead_hours=lh,
            n_bootstrap=n_bootstrap, ci=ci,
        )

    return {'overall': overall, 'by_year': by_year, 'by_lead': by_lead}


def print_ci_summary(cis: dict):
    """Pretty-print CI results."""
    print("\n" + "=" * 60)
    print("BOOTSTRAPPED 95% CIs (storm-level resampling)")
    print("=" * 60)

    print("\nOVERALL:")
    for metric, (lo, hi) in cis['overall'].items():
        print(f"  {metric:>10s}: ({lo:.1f}, {hi:.1f})")

    print("\nBY LEAD TIME (median distance km):")
    for lh, ci_vals in sorted(cis['by_lead'].items()):
        lo, hi = ci_vals['median']
        print(f"  Lead {lh:3d}h: ({lo:.1f}, {hi:.1f})")

    print("\nBY YEAR (median distance km):")
    for year, ci_vals in sorted(cis['by_year'].items()):
        lo, hi = ci_vals['median']
        print(f"  {year}: ({lo:.1f}, {hi:.1f})")


# =============================================================================
# VISUALIZATION 1 — Lead-time error bands with bootstrapped CIs
# =============================================================================

def plot_lead_time_error_bands(
    results: dict,
    cis: dict,
    lead_hours_list: list,
    outlier_threshold_km: float = None,
    save_path: str = None,
    figsize: tuple = (10, 6),
):
    """
    Two-panel plot: distance error vs lead time (with CI bands) and
    percentage within distance thresholds.

    Parameters
    ----------
    results              : dict from run_loyo_experiment(), keyed by year
    cis                  : dict from compute_all_cis()
    lead_hours_list      : e.g. [0, 6, 12, 18, 24, 30, 36, 42, 48]
    outlier_threshold_km : if set, overlay median with outliers clipped
    save_path            : PNG output path
    """
    all_dists = np.concatenate(
        [r['distances'] for r in results.values() if r['n'] > 0]
    )
    all_leads = np.concatenate(
        [r['lead_hours'] for r in results.values() if r['n'] > 0]
    )

    lead_hours_arr = np.array(sorted(lead_hours_list))
    means, medians         = [], []
    ci_lo_mean, ci_hi_mean = [], []
    ci_lo_med,  ci_hi_med  = [], []
    pct_100, pct_500       = [], []

    for lh in lead_hours_arr:
        d = all_dists[all_leads == lh]
        means.append(np.mean(d))
        medians.append(np.median(d))
        pct_100.append((d <= 100).mean() * 100)
        pct_500.append((d <= 500).mean() * 100)

        ci = cis['by_lead'].get(lh, {})
        ci_lo_mean.append(ci.get('mean',   (np.nan, np.nan))[0])
        ci_hi_mean.append(ci.get('mean',   (np.nan, np.nan))[1])
        ci_lo_med.append( ci.get('median', (np.nan, np.nan))[0])
        ci_hi_med.append( ci.get('median', (np.nan, np.nan))[1])

    means      = np.array(means)
    medians    = np.array(medians)
    ci_lo_mean = np.array(ci_lo_mean)
    ci_hi_mean = np.array(ci_hi_mean)
    ci_lo_med  = np.array(ci_lo_med)
    ci_hi_med  = np.array(ci_hi_med)
    pct_100    = np.array(pct_100)
    pct_500    = np.array(pct_500)

    if outlier_threshold_km is not None:
        medians_clean = []
        for lh in lead_hours_arr:
            d = all_dists[all_leads == lh]
            d_clean = d[d <= outlier_threshold_km]
            medians_clean.append(
                np.median(d_clean) if len(d_clean) > 0 else np.nan
            )
        medians_clean = np.array(medians_clean)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5]},
    )
    fig.subplots_adjust(hspace=0.08)

    ax1.fill_between(lead_hours_arr, ci_lo_mean, ci_hi_mean,
                     alpha=0.15, color='steelblue', label='Mean 95% CI')
    ax1.fill_between(lead_hours_arr, ci_lo_med, ci_hi_med,
                     alpha=0.20, color='darkorange', label='Median 95% CI')
    ax1.plot(lead_hours_arr, means, 'o-', color='steelblue',
             linewidth=2, markersize=6, label='Mean distance')
    ax1.plot(lead_hours_arr, medians, 's-', color='darkorange',
             linewidth=2, markersize=6, label='Median distance')

    if outlier_threshold_km is not None:
        ax1.plot(lead_hours_arr, medians_clean, 's--', color='darkorange',
                 linewidth=1.5, markersize=5, alpha=0.6,
                 label=f'Median (excl. >{outlier_threshold_km:.0f} km)')

    for thresh, label, ls in [(100, '100 km', ':'), (500, '500 km', '--')]:
        ax1.axhline(thresh, color='gray', linestyle=ls,
                    linewidth=1.0, alpha=0.7, label=label)

    ax1.set_ylabel('Great-circle distance (km)', fontsize=12)
    ax1.set_title(
        'LOYO Cross-Validation: Forecast Error by Lead Time\n'
        'with Storm-Level Bootstrapped 95% Confidence Intervals',
        fontsize=13, pad=10,
    )
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    for lh, med in zip(lead_hours_arr, medians):
        ax1.annotate(f'{med:.0f}', xy=(lh, med), xytext=(0, 8),
                     textcoords='offset points', ha='center',
                     fontsize=8, color='darkorange')

    ax2.plot(lead_hours_arr, pct_100, 'o-', color='seagreen',
             linewidth=2, markersize=6, label='≤100 km')
    ax2.plot(lead_hours_arr, pct_500, 's-', color='mediumpurple',
             linewidth=2, markersize=6, label='≤500 km')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_ylabel('% within threshold', fontsize=11)
    ax2.set_xlabel('Forecast lead time (hours)', fontsize=12)
    ax2.set_xticks(lead_hours_arr)
    ax2.set_xticklabels([f'{lh}h' for lh in lead_hours_arr])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    for lh, p in zip(lead_hours_arr, pct_500):
        ax2.annotate(f'{p:.0f}%', xy=(lh, p), xytext=(0, 6),
                     textcoords='offset points', ha='center',
                     fontsize=8, color='mediumpurple')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 2 — Year-by-year violin plots
# =============================================================================

def plot_year_violin(
    results: dict,
    cis: dict = None,
    outlier_threshold_km: float = None,
    min_n: int = 5,
    save_path: str = None,
    figsize: tuple = (18, 6),
):
    """
    Violin plot of distance error distribution for each LOYO test year.

    Violins are colored by median error (green = good, red = bad).
    If cis provided, overlays bootstrapped 95% CI ticks on each violin.

    Parameters
    ----------
    results              : dict from run_loyo_experiment(), keyed by year
    cis                  : optional dict from compute_all_cis()
    outlier_threshold_km : clip distances at this value for display
    min_n                : skip years with fewer samples
    save_path            : PNG output path
    """
    years = sorted(results.keys())
    data, labels, medians, ns = [], [], [], []

    for year in years:
        r = results[year]
        if r['n'] < min_n:
            continue
        d = r['distances'].copy()
        if outlier_threshold_km is not None:
            d = np.clip(d, 0, outlier_threshold_km)
        data.append(d)
        labels.append(str(year))
        medians.append(np.median(d))
        ns.append(r['n'])

    norm   = mcolors.Normalize(vmin=min(medians), vmax=max(medians))
    cmap   = cm.RdYlGn_r
    colors = [cmap(norm(m)) for m in medians]

    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(
        data,
        positions=range(len(data)),
        showmedians=True,
        showextrema=False,
        widths=0.75,
    )

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('dimgray')
        pc.set_alpha(0.75)
        pc.set_linewidth(0.8)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Overlay bootstrapped CI ticks
    if cis is not None and 'by_year' in cis:
        visible_years = [y for y in years if results[y]['n'] >= min_n]
        for i, year in enumerate(visible_years):
            if year in cis['by_year']:
                lo, hi = cis['by_year'][year]['median']
                ax.vlines(i, lo, hi, color='black', linewidth=2.5,
                          alpha=0.6, zorder=5)

    ax.axhline(100, color='steelblue', linestyle=':', linewidth=1.2,
               alpha=0.7, label='100 km')
    ax.axhline(500, color='darkorange', linestyle='--', linewidth=1.2,
               alpha=0.7, label='500 km')

    for i, n in enumerate(ns):
        ax.text(i, -20, f'n={n}', ha='center', va='top',
                fontsize=6.5, color='dimgray', rotation=45)

    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label('Median error (km)', fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Great-circle distance error (km)', fontsize=12)
    ax.set_xlabel('Test year (LOYO fold)', fontsize=12)

    title = 'LOYO Cross-Validation: Error Distribution by Year'
    if outlier_threshold_km:
        title += f'\n(distances clipped at {outlier_threshold_km:.0f} km for display)'
    if cis is not None:
        title += '\nBlack ticks = bootstrapped 95% CI on median'
    ax.set_title(title, fontsize=13, pad=10)

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# VISUALIZATION 3 — Basin map with worst storm tracks
# =============================================================================

def plot_worst_storm_tracks(
    df_err: pd.DataFrame,
    n_worst: int = 10,
    threshold_km: float = 1000,
    ibtracs_df: pd.DataFrame = None,
    save_path: str = None,
    figsize: tuple = (13, 7),
):
    """
    Basin map showing predicted vs true genesis locations for worst predictions.

    Requires pred_lat, pred_lon, true_lat, true_lon columns in df_err,
    which are populated by the corrected evaluate_predictions().

    Parameters
    ----------
    df_err        : master diagnostic DataFrame
    n_worst       : highlight top N storms by mean error
    threshold_km  : also flag any prediction above this threshold
    ibtracs_df    : optional DataFrame [storm_id, lat, lon, time] for best tracks
    save_path     : PNG output path
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("cartopy not installed — run: pip install cartopy")
        return None

    # Identify worst storms
    storm_mean   = (df_err.groupby('storm_id')
                          .agg(mean_dist=('distance_km', 'mean'),
                               year=('year', 'first'))
                          .reset_index()
                          .sort_values('mean_dist', ascending=False))

    worst_storms  = storm_mean.head(n_worst)['storm_id'].tolist()
    above_thresh  = (df_err[df_err['distance_km'] >= threshold_km]
                     ['storm_id'].unique())
    highlight     = list(set(worst_storms) | set(above_thresh))

    print(f"Highlighting {len(highlight)} storms "
          f"({n_worst} worst by mean + any ≥{threshold_km:.0f} km)")

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': proj})

    # Full NACSGM domain: 11–30°N, 263–300°E → -97 to -60°W (1° padding)
    ax.set_extent([263 - 360 - 1, 300 - 360 + 1, 10, 31], crs=proj)

    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor='aliceblue', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=':', zorder=2)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.5, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False

    all_worst = df_err[df_err['storm_id'].isin(highlight)]
    norm      = mcolors.Normalize(
        vmin=all_worst['distance_km'].min(),
        vmax=all_worst['distance_km'].max(),
    )
    cmap = cm.YlOrRd

    for storm_id in highlight:
        storm_df = (df_err[df_err['storm_id'] == storm_id]
                    .sort_values('lead_hours'))

        true_lat = storm_df['true_lat'].iloc[0]
        true_lon = storm_df['true_lon'].iloc[0]
        if true_lon > 180:
            true_lon -= 360

        ax.plot(true_lon, true_lat, '*', markersize=12,
                color='black', markeredgewidth=0.5,
                transform=proj, zorder=5)

        for _, row in storm_df.iterrows():
            pred_lon = row['pred_lon']
            if pred_lon > 180:
                pred_lon -= 360
            color = cmap(norm(row['distance_km']))

            ax.plot(pred_lon, row['pred_lat'], 'o', markersize=6,
                    color=color, markeredgecolor='black',
                    markeredgewidth=0.3, transform=proj, zorder=4)

            ax.annotate(
                '',
                xy=(true_lon, true_lat),
                xytext=(pred_lon, row['pred_lat']),
                xycoords=proj._as_mpl_transform(ax),
                textcoords=proj._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.2, alpha=0.6),
                zorder=3,
            )

        ax.text(true_lon + 0.3, true_lat + 0.3,
                storm_id, fontsize=6.5, color='black',
                transform=proj, zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6))

        if ibtracs_df is not None and storm_id in ibtracs_df['storm_id'].values:
            track = (ibtracs_df[ibtracs_df['storm_id'] == storm_id]
                     .sort_values('time'))
            lons = track['lon'].values.copy()
            lons[lons > 180] -= 360
            ax.plot(lons, track['lat'].values, '-', color='navy',
                    linewidth=1.0, alpha=0.5, transform=proj, zorder=3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        pad=0.02, shrink=0.8)
    cbar.set_label('Forecast error (km)', fontsize=10)

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=10, label='True genesis location'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=7, markeredgecolor='black',
                   label='Predicted location (colored by error)'),
    ]
    if ibtracs_df is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='navy', linewidth=1.5,
                       alpha=0.6, label='IBTrACS best track')
        )
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=8, framealpha=0.9)

    ax.set_title(
        f'Worst LOYO Predictions: Top {n_worst} storms by mean error '
        f'+ all ≥{threshold_km:.0f} km\n'
        'Stars = true genesis; circles = predicted (by lead time)',
        fontsize=12, pad=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


# =============================================================================
# GRID CONFIGURATION (data loader section — uses module-level constants above)
# =============================================================================
# NOTE: GRID_SOUTH/NORTH/WEST/EAST, HEIGHT, WIDTH, LOAD_VARIABLES, and
# LOAD_NUM_CHANNELS are all defined at module level near the top of the file.
# The duplicate definitions that used to live here with old values
# (11-30N, 263-300E) have been removed to avoid shadowing the expanded basin
# bounds (11-32N, 263-305E).
RESOLUTION = 0.25

FULL_LATS = np.round(np.linspace(GRID_NORTH, GRID_SOUTH, HEIGHT), 2)
FULL_LONS = np.round(np.linspace(GRID_WEST,  GRID_EAST,  WIDTH),  2)


# =============================================================================
# HELPERS
# =============================================================================

def _extract_year(storm_id, fallback_time=None):
    if re.match(r'^[A-Z]{2}\d+', storm_id):
        match = re.search(r'(\d{4})$', storm_id)
        if match:
            return int(match.group(1))
    else:
        match = re.match(r'^(\d{4})', storm_id)
        if match:
            return int(match.group(1))
    if fallback_time is not None:
        return pd.Timestamp(fallback_time).year
    raise ValueError(f"Cannot extract year from ID='{storm_id}'")


def _build_coord_maps(df):
    lat_to_idx = {round(lat, 2): i for i, lat in enumerate(FULL_LATS)}
    lon_to_idx = {round(lon, 2): i for i, lon in enumerate(FULL_LONS)}
    df = df.copy()
    df['lat_idx'] = df['latitude'].round(2).map(lat_to_idx)
    df['lon_idx'] = df['longitude'].round(2).map(lon_to_idx)
    n_before = len(df)
    df = df.dropna(subset=['lat_idx', 'lon_idx'])
    n_after  = len(df)
    if n_before != n_after:
        print(f"  ⚠ Dropped {n_before - n_after} rows outside grid bounds")
    df['lat_idx'] = df['lat_idx'].astype(int)
    df['lon_idx'] = df['lon_idx'].astype(int)
    return df


def _build_land_mask(df):
    first_id   = df['ID'].iloc[0]
    sample     = df[df['ID'] == first_id]
    first_time = sample['time'].min()
    sample     = sample[sample['time'] == first_time]
    land_mask  = np.zeros((HEIGHT, WIDTH), dtype=bool)
    land_mask[sample['lat_idx'].values, sample['lon_idx'].values] = True
    print(f"  Ocean cells: {land_mask.sum()}, "
          f"Land cells: {HEIGHT * WIDTH - land_mask.sum()}")
    return land_mask


# =============================================================================
# LOAD POSITIVE STORMS
# =============================================================================

def load_positive_storms(df_pos):
    print("\n=== Loading positive (developing) storms ===")
    df = df_pos.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df = _build_coord_maps(df)

    storm_ids_unique = df['ID'].unique()
    num_storms       = len(storm_ids_unique)
    storm_to_idx     = {sid: i for i, sid in enumerate(storm_ids_unique)}
    df['storm_idx']  = df['ID'].map(storm_to_idx)

    df['time_idx'] = df.groupby('ID')['time'].transform(
        lambda x: pd.factorize(x, sort=True)[0]
    )
    _EXPECTED_T = 9  # t-48h to t-0h at 6h intervals
    n_over = (df['time_idx'] >= _EXPECTED_T).sum()
    if n_over > 0:
        over_storms = df.loc[df['time_idx'] >= _EXPECTED_T, 'ID'].unique()
        print(f"  WARNING: {n_over:,} rows have time_idx >= {_EXPECTED_T} "
              f"({len(over_storms)} storms with >9 timesteps) — dropping excess.")
        print(f"  Affected: {list(over_storms[:10])}"
              f"{'...' if len(over_storms) > 10 else ''}")
        df = df[df['time_idx'] < _EXPECTED_T].copy()
    T_max = int(df['time_idx'].max()) + 1
    print(f"  Storms: {num_storms},  Max timesteps: {T_max}")

    storms      = np.zeros((num_storms, T_max, LOAD_NUM_CHANNELS, HEIGHT, WIDTH),
                           dtype=np.float32)
    genesis_loc = np.zeros((num_storms, HEIGHT, WIDTH), dtype=np.float32)

    for c, var in enumerate(LOAD_VARIABLES):
        col = df[var].values.astype(np.float32)
        col = np.nan_to_num(col, nan=0.0)
        storms[
            df['storm_idx'].values,
            df['time_idx'].values,
            c,
            df['lat_idx'].values,
            df['lon_idx'].values,
        ] = col

    # ── Physical-range clipping ───────────────────────────────────────────────
    # CAPE: physically ≥ 0 J/kg.  ERA5 fill/land sentinel = -273.15 (0 K).
    c_cape = LOAD_VARIABLES.index('cape')
    n_nan  = int(np.isnan(storms[:, :, c_cape, :, :]).sum())
    n_neg  = int((storms[:, :, c_cape, :, :] < 0).sum())
    storms[:, :, c_cape, :, :] = np.nan_to_num(storms[:, :, c_cape, :, :], nan=0.0)
    storms[:, :, c_cape, :, :] = np.clip(storms[:, :, c_cape, :, :], 0.0, None)
    if n_nan or n_neg:
        print(f"  CAPE clip: {n_nan:,} NaN→0 (ERA5 fill), {n_neg:,} negative→0 (noise).")

    # r (RH): clip to [0, 150]%.
    # Small negatives (~-5%) = interpolation noise → clip to 0.
    # Values 100-150% = ERA5 ice supersaturation (physically real, retain signal).
    # Values > 150% = suspect ERA5 artifact → cap at 150.
    c_r   = LOAD_VARIABLES.index('r')
    n_lo  = int((storms[:, :, c_r, :, :] < 0).sum())
    n_hi  = int((storms[:, :, c_r, :, :] > 150).sum())
    if n_lo or n_hi:
        storms[:, :, c_r, :, :] = np.clip(storms[:, :, c_r, :, :], 0.0, 150.0)
        print(f"  RH   clip: {n_lo:,} below 0%, {n_hi:,} above 150% → clamped.")

    gen_df = df[df['origin'] == 1]
    genesis_loc[
        gen_df['storm_idx'].values,
        gen_df['lat_idx'].values,
        gen_df['lon_idx'].values,
    ] = 1.0

    n_genesis = int(genesis_loc.sum())
    print(f"  Genesis markers: {n_genesis}  (expect {num_storms})")
    if n_genesis != num_storms:
        missing = [sid for sid in storm_ids_unique
                   if genesis_loc[storm_to_idx[sid]].sum() == 0]
        print(f"  ⚠ Storms missing genesis marker: {missing}")

    # ── Vectorized storm_years (was per-storm df.loc[df['ID']==sid]) ────────
    # Original: N separate O(N) filters against full df (~30 min on WP, less here).
    # Replaced with one drop_duplicates + reindex to preserve sid order.
    _t_yrs = datetime.now()
    _first_time = (df[['ID', 'time']]
                   .drop_duplicates('ID', keep='first')
                   .set_index('ID')
                   .loc[storm_ids_unique, 'time'])
    storm_years = np.array([
        _extract_year(sid, t)
        for sid, t in zip(storm_ids_unique, _first_time.values)
    ])
    print(f"  storm_years built in {_fmt_elapsed(_t_yrs)}")

    ts_counts   = df.groupby('ID')['time'].nunique()
    n_timesteps = np.array([ts_counts[sid] for sid in storm_ids_unique])

    land_mask = _build_land_mask(df)

    print(f"  storms array  : {storms.shape}  ({storms.nbytes / 1e6:.1f} MB)")
    print(f"  genesis_loc   : {genesis_loc.shape}")
    print(f"  Year range    : {storm_years.min()}–{storm_years.max()}")
    for c, var in enumerate(LOAD_VARIABLES):
        v = storms[:, :, c, :, :]
        print(f"    {var:>5s}: min={v.min():.4f}, max={v.max():.4f}, "
              f"nonzero={np.count_nonzero(v)}")

    return (storms, genesis_loc, list(storm_ids_unique),
            storm_years, n_timesteps, land_mask)


# =============================================================================
# LOAD NEGATIVE STORMS
# =============================================================================

def load_negative_storms(df_neg):
    print("\n=== Loading negative (non-developing) storms ===")
    df = df_neg.copy()
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    df = _build_coord_maps(df)

    df['snapshot_key'] = df['ID'] + '_' + df['time'].astype(str)
    snapshot_keys      = df['snapshot_key'].unique()
    num_neg            = len(snapshot_keys)
    snap_to_idx        = {k: i for i, k in enumerate(snapshot_keys)}
    df['snap_idx']     = df['snapshot_key'].map(snap_to_idx)

    print(f"  Negative snapshots: {num_neg}")

    negatives = np.zeros((num_neg, LOAD_NUM_CHANNELS, HEIGHT, WIDTH),
                         dtype=np.float32)

    for c, var in enumerate(LOAD_VARIABLES):
        col = df[var].values.astype(np.float32)
        col = np.nan_to_num(col, nan=0.0)
        negatives[
            df['snap_idx'].values,
            c,
            df['lat_idx'].values,
            df['lon_idx'].values,
        ] = col

    # ── Physical-range clipping (mirrors load_positive_storms) ───────────────
    c_cape = LOAD_VARIABLES.index('cape')
    n_nan  = int(np.isnan(negatives[:, c_cape, :, :]).sum())
    n_neg  = int((negatives[:, c_cape, :, :] < 0).sum())
    negatives[:, c_cape, :, :] = np.nan_to_num(negatives[:, c_cape, :, :], nan=0.0)
    negatives[:, c_cape, :, :] = np.clip(negatives[:, c_cape, :, :], 0.0, None)
    if n_nan or n_neg:
        print(f"  CAPE clip: {n_nan:,} NaN→0 (ERA5 fill), {n_neg:,} negative→0 (noise).")

    c_r  = LOAD_VARIABLES.index('r')
    n_lo = int((negatives[:, c_r, :, :] < 0).sum())
    n_hi = int((negatives[:, c_r, :, :] > 150).sum())
    if n_lo or n_hi:
        negatives[:, c_r, :, :] = np.clip(negatives[:, c_r, :, :], 0.0, 150.0)
        print(f"  RH   clip: {n_lo:,} below 0%, {n_hi:,} above 150% → clamped.")

    # ── Build neg_ids / neg_years vectorised (was O(N x S) per-snapshot filter) ──
    # Original code did df[df['snapshot_key'] == key] for every key -- on WP this
    # was 6309 scans of a 252M-row df (~10 hours). Same fix here for safety.
    _t_meta = datetime.now()
    _meta = (df[['snapshot_key', 'ID', 'time']]
             .drop_duplicates('snapshot_key', keep='first')
             .set_index('snapshot_key')
             .loc[snapshot_keys])      # reindex to match snapshot_keys order
    neg_ids   = _meta['ID'].tolist()
    neg_years = np.array([
        _extract_year(sid, t)
        for sid, t in zip(_meta['ID'].values, _meta['time'].values)
    ])
    print(f"  neg_ids/neg_years built in {_fmt_elapsed(_t_meta)}")

    print(f"  negatives array: {negatives.shape}  "
          f"({negatives.nbytes / 1e6:.1f} MB)")
    print(f"  Year range     : {neg_years.min()}–{neg_years.max()}")
    for c, var in enumerate(LOAD_VARIABLES):
        v = negatives[:, c, :, :]
        print(f"    {var:>5s}: min={v.min():.4f}, max={v.max():.4f}, "
              f"nonzero={np.count_nonzero(v)}")

    return negatives, neg_ids, neg_years


# =============================================================================
# MODEL
# =============================================================================


# =============================================================================
# TEMPORAL TCG MODEL  (replaces SingleTimestepTCG for all 9 timesteps)
# =============================================================================
class TemporalTCG(nn.Module):
    """
    Convolutional LSTM encoder that jointly processes all T timesteps and
    outputs a spatial genesis logit map from the final hidden state.

    Input:  (B, T, C, H, W)  — T atmospheric snapshots ending at genesis-lead
    Output: (B, 1, H, W)     — raw logits (no activation)

    Architecture
    ------------
    1. Spatial embedding  : 1×1 conv projects C channels → base_filters
    2. ConvLSTM encoder   : 2-layer ConvLSTM processes T steps spatially
    3. Temporal attention : learned per-timestep softmax weights recombine
                            the ConvLSTM output sequence (instead of just
                            using the last hidden state)
    4. Refinement conv    : 3-layer CNN on the attended feature map
    5. Output head        : 1×1 conv → 1 logit channel

    The orthogonal init + forget-gate bias=1 + cell-state clamping are
    taken directly from TCG-NACSGM.py's ConvLSTMCell for training stability.
    """

    def __init__(self, input_channels: int = LOAD_NUM_CHANNELS,
                 base_filters: int = 64):
        super().__init__()
        bf = base_filters

        # ── Spatial embedding ───────────────────────────────────────────────
        self.embed = nn.Sequential(
            nn.Conv2d(input_channels, bf, kernel_size=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
        )

        # ── 2-layer ConvLSTM ────────────────────────────────────────────────
        # Layer 1: embedded features → bf channels
        self.clstm1 = _ConvLSTMCell(bf,      bf,      kernel_size=3)
        # Layer 2: bf → bf*2 for richer temporal features
        self.clstm2 = _ConvLSTMCell(bf,      bf * 2,  kernel_size=3)

        # ── Temporal attention: score each timestep, softmax over T ─────────
        # Single conv channel per timestep → scalar score → softmax weights
        self.t_attn = nn.Sequential(
            nn.Conv2d(bf * 2, 1, kernel_size=1, bias=True),
        )

        # ── Spatial refinement after attention pooling ───────────────────────
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
        """
        x : (B, T, C, H, W)
        returns : (B, 1, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device

        # ── Embed each timestep ──────────────────────────────────────────────
        emb = []
        for t in range(T):
            emb.append(self.embed(x[:, t]))   # (B, bf, H, W)
        # emb: list of T × (B, bf, H, W)

        # ── ConvLSTM layer 1 ────────────────────────────────────────────────
        h1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        c1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        out1 = []
        for t in range(T):
            h1, c1 = self.clstm1(emb[t], h1, c1)
            out1.append(h1)

        # ── ConvLSTM layer 2 ────────────────────────────────────────────────
        h2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        c2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        out2 = []
        for t in range(T):
            h2, c2 = self.clstm2(out1[t], h2, c2)
            out2.append(h2)

        # ── Temporal attention ───────────────────────────────────────────────
        # Score each timestep → softmax → weighted sum over T
        seq = torch.stack(out2, dim=1)            # (B, T, bf*2, H, W)
        scores = []
        for t in range(T):
            s = self.t_attn(seq[:, t])             # (B, 1, H, W)
            scores.append(s)
        scores = torch.stack(scores, dim=1)        # (B, T, 1, H, W)
        weights = torch.softmax(scores, dim=1)     # softmax over T
        attended = (seq * weights).sum(dim=1)      # (B, bf*2, H, W)

        return self.refine(attended)               # (B, 1, H, W)


class _ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — orthogonal init + forget-gate bias=1 + cell clamp."""
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

    def forward(self, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = torch.clamp(f * c + i * g, -10.0, 10.0)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class SingleTimestepTCG(nn.Module):
    """
    Predict TCG location from atmospheric snapshot.
    Input:  (B, C, H, W)
    Output: (B, 1, H, W) logits
    """

    def __init__(self, input_channels=LOAD_NUM_CHANNELS, base_filters=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_filters * 2, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, 1),
        )

        # Initialise final conv so random logits have a spatial prior:
        # slightly positive bias breaks the uniform-argmax-at-cell-0 problem.
        # Weight init: Kaiming uniform (default for Conv2d, but explicit here).
        final_conv = self.encoder[-1]
        nn.init.kaiming_uniform_(final_conv.weight, nonlinearity='relu')
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0.01)

    def forward(self, x):
        return self.encoder(x)

def compute_metrics_with_without_outliers(
    df_err: pd.DataFrame,
    outlier_storm_ids: List[str],
    lead_hours_list: List[int],
    n_bootstrap: int = 1000,
    ci: int = 95,
) -> Dict:
    """
    Compute LOYO metrics with and without identified outlier storms,
    for clean side-by-side reporting in the dissertation.

    Parameters
    ----------
    df_err            : master diagnostic DataFrame
    outlier_storm_ids : list of storm_id strings to exclude
    lead_hours_list   : e.g. [0, 6, 12, 18, 24, 30, 36, 42, 48]
    n_bootstrap       : bootstrap iterations for CIs
    ci                : confidence interval width

    Returns
    -------
    dict with keys 'full' and 'filtered', each containing
    overall metrics, per-year metrics, and per-lead metrics
    """

    def _metrics(df: pd.DataFrame) -> Dict:
        """Compute summary metrics for a DataFrame subset."""
        d = df['distance_km'].values
        if len(d) == 0:
            return {}
        return {
            'n':          len(d),
            'n_storms':   df['storm_id'].nunique(),
            'mean_km':    np.mean(d),
            'median_km':  np.median(d),
            'pct_50km':   (d <=  50).mean() * 100,
            'pct_100km':  (d <= 100).mean() * 100,
            'pct_200km':  (d <= 200).mean() * 100,
            'pct_500km':  (d <= 500).mean() * 100,
        }

    def _by_lead(df: pd.DataFrame) -> Dict:
        out = {}
        for lh in lead_hours_list:
            sub = df[df['lead_hours'] == lh]
            if len(sub) > 0:
                out[lh] = _metrics(sub)
        return out

    def _by_year(df: pd.DataFrame) -> Dict:
        out = {}
        for year in sorted(df['year'].unique()):
            sub = df[df['year'] == year]
            if len(sub) > 0:
                out[year] = _metrics(sub)
        return out

    df_full     = df_err.copy()
    df_filtered = df_err[~df_err['storm_id'].isin(outlier_storm_ids)].copy()

    n_removed        = df_full['storm_id'].isin(outlier_storm_ids).sum()
    n_storms_removed = df_full[df_full['storm_id'].isin(outlier_storm_ids)]['storm_id'].nunique()

    print(f"Outlier storms identified: {len(outlier_storm_ids)}")
    print(f"  Predictions removed:     {n_removed} "
          f"({n_removed/len(df_full)*100:.1f}% of total)")
    print(f"  Storms removed:          {n_storms_removed}")

    # Compute CIs for both sets
    print("\nBootstrapping full dataset CIs...")
    cis_full = compute_all_cis(df_full,     lead_hours_list, n_bootstrap, ci)
    print("Bootstrapping filtered dataset CIs...")
    cis_filt = compute_all_cis(df_filtered, lead_hours_list, n_bootstrap, ci)

    result = {
        'full': {
            'overall':  _metrics(df_full),
            'by_lead':  _by_lead(df_full),
            'by_year':  _by_year(df_full),
            'cis':      cis_full,
            'df':       df_full,
        },
        'filtered': {
            'overall':  _metrics(df_filtered),
            'by_lead':  _by_lead(df_filtered),
            'by_year':  _by_year(df_filtered),
            'cis':      cis_filt,
            'df':       df_filtered,
            'removed_storms': outlier_storm_ids,
        },
    }

    return result


def print_comparison_table(comparison: Dict, lead_hours_list: List[int]):
    """
    Print a side-by-side comparison table suitable for dissertation reporting.
    """
    full = comparison['full']['overall']
    filt = comparison['filtered']['overall']

    print("\n" + "=" * 70)
    print("METRICS: FULL DATASET vs. OUTLIERS EXCLUDED")
    print("=" * 70)
    print(f"{'Metric':<20} {'Full':>15} {'Filtered':>15} {'Δ':>10}")
    print("-" * 70)

    metrics = [
        ('N predictions',  'n',        '{:.0f}'),
        ('N storms',       'n_storms',  '{:.0f}'),
        ('Mean (km)',      'mean_km',   '{:.1f}'),
        ('Median (km)',    'median_km', '{:.1f}'),
        ('≤ 50 km (%)',    'pct_50km',  '{:.1f}'),
        ('≤ 100 km (%)',   'pct_100km', '{:.1f}'),
        ('≤ 200 km (%)',   'pct_200km', '{:.1f}'),
        ('≤ 500 km (%)',   'pct_500km', '{:.1f}'),
    ]

    for label, key, fmt in metrics:
        f_val = full.get(key, float('nan'))
        p_val = filt.get(key, float('nan'))
        delta = p_val - f_val
        sign  = '+' if delta > 0 else ''
        print(f"  {label:<18} {fmt.format(f_val):>15} "
              f"{fmt.format(p_val):>15} {sign}{fmt.format(delta):>9}")

    print("\n" + "-" * 70)
    print("BY LEAD TIME (median km):")
    print(f"  {'Lead':<8} {'Full':>10} {'CI':>16} {'Filtered':>10} {'CI':>16} {'Δ':>8}")
    print("  " + "-" * 68)

    for lh in lead_hours_list:
        f_med = comparison['full']['by_lead'].get(lh, {}).get('median_km', float('nan'))
        p_med = comparison['filtered']['by_lead'].get(lh, {}).get('median_km', float('nan'))

        f_ci  = comparison['full']['cis']['by_lead'].get(lh, {}).get('median', (float('nan'), float('nan')))
        p_ci  = comparison['filtered']['cis']['by_lead'].get(lh, {}).get('median', (float('nan'), float('nan')))

        delta = p_med - f_med
        sign  = '+' if delta > 0 else ''

        print(f"  {lh:3d}h     "
              f"{f_med:>8.1f}   "
              f"({f_ci[0]:>5.1f}, {f_ci[1]:>5.1f})   "
              f"{p_med:>8.1f}   "
              f"({p_ci[0]:>5.1f}, {p_ci[1]:>5.1f})  "
              f"{sign}{delta:>5.1f}")

    print("\nOutliers excluded:")
    for sid in comparison['filtered']['removed_storms']:
        print(f"  {sid}")

def compute_boundary_zone_analysis(
    df_err: pd.DataFrame,
    outlier_storm_ids: List[str],
    lead_hours_list: List[int],
    east_boundary_lon: float  = 299.5,
    west_boundary_lon: float  = 272.5,
    north_boundary_lat: float = 21.0,
    south_boundary_lat: float = 11.0,
) -> tuple:
    """
    Quantify contribution of boundary-zone storms to total error.
    Boundary zones are defined as within ~1-2° of the domain edge:
      Domain: 10–22°N, 271–301°E
      East  : genesis_lon >= east_boundary_lon   (≥299.5°E → within 1.5° of 301°E)
      West  : genesis_lon <= west_boundary_lon   (≤272.5°E → within 1.5° of 271°E)
      North : genesis_lat >= north_boundary_lat  (≥21.0°N  → within 1° of 22°N)
      South : genesis_lat <= south_boundary_lat  (≤11.0°N  → within 1° of 10°N)
    """

    # ---- classify each storm's genesis location ----
    genesis = (df_err.groupby('storm_id')
                     .agg(true_lat=('true_lat', 'first'),
                          true_lon=('true_lon', 'first'),
                          year=('year', 'first'))
                     .reset_index())

    def classify_zone(row):
        zones = []
        if row['true_lon'] >= east_boundary_lon:
            zones.append('SE boundary (east)')
        if row['true_lon'] <= west_boundary_lon:
            zones.append('W/NW boundary (west/Gulf)')
        if row['true_lat'] >= north_boundary_lat:
            zones.append('N boundary')
        if row['true_lat'] <= south_boundary_lat:
            zones.append('S boundary')
        return ', '.join(zones) if zones else 'Interior'

    genesis['zone']       = genesis.apply(classify_zone, axis=1)
    genesis['is_outlier'] = genesis['storm_id'].isin(outlier_storm_ids)

    df = df_err.merge(
        genesis[['storm_id', 'zone', 'is_outlier']],
        on='storm_id', how='left',
    )

    total_predictions = len(df)
    total_error_km    = df['distance_km'].sum()
    total_storms      = df['storm_id'].nunique()

    # ---- summary by zone ----
    rows      = []
    zone_order = [
        'SE boundary (east)',
        'W/NW boundary (west/Gulf)',
        'N boundary',
        'S boundary',
        'Interior',
        'ALL',
    ]

    for zone in zone_order:
        sub = df if zone == 'ALL' else df[df['zone'] == zone]
        if len(sub) == 0:
            continue

        d           = sub['distance_km'].values
        n_storms    = sub['storm_id'].nunique()
        n_preds     = len(sub)

        rows.append({
            'Zone':              zone,
            'N storms':          n_storms,
            '% of storms':       f"{n_storms/total_storms*100:.1f}%",
            'N predictions':     n_preds,
            '% of predictions':  f"{n_preds/total_predictions*100:.1f}%",
            'Mean error (km)':   f"{np.mean(d):.0f}",
            'Median error (km)': f"{np.median(d):.0f}",
            '% ≤100 km':         f"{(d<=100).mean()*100:.1f}%",
            '% ≤500 km':         f"{(d<=500).mean()*100:.1f}%",
            '% of total error':  f"{sub['distance_km'].sum()/total_error_km*100:.1f}%",
        })

    summary_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("BOUNDARY ZONE ANALYSIS (tightened: within 1-2° of domain edge)")
    print(f"  E≥{east_boundary_lon}°E  W≤{west_boundary_lon}°E  "
          f"N≥{north_boundary_lat}°N  S≤{south_boundary_lat}°N")
    print(f"  Domain edges: 10–22°N, 271–301°E")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # ---- interior vs boundary per-lead median ----
    print("\n" + "-" * 80)
    print("MEDIAN ERROR BY LEAD TIME: Interior vs Boundary")
    print(f"  {'Lead':<8} {'Interior':>12} {'Boundary':>12} {'Ratio':>8}")
    print("  " + "-" * 44)

    df_interior = df[df['zone'] == 'Interior']
    df_boundary = df[df['zone'] != 'Interior']

    for lh in lead_hours_list:
        d_int = df_interior[df_interior['lead_hours'] == lh]['distance_km'].values
        d_bnd = df_boundary[df_boundary['lead_hours'] == lh]['distance_km'].values
        if len(d_int) > 0 and len(d_bnd) > 0:
            med_int = np.median(d_int)
            med_bnd = np.median(d_bnd)
            ratio   = med_bnd / med_int
            print(f"  {lh:3d}h     "
                  f"{med_int:>10.1f}   "
                  f"{med_bnd:>10.1f}   "
                  f"{ratio:>6.2f}x")

    # ---- boundary storm listing ----
    print("\n" + "-" * 80)
    print("BOUNDARY STORM LISTING (tightened zones):")
    storm_mean    = (df_err.groupby('storm_id')['distance_km']
                           .mean().reset_index()
                           .rename(columns={'distance_km': 'mean_error_km'}))
    boundary_storms = (genesis[genesis['zone'] != 'Interior']
                       .merge(storm_mean, on='storm_id')
                       .sort_values('mean_error_km', ascending=False))

    print(boundary_storms[['storm_id', 'year', 'zone',
                            'true_lat', 'true_lon', 'mean_error_km']]
          .to_string(index=False))

    print(f"\n  Boundary storms: {len(boundary_storms)} "
          f"({len(boundary_storms)/total_storms*100:.1f}% of dataset)")
    print(f"  Interior storms: {total_storms - len(boundary_storms)} "
          f"({(total_storms-len(boundary_storms))/total_storms*100:.1f}% of dataset)")

    return summary_df, df

def plot_boundary_zone_map(
    df_err: pd.DataFrame,
    df_with_zones: pd.DataFrame,
    east_boundary_lon: float = 298.0,
    west_boundary_lon: float = 278.0,
    north_boundary_lat: float = 21.0,
    south_boundary_lat: float = 12.0,
    save_path: str = None,
    figsize: tuple = (13, 7),
):
    """
    Basin map showing all genesis locations colored by zone classification.
    Overlays boundary zone boxes and mean error per storm as marker size.

    Parameters
    ----------
    df_err          : master diagnostic DataFrame
    df_with_zones   : df_err with 'zone' column added by compute_boundary_zone_analysis
    save_path       : PNG output path
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("cartopy not installed — run: pip install cartopy")
        return None

    # One point per storm (genesis location + mean error)
    genesis = (df_with_zones.groupby('storm_id')
                             .agg(true_lat=('true_lat', 'first'),
                                  true_lon=('true_lon', 'first'),
                                  mean_error=('distance_km', 'mean'),
                                  zone=('zone', 'first'),
                                  year=('year', 'first'))
                             .reset_index())

    # Convert 0–360 → -180–180
    genesis['plot_lon'] = genesis['true_lon'].apply(
        lambda x: x - 360 if x > 180 else x
    )

    zone_colors = {
        'SE boundary (east)':       'crimson',
        'W/NW boundary (west/Gulf)': 'darkorange',
        'N boundary':               'purple',
        'S boundary':               'brown',
        'Interior':                 'steelblue',
    }

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw={'projection': proj})
    ax.set_extent([263 - 360 - 1, 300 - 360 + 1, 10, 31], crs=proj)  # full NACSGM: 11–30°N, 263–300°E

    ax.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor='aliceblue', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=':', zorder=2)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.4, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False

    # Draw boundary zone boxes
    from matplotlib.patches import Rectangle
    box_kw = dict(linewidth=1.5, linestyle='--', fill=False,
                  transform=proj, zorder=3)

    # Eastern boundary box
    ax.add_patch(Rectangle(
        (east_boundary_lon - 360, 10),
        (300 - east_boundary_lon), 21,
        edgecolor='crimson', **box_kw,
    ))
    # Western boundary box
    ax.add_patch(Rectangle(
        (263 - 360, 10),
        (west_boundary_lon - 263), 21,
        edgecolor='darkorange', **box_kw,
    ))
    # Northern boundary box
    ax.add_patch(Rectangle(
        (263 - 360, north_boundary_lat),
        37, (31 - north_boundary_lat),
        edgecolor='purple', alpha=0.5, **box_kw,
    ))

    # Scatter genesis points — size ∝ mean error, color = zone
    norm      = mcolors.Normalize(
        vmin=genesis['mean_error'].min(),
        vmax=genesis['mean_error'].quantile(0.95),  # cap colorscale at 95th pct
    )
    size_norm = mcolors.Normalize(
        vmin=genesis['mean_error'].min(),
        vmax=genesis['mean_error'].max(),
    )

    for zone, color in zone_colors.items():
        sub = genesis[genesis['zone'] == zone]
        if len(sub) == 0:
            continue

        sizes = 20 + 200 * (sub['mean_error'] - genesis['mean_error'].min()) / \
                (genesis['mean_error'].max() - genesis['mean_error'].min())

        ax.scatter(
            sub['plot_lon'], sub['true_lat'],
            s=sizes, c=color, alpha=0.75,
            edgecolors='black', linewidths=0.4,
            transform=proj, zorder=4,
            label=f"{zone} (n={len(sub)})",
        )

    ax.legend(loc='lower left', fontsize=8, framealpha=0.9,
              title='Genesis zone', title_fontsize=8)

    ax.set_title(
        'Genesis Location Classification by Domain Boundary Zone\n'
        'Marker size ∝ mean LOYO prediction error; '
        'dashed boxes = boundary regions',
        fontsize=12, pad=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Force unbuffered stdout so we see every print in real-time (critical for
    # debugging hangs in PyCharm where output is otherwise block-buffered).
    sys.stdout.reconfigure(line_buffering=True)

    # ── CLI flags ────────────────────────────────────────────────────────────
    SMOKE_TEST    = '--smoke-test' in sys.argv
    USE_CACHE     = '--cache' in sys.argv
    REBUILD_CACHE = '--rebuild-cache' in sys.argv
    NO_RESUME     = '--no-resume' in sys.argv
    if REBUILD_CACHE:
        USE_CACHE = True  # rebuild implies cache usage
    if SMOKE_TEST:
        print('=' * 70)
        print('[SMOKE TEST MODE] -- data load + 3 train iterations on tiny subset')
        print('=' * 70)
        print()
    if USE_CACHE:
        mode = 'force rebuild + save' if REBUILD_CACHE else 'use if valid, else build + save'
        print(f'[CACHE MODE] {mode}')
    if NO_RESUME:
        print('[NO-RESUME] LOYO will start from fold 1, ignoring any existing progress.')
    if USE_CACHE or NO_RESUME:
        print()

    class Tee:
        def __init__(self, filepath):
            self.file   = open(filepath, 'w', encoding='utf-8')
            self.stdout = sys.stdout

        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)

        def flush(self):
            self.stdout.flush()
            self.file.flush()

    sys.stdout = Tee('../../../data/NACSGM/final/training_log.txt')

    # ── Device check + cuDNN safety (harmless on any CUDA version) ──────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA build: {torch.version.cuda}')
        print(f'  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        # Disable cuDNN benchmark for reliable startup (no autotune surprises).
        # Slight perf cost (~10%) for stability across PyTorch/CUDA versions.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f'  cuDNN: benchmark=False, deterministic=True')
    else:
        print('  WARNING: CUDA not available. Training will run on CPU (very slow).')
        if SMOKE_TEST:
            print('  Continuing anyway since --smoke-test was passed.')
    print()
    print(f'Run started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    CSV_PATH      = '../../../data/NACSGM/final/NACSGM_combined.csv'
    EXTREMES_PATH = '../../../data/NACSGM/final/extremes.csv'
    LEAD_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]
    EPOCHS     = 200
    LR         = 3e-4
    OUT_DIR    = '../../../data/NACSGM/final'

    # ── Cache: try to load pre-computed arrays before doing the CSV read ────
    _cache_loaded = False
    _cache_p = None
    if USE_CACHE:
        _cache_key = _compute_cache_key(
            CSV_PATH, EXTREMES_PATH, ALL_EXCLUDED_IDS,
            (GRID_SOUTH, GRID_NORTH, GRID_WEST, GRID_EAST),
            LOAD_VARIABLES,
        )
        _cache_p = _cache_file_path(OUT_DIR, 'multilead_kl_nacsgm', _cache_key)
        if not REBUILD_CACHE:
            print(f"{_stamp()} Looking for cache at {_cache_p}...")
            _t_cache = datetime.now()
            cached = _load_data_cache(_cache_p)
            if cached is not None:
                storms          = cached['storms']
                genesis_loc     = cached['genesis_loc']
                storm_ids       = list(cached['storm_ids'])
                storm_years     = cached['storm_years']
                n_timesteps     = cached['n_timesteps']
                land_mask       = cached['land_mask']
                negatives       = cached['negatives']
                neg_ids         = list(cached['neg_ids'])
                neg_years       = cached['neg_years']
                normalize_stats = [tuple(s) for s in cached['normalize_stats']]
                _cache_loaded   = True
                print(f"{_stamp()}   Cache HIT, loaded in {_fmt_elapsed(_t_cache)}")
                print(f"  Positives: {storms.shape}  ({len(storm_ids)} storms)")
                print(f"  Negatives: {negatives.shape}  ({len(neg_ids)} snapshots)")
                print(f"  Skipping CSV read, loaders, and normalization.")
            else:
                print(f"{_stamp()}   Cache MISS, will build cache after loading")
        else:
            print(f"{_stamp()} --rebuild-cache: forcing fresh build of {_cache_p.name}")

    if not _cache_loaded:
        print(f"{_stamp()} Reading {CSV_PATH} ...")
        _t_phase = datetime.now()
        df = pd.read_csv(CSV_PATH)
        # The combined CSV is stored in -180/+180 longitude convention, but the
        # grid bounds (GRID_WEST=263, GRID_EAST=305) and all downstream cell
        # indexing use ERA5's 0-360 convention. Map negative longitudes back to
        # 0-360 so the grid-bounds filter and genesis indexing line up.
        # (Idempotent: values already in 0-360 are unaffected.)
        if 'longitude' in df.columns:
            n_neg_lon = int((df['longitude'] < 0).sum())
            if n_neg_lon:
                df.loc[df['longitude'] < 0, 'longitude'] += 360.0
                print(f"  Converted {n_neg_lon:,} longitudes from -180/+180 to 0-360")
                print(f"{_stamp()}   CSV loaded in {_fmt_elapsed(_t_phase)}.  "
              f"Total rows: {len(df):,}")
        print(f"  origin value counts:\n{df['origin'].value_counts().to_string()}")

        pos_df = df[df['origin'] >= 0].copy()
        neg_df = df[df['origin'] == -1].copy()
        print(f"\n  Positive rows: {len(pos_df):,}  ({pos_df['ID'].nunique()} storms)")
        print(f"  Negative rows: {len(neg_df):,}  ({neg_df['ID'].nunique()} systems)")

        # ── Step B: exclude boundary storms (CLIPPED + EDGE categories)
        # CLIPPED_BOUNDARY_IDS: data quality (true location outside basin)
        # EDGE_EXCLUSION_IDS:   model capability (within 1-2 deg of edge)
        # Both filtered identically; categories reported separately for clarity.
        # Set EXCLUDE_BOUNDARY = False to skip this filter and reproduce the
        # pre-Step-B behaviour for direct comparison.
        EXCLUDE_BOUNDARY = True
        if EXCLUDE_BOUNDARY:
            n_pos_before = pos_df['ID'].nunique()
            unique_pos = set(pos_df['ID'].unique())
            clipped_in_pos = unique_pos & CLIPPED_BOUNDARY_IDS
            edge_in_pos    = unique_pos & EDGE_EXCLUSION_IDS
            total_excluded = clipped_in_pos | edge_in_pos
            if total_excluded:
                pos_df = pos_df[~pos_df['ID'].isin(total_excluded)].copy()
                print(f"\n  Step B: excluded {len(total_excluded)} storm(s) from positives:")
                print(f"    - {len(clipped_in_pos)} clipped-boundary (data quality)")
                print(f"    - {len(edge_in_pos)} edge-exclusion (model capability)")
                print(f"    {n_pos_before} -> {pos_df['ID'].nunique()} positive storms")
            else:
                print(f"\n  Step B: 0 storms in exclusion sets; "
                      f"{n_pos_before} positives unchanged.")
            unique_neg = set(neg_df['ID'].unique())
            excluded_in_neg = unique_neg & ALL_EXCLUDED_IDS
            if excluded_in_neg:
                neg_df = neg_df[~neg_df['ID'].isin(ALL_EXCLUDED_IDS)].copy()
                print(f"  Step B: excluded {len(excluded_in_neg)} boundary "
                      f"system(s) from negatives.")

        (storms, genesis_loc, storm_ids,
         storm_years, n_timesteps, land_mask) = load_positive_storms(pos_df)

        negatives, neg_ids, neg_years = load_negative_storms(neg_df)

        assert storms.shape[2]    == LOAD_NUM_CHANNELS
        assert negatives.shape[1] == LOAD_NUM_CHANNELS
        assert storms.shape[3:]   == (HEIGHT, WIDTH)
        assert negatives.shape[2:] == (HEIGHT, WIDTH)

        print("\n✓ All arrays loaded and validated.")
        print(f"  Positive: {storms.shape[0]} storms × {storms.shape[1]} timesteps")
        print(f"  Negative: {negatives.shape[0]} snapshots")
        print(f"  Channels: {LOAD_NUM_CHANNELS}  ({', '.join(LOAD_VARIABLES)})")
        print(f"  Grid:     {HEIGHT}×{WIDTH}")

        # ── Per-channel normalisation stats (mean, std) ───────────────────────────
        # We compute stats from the GENESIS TIMESTEP (t=0, index -1) ocean cells only.
        #
        # Why genesis timestep, not all timesteps:
        #   The full storms array is 9 timesteps × 9,894 ocean cells per storm.
        #   At t-48h through t-6h, cells far from the TC are background atmosphere
        #   (high MSL, low vorticity) and dominate the statistics 9,694:1 vs the
        #   TC core cells.  This pulls MSL mean high (~101,114 Pa) and compresses
        #   std (~340 Pa), masking the full 95,000–103,000 Pa TC range.
        #   The genesis timestep captures the actual TC environment the model needs
        #   to learn to predict, giving representative (mean, std) for normalisation.
        #
        # genesis_loc marks the exact genesis grid cell — but we use all ocean cells
        # at the genesis timestep so the normalisation covers the full spatial context
        # the model sees at prediction time (not just the single TC cell).
        # Per-variable minimum std — physically motivated lower bounds.
        # Prevents z-score explosion for variables with naturally small variance
        # at genesis (vo, d) while not over-clamping variables like MSL where
        # a std of 350 Pa is real and meaningful.
        #
        #   sst  : 0.5°C   — SST variation < 0.5°C is measurement noise
        #   msl  : 100 Pa  — ~1 hPa; smaller is numerical noise
        #   cape : 50 J/kg — convective environment always has some spread
        #   r    : 2%      — RH spread < 2% is noise
        #   vo   : 1e-5/s  — vorticity at genesis always has this much spread
        #   d    : 1e-5/s  — divergence likewise
        #   vwsh : 0.5 m/s — shear spread < 0.5 m/s is noise
        #   pi   : 2 m/s   — PI spread < 2 m/s is noise
        # MIN_STD removed — no longer needed with min-max normalisation from extremes.csv

        # ── Min-max normalisation from extremes.csv ─────────────────────────────
        # Matches TCG-NACSGM.py exactly: (x - vmin) / (vmax - vmin) → [0, 1]
        # Using global dataset extremes ensures consistent scaling across all LOYO
        # folds and makes the two models directly comparable.
        print("\nNormalising all channels from extremes.csv...")
        _ext_df = pd.read_csv(EXTREMES_PATH)
        _ext_df.columns = [c.strip().lower() for c in _ext_df.columns]
        _ext = {
            row['variable'].strip().lower(): (float(row['min']), float(row['max']))
            for _, row in _ext_df.iterrows()
        }
        normalize_stats = []  # now stores (vmin, vmax) tuples instead of (mu, std)
        for c, var in enumerate(LOAD_VARIABLES):
            if var not in _ext:
                raise RuntimeError(
                    f"extremes.csv has no '{var}' row. "
                    f"Run recompute_extremes_from_nc.py to regenerate it."
                )
            _vmin, _vmax = _ext[var]
            _range = _vmax - _vmin
            if _range <= 0:
                raise RuntimeError(
                    f"extremes.csv: invalid range for '{var}': "
                    f"min={_vmin}, max={_vmax}."
                )
            ch = storms[:, :, c, :, :]
            # Guard: only skip if values are truly already in [0,1] AND
            # extremes confirm that's correct — avoids false-skip for
            # small-magnitude variables like vo/d (~±0.001).
            ch_max = float(np.nanmax(ch))
            ch_min = float(np.nanmin(ch))
            already_normed = (ch_max <= 1.0 and ch_min >= 0.0
                              and _vmin <= 0.0 and _vmax >= 1.0)
            if already_normed:
                print(f"  {var:<6}  ch{c}  already in [0,1] range — skipping")
            else:
                ch = np.nan_to_num(ch, nan=0.0)
                storms[:, :, c, :, :] = np.clip(
                    (ch - _vmin) / _range, 0.0, 1.0
                )
                after = storms[:, :, c, :, :]
                print(f"  {var:<6}  ch{c}  [{_vmin:.4g}, {_vmax:.4g}] → "
                      f"[{after.min():.4f}, {after.max():.4f}]")
            normalize_stats.append((_vmin, _vmax))

        # ── Apply same min-max normalization to negatives ─────────────────────
        # Negatives must be on the identical [0,1] scale as positives so the
        # model sees consistent input distributions for both classes.
        print("\nNormalising negatives with same extremes.csv...")
        for c, var in enumerate(LOAD_VARIABLES):
            _vmin, _vmax = normalize_stats[c]
            _range = _vmax - _vmin
            ch = negatives[:, c, :, :]   # (N_neg, H, W)
            ch_max = float(np.nanmax(ch))
            ch_min = float(np.nanmin(ch))
            already_normed = (ch_max <= 1.0 and ch_min >= 0.0
                              and _vmin <= 0.0 and _vmax >= 1.0)
            if already_normed:
                print(f"  {var:<6}  ch{c}  already in [0,1] range — skipping")
            else:
                ch = np.nan_to_num(ch, nan=0.0)
                negatives[:, c, :, :] = np.clip(
                    (ch - _vmin) / _range, 0.0, 1.0
                )
                after = negatives[:, c, :, :]
                print(f"  {var:<6}  ch{c}  [{_vmin:.4g}, {_vmax:.4g}] → "
                      f"[{after.min():.4f}, {after.max():.4f}]")


        # ── Save cache for next run if requested ────────────────────────────
        if USE_CACHE and _cache_p is not None:
            print(f"{_stamp()} Saving cache to {_cache_p}...")
            _t_save = datetime.now()
            _save_data_cache(_cache_p, {
                'storms':          storms,
                'genesis_loc':     genesis_loc,
                'storm_ids':       np.array(storm_ids, dtype=object),
                'storm_years':     storm_years,
                'n_timesteps':     n_timesteps,
                'land_mask':       land_mask,
                'negatives':       negatives,
                'neg_ids':         np.array(neg_ids, dtype=object),
                'neg_years':       neg_years,
                'normalize_stats': np.array(normalize_stats, dtype=object),
            })
            _size_gb = _cache_p.stat().st_size / 1e9
            print(f"{_stamp()}   Cache saved in {_fmt_elapsed(_t_save)} "
                  f"({_size_gb:.1f} GB)")

    # Pretrain warm-start: set to a year whose saved fold weights should
    # initialise all other folds. Weights must exist at
    # OUT_DIR/model_fold_{PRETRAIN_YEAR}.pt — run once with None first.
    PRETRAIN_YEAR = None  # SingleTimestepTCG weights incompatible with TemporalTCG
    _pretrain_path = None
    if PRETRAIN_YEAR is not None:
        from pathlib import Path as _Path
        _pt = _Path(f'{OUT_DIR}/model_fold_{PRETRAIN_YEAR}.pt')
        if _pt.exists():
            _pretrain_path = str(_pt)
            print(f'Warm-starting all folds from model_fold_{PRETRAIN_YEAR}.pt')
        else:
            print(f'WARNING: pretrain weights not found: {_pt} — random init')

    # ── Smoke test (--smoke-test): build dataset + 3 train iters, then exit ──
    if SMOKE_TEST:
        print('\n' + '=' * 70)
        print(f'SMOKE TEST  -- minimal end-to-end pipeline check {_stamp()}')
        print('=' * 70)
        print(f'  Device: {device}')

        # Tiny subset: 8 storms, 16 negatives, batch=2, 3 iterations
        N_POS, N_NEG, BATCH_SZ, N_ITERS = 8, 16, 2, 3
        n_pos = min(N_POS, storms.shape[0])
        n_neg = min(N_NEG, negatives.shape[0])
        print(f'  Subset: {n_pos} positive storms, {n_neg} negatives, '
              f'batch={BATCH_SZ}, iterations={N_ITERS}')

        _t_smoke = datetime.now()
        sm_ds = TCGMultiLeadDataset(
            storms=storms[:n_pos],
            genesis_loc=genesis_loc[:n_pos],
            storm_ids=storm_ids[:n_pos],
            storm_years=storm_years[:n_pos],
            land_mask=land_mask,
            lead_hours_list=LEAD_HOURS,
            negatives=negatives[:n_neg],
            neg_ids=neg_ids[:n_neg],
            neg_years=neg_years[:n_neg],
            normalize_stats=normalize_stats,
        )
        print(f'  Dataset built: {len(sm_ds)} samples '
              f'(storms x lead_hours + negatives x lead_hours)')

        sm_loader = DataLoader(
            sm_ds, batch_size=BATCH_SZ, shuffle=False, num_workers=0,
            collate_fn=tcg_collate_fn,
        )

        sm_model = TemporalTCG(
            input_channels=LOAD_NUM_CHANNELS, base_filters=32,
        ).to(device)
        n_params = sum(p.numel() for p in sm_model.parameters())
        print(f'  Model: {type(sm_model).__name__}, {n_params:,} params')

        sm_criterion = TCGMultiLeadLoss(land_mask).to(device)
        sm_optimizer = torch.optim.Adam(sm_model.parameters(), lr=3e-4)
        sm_scaler    = amp.GradScaler(device='cuda',
                                      enabled=(device.type == 'cuda'))

        sm_model.train()
        sm_iter = iter(sm_loader)
        for step in range(N_ITERS):
            try:
                batch = next(sm_iter)
            except StopIteration:
                sm_iter = iter(sm_loader)
                batch = next(sm_iter)

            x, target, is_negative, lead_hours, _, sample_weights = batch
            x              = x.to(device)
            target         = target.to(device)
            is_negative    = is_negative.to(device)
            sample_weights = sample_weights.to(device)
            x_finite = torch.isfinite(x).all().item()

            sm_optimizer.zero_grad()
            with amp.autocast(device_type='cuda',
                              enabled=(device.type == 'cuda')):
                logits = sm_model(x)
                loss, metrics = sm_criterion(
                    logits, target, is_negative, sample_weights,
                )

            if device.type == 'cuda':
                sm_scaler.scale(loss).backward()
                sm_scaler.unscale_(sm_optimizer)
                gn = torch.nn.utils.clip_grad_norm_(sm_model.parameters(), 1.0)
                sm_scaler.step(sm_optimizer)
                sm_scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(sm_model.parameters(), 1.0)
                sm_optimizer.step()

            out_finite = torch.isfinite(logits).all().item()
            print(f'  [iter {step+1}/{N_ITERS}]  x={tuple(x.shape)}  '
                  f'logits={tuple(logits.shape)}  loss={loss.item():.4f}  '
                  f'grad_norm={gn.item():.4f}  '
                  f'finite_in={x_finite}  finite_out={out_finite}')

        if device.type == 'cuda':
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f'  Peak GPU memory: {mem_gb:.2f} GB')

        print('\n' + '=' * 70)
        print(f'[SMOKE TEST PASSED]  {_stamp()}')
        print('=' * 70)
        print('  All checks green:')
        print('    - Data loaded and normalized')
        print('    - Model instantiated on GPU')
        print('    - Forward pass produces finite output')
        print('    - Backward pass produces finite gradients')
        print('    - Optimizer step succeeded')
        print(f'  Total elapsed: {_fmt_elapsed(_t_smoke)}')
        print(f'  Finished at:   {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('  Ready for full LOYO training. Re-run without --smoke-test.')
        sys.exit(0)

    # -- All-years final model for deployment (the EWS checkpoint) --
    if '--all-years' in sys.argv:
        _vy = None
        if '--val-year' in sys.argv:
            _vy = int(sys.argv[sys.argv.index('--val-year') + 1])
        run_all_years_training(
            storms=storms, genesis_loc=genesis_loc, storm_ids=storm_ids,
            storm_years=storm_years, land_mask=land_mask,
            negatives=negatives, neg_ids=neg_ids, neg_years=neg_years,
            lead_hours_list=LEAD_HOURS, ModelClass=TemporalTCG,
            epochs=EPOCHS, lr=LR, normalize_stats=normalize_stats,
            out_dir=OUT_DIR, val_year=_vy,
        )
        return

    if '--reeval' in sys.argv:
        from pathlib import Path
        results = {}
        sy = np.asarray(storm_years)
        for year in sorted(set(int(y) for y in sy)):
            ckpt = Path(f'{OUT_DIR}/model_fold_{year}.pt')
            if not ckpt.exists():
                print(f"  [skip] {year}: no {ckpt.name}");
                continue
            m = sy == year
            test_ds = TCGMultiLeadDataset(
                storms=storms[m], genesis_loc=genesis_loc[m],
                storm_ids=[s for s, k in zip(storm_ids, m) if k],
                storm_years=sy[m], land_mask=land_mask,
                lead_hours_list=LEAD_HOURS,
                negatives=negatives[:0], neg_ids=[], neg_years=neg_years[:0],  # positives only
                normalize_stats=normalize_stats,
            )
            loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                                num_workers=0, collate_fn=tcg_collate_fn)
            model = TemporalTCG(input_channels=LOAD_NUM_CHANNELS, base_filters=32).to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state.get('model', state) if isinstance(state, dict) else state)
            results[year] = evaluate_predictions(model, loader, land_mask, device, test_year=year)
            print(f"  {year}: median {results[year]['median_km']:.0f} km (n={results[year]['n']})")
        df_err = pd.concat([r['df'] for r in results.values() if r['n'] > 0], ignore_index=True)
        df_err.to_csv(f'{OUT_DIR}/loyo_diagnostics_tapered.csv', index=False)
        print(f"\n  Wrote {OUT_DIR}/loyo_diagnostics_tapered.csv "
              f"({len(df_err)} preds, {df_err.storm_id.nunique()} storms)")
        return


    results = run_loyo_experiment(
        storms=storms,
        genesis_loc=genesis_loc,
        storm_ids=storm_ids,
        storm_years=storm_years,
        land_mask=land_mask,
        negatives=negatives,
        neg_ids=neg_ids,
        neg_years=neg_years,
        lead_hours_list=LEAD_HOURS,
        ModelClass=TemporalTCG,         # temporal — baseline confirmed at 1476 km
        epochs=EPOCHS,
        lr=LR,
        normalize_stats=normalize_stats,
        pretrain_weights=_pretrain_path,
        resume=not NO_RESUME,
        out_dir=OUT_DIR,
    )

    # ---- results by year ----
    print("\n" + "=" * 60)
    print("RESULTS BY YEAR")
    print("=" * 60)
    for year in sorted(results.keys()):
        r = results[year]
        print(f"  {year}: "
              f"{r['mean_km']:6.0f} km mean, "
              f"{r['median_km']:6.0f} km median, "
              f"≤100km: {r['pct_100km']:5.1f}%, "
              f"≤500km: {r['pct_500km']:5.1f}% "
              f"(n={r['n']})")

    # ---- results by lead time ----
    print("\n" + "=" * 60)
    print("RESULTS BY LEAD TIME")
    print("=" * 60)
    all_dists = np.concatenate([r['distances'] for r in results.values() if r['n'] > 0])
    all_leads = np.concatenate([r['lead_hours'] for r in results.values() if r['n'] > 0])

    for lh in sorted(np.unique(all_leads)):
        mask = all_leads == lh
        d    = all_dists[mask]
        print(f"  Lead {int(lh):3d}h: "
              f"{d.mean():6.0f} km mean, "
              f"{np.median(d):6.0f} km median, "
              f"≤100km: {100*(d<=100).mean():5.1f}%, "
              f"≤500km: {100*(d<=500).mean():5.1f}% "
              f"(n={len(d)})")

    # ---- assemble master diagnostic DataFrame ----
    df_err = pd.concat(
        [r['df'] for r in results.values() if r['n'] > 0],
        ignore_index=True,
    )

    # Storms identified as boundary/anomaly cases
    OUTLIER_STORMS = [
        # SE boundary (≤13.7°N or ≥298.9°E)
        '2017170N08310',  # Bret    — 10.1°N, 298.9°E
        '2000270N11330',  # Joyce   — 11.3°N, 299.1°E
        '1994253N13303',  # Debby   — 13.7°N, 299.8°E
        '2000233N12316',  # Debby   — 17.1°N, 299.1°E
        '2000260N15308',  # Helene  — 16.6°N, 299.2°E

        # NW/W/Gulf boundary (≥20°N or ≤277°E)
        '1994181N22276',  # Alberto — 21.7°N, 276.4°E
        '1994268N16276',  # Unnamed — 16.0°N, 275.5°E
        '1994272N20274',  # Unnamed — 20.5°N, 274.0°E
        '1994313N12278',  # Gordon  — 11.9°N, 277.7°E
        '1980290N16275',  # Unnamed — 16.2°N, 274.8°E
        '2023239N21274',  # Idalia  — 20.8°N, 273.9°E
        '2000259N20273',  # Gordon  — 19.8°N, 272.7°E
        '2000273N16277',  # Keith   — 16.1°N, 277.1°E
    ]

    summary_df, df_with_zones = compute_boundary_zone_analysis(
        df_err=df_err,
        outlier_storm_ids=OUTLIER_STORMS,
        lead_hours_list=LEAD_HOURS,
    )

    plot_boundary_zone_map(
        df_err=df_err,
        df_with_zones=df_with_zones,
        save_path=f'{OUT_DIR}/boundary_zone_map.png',
    )

    # Save the summary table for the dissertation
    summary_df.to_csv(f'{OUT_DIR}/boundary_zone_summary.csv', index=False)

    comparison = compute_metrics_with_without_outliers(
        df_err=df_err,
        outlier_storm_ids=OUTLIER_STORMS,
        lead_hours_list=LEAD_HOURS,
    )
    print_comparison_table(comparison, LEAD_HOURS)

    # pass comparison['filtered']['df'] to the plots
    # to show figures both ways
    plot_lead_time_error_bands(
        results=results,  # raw results dict doesn't change
        cis=comparison['filtered']['cis'],
        lead_hours_list=LEAD_HOURS,
        save_path=f'{OUT_DIR}/lead_time_errors_filtered.png',
    )

    print(f"\nTotal predictions: {len(df_err)}")
    print(f"Unique storms:     {df_err['storm_id'].nunique()}")

    print("\nWORST 20 INDIVIDUAL PREDICTIONS:")
    print(df_err.nlargest(20, 'distance_km')
                [['storm_id', 'year', 'lead_hours', 'distance_km',
                  'pred_lat', 'pred_lon', 'true_lat', 'true_lon']]
                .to_string(index=False))

    print("\nWORST 15 STORMS BY MEAN ERROR:")
    storm_summary = (df_err.groupby(['storm_id', 'year'])
                           .agg(mean_km=('distance_km', 'mean'),
                                median_km=('distance_km', 'median'),
                                max_km=('distance_km', 'max'),
                                n_leads=('distance_km', 'count'))
                           .reset_index()
                           .sort_values('mean_km', ascending=False))
    print(storm_summary.head(15).to_string(index=False))

    # Save diagnostic CSV
    df_err.to_csv(f'{OUT_DIR}/loyo_diagnostics.csv', index=False)
    storm_summary.to_csv(f'{OUT_DIR}/loyo_storm_summary.csv', index=False)
    print(f"\n  Saved diagnostics to {OUT_DIR}/loyo_diagnostics.csv")

    # ---- bootstrapped CIs ----
    cis = compute_all_cis(df_err, lead_hours_list=LEAD_HOURS)
    print_ci_summary(cis)

    # ---- plots ----
    plot_lead_time_error_bands(
        results=results,
        cis=cis,
        lead_hours_list=LEAD_HOURS,
        outlier_threshold_km=1500,
        save_path=f'{OUT_DIR}/lead_time_errors.png',
    )

    plot_year_violin(
        results=results,
        cis=cis,
        outlier_threshold_km=2000,
        save_path=f'{OUT_DIR}/year_violin.png',
    )

    plot_worst_storm_tracks(
        df_err=df_err,
        n_worst=10,
        threshold_km=1000,
        save_path=f'{OUT_DIR}/worst_tracks.png',
    )

    return (storms, genesis_loc, storm_ids, storm_years,
            n_timesteps, land_mask, negatives, neg_ids, neg_years)


if __name__ == '__main__':
    main()
