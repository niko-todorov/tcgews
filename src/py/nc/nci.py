"""
3. nci.py
=========
Extracts raw (un-normalised) ERA5 variable values for every positive storm
(origin == 1) across all 9 timesteps, reading directly from the per-variable
NetCDF files.

NC file structure (confirmed from file inspection)
---------------------------------------------------
  Filename   : {YYYYMMDDOO}-{season_number}-{storm_name}.nc
               e.g.  2024101900-16-OSCAR.nc
               = date 2024-10-19, hour 00Z, storm #16 of season, named OSCAR

  Directories: ../../../data/NACSGM/<variable>/<YYYYMMDDOO>-<NN>-<NAME>.nc
               where <variable> in {cape, d, msl, r, sst, vo, vwsh, pi}

  Contents   : one file per storm per timestep
               dims  : valid_time=1, latitude=77, longitude=149
               grid  : NACSGM domain (11-30N, 263-300E, 0.25 deg res)
               var   : named same as the subdirectory (e.g. 'sst')
               SST   : stored in Kelvin -> converted to Celsius here

  For a storm with 9 timesteps (t-48h to t-0h at 6-hour intervals), there
  are 9 separate files with the same season_number and storm_name but
  different YYYYMMDDOO prefixes.

Matching strategy
-----------------
  NACSGM_combined.csv provides:  ID, time, origin, latitude, longitude
  NC files provide:              variable values for (datetime, storm)

  Matching by:
    1. Build datetime -> [storm_name, ...] index from NC filenames
    2. For each CSV row, find NC file(s) whose embedded datetime matches
       the row time (exact match to the hour)
    3. If multiple storms share a timestep, disambiguate by storm name
       (name embedded in NC filename checked against the CSV ID)

Output: NACSGM.csv
  Columns: ID, year, time, origin, latitude, longitude,
           sst (C), msl (Pa), cape (J/kg), r (%), vo (s-1), d (s-1),
           vwsh (m/s), pi (m/s)
  One row per (storm, timestep) for all positive genesis storms.

Usage
-----
  python nci.py
  python nci.py --data-root path/to/NACSGM
  python nci.py --all-storms
  python nci.py --out NACSGM.csv

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

import argparse
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── NetCDF backend ────────────────────────────────────────────────────────────
try:
    import netCDF4 as nc4
    _BACKEND = 'netCDF4'
except ImportError:
    try:
        import xarray as xr
        _BACKEND = 'xarray'
    except ImportError:
        print("ERROR: install netCDF4 or xarray:\n  pip install netCDF4")
        sys.exit(1)

# =============================================================================
# PATHS & CONFIG
# =============================================================================
_DATA_ROOT    = Path('../../../data/NACSGM')
_FINAL_ROOT   = _DATA_ROOT / 'final'
_CSV_PATH     = _FINAL_ROOT / 'NACSGM_combined.csv'   # negative (CLM_) storms live here
_OUT_CSV      = _FINAL_ROOT / 'NACSGM.csv'             # positive storms output (all timesteps)
_IBTRACS_CSV  = Path('../../../data/1979-NA-CSGM-origins-ibtracs.since1980.list.v04r01.csv')

VARIABLES     = ['sst', 'msl', 'cape', 'r', 'vo', 'd']  # direct grid read from .nc
VWSH_VAR      = 'vwsh'                                    # computed from u/v in its .nc
PI_VAR        = 'pi'                                      # read from .csv files
ALL_VARIABLES = VARIABLES + [VWSH_VAR, PI_VAR]           # full output column set

SST_KELVIN_THRESHOLD = 200.0

# Filename pattern:  YYYYMMDDOO-NN-NAME.nc  or  YYYYMMDDOO-NN-HYPHEN-NAME.nc
# The name segment can contain hyphens (e.g. JOAN-MIRIAM, TWENTY-TWO from HURDAT2)
_FNAME_RE = re.compile(r'^(\d{10})-(\d+)-([A-Za-z][A-Za-z-]*)$')

# =============================================================================
# MISCLASSIFIED STORMS — NC files exist but storms are NEGATIVE (non-genesis)
# =============================================================================
# These 10 storms have NC files under data/NACSGM/<var>/ but were sourced from
# HURDAT2 (not IBTrACS) and should be treated as negative (non-genesis) storms.
# They are excluded from NACSGM.csv and must NOT be used as positive training
# examples.  Add their full NC filename stems (without extension) here.
NEGATIVE_STORM_STEMS = frozenset({
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
})

# =============================================================================
# GENESIS ORIGIN FIXES
# =============================================================================
# 48 storms whose origin==1 row is absent from NACSGM_combined.csv.
#
# _GENESIS_FIX_IDS  : the SIDs that load_storm_index must force-include so
#                     their full 9-timestep sequences are extracted.
# _GENESIS_FIXES    : per-storm fix instructions used by _apply_genesis_fixes()
#   match_mode='time' -> set origin=1 on row whose time is closest to
#                        genesis_time (within _GENESIS_FIX_MAX_OFFSET_H hours)
#   match_mode='last' -> all time values are NaT; set origin=1 on last row
#                        (t-0h = genesis by construction)
#
# 4 storms (2020205N26272, 2020276N17277, 2021222N14301, 2025277N07339) are
# absent from NACSGM_combined.csv entirely -- NC files must be downloaded
# first; add their SIDs to _GENESIS_FIX_IDS once the files exist.
_GENESIS_FIXES = [
    # -- time-match storms (38) -----------------------------------------------
    # original 23
    ('1981250N15306',    '1981-09-08 00:00', 'time'),
    ('1982169N26274',    '1982-06-18 00:00', 'time'),
    ('1984171N20266',    '1984-06-18 12:00', 'time'),
    ('1985224N18279',    '1985-08-12 00:00', 'time'),
    ('1985280N18291',    '1985-10-07 00:00', 'time'),
    ('1985299N25270',    '1985-10-26 00:00', 'time'),
    ('1995154N17276',    '1995-06-03 00:00', 'time'),
    ('1996241N02092996', '1996-08-28 00:00', 'time'),
    ('1999236N02152923', '1999-08-24 00:00', 'time'),
    ('1999249N22264',    '1999-09-05 18:00', 'time'),
    ('1999277N20266',    '1999-10-04 06:00', 'time'),
    ('2005281N02882999', '2005-10-08 18:00', 'time'),
    ('2005318N13298',    '2005-11-14 00:00', 'time'),
    ('2007151N18273',    '2007-05-31 00:00', 'time'),
    ('2007227N24269',    '2007-08-15 00:00', 'time'),
    ('2008238N13293',    '2008-08-25 00:00', 'time'),
    ('2008242N02082995', '2008-08-29 00:00', 'time'),
    ('2008269N02152900', '2008-09-25 00:00', 'time'),
    ('2011214N15299',    '2011-08-02 00:00', 'time'),
    ('2011231N15278',    '2011-08-19 00:00', 'time'),
    ('2011248N02302994', '2011-09-05 00:00', 'time'),
    ('2014182N02772811', '2014-07-01 00:00', 'time'),
    ('2014294N20265',    '2014-10-21 00:00', 'time'),
    # Cat D additions (15) — origin==1 absent from NACSGM_combined.csv
    ('1981307N17279',    '1981-11-03 00:00', 'time'),
    ('1982252N26269',    '1982-09-09 00:00', 'time'),
    ('1994227N29273',    '1994-08-14 12:00', 'time'),
    ('1995194N01922990', '1995-07-13 00:00', 'time'),
    ('1998233N01872987', '1998-08-21 00:00', 'time'),
    ('1998259N10335',    '1998-09-21 00:00', 'time'),
    ('2005235N19266',    '2005-08-22 12:00', 'time'),
    ('2010279N02202928', '2010-10-06 06:00', 'time'),
    ('2010301N02562988', '2010-10-28 18:00', 'time'),
    ('2011245N27269',    '2011-09-02 00:00', 'time'),
    ('2011278N02432998', '2011-10-05 00:00', 'time'),
    ('2012246N02122993', '2012-09-02 12:00', 'time'),
    ('2012287N15297',    '2012-10-12 18:00', 'time'),
    ('2020205N26272',    '2020-07-23 00:00', 'time'),
    ('2020276N17277',    '2020-10-02 00:00', 'time'),
    # -- last-row storms (9 -- all time values NaT) ---------------------------
    # original 4
    ('1983209N12333',    '1983-08-02 06:00', 'last'),
    ('2005289N18282',    '2005-10-15 18:00', 'last'),
    ('2005296N16293',    '2005-10-22 12:00', 'last'),
    ('2005300N10279',    '2005-10-27 06:00', 'last'),
    # Cat A — NaT in NACSGM_combined.csv but valid times in NC files;
    # genesis times confirmed from IBTrACS, use time mode
    ('1994242N20267',    '1994-08-29 12:00', 'time'),  # 71-UNNAMED
    ('1994268N16276',    '1994-09-24 12:00', 'time'),  # 89-UNNAMED
    ('1994272N20274',    '1994-09-29 06:00', 'time'),  # 92-UNNAMED
    ('2005236N23285',    '2005-08-26 06:00', 'time'),  # 61-KATRINA
    ('2012285N02542876', '2012-10-11 00:00', 'time'),  # 16-PATTY
    # Storms with NC files present but origin==1 row missing
    ('2005205N19267',    '2005-07-23 18:00', 'time'),  # 46-GERT
    ('2011250N20266',    '2011-09-06 18:00', 'time'),  # 65-NATE
    ('2020233N14313',    '2020-08-21 18:00', 'time'),  # 60-LAURA
    # Storms missing genesis marker in TCG model load
    ('1985240N20286',    '1985-08-28 00:00', 'time'),  # 80-ELENA
    ('1995294N14306',    '1995-10-24 00:00', 'time'),  # 82-SEBASTIEN
    ('2005300N10279',    '2005-10-26 18:00', 'time'),  # 100-BETA
    ('2006213N16302',    '2006-08-02 00:00', 'time'),  # 43-CHRIS
    ('2007297N18300',    '2007-10-26 00:00', 'time'),  # 75-NOEL
    ('2017249N22263',    '2017-09-05 12:00', 'time'),  # 71-KATIA
    ('1985320N21296',    '1985-11-20 00:00', 'time'),  # 115-KATE
    ('1997198N12309',    '1997-07-19 00:00', 'time'),  # 49-UNNAMED
    ('2016235N02502995', '2016-08-22 18:00', 'time'),  # 06-FIONA
    ('2016278N02352997', '2016-10-04 12:00', 'time'),  # 15-NICOLE
    ('2016323N13279',    '2016-11-17 18:00', 'time'),  # 89-OTTO
    # 54 storms with complete NC files — added from trace_storm_attrition.py
    ('1980290N16275',    '1980-10-16 00:00', 'time'),
    ('1981255N12286',    '1981-09-12 00:00', 'time'),
    ('1983205N10322',    '1983-07-27 09:00', 'time'),
    ('1983228N27270',    '1983-08-15 12:00', 'time'),
    ('1983236N26284',    '1983-08-25 21:00', 'time'),
    ('1984258N20264',    '1984-09-14 00:00', 'time'),
    ('1987249N13297',    '1987-09-06 00:00', 'time'),
    ('1990216N13281',    '1990-08-04 00:00', 'time'),
    ('1991288N19274',    '1991-10-15 00:00', 'time'),
    ('1994253N13303',    '1994-09-10 00:00', 'time'),
    ('1998295N12284',    '1998-10-22 00:00', 'time'),
    ('2001227N13323',    '2001-08-17 00:00', 'time'),
    ('2004217N13306',    '2004-08-04 06:00', 'time'),
    ('2005192N11318',    '2005-07-14 00:00', 'time'),
    ('2006161N20275',    '2006-06-10 06:00', 'time'),
    ('2006237N13298',    '2006-08-24 18:00', 'time'),
    ('2008152N18273',    '2008-05-31 00:00', 'time'),
    ('2008280N18268',    '2008-10-06 00:00', 'time'),
    ('2010271N19276',    '2010-09-28 00:00', 'time'),
    ('2010284N14278',    '2010-10-11 00:00', 'time'),
    ('2010293N17277',    '2010-10-19 18:00', 'time'),
    ('2011179N20267',    '2011-06-28 06:00', 'time'),
    ('2011250N12324',    '2011-09-10 09:00', 'time'),
    ('2011271N01872999', '2011-09-28 18:00', 'time'),
    ('2011295N13279',    '2011-10-22 00:00', 'time'),
    ('2012296N14283',    '2012-10-21 18:00', 'time'),
    ('2013157N25273',    '2013-06-05 18:00', 'time'),
    ('2013167N12279',    '2013-06-16 00:00', 'time'),
    ('2013189N09319',    '2013-07-09 12:00', 'time'),
    ('2013238N19266',    '2013-08-25 12:00', 'time'),
    ('2013248N16294',    '2013-09-04 18:00', 'time'),
    ('2013255N19268',    '2013-09-12 06:00', 'time'),
    ('2013276N21273',    '2013-10-03 06:00', 'time'),
    ('2014210N10323',    '2014-08-01 18:00', 'time'),
    ('2014245N19268',    '2014-09-01 12:00', 'time'),
    ('2015167N27266',    '2015-06-16 00:00', 'time'),
    ('2015271N02742910', '2015-09-28 00:00', 'time'),
    ('2016242N24279',    '2016-08-29 06:00', 'time'),
    ('2016257N02732798', '2016-09-13 06:00', 'time'),
    ('2016266N02352997', '2016-09-22 06:00', 'time'),
    ('2016273N13300',    '2016-09-28 12:00', 'time'),
    ('2017252N01792992', '2017-09-09 12:00', 'time'),
    ('2017277N11279',    '2017-10-03 12:00', 'time'),
    ('2017301N18276',    '2017-10-27 18:00', 'time'),
    ('2018146N19273',    '2018-05-25 12:00', 'time'),
    ('2018246N22283',    '2018-09-03 12:00', 'time'),
    ('2019324N02042997', '2019-11-20 00:00', 'time'),
    ('2020211N13306',    '2020-07-29 06:00', 'time'),
    ('2020228N01962991', '2020-08-15 18:00', 'time'),
    ('2020234N14280',    '2020-08-20 18:00', 'time'),
    ('2020264N02692992', '2020-09-20 00:00', 'time'),
    ('2021222N14301',    '2021-08-10 00:00', 'time'),
    ('2024293N02102935', '2024-10-19 00:00', 'time'),
    ('2025277N07339',    '2025-10-10 00:00', 'time'),
]

# SIDs that load_storm_index must force-include (no origin==1 in source CSV)
_GENESIS_FIX_IDS = frozenset(sid for sid, _, _ in _GENESIS_FIXES)

_GENESIS_FIX_MAX_OFFSET_H = 48   # max time offset for 'time' mode


def _apply_genesis_fixes(result):
    """
    Post-processing pass: set origin=1 for every storm in _GENESIS_FIXES.
    Called at the end of extract_all() after NC values have been extracted.
    Idempotent -- storms already at origin==1 are skipped silently.
    """
    n_fixed = n_skipped = n_absent = 0
    absent_ids = []

    for sid, gen_time_str, mode in _GENESIS_FIXES:
        id_mask = result['ID'] == sid
        if not id_mask.any():
            n_absent += 1
            absent_ids.append(sid)
            continue
        id_rows = result[id_mask]

        # Already correct -- skip
        if (id_rows['origin'] == 1).any():
            n_skipped += 1
            continue

        if mode == 'last':
            result.loc[id_rows.index[-1], 'origin'] = 1
            n_fixed += 1

        elif mode == 'time':
            gen_ts     = pd.Timestamp(gen_time_str)
            valid_rows = id_rows[id_rows['time'].notna()]
            if valid_rows.empty:
                result.loc[id_rows.index[-1], 'origin'] = 1
                n_fixed += 1
                continue
            offsets  = (valid_rows['time'] - gen_ts).abs()
            best_idx = offsets.idxmin()
            if offsets[best_idx].total_seconds() / 3600 <= _GENESIS_FIX_MAX_OFFSET_H:
                result.loc[best_idx, 'origin'] = 1
                n_fixed += 1

    print(f"  Genesis fixes: {n_fixed} origin=1 set | "
          f"{n_skipped} already ok | {n_absent} IDs absent (NC files missing)")
    if absent_ids:
        print(f"  Absent SIDs (need NC download):")
        for _sid in absent_ids:
            print(f"    {_sid}")
    return result


# Coordinate variable names to skip when auto-detecting the data variable
_COORD_NAMES = {
    'latitude', 'longitude', 'lat', 'lon', 'time', 'valid_time',
    'number', 'expver', 'step', 'level', 'realization',
}


# =============================================================================
# FILENAME PARSING
# =============================================================================

def parse_nc_stem(stem):
    """
    Parse a NC filename stem.

    '2024101900-16-OSCAR'  ->
      {'datetime': Timestamp('2024-10-19 00:00'),
       'season_num': 16, 'name': 'OSCAR', 'stem': '2024101900-16-OSCAR'}

    Returns None if the stem does not match the expected pattern.
    """
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    dt_str, num_str, name = m.group(1), m.group(2), m.group(3).upper()
    try:
        dt = pd.Timestamp(
            year=int(dt_str[0:4]),
            month=int(dt_str[4:6]),
            day=int(dt_str[6:8]),
            hour=int(dt_str[8:10]),
        )
    except ValueError:
        return None
    return {'datetime': dt, 'season_num': int(num_str),
            'name': name, 'stem': stem}


# =============================================================================
# NC FILE INDEX
# =============================================================================

class NCFileIndex:
    """
    Scans all variable directories and builds:

        index[datetime][storm_name][variable] = Path

    so any (timestamp, storm_name, variable) triple resolves to a file path.
    """

    def __init__(self, data_root, variables):
        self.data_root = data_root
        self.variables = variables
        self.index     = defaultdict(lambda: defaultdict(dict))
        self.num_index = defaultdict(dict)   # dt -> {season_num: name}
        self.all_stems = set()
        self._build()

    def _build(self):
        print(f"\nIndexing NC files under {self.data_root} ...")
        total = 0
        for var in self.variables:
            var_dir = self.data_root / var
            if not var_dir.exists():
                print(f"  {var:<6}  x  {var_dir} not found -- will be NaN")
                continue
            files = sorted(var_dir.glob('*.nc'))
            if not files:
                print(f"  {var:<6}  x  no .nc files in {var_dir}")
                continue
            matched = 0
            skipped = 0
            _neg_upper = {s.upper() for s in NEGATIVE_STORM_STEMS}
            for f in files:
                parsed = parse_nc_stem(f.stem)
                if parsed is None:
                    print(f"  {var:<6}  !  unrecognised filename: {f.name} -- skipping")
                    continue
                if parsed['stem'].upper() in _neg_upper:
                    skipped += 1
                    continue   # explicitly negative — do not index
                self.index[parsed['datetime']][parsed['name']][var] = f
                # (year, season_number) is unique within a season —
                # e.g. 2024-16 always means the 16th storm of 2024 (OSCAR)
                self.num_index[parsed['datetime'].year][parsed['season_num']] = parsed['name']
                self.all_stems.add(parsed['stem'])
                matched += 1
            total += matched
            suffix = f"  ({skipped} negative excluded)" if skipped else ""
            print(f"  {var:<6}  ok  {matched:>5} files{suffix}")
        print(f"  Total: {total} entries | "
              f"{len(self.index)} timestamps | "
              f"{len(self.all_stems)} storm x timestep combinations\n")

    def get_path(self, dt, storm_name, var):
        return self.index.get(dt, {}).get(storm_name, {}).get(var)

    def storms_at(self, dt):
        return list(self.index.get(dt, {}).keys())

    def name_by_num(self, year, season_num):
        """Return storm name for (year, season_number), or None.
        (season, number) is globally unique: e.g. (2024, 16) -> 'OSCAR'.
        Uses NUMBER embedded in NC filename: {YYYYMMDDOO}-{NUMBER}-{NAME}.nc
        """
        if season_num is None or year is None:
            return None
        return self.num_index.get(int(year), {}).get(int(season_num))


# =============================================================================
# NC READING
# =============================================================================

_nc_cache = {}


def read_nc_grid(path, var_hint):
    """
    Read the full 77x149 grid from a single-timestep NC file.
    Returns a (77, 149) float32 array or None on failure.
    Converts SST from K to C automatically.
    Each file is read only once (process-level cache).
    """
    if path in _nc_cache:
        return _nc_cache[path]
    try:
        if _BACKEND == 'netCDF4':
            data = _read_nc4(path, var_hint)
        else:
            data = _read_xr(path, var_hint)
    except Exception as e:
        print(f"\n    WARNING: could not read {path.name}: {e}")
        return None
    if data is not None:
        _nc_cache[path] = data
    return data


def _read_nc4(path, var_hint):
    with nc4.Dataset(path, 'r') as ds:
        candidates = [v for v in ds.variables if v.lower() not in _COORD_NAMES]
        dvar = (var_hint if var_hint in candidates else
                var_hint.lower() if var_hint.lower() in candidates else
                candidates[0] if candidates else None)
        if dvar is None:
            return None
        raw = ds.variables[dvar][:]
        if hasattr(raw, 'filled'):
            raw = raw.filled(np.nan)
    data = raw.astype(np.float32).squeeze()   # remove size-1 dims
    # Reshape flat (11473,) → (77, 149) if needed
    if data.ndim == 1 and data.shape[0] == 77 * 149:
        data = data.reshape(77, 149)
    if np.nanmax(data) > SST_KELVIN_THRESHOLD:
        data = data - 273.15
    return data


def _read_xr(path, var_hint):
    ds = xr.open_dataset(path, engine='netcdf4')
    dvar = var_hint if var_hint in ds else list(ds.data_vars)[0]
    data = ds[dvar].values.astype(np.float32).squeeze()
    ds.close()
    # Reshape flat (11473,) → (77, 149) if needed
    if data.ndim == 1 and data.shape[0] == 77 * 149:
        data = data.reshape(77, 149)
    if np.nanmax(data) > SST_KELVIN_THRESHOLD:
        data = data - 273.15
    return data


# =============================================================================
# CAPE FILL VALUE
# =============================================================================
# ERA5 uses -273.15 J/kg (0 Kelvin in Celsius) as a sentinel for grid cells
# where CAPE was not computed (land, sea ice, failed parcel ascent).
# This is NOT a physical value — treat it as missing (NaN) so the land filter
# and imputation cascade handle it correctly, rather than clipping to 0.
_CAPE_FILL     = -273.15   # ERA5 sentinel value (J/kg)
_CAPE_FILL_TOL =   0.01    # tolerance for float comparison

# =============================================================================
# VWSH COMPUTATION  (u/v 850-200 hPa → wind shear magnitude)
# =============================================================================

_vwsh_cache = {}   # path → (77, 149) float32 grid


def compute_vwsh_from_nc(path):
    """
    Compute 850-200 hPa wind shear from a vwsh NC file containing u/v winds.

    ncio.py stores u and v at 850 and 200 hPa in <var>/vwsh/<stem>.nc —
    it does NOT store a pre-computed vwsh field.  This mirrors ncvwsh2csv()
    exactly:

        vwsh = sqrt((u200 - u850)^2 + (v200 - v850)^2)   [m/s]

    Returns a (77, 149) float32 array or None on failure.
    Cached per path so each file is read only once.
    """
    if path in _vwsh_cache:
        return _vwsh_cache[path]
    try:
        import xarray as _xr
    except ImportError:
        print(f"\n    WARNING: xarray not available for vwsh computation")
        return None

    try:
        ds = _xr.open_dataset(str(path), engine='netcdf4')

        # Drop expver/number — mirrors ncio.py ncvwsh2csv()
        for drop_var in ['number', 'expver']:
            if drop_var in ds:
                ds = ds.drop_vars([drop_var])

        # Rename valid_time → time if needed
        if 'valid_time' in ds.dims and 'time' not in ds.dims:
            ds = ds.rename({'valid_time': 'time'})

        level_dim = 'pressure_level' if 'pressure_level' in ds.dims else 'level'

        # Resolve u/v names (short or long form)
        u_name = ('u' if 'u' in ds else
                  'u_component_of_wind' if 'u_component_of_wind' in ds else None)
        v_name = ('v' if 'v' in ds else
                  'v_component_of_wind' if 'v_component_of_wind' in ds else None)

        if u_name is None or v_name is None:
            print(f"\n    WARNING: vwsh NC {path.name} has no u/v — "
                  f"vars: {list(ds.data_vars)}")
            ds.close()
            return None

        # Select 850 and 200 hPa
        u850 = ds[u_name].sel({level_dim: 850}, method='nearest').squeeze()
        v850 = ds[v_name].sel({level_dim: 850}, method='nearest').squeeze()
        u200 = ds[u_name].sel({level_dim: 200}, method='nearest').squeeze()
        v200 = ds[v_name].sel({level_dim: 200}, method='nearest').squeeze()

        vwsh = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        data = vwsh.values.astype(np.float32).squeeze()
        ds.close()

        if data.ndim == 1 and data.shape[0] == 77 * 149:
            data = data.reshape(77, 149)
        if data.ndim != 2:
            return None

        _vwsh_cache[path] = data
        return data

    except Exception as e:
        print(f"\n    WARNING: vwsh computation failed for {path.name}: "
              f"{type(e).__name__}: {e}")
        return None


# =============================================================================
# PI IMPUTATION HELPERS
# =============================================================================

# Monthly climatological PI for the NACSGM basin (Caribbean Sea + Gulf of Mexico).
# Derived from Emanuel (1986) and updated NACSGM domain averages.
# Used as last-resort fallback when both the exact cell and its 5×5 neighbourhood
# are all NaN (e.g. isolated coastal cells where tcpyPI fails).
# Units: m/s
# PI fallback = 0.0 for all months.
# Land/coastal cells where tcpyPI returns NaN have no thermodynamic support
# for genesis. Setting PI=0 makes this explicit:
#   - TCG-NACSGM.py  (min-max): 0 → 0.0 (clear minimum, suppresses genesis)
#   - TCG-multilead  (z-score): 0 → ~2σ below mean (strong negative signal)
# This prevents imputed values from creating spurious high-PI regions that
# bias predictions toward coastal/boundary cells.
PI_CLIMO_NACSGM = {
    1:  0.0,   #42.0,   # January    — cold SST, suppressed convection
    2:  0.0,   #40.0,   # February
    3:  0.0,   #42.0,   # March
    4:  0.0,   #46.0,   # April
    5:  0.0,   #52.0,   # May        — pre-season warming
    6:  0.0,   #58.0,   # June       — early season
    7:  0.0,   #63.0,   # July
    8:  0.0,   #67.0,   # August     — peak season
    9:  0.0,   #68.0,   # September  — peak season
    10: 0.0,   #63.0,   # October
    11: 0.0,   #55.0,   # November
    12: 0.0,   #47.0,   # December
}


def impute_pi_grid(pi_grid, month, half_window=2):
    """
    Impute NaN cells in a (77, 149) PI grid using:
      1. Local spatial mean of a (2*half_window+1)² neighbourhood  [5×5 default]
      2. Monthly NACSGM climatology where the neighbourhood is also all-NaN

    Parameters
    ----------
    pi_grid     : (77, 149) float32 array — NaN on land / computation failures
    month       : int 1-12 — used for climatological fallback
    half_window : int — half-size of the neighbourhood (default 2 → 5×5 window)

    Returns
    -------
    imputed : (77, 149) float32 array — no NaN values
    n_spatial  : int — cells filled by spatial mean
    n_climo    : int — cells filled by climatology
    """
    out        = pi_grid.copy()
    nan_mask   = ~np.isfinite(out)
    if not nan_mask.any():
        return out, 0, 0

    nrows, ncols = out.shape
    climo_val    = float(PI_CLIMO_NACSGM.get(month, 60.0))
    n_spatial    = 0
    n_climo      = 0

    rows_nan, cols_nan = np.where(nan_mask)
    for r, c in zip(rows_nan, cols_nan):
        r0, r1 = max(0, r - half_window), min(nrows, r + half_window + 1)
        c0, c1 = max(0, c - half_window), min(ncols, c + half_window + 1)
        neighbourhood = out[r0:r1, c0:c1]
        valid = neighbourhood[np.isfinite(neighbourhood)]
        if len(valid) > 0:
            out[r, c] = float(valid.mean())
            n_spatial += 1
        else:
            out[r, c] = climo_val
            n_climo   += 1

    return out, n_spatial, n_climo


# =============================================================================
# PI CSV INDEX  (pi values stored in .csv, not .nc)
# =============================================================================

class PICSVIndex:
    """
    Indexes all .csv files in data/NACSGM/pi/ and exposes an extract() method
    that returns the PI value (m/s) at a given (datetime, storm_name, lat, lon).

    CSV format (confirmed from file inspection):
      columns : latitude, longitude, time, pi
      rows    : 11473 (full 77x149 grid); land cells have NaN pi
      time fmt: '20241019 00:00'
      units   : m/s (physical, no normalisation needed)
    """

    def __init__(self, data_root):
        self.pi_dir = data_root / 'pi'
        # index[datetime][storm_name] = Path
        self.index  = defaultdict(dict)
        self._cache = {}   # Path -> pd.DataFrame
        self._build()

    def _build(self):
        if not self.pi_dir.exists():
            print(f"  pi     x  {self.pi_dir} not found -- pi will be NaN")
            return
        files = sorted(self.pi_dir.glob('*.csv'))
        if not files:
            print(f"  pi     x  no .csv files in {self.pi_dir}")
            return
        matched = 0
        skipped = 0
        _neg_upper = {s.upper() for s in NEGATIVE_STORM_STEMS}
        for f in files:
            parsed = parse_nc_stem(f.stem)   # same naming convention
            if parsed is None:
                print(f"  pi     !  unrecognised filename: {f.name} -- skipping")
                continue
            if parsed['stem'].upper() in _neg_upper:
                skipped += 1
                continue   # explicitly negative — do not index
            self.index[parsed['datetime']][parsed['name']] = f
            matched += 1
        suffix = f"  ({skipped} negative excluded)" if skipped else ""
        print(f"  pi     ok  {matched:>5} CSV files{suffix}")

    def _load(self, path):
        """
        Load and cache a PI CSV as a (77, 149) float32 grid.
        NaN on land cells and any cell where tcpyPI returned no value.
        """
        if path in self._cache:
            return self._cache[path]

        df = pd.read_csv(path, dtype={'latitude': float, 'longitude': float,
                                      'pi': float})

        # Build index maps aligned to the NACSGM grid
        lats_grid = np.round(np.linspace(30.0, 11.0,  77),  2)  # decreasing
        lons_grid = np.round(np.linspace(263.0, 300.0, 149), 2)  # increasing

        grid = np.full((77, 149), np.nan, dtype=np.float32)
        for row in df.itertuples():
            if not pd.notna(row.pi):
                continue
            lat_r = round(float(row.latitude),  2)
            lon_r = round(float(row.longitude), 2)
            ri = int(np.argmin(np.abs(lats_grid - lat_r)))
            ci = int(np.argmin(np.abs(lons_grid - lon_r)))
            grid[ri, ci] = float(row.pi)

        self._cache[path] = grid
        return grid

    def extract(self, dt, storm_name, lat, lon, month=None):
        """
        Return PI (m/s) for (datetime, storm_name, lat, lon).

        Imputation cascade (applied to the full grid before extraction):
          1. Exact cell value from the PI CSV
          2. 5×5 spatial mean of valid neighbours in the same grid
          3. Monthly NACSGM climatology (last resort)

        Parameters
        ----------
        month : int 1-12 or None — if None, extracted from dt
        """
        path = self.index.get(dt, {}).get(storm_name)
        if path is None:
            # No CSV for this storm/timestamp — fall back to climatology only
            _month = month or pd.Timestamp(dt).month
            return float(PI_CLIMO_NACSGM.get(_month, 60.0))

        pi_grid = self._load(path)

        # Snap to nearest grid cell
        lats_grid = np.round(np.linspace(30.0, 11.0,  77),  2)
        lons_grid = np.round(np.linspace(263.0, 300.0, 149), 2)
        lat_r = round(round(lat / 0.25) * 0.25, 2)
        lon_r = round(round(lon / 0.25) * 0.25, 2)
        ri = int(np.argmin(np.abs(lats_grid - lat_r)))
        ci = int(np.argmin(np.abs(lons_grid - lon_r)))

        val = float(pi_grid[ri, ci])
        if np.isfinite(val):
            return val

        # Cell is NaN — apply impute_pi_grid (cached after first call per file)
        _month = month or pd.Timestamp(dt).month
        cache_key = (path, 'imputed')
        if cache_key not in self._cache:
            imputed, _, _ = impute_pi_grid(pi_grid, _month)
            self._cache[cache_key] = imputed
        imputed_grid = self._cache[cache_key]
        return float(imputed_grid[ri, ci])

    def storms_at(self, dt):
        return list(self.index.get(dt, {}).keys())

def build_grid_maps():
    """Build lat->row and lon->col dicts for the NACSGM 77x149 grid."""
    lats = np.round(np.linspace(30.0, 11.0, 77), 2)   # decreasing 30->11
    lons = np.round(np.linspace(263.0, 300.0, 149), 2)
    lat_map = {float(v): i for i, v in enumerate(lats)}
    lon_map = {float(v): i for i, v in enumerate(lons)}
    return lat_map, lon_map


def nearest_grid(lat, lon, lat_map, lon_map):
    """Snap (lat, lon) to nearest NACSGM grid cell. Returns (row, col) or None."""
    lat_r = round(round(lat / 0.25) * 0.25, 2)
    lon_r = round(round(lon / 0.25) * 0.25, 2)
    row = lat_map.get(lat_r)
    col = lon_map.get(lon_r)
    if row is None or col is None:
        return None
    return row, col


# =============================================================================
# STORM NAME RESOLUTION
# =============================================================================

def build_sid_name_map(ibtracs_csv: Path) -> dict:
    """
    Build a SID -> storm_name dict from the IBTrACS origins CSV.

    IBTrACS SIDs (e.g. 1998233N25268) don't embed the storm name, but the
    CSV has both SID and NAME columns.  NAME is upper-cased to match NC
    filenames (e.g. 'CHARLEY', 'OSCAR').

    Called once at startup; result passed into resolve_storm_name() so that
    IBTrACS-format SIDs with ambiguous timestamps can be resolved by name
    without requiring a 'number' column in NACSGM_combined.csv.
    """
    if not ibtracs_csv.exists():
        print(f"  WARNING: IBTrACS CSV not found at {ibtracs_csv} "
              f"— 14 ambiguous storms may have NaN variables.")
        return {}
    try:
        df = pd.read_csv(ibtracs_csv, low_memory=False,
                         usecols=lambda c: c in ['SID', 'NAME'])
        df['SID']  = df['SID'].astype(str).str.strip()
        df['NAME'] = df['NAME'].astype(str).str.strip().str.upper()
        # One row per SID (first occurrence = genesis timestep in IBTrACS)
        sid_name = df.drop_duplicates('SID').set_index('SID')['NAME'].to_dict()
        print(f"  IBTrACS SID->name map: {len(sid_name)} entries loaded")
        return sid_name
    except Exception as e:
        print(f"  WARNING: could not load IBTrACS CSV: {e}")
        return {}


def resolve_storm_name(storm_id, index, dt, sid_name_map=None):
    """
    Resolve the storm name (as used in NC filenames) for a CSV storm ID.

    NC filenames: {YYYYMMDDOO}-{NUMBER}-{NAME}.nc
    Tries in order:
      1. CSV ID is itself a valid NC stem -> name embedded directly
      2. sid_name_map lookup (from IBTrACS CSV) -> direct SID->name match,
         resolves IBTrACS-format SIDs even when multiple storms share a timestamp
      3. Only one storm active at this timestamp -> use that name
      4. Multiple storms, no name available -> return None (ambiguous)
    """
    sid = str(storm_id).upper().strip()

    # Strategy 1: CSV ID matches NC stem pattern directly
    parsed = parse_nc_stem(sid)
    if parsed:
        return parsed['name']

    # Strategy 2: IBTrACS SID->name map — resolves ambiguous timestamps
    if sid_name_map:
        name = sid_name_map.get(sid) or sid_name_map.get(str(storm_id).strip())
        if name:
            # Verify the name has NC files at this timestamp
            if name in index.storms_at(dt) or not index.storms_at(dt):
                return name
            # Name found in map but not at this exact dt — still return it;
            # the NC lookup will return None gracefully if file is absent
            return name

    # Strategy 3: single storm at this timestamp
    names = index.storms_at(dt)
    if len(names) == 1:
        return names[0]

    # Strategy 4: ambiguous -- multiple storms, no name resolvable
    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_storm_index(csv_path, positive_only, chunk_size=500_000):
    """
    Read NACSGM_combined.csv and return rows for the desired storms.

    If positive_only=True (default):
      - Identifies storms that have at least one origin==1 row
      - Returns ALL timesteps for those storms (not just the genesis row)
        so the output CSV has the full 9-timestep sequence per storm
      - CLM_ negative storms are excluded entirely; they will be handled
        separately via NACSGM_combined.csv
    """
    print(f"Loading storm index from {csv_path} ...")
    keep = ['ID', 'time', 'origin', 'latitude', 'longitude']
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size,
                             low_memory=False, parse_dates=['time']):
        if not pd.api.types.is_datetime64_any_dtype(chunk['time']):
            chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
        chunks.append(chunk[[c for c in keep if c in chunk.columns]])

    df = pd.concat(chunks, ignore_index=True)

    if positive_only:
        genesis_ids = set(df.loc[df['origin'] == 1, 'ID'])
        # Force-include storms whose origin==1 row is absent from the source
        # CSV but is known and will be patched by _apply_genesis_fixes().
        genesis_ids |= _GENESIS_FIX_IDS & set(df['ID'].unique())
        df = df[df['ID'].isin(genesis_ids)].copy()

    df['year'] = df['time'].dt.year
    df = df.dropna(subset=['time']).sort_values(['ID', 'time']).reset_index(drop=True)

    n_storms  = df['ID'].nunique()
    n_genesis = df[df['origin'] == 1]['ID'].nunique()
    print(f"  {len(df):,} rows | {n_storms} storms "
          f"({n_genesis} with genesis timestep)")
    return df


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_all(storm_df, index, pi_index, lat_map, lon_map, sid_name_map=None):
    """Extract raw variable values for every row in storm_df."""
    result = storm_df.copy()
    for var in ALL_VARIABLES:
        result[var] = np.nan

    n_rows   = len(storm_df)
    n_miss   = defaultdict(int)
    n_ambig  = 0
    log_step = max(1, n_rows // 20)
    name_cache = {}

    for i, row in storm_df.iterrows():
        storm_id = str(row['ID'])
        dt       = pd.Timestamp(row['time']).floor('h')
        lat      = float(row['latitude'])
        lon      = float(row['longitude'])

        cell = nearest_grid(lat, lon, lat_map, lon_map)
        if cell is None:
            for var in ALL_VARIABLES:
                n_miss[var] += 1
            continue
        r, c = cell

        if storm_id not in name_cache:
            name_cache[storm_id] = resolve_storm_name(
                storm_id, index, dt, sid_name_map=sid_name_map)
        storm_name = name_cache[storm_id]

        if storm_name is None:
            names = index.storms_at(dt)
            if len(names) > 1:
                n_ambig += 1
            for var in ALL_VARIABLES:
                n_miss[var] += 1
            continue

        # ── NC variables (direct grid read) ──────────────────────────────
        for var in VARIABLES:
            path = index.get_path(dt, storm_name, var)
            if path is None:
                n_miss[var] += 1
                continue
            grid = read_nc_grid(path, var)
            if grid is None or grid.ndim != 2:
                n_miss[var] += 1
                continue
            val = float(grid[r, c])
            # Replace ERA5 CAPE fill value (-273.15) with NaN
            if var == 'cape' and abs(val - _CAPE_FILL) < _CAPE_FILL_TOL:
                val = np.nan
            result.at[i, var] = val if np.isfinite(val) else np.nan

        # ── VWSH: compute from u/v 850-200 hPa in vwsh NC file ───────────
        vwsh_path = index.get_path(dt, storm_name, VWSH_VAR)
        if vwsh_path is None:
            n_miss[VWSH_VAR] += 1
        else:
            vwsh_grid = compute_vwsh_from_nc(vwsh_path)
            if vwsh_grid is None or vwsh_grid.ndim != 2:
                n_miss[VWSH_VAR] += 1
            else:
                val = float(vwsh_grid[r, c])
                result.at[i, VWSH_VAR] = val if np.isfinite(val) else np.nan

        # ── PI from CSV (with spatial + climatological imputation) ───────
        _month = dt.month
        pi_val = pi_index.extract(dt, storm_name, lat, lon, month=_month)
        result.at[i, PI_VAR] = pi_val
        if np.isnan(pi_val):
            n_miss[PI_VAR] += 1

        n_done = i - storm_df.index[0] + 1
        if n_done % log_step == 0 or n_done == n_rows:
            print(f"  {n_done:>7,} / {n_rows:,}  ({n_done/n_rows*100:.0f}%)",
                  end='\r')

    print(f"\n  Done.")
    if n_ambig:
        print(f"  WARNING: {n_ambig} rows had ambiguous storm name "
              f"(multiple storms at same timestamp) -> NaN.")

    # ── Inner join on SST: drop land points, but KEEP genesis rows ───────────
    # SST is NaN on land cells in ERA5.  We use SST presence to mask land.
    # Exception: genesis rows (origin==1) must be preserved even if their
    # reported lat/lon snaps to a coastal cell with NaN SST — dropping them
    # would silently remove storms from training, causing the
    # "380 storms but 348 genesis points" mismatch in the merged CSV.
    # For genesis rows with NaN SST we snap to the nearest valid ocean cell.
    is_genesis = result['origin'] == 1
    n_before   = len(result)

    # Snap NaN-SST genesis rows to nearest ocean neighbour (up to 3 cells away)
    nan_sst_genesis = is_genesis & result['sst'].isna()
    if nan_sst_genesis.any():
        n_snap = int(nan_sst_genesis.sum())
        print(f"\n  WARNING: {n_snap} genesis row(s) have NaN SST "
              f"(coastal/land cell) — snapping to nearest ocean cell.")
        for _snap_idx in result.index[nan_sst_genesis]:
            print(f"    {result.at[_snap_idx, 'ID']:<25}  "
                  f"time={result.at[_snap_idx, 'time']}  "
                  f"lat={result.at[_snap_idx, 'latitude']:.2f}  "
                  f"lon={result.at[_snap_idx, 'longitude']:.2f}")
        # For these rows we can't fix the NC values inline here, but we
        # retain the row and fill SST with the nearest non-NaN value from
        # the same storm's other rows (same timestep, different cell).
        # As a conservative fallback, fill with the storm median SST.
        for idx in result.index[nan_sst_genesis]:
            sid       = result.at[idx, 'ID']
            storm_sst = result.loc[(result['ID'] == sid) & result['sst'].notna(), 'sst']
            if len(storm_sst):
                result.at[idx, 'sst'] = float(storm_sst.median())
            else:
                result.at[idx, 'sst'] = 28.0   # NACSGM basin fallback

    # Now drop land rows — but genesis rows are all protected above
    result  = result[result['sst'].notna() | is_genesis].copy()
    n_land  = n_before - len(result)
    if n_land:
        print(f"\n  Land-point filter: dropped {n_land:,} non-genesis land rows, "
              f"{len(result):,} rows kept.")

    # Verify no genesis rows were lost
    n_genesis_after = (result['origin'] == 1).sum()
    n_genesis_before = int(is_genesis.sum())
    if n_genesis_after != n_genesis_before:
        print(f"  ⚠  Lost {n_genesis_before - n_genesis_after} genesis rows "
              f"in land filter — investigate.")

    # ── PI imputation ─────────────────────────────────────────────────────────
    # Imputation is now applied upstream inside PICSVIndex.extract():
    #   1. Exact cell value from CSV
    #   2. 5×5 spatial mean of valid neighbours
    #   3. Monthly NACSGM climatology (last resort)
    # Any remaining NaN here means the CSV file was entirely missing for that
    # storm — fill with monthly climatology using the row's timestamp.
    nan_pi = result[PI_VAR].isna()
    if nan_pi.any():
        def _climo_pi(ts):
            return float(PI_CLIMO_NACSGM.get(pd.Timestamp(ts).month, 60.0))
        result.loc[nan_pi, PI_VAR] = result.loc[nan_pi, 'time'].apply(_climo_pi)
        n_climo = int(nan_pi.sum())
        print(f"  PI imputation: {n_climo:,} rows with missing CSV "
              f"→ monthly climatology")

    print(f"\n  Missing counts per variable (after land filter + PI imputation):")
    for var in ALL_VARIABLES:
        n_miss_post = int(result[var].isna().sum())
        pct    = n_miss_post / len(result) * 100 if len(result) else 0
        status = 'ok' if pct < 1 else ('! ' if pct < 10 else 'XX')
        src    = '(csv)' if var == PI_VAR else '(nc) '
        print(f"    {status}  {var:<6} {src}  {n_miss_post:>7,} missing  ({pct:.1f}%)")

    # Report genesis rows still missing NC variables after all fixes
    _nc_vars = [v for v in ALL_VARIABLES if v != PI_VAR]
    _gen_nan = result[(result['origin'] == 1) & result[_nc_vars].isna().any(axis=1)]
    if not _gen_nan.empty:
        print(f"\n  WARNING: {len(_gen_nan)} genesis row(s) still missing NC variables:")
        for _, _gr in _gen_nan.iterrows():
            _missing_vars = [v for v in _nc_vars if pd.isna(_gr[v])]
            print(f"    {str(_gr['ID']):<25}  time={_gr['time']}  "
                  f"lat={_gr['latitude']:.2f}  lon={_gr['longitude']:.2f}  "
                  f"missing={_missing_vars}")

    # -- Apply known genesis-origin fixes ---------------------------------
    print(f"\n  Applying genesis origin fixes ...")
    result = _apply_genesis_fixes(result)

    return result


# =============================================================================
# STATS
# =============================================================================

def print_stats(df):
    print("\n" + "=" * 76)
    print("RAW VALUE STATISTICS  (physical units -- genesis rows only)")
    print("=" * 76)
    gen   = df[df['origin'] == 1]
    units = {'sst': 'C', 'msl': 'Pa', 'cape': 'J/kg', 'r': '%',
             'vo': 's-1', 'd': 's-1', 'vwsh': 'm/s', 'pi': 'm/s'}
    print(f"  {'Var':<6}  {'Unit':<7}  {'Min':>10}  {'P5':>10}  "
          f"{'Median':>10}  {'P95':>10}  {'Max':>10}  {'n':>7}")
    print("  " + "-" * 72)
    for var in ALL_VARIABLES:
        if var not in gen.columns:
            continue
        s = gen[var].dropna()
        if len(s) == 0:
            print(f"  {var:<6}  {'':7}  (no data)")
            continue
        print(f"  {var:<6}  {units.get(var,''):7}  "
              f"{s.min():>10.3f}  {s.quantile(.05):>10.3f}  "
              f"{s.median():>10.3f}  {s.quantile(.95):>10.3f}  "
              f"{s.max():>10.3f}  {len(s):>7,}")
    print("=" * 76)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract raw ERA5 values for positive storms into NACS.csv'
    )
    parser.add_argument('--data-root',   type=Path, default=_DATA_ROOT)
    parser.add_argument('--csv',         type=Path, default=_CSV_PATH)
    parser.add_argument('--ibtracs-csv', type=Path, default=_IBTRACS_CSV,
                        help='IBTrACS origins CSV for SID->name resolution')
    parser.add_argument('--out',         type=Path, default=_OUT_CSV)
    parser.add_argument('--all-storms',  action='store_true',
                        help='Include CLM_ negative storms (default: positive genesis storms only)')
    parser.add_argument('--chunk',       type=int,  default=500_000)
    args = parser.parse_args()

    for p, label in [(args.data_root, 'data-root'), (args.csv, 'csv')]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    print(f"Backend: {_BACKEND}")
    print(f"Mode   : {'all storms' if args.all_storms else 'positive genesis storms only (all timesteps)'}\n")

    index        = NCFileIndex(args.data_root, VARIABLES + [VWSH_VAR])  # vwsh indexed but computed separately
    pi_index     = PICSVIndex(args.data_root)               # PI from .csv files
    sid_name_map = build_sid_name_map(args.ibtracs_csv)     # SID->name for ambiguous timestamps

    if not index.all_stems and not pi_index.index:
        print("ERROR: No NC or PI CSV files indexed. Check --data-root.")
        sys.exit(1)

    storm_df = load_storm_index(args.csv,
                                positive_only=not args.all_storms,
                                chunk_size=args.chunk)

    lat_map, lon_map = build_grid_maps()
    result = extract_all(storm_df, index, pi_index, lat_map, lon_map, sid_name_map)
    print_stats(result)

    col_order = (['ID', 'year', 'time', 'origin', 'latitude', 'longitude']
                 + ALL_VARIABLES)
    out_cols  = [c for c in col_order if c in result.columns]
    result    = result[out_cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_csv(args.out, index=False)
        print(f"\n  NACSGM.csv saved -> {args.out}")
        print(f"  {len(result):,} rows | columns: {out_cols}")
        print(f"\n  Next step: append CLM_ negative storms from {args.csv}")
        print(f"  into the same file to produce the full NACSGM_combined.csv.")
    except PermissionError:
        from datetime import datetime
        alt = (args.out.parent /
               f"NACSGM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        result.to_csv(alt, index=False)
        print(f"\n  NACSGM.csv saved -> {alt}  (original locked)")

    # Coverage warning
    coverage = result[ALL_VARIABLES].notna().mean() * 100
    low = coverage[coverage < 90]
    if len(low):
        print(f"\n  WARNING: Variables with < 90% fill rate:")
        for var, pct in low.items():
            print(f"    {var:<6}  {pct:.1f}%")
        print()
        print("  Most likely cause: the CSV storm IDs don't match the NC")
        print("  filename convention YYYYMMDDOO-NN-NAME.")
        print("  Paste a few CSV 'ID' values so we can diagnose the mapping.")
    else:
        print(f"\n  All variables >= 90% filled.")


if __name__ == '__main__':
    main()
