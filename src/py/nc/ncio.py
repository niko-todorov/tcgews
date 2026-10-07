"""
2. ncio.py
=============
NetCDFs I/O tool for 7 ERA5 variables + 1 Kerry Emanuel's PI calculated variable

Run once:
    python ncio.py

Input:  <NONE>
Output: NACS.csv with origin=1 at 121 grid points per storm (11x11 around the IBTrACS lat/lon 1st record)

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""
#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import track
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add tcpyPI to path for PI calculations
tcpypi_path = os.path.relpath(r'.\\tcpyPI\\src')
# print(f'tcpypi_path: {tcpypi_path}\nsys.path: {sys.path}\n')
if tcpypi_path not in sys.path:
    sys.path.insert(0, tcpypi_path)

try:
    from tcpyPI import pi
    TCPYPI_AVAILABLE = True
    print("✓ tcpyPI successfully imported")
except ImportError:
    TCPYPI_AVAILABLE = False
    print("⚠ Warning: tcpyPI not found. PI calculations will be skipped.")
# https://rammb.cira.colostate.edu/projects/gparm/description.asp

PATH = '..\\..\\..\\data\\'
C = 273.15
grid_inc: float = 0.25  # Grid resolution (degrees)
# Define the grid parameters
# _basin:    str = 'NA' # assumed, hardcoded for now
# _subbasin: str = 'GM' # assumed, hardcoded for now
# _south: float = 18 # NAGM Starting latitude (degrees)
# _north: float = 30 # NAGM Ending latitude (degrees)
# _west: float = 263 # NAGM Starting longitude (degrees)
# _east: float = 279 # NAGM Ending longitude (degrees)

# _BASIN:    str = 'NA'  # assumed, hardcoded for now
# _SUBBASIN: str = 'CS'  # assumed, hardcoded for now
# _south: float = 10  # NACS Starting latitude (degrees)
# _north: float = 22  # NACS Ending latitude (degrees) 49
# _west: float = 271  # NACS Starting longitude (degrees)
# _east: float = 301  # NACS Ending longitude (degrees) 121

_BASIN:    str   = 'NA'
_SUBBASIN: str   = 'CSGM' # CS+GM combined
_south:    float = 11.0
_north:    float = 32.0   # +2° buffer (avoids US coast landfall boundary)
_west:     float = -97.0  # +1° buffer (avoids western Gulf boundary noise)  [was 263.0 in 0/360]
_east:     float = -55.0  # +4° buffer (adds 5 eastern Caribbean boundary)   [was 305.0 in 0/360]

# _BASIN:    str = 'MD' # assumed, hardcoded for now
# _SUBBASIN: str = 'MD' # assumed, hardcoded for now
# _south: float = 30 # MDMD Starting latitude (degrees)
# _north: float = 46 # MDMD Ending latitude (degrees)
# _west: float = 5   # MDMD Starting longitude (degrees)
# _east: float = 36  # MDMD Ending longitude (degrees)

# _BASIN:    str = 'WP' # assumed, hardcoded for now
# _SUBBASIN: str = 'MM' # assumed, hardcoded for now
# _south: float =  5 # WPMM Starting latitude (degrees)
# _north: float = 45 # WPMM Ending latitude (degrees)
# _west: float = 105 # WPMM Starting longitude (degrees)
# _east: float = 180 # WPMM Ending longitude (degrees)

# _basin:    str = 'EP' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =  0 # EPMM Starting latitude (degrees)
# _north: float = 30 # EPMM Ending latitude (degrees)
# _west: float = 180 # EPMM Starting longitude (degrees)
# _east: float = 270 # EPMM Ending longitude (degrees)

# _basin:    str = 'SP' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float = -50 #  Starting latitude (degrees)
# _north: float = 0   #  Ending latitude (degrees)
# _west: float = 150  #  Starting longitude (degrees)
# _east: float = 270  #  Ending longitude (degrees)

# _basin:    str = 'SA' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =-30 #  Starting latitude (degrees)
# _north: float =  0 #  Ending latitude (degrees)
# _west: float = 317 #  Starting longitude (degrees)
# _east: float = 360 #  Ending longitude (degrees)

# _basin:    str = 'SI' # assumed, hardcoded for now
# _subbasin: str = 'WA' # assumed, hardcoded for now
# _south: float = -19.5 # SIWA Starting latitude (degrees)
# _north: float = -3 # SIWA Ending latitude (degrees)
# _west: float =  90 # SIWA Starting longitude (degrees)
# _east: float = 135 # SIWA Ending longitude (degrees)

# _basin:    str = 'SI' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =-35 # SIMM Starting latitude (degrees)
# _north: float = -2 # SIMM Ending latitude (degrees)
# _west: float =  33 # SIMM Starting longitude (degrees)
# _east: float =  90 # SIMM Ending longitude (degrees)

# _basin:    str = 'NI' # assumed, hardcoded for now
# _subbasin: str = 'AS' # AS, BB
# _south: float =  0   #  NIAS Starting latitude (degrees)
# _north: float = 25   #  NIAS Ending latitude (degrees)
# _west: float =  50   #  NIAS Starting longitude (degrees)
# _east: float =  77.5 #  NIAS Ending longitude (degrees)

# _basin:    str = 'NI' # assumed, hardcoded for now
# _subbasin: str = 'BB' # AS, BB
# _south: float =  0   # NIBB Starting latitude (degrees)
# _north: float = 25   # NIBB Ending latitude (degrees)
# _west: float =  77.5 # NIBB Starting longitude (degrees)
# _east: float = 101   # NIBB Ending longitude (degrees)

# origins_df = track.basin_grid(_BASIN, _SUBBASIN)
#
# origins_df['NAME'] = origins_df['NAME'].str.replace(":", "-")

# ── Boundary re-download mode ─────────────────────────────────────────────────
# Set to True when running ONLY the 9 boundary storms that need re-download.
# These storms are NOT in the main origins CSV — they have updated entry coords
# in boundary_redownload_origins.csv which is loaded directly below.
# Leave False for a normal full run (loads via track.basin_grid as usual).
BOUNDARY_REDOWNLOAD_MODE: bool = False
BOUNDARY_REDOWNLOAD_CSV:  str  = f'{PATH}boundary_redownload_origins.csv'

# ── Origin-repair mode ────────────────────────────────────────────────────────
# When True, main() SKIPS all CDS downloads and instead re-marks the `origin`
# column on the per-storm CSVs already on disk, using the fixed convention-
# agnostic longitude match (_lon_close) and snapped genesis time. Use this to
# repair an existing NACSGM.csv whose origin column is all zeros WITHOUT
# re-downloading ERA5 data. After repair, the per-storm CSVs are re-stacked
# into final/NACSGM.csv exactly as a normal run would.
#
# This mode iterates the SAME origins_df as a normal/boundary run, so set
# BOUNDARY_REDOWNLOAD_MODE appropriately:
#   - Repair only the 19 boundary storms : BOUNDARY_REDOWNLOAD_MODE = True
#   - Repair ALL 421 positive storms     : BOUNDARY_REDOWNLOAD_MODE = False
ORIGIN_REPAIR_MODE: bool = True

# ── Quarantine-malformed mode ─────────────────────────────────────────────────
# When True, main()/repair are SKIPPED. Instead, every positive-style per-storm
# CSV (stamp-number-NAME.csv) whose ID column contains a malformed IBTrACS SID
# (length != 13 — a coordinate-injected synthetic ID, e.g. the
# YYYYDDDN[lat*100][lon*10] pattern) is MOVED (not deleted) into
# data/NACSGM/quarantine/, then final/NACSGM.csv is re-stacked from the cleaned
# directory. Reversible: the files are preserved, just relocated.
#
# Background: a SID-construction bug produced 95 such files for storms whose
# genesis was logged in the deep tropics (~2°N), well outside NA-CSGM. They
# inflate the stack and pollute the positive set, so they are quarantined here.
# CLM_*/AL_* negative files are never touched.
QUARANTINE_MALFORMED_MODE: bool = False

# ── Missing-redownload mode ───────────────────────────────────────────────────
# When True, main() downloads ONLY the positive storms whose per-storm CSV is
# absent from data/NACSGM/ (auto-detected — not a hardcoded list). Use this to
# fill genuine gaps in the positive set: e.g. the 7 storms that never produced a
# file (BRET-1993, BRET-2017, UNNAMED-2013, GONZALO-2020, IDALIA-2023,
# PHILIPPE-2023, TAMMY-2023). It iterates the full filtered origins_df but skips
# any storm whose file already exists, so only the missing ones are fetched,
# then re-stacks. Requires BOUNDARY_REDOWNLOAD_MODE = False (so origins_df is the
# full 266-storm set, not the 19-storm boundary subset).
MISSING_REDOWNLOAD_MODE: bool = False

# BOUNDARY_REDOWNLOAD_SIDS: set = {
#     "2016273N13300",  # east  Δ3h   MATTHEW
#     "2003188N11307",  # east  Δ21h  CLAUDETTE
#     "2004217N13306",  # east  Δ21h  BONNIE
#     "1996319N11283",  # south Δ6h   MARCO
#     "2017277N11279",  # south Δ6h   NATE
#     "1993258N11279",  # south Δ12h  GERT
#     "2005300N10279",  # south Δ12h  BETA
#     "2002258N10300",  # south Δ18h  ISIDORE
#     "2017249N22263",  # west  Δ6h   KATIA
# }
if BOUNDARY_REDOWNLOAD_MODE:
    origins_df = pd.read_csv(BOUNDARY_REDOWNLOAD_CSV, low_memory=False)
    origins_df['NAME'] = origins_df['NAME'].str.replace(":", "-")
    print(f"BOUNDARY_REDOWNLOAD_MODE: {len(origins_df)} storms queued from {BOUNDARY_REDOWNLOAD_CSV}")
else:
    origins_df = track.basin_grid(_BASIN, _SUBBASIN) # 1979-NA-CSGM-origins-ibtracs.since1980.list.v04r01.csv
    origins_df['NAME'] = origins_df['NAME'].str.replace(":", "-")
# Directory containing storm CSV files
_path = f'{PATH}{_BASIN}{_SUBBASIN}\\'
os.makedirs(f'{_path}final\\', exist_ok=True)

grid_lat: float = 0.
grid_lon: float = 0.
# og_time: datetime = ""

def ncs2csv(_path: str, _atmvar: str, _var: str, _year: str, _mo: str,
            _day: str, _time: str, _north, _east, _south, _west, C: float = 0.0, _low: bool = True):
    c = cdsapi.Client()
    _type = "reanalysis-era5-single-levels"
    nc_path = _path + '.nc'
    csv_path = _path + '.csv'
    # Skip CDS download if .nc already exists (crash-resume / idempotency)
    if not os.path.exists(nc_path):
        import time as _time_mod
        _max_retries = 4
        for _attempt in range(1, _max_retries + 1):
            try:
                c.retrieve(_type, {
                "variable": _atmvar,
                "product_type": "reanalysis",
                "year": f'{_year}',
                "month": f'{_mo}',
                "day": f'{_day}',
                "time": f'{_time}',
                "area": [_north, _west, _south, _east],  # North/West/South/East
                "data_format": "netcdf"
                }, nc_path)
                break  # success — exit retry loop
            except Exception as e:
                # MarsNoDataError means ERA5 has no data for this timestamp
                # (e.g. sub-hourly offsets). Return an empty DataFrame gracefully.
                if 'MarsNoDataError' in str(e):
                    print(f'  Warning: No ERA5 data for {_year}-{_mo}-{_day} {_time} '
                          f'({_var}), skipping (MarsNoDataError).')
                    empty_df = pd.DataFrame({'latitude': [], 'longitude': [], 'time': [], _var: []})
                    empty_df.to_csv(csv_path, index=False)
                    return empty_df
                if _attempt < _max_retries:
                    _wait = 30 * _attempt
                    print(f'  CDS attempt {_attempt}/{_max_retries} failed — retrying in {_wait}s: {e}')
                    _time_mod.sleep(_wait)
                else:
                    raise Exception(f'NCIO ncs2csv: CDSAPI failed after {_max_retries} attempts: {e}')
    # end: if not os.path.exists(nc_path)

    # Load the netCDF data
    ds = xr.open_dataset(nc_path)
    ds = ds.drop_vars(['number', 'expver'])
    ds = ds.rename({'valid_time': 'time'})

    # # Apply the NACS-NP condition to filter the dataset
    # condition = (ds.longitude > 276) | (ds.latitude > 15.5)
    # ds = ds.where(condition, drop=True)
    da = ds[_var].isel(time=0)  # DataArray with scalar time coord
    # Capture the scalar time value before dropping it from the DataFrame
    _time_val = pd.to_datetime(da.time.values).strftime('%Y-%m-%d %H:%M')
    da = da - C  # convert Kelvin → Celsius

    # Convert the xarray DataArray to a pandas DataFrame
    df = da.to_dataframe(name=_var).reset_index()
    # print(f'DEBUG df[{_var}]: {df.columns.tolist}')

    # Replace any leftover time column (may be scalar coord or absent) with
    # the correctly formatted string. This avoids the (1, 14365) transposition
    # bug caused by iterating over a scalar datetime object's characters.
    df['time'] = _time_val

    # Save the NCS to CSV
    csv_path = _path + '.csv'
    df.to_csv(csv_path, index=False)

    # Close the dataset
    ds.close()
    return df


def ncp2csv(_path: str, _atmvar: str, _var: str, _level: str,
            _year: str, _mo: str, _day: str, _time: str, _north, _east, _south, _west):
    c = cdsapi.Client()
    _type = "reanalysis-era5-pressure-levels"
    nc_path = _path + '.nc'
    # Skip CDS download if .nc already exists (crash-resume / idempotency)
    if not os.path.exists(nc_path):
        import time as _time_mod
        _max_retries = 4
        for _attempt in range(1, _max_retries + 1):
            try:
                c.retrieve(_type, {
                    "variable": _atmvar,
                    "pressure_level": f'{_level}',
                "product_type": "reanalysis",
                "year": f'{_year}',
                "month": f'{_mo}',
                "day": f'{_day}',
                "time": f'{_time}',
                "area": [_north, _west, _south, _east],  # North/West/South/East
                "data_format": "netcdf"}, nc_path)
                break
            except Exception as e:
                if 'MarsNoDataError' in str(e):
                    print(f'  Warning: No ERA5 data for {_year}-{_mo}-{_day} {_time} '
                          f'({_var}@{_level}hPa), skipping (MarsNoDataError).')
                    csv_path = _path + '.csv'
                    empty_df = pd.DataFrame({'latitude': [], 'longitude': [], 'time': [], _var: []})
                    empty_df.to_csv(csv_path, index=False)
                    return empty_df
                if _attempt < _max_retries:
                    _wait = 30 * _attempt
                    print(f'  CDS attempt {_attempt}/{_max_retries} failed — retrying in {_wait}s: {e}')
                    _time_mod.sleep(_wait)
                else:
                    raise Exception(f'NCIO ncp2csv: CDSAPI failed after {_max_retries} attempts: {e}')
    # end: if not os.path.exists(nc_path)

    # Load the netCDF data
    ds = xr.open_dataset(nc_path)
    ds = ds.drop_vars(['number', 'expver', 'pressure_level'])
    # ds = ds.drop_vars(['number', 'expver'])
    ds = ds.rename({'valid_time': 'time'})

    # # Apply the NACS-NP condition to filter the dataset
    # condition = (ds.longitude > 276) | (ds.latitude > 15.5)
    # ds = ds.where(condition, drop=True)

    # Extract the data for the specific time point
    da = ds[_var].isel(time=0)
    _time_val = pd.to_datetime(da.time.values).strftime('%Y-%m-%d %H:%M')

    # Convert the xarray DataArray to a pandas DataFrame
    df = da.to_dataframe(name=_var).reset_index()

    # # remove temp col
    df = df.drop(['pressure_level'], axis=1)

    # print(f'DEBUG df[{_var}]: {df.columns.tolist}')
    df['time'] = _time_val

    # Save the NCP to CSV
    csv_path = _path + '.csv'
    df.to_csv(csv_path, index=False)

    # Close the dataset
    ds.close()
    return df


def ncvwsh2csv(_path: str, _var: str, _level: list, _year: str,
               _mo: str, _day: str, _time: str, _north, _east, _south, _west):
    request = {
        "product_type": "reanalysis",
        'variable': [
            'u_component_of_wind',  # Eastward wind
            'v_component_of_wind',  # Northward wind
        ],
        'pressure_level': [
            '850',  # 850 hPa
            '200',  # 200 hPa
        ],
        'year': f'{_year}',
        'month': f'{_mo}',
        'day': f'{_day}',
        'time': f'{_time}', # DEBUG
        'area': [_north, _west, _south, _east],  # North/West/South/East
        'data_format': 'netcdf'
    }
    # print(request)
    c = cdsapi.Client()
    _type = "reanalysis-era5-pressure-levels"
    nc_path = _path + '.nc'
    # Skip CDS download if .nc already exists (crash-resume / idempotency)
    if not os.path.exists(nc_path):
        import time as _time_mod
        _max_retries = 4
        for _attempt in range(1, _max_retries + 1):
            try:
                c.retrieve(_type, request, nc_path)
                break
            except Exception as e:
                if 'MarsNoDataError' in str(e):
                    print(f'  Warning: No ERA5 data for {request["year"]}-{request["month"]}-{request["day"]} '
                          f'{request["time"]} (vwsh), skipping (MarsNoDataError).')
                    csv_path = _path + '.csv'
                    empty_df = pd.DataFrame({'latitude': [], 'longitude': [], 'time': [], _var: []})
                    empty_df.to_csv(csv_path, index=False)
                    return empty_df
                if _attempt < _max_retries:
                    _wait = 30 * _attempt
                    print(f'  CDS attempt {_attempt}/{_max_retries} failed — retrying in {_wait}s: {e}')
                    _time_mod.sleep(_wait)
                else:
                    raise Exception(f'NCIO ncvwsh2csv: CDSAPI failed after {_max_retries} attempts: {e}')
    # end: if not os.path.exists(nc_path)

    # Load the netCDF data
    ds = xr.open_dataset(nc_path)
    ds = ds.drop_vars(['number', 'expver'])
    ds = ds.rename({'valid_time': 'time'})

    # # Apply the NACS-NP condition to filter the dataset
    # condition = (ds.longitude > 276) | (ds.latitude > 15.5)
    # ds = ds.where(condition, drop=True)

    # Select single time step before computing shear (avoids multi-time edge cases)
    ds_t = ds.isel(time=0)
    _time_val = pd.to_datetime(ds_t.time.values).strftime('%Y-%m-%d %H:%M')

    # Select the wind components at two different pressure levels
    u850: float = ds_t['u'].sel(pressure_level=850)  # u-component at 850 hPa
    v850: float = ds_t['v'].sel(pressure_level=850)  # v-component at 850 hPa
    u200: float = ds_t['u'].sel(pressure_level=200)  # u-component at 200 hPa
    v200: float = ds_t['v'].sel(pressure_level=200)  # v-component at 200 hPa

    # Calculate the vertical wind shear
    vwsh: float = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)

    # Convert the xarray DataArray to a pandas DataFrame
    df = vwsh.to_dataframe(name=_var).reset_index()
    # print(f'DEBUG df[{_var}]: {df.columns.tolist}')

    df['time'] = _time_val

    # Save the NCVWSH to CSV
    csv_path = _path + '.csv'
    df.to_csv(csv_path, index=False)

    # Close the dataset
    ds.close()
    return df


import cdsapi
import os


def pi_pl(output_file: str,
          year: str, month: str, day: str, time: str,
          north: float, east: float, south: float, west: float,
          variables: list = None,
          levels: list = None) -> bool:
    """
    Download ERA5 pressure level data

    Args:
        output_file: Path to save the NetCDF file
        year: Year as string (e.g., '2020')
        month: Month as string with zero padding (e.g., '01')
        day: Day as string with zero padding (e.g., '15')
        time: Hour as string with zero padding (e.g., '00', '12')
        north: Northern latitude boundary
        east: Eastern longitude boundary
        south: Southern latitude boundary
        west: Western longitude boundary
        variables: List of variables to download (default: temperature, specific_humidity)
        levels: List of pressure levels in hPa (default: standard levels from 1000 to 100)

    Returns:
        bool: True if successful, False otherwise
    """

    # Default variables for PI calculation
    if variables is None:
        variables = ['temperature', 'specific_humidity']

    # Default pressure levels
    if levels is None:
        levels = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]

    # Convert levels to strings
    levels_str = [str(level) for level in levels]

    # Ensure time is in HH:MM format
    if len(time) == 2:
        time_str = f'{time}'
    else:
        time_str = time

    if ':' not in time:
        time_str = f'{time}:00'  # Convert '00' to '00:00'
    else:
        time_str = time

    try:
        # Initialize CDS API client
        c = cdsapi.Client()

        print(f'  Requesting ERA5 pressure level data from CDS...')
        print(f'    Variables: {variables}')
        print(f'    Levels: {len(levels)} pressure levels')
        print(f'    Domain: (S:{south},W:{west}) to (N:{north},E:{east})')
        print(f'    Time: {year}-{month}-{day} {time_str}')

        # Download data
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': variables,
                'pressure_level': levels_str,
                'year': year,
                'month': month,
                'day': day,
                'time': time_str,
                'area': [north, west, south, east],  # North, West, South, East
            },
            output_file
        )

        print(f'  Successfully downloaded pressure level data to {output_file}')
        return True

    except Exception as e:
        print(f'  Error downloading pressure level data: {e}')
        return False


def pi_sl(output_file: str,
          year: str, month: str, day: str, time: str,
          north: float, east: float, south: float, west: float,
          variables: list = None) -> bool:
    """
    Download ERA5 surface level data

    Args:
        output_file: Path to save the NetCDF file
        year: Year as string (e.g., '2020')
        month: Month as string with zero padding (e.g., '01')
        day: Day as string with zero padding (e.g., '15')
        time: Hour as string with zero padding (e.g., '00', '12')
        north: Northern latitude boundary
        east: Eastern longitude boundary
        south: Southern latitude boundary
        west: Western longitude boundary
        variables: List of variables to download (default: SST, MSL pressure)

    Returns:
        bool: True if successful, False otherwise
    """

    # Default variables for PI calculation
    if variables is None:
        variables = ['sea_surface_temperature', 'mean_sea_level_pressure']

    # Ensure time is in HH:MM format
    if len(time) == 2:
        time_str = f'{time}'
    else:
        time_str = time

    try:
        # Initialize CDS API client
        c = cdsapi.Client()

        print(f'  Requesting ERA5 surface data from CDS...')
        print(f'    Variables: {variables}')
        print(f'    Domain: (S:{south},W:{west}) to (N:{north},E:{east})')
        print(f'    Time: {year}-{month}-{day} {time_str}')

        # Download data
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': variables,
                'year': year,
                'month': month,
                'day': day,
                'time': time_str,
                'area': [north, west, south, east],  # North, West, South, East
            },
            output_file
        )

        print(f'  Successfully downloaded surface data to {output_file}')
        return True

    except Exception as e:
        print(f'  Error downloading surface data: {e}')
        return False


def pi2csv(_path: str, _var: str,
           _year: str, _mo: str, _day: str, _time: str,
           _north, _east, _south, _west) -> pd.DataFrame:
    """
    Calculate PI (Potential Intensity) and save to CSV
    Following the same pattern as ncp2csv, ncs2csv, ncvwsh2csv

    Returns:
        pd.DataFrame: DataFrame with latitude, longitude, and PI values
    """
    # Check if tcpyPI is available
    if not TCPYPI_AVAILABLE:
        print(f'  Warning: tcpyPI not available, creating empty PI dataframe')
        empty_df = pd.DataFrame({'latitude': [], 'longitude': [], 'time': [], 'pi': []})
        empty_df.to_csv(_path, index=False)
        return empty_df

    # Construct file paths for pressure and surface data (following existing pattern)
    pl_file = f'{_path}_pl.nc'
    sl_file = f'{_path}_sl.nc'

    # Download pressure level data if it doesn't exist
    try:
        # Download pressure level data (temperature, specific humidity)
        if not os.path.exists(pl_file):
            print(f'  Downloading pressure level data to {pl_file}')
            pi_pl(
                pl_file,
                _year, _mo, _day, _time,
                _north, _east, _south, _west,
                # variables=['temperature', 'specific_humidity'],
                variables=['t', 'q'],
                levels=[1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]
            )

        # Download surface level data (SST, MSL pressure)
        if not os.path.exists(sl_file):
            print(f'  Downloading surface level data to {sl_file}')
            pi_sl(
                sl_file,
                _year, _mo, _day, _time,
                _north, _east, _south, _west,
                # variables=['sea_surface_temperature', 'mean_sea_level_pressure']
                variables = ['sst', 'msl']
            )
    except Exception as e:
        print(f'  Error downloading data for PI calculation: {e}')

    # Check if both files exist after download attempt
    if not os.path.exists(pl_file) or not os.path.exists(sl_file):
        print(f'  Warning: Missing data files for PI calculation after download attempt\n{pl_file}\n{sl_file}')
        empty_df = pd.DataFrame({'latitude': [], 'longitude': [], 'time': [], 'pi': []})
        empty_df.to_csv(_path, index=False)
        return empty_df

    # Load the data
    try:
        pl_da = xr.open_dataset(pl_file, engine='netcdf4')
        sl_da = xr.open_dataset(sl_file, engine='netcdf4')

        # Convert PL longitudes from -180/180 to 0/360 format
        if pl_da.longitude.values.min() < 0:
            # print(f'  Converting PL longitudes from -180/180 to 0/360 format...')
            new_lons = xr.where(pl_da.longitude < 0, pl_da.longitude + 360, pl_da.longitude)
            pl_da = pl_da.assign_coords(longitude=new_lons)

            # Sort by longitude to maintain proper ordering
            pl_da = pl_da.sortby('longitude')

        # Verify SL is already in 0-360 format (or convert if needed)
        if sl_da.longitude.values.min() < 0:
            # print(f'  Converting SL longitudes from -180/180 to 0/360 format...')
            new_lons = xr.where(sl_da.longitude < 0, sl_da.longitude + 360, sl_da.longitude)
            sl_da = sl_da.assign_coords(longitude=new_lons)
            sl_da = sl_da.sortby('longitude')

        # Rename dimensions
        if 'valid_time' in pl_da.dims:
            pl_da = pl_da.rename({'valid_time': 'time'})
        if 'pressure_level' in pl_da.dims:
            pl_da = pl_da.rename({'pressure_level': 'level'})
        if 'valid_time' in sl_da.dims:
            sl_da = sl_da.rename({'valid_time': 'time'})

        _pl = pl_da.to_dataframe().reset_index() # {'valid_time': 1, 'pressure_level': 21, 'latitude': 49, 'longitude': 121}
        # remove cols ['time', 'pressure_level', 'latitude', 'longitude', 'number', 'expver', 't', 'q']
        _pl = _pl.drop(['number', 'expver'], axis=1)
        # print(f'  DEBUG _pl[{_var}]: {_pl.columns.tolist}') # ['time', 'level', 'latitude', 'longitude', 't', 'q']
        # _pl.to_csv(f'{_path}_pl.csv', index=False)
        _sl = sl_da.to_dataframe().reset_index()
        # remove cols ['time', 'latitude', 'longitude', 'number', 'expver', 'sst', 'msl']
        _sl = _sl.drop(['number', 'expver'], axis=1)
        # print(f'  DEBUG _sl[{_var}]: {_sl.columns.tolist}') # ['time', 'latitude', 'longitude', 'sst', 'msl']
        # _sl.to_csv(f'{_path}_sl.csv', index=False)

    except Exception as e:
        print(f'  Error loading data for PI calculation: {e}')
        empty_df = pd.DataFrame({'latitude': [], 'longitude': [], f'{_var}_{_year}{_mo}{_day}{_time}': []})
        empty_df.to_csv(_path, index=False)
        return empty_df

    # Select time
    # Build ISO time string — _time is already 'HH:MM' so append ':00' once only
    # e.g. _time='12:00' → '2016-09-28T12:00:00', NOT '2016-09-28T12:00:00:00'
    _time_hhmm = _time if ':' in _time else f'{_time}:00'
    time_str = f'{_year}-{_mo}-{_day}T{_time_hhmm}:00'
    try:
        pl_da = pl_da.sel(time=time_str, method='nearest')
        sl_da = sl_da.sel(time=time_str, method='nearest')
    except:
        try:
            time_str_ns = f'{_year}-{_mo}-{_day}T{_time_hhmm}:00.000000000'
            pl_da = pl_da.sel(time=time_str_ns, method='nearest')
            sl_da = sl_da.sel(time=time_str_ns, method='nearest')
        except:
            print(f'  Warning: Could not select time {time_str}')

    # Select spatial domain (following pattern from other functions)
    # If lons were converted to 0/360, _west/_east must also be converted
    _west_sel = _west % 360 if _west < 0 and pl_da.longitude.values.min() >= 0 else _west
    _east_sel = _east % 360 if _east < 0 and pl_da.longitude.values.min() >= 0 else _east

    print(f"  DEBUG PI subset: west_sel={_west_sel}, east_sel={_east_sel}, pl lons=[{pl_da.longitude.values.min():.2f},{pl_da.longitude.values.max():.2f}]")

    try:
        pl_da = pl_da.sel(
            latitude=slice(_north, _south),
            longitude=slice(_west_sel, _east_sel)
        )
        sl_da = sl_da.sel(
            latitude=slice(_north, _south),
            longitude=slice(_west_sel, _east_sel)
        )
    except Exception as e:
        print(f'  Warning: Could not spatially subset domain: {e}')

    # print(f'  After subsetting:')
    # print(f'    PL shape: {pl_da.dims}')
    # print(f'    SL shape: {sl_da.dims}')

    print(f'  PI for domain: (S:{_south}°N,W:{_west}°E) to (N:{_north}°N,E:{_east}°E)')
    # print(f'  Data shape: {pl_da.dims}')

    # Get dimensions
    lats = pl_da.latitude.values
    lons = pl_da.longitude.values
    levels = pl_da.level.values if 'level' in pl_da.dims else None

    # Initialize arrays for results
    pi_values = []
    lat_list = []
    lon_list = []
    successful_pi = 0
    failed_sst_cold = 0
    failed_nan = 0
    failed_pi = 0

    # Process each grid point
    successful_pi = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            lat_list.append(lat)
            lon_list.append(lon)

            try:
                # Get SST (scalar value)
                sst = float(sl_da['sst'].isel(latitude=i, longitude=j).values.item()) if 'sst' in sl_da else np.nan
                if sst > 200:  # Convert K to C
                    sst = sst - 273.15
                # print(f'  {lat}, {lon} sst={sst}')

                # Get MSL (scalar value)
                msl = float(sl_da['msl'].isel(latitude=i, longitude=j).values.item()) if 'msl' in sl_da else np.nan

                # Convert Pa to hPa if needed
                if msl > 10000:
                    msl = msl / 100

                # Check validity
                if np.isnan(sst) or np.isnan(msl):
                    pi_values.append(np.nan)
                    failed_nan += 1
                    continue

                if sst < 5.0:
                    pi_values.append(np.nan)
                    failed_sst_cold += 1
                    continue

                # Get profiles - IMPORTANT: ensure they are 1D arrays
                if 't' in pl_da:
                    t_profile = pl_da['t'].isel(latitude=i, longitude=j).values
                    # Ensure 1D by squeezing or selecting the values directly
                    t_profile = np.squeeze(t_profile)
                    if t_profile.ndim != 1:
                        t_profile = t_profile.flatten()
                else:
                    pi_values.append(np.nan)
                    continue

                if 'q' in pl_da:
                    q_profile = pl_da['q'].isel(latitude=i, longitude=j).values
                    # Ensure 1D by squeezing or selecting the values directly
                    q_profile = np.squeeze(q_profile)
                    if q_profile.ndim != 1:
                        q_profile = q_profile.flatten()
                else:
                    pi_values.append(np.nan)
                    continue

                # Check for all NaN
                if np.all(np.isnan(t_profile)) or np.all(np.isnan(q_profile)):
                    pi_values.append(np.nan)
                    failed_nan += 1
                    continue


                # Convert temperature to Celsius if needed
                if np.nanmax(t_profile) > 200:
                    t_profile = t_profile - 273.15

                # Ensure correct vertical order (surface to top)
                # Also ensure levels is 1D
                levels_pi = np.array(levels).flatten()

                if levels[0] < levels[-1]:
                    levels_pi = levels[::-1]
                    t_profile = t_profile[::-1]
                    q_profile = q_profile[::-1]
                else:
                    levels_pi = levels.copy()

                # Convert specific humidity to mixing ratio
                q_profile = np.clip(q_profile, 0, 0.999)
                mixing_ratio = (q_profile / (1 - q_profile)) * 1000  # g/kg

                # Ensure all arrays are 1D and proper numpy arrays
                levels_pi = np.asarray(levels_pi, dtype=np.float64)
                t_profile = np.asarray(t_profile, dtype=np.float64)
                mixing_ratio = np.asarray(mixing_ratio, dtype=np.float64)

                # Final check that arrays are 1D
                assert levels_pi.ndim == 1, f"levels_pi has {levels_pi.ndim} dimensions"
                assert t_profile.ndim == 1, f"t_profile has {t_profile.ndim} dimensions"
                assert mixing_ratio.ndim == 1, f"mixing_ratio has {mixing_ratio.ndim} dimensions"

                # Calculate PI using tcpyPI
                from tcpyPI import pi
                # print(f'DEBUG before pi(sst={sst}, msl={msl}, q={q_profile}, t={t_profile}, mixing_ratio={mixing_ratio})')
                VMAX, PMIN, IFL, TO, LNB = pi(
                    sst,  # scalar, Sea surface temperature (C)
                    msl,  # scalar, Mean sea level pressure (hPa)
                    levels_pi,  # 1D array, Pressure levels (hPa)
                    t_profile,  # 1D array, Temperature profile (C)
                    mixing_ratio,  # 1D array, Mixing ratio (g/kg)
                    CKCD=0.9,  # Surface drag coefficient ratio
                    ascent_flag=0,  # Reversible ascent
                    diss_flag=1,  # Include dissipative heating
                    V_reduc=0.8,  # Reduction factor for gradient wind
                    miss_handle=1  # Conservative missing data handling
                )

                # Store result
                if IFL == 1:  # Successful calculation
                    pi_values.append(VMAX)
                    successful_pi += 1
                    # print(f'  DEBUG {successful_pi}. successful for {lat}, {lon} pi() VMAX={VMAX}, PMIN={PMIN}, IFL={IFL}, TO={TO}, LNB={LNB}')
                else:
                    pi_values.append(np.nan)
                    failed_pi += 1
                    # print(f'  DEBUG unsuccessful for {lat}, {lon} pi() VMAX={VMAX}, PMIN={PMIN}, IFL={IFL}, TO={TO}, LNB={LNB}')

            except Exception as e:
                pi_values.append(np.nan)
                failed_nan += 1
                if successful_pi == 0: print(f'  DEBUG {e}')

    print(f'  {successful_pi}/{len(lat_list)} PI-calculated grid points')

    if failed_sst_cold > 0:
        print(f'  {failed_sst_cold} PI-calc failed due to cold SST (<5°C)')
    if failed_nan > 0:
        print(f'  {failed_nan} PI-calc failed due to NaN (SST over land)')
    if failed_pi > 0:
        print(f'  {failed_pi} PI-calc failed (IFL!=1)')

    # Create DataFrame (following pattern from other functions)
    pi_df = pd.DataFrame({
        'latitude': lat_list,
        'longitude': lon_list,
        'time': f'{_year}{_mo}{_day} {_time}',
        'pi':  pi_values
    })

    # Save PI to CSV
    print(f'  Writing PI to {_path}')
    pi_df.to_csv(_path+'.csv', index=False)

    # Close datasets
    pl_da.close()
    sl_da.close()

    return pi_df

COORD_TOL = grid_inc / 4   # 0.0625° — tight enough to match only the target cell

def _lon_close(a, b, tol):
    """
    Convention-agnostic longitude proximity test.

    Compares two longitudes using minimum circular distance, so it
    returns the correct answer regardless of whether either value is
    expressed in -180/+180 or 0-360 convention. e.g. -79.0 and 281.0
    are the SAME meridian (circular distance 0), even though their
    arithmetic difference is 360.
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d) < tol


def beefup_origin(row, og_time):
    """
    Mark origin=1 at the SINGLE nearest grid point to the genesis
    location, at t=0 only. Uses tolerance-based coordinate comparison
    to avoid IEEE 754 floating-point equality failures.

    Longitude is matched with circular distance (_lon_close) so the
    -180/+180 data convention and any 0-360 grid_lon value still match.

    The sigma-widened Gaussian target is constructed at training time
    by TCGMultiLeadDataset.
    """
    if row['time'] != og_time:
        return 0
    lat_match = abs(row['latitude'] - grid_lat) < COORD_TOL
    lon_match = _lon_close(row['longitude'], grid_lon, COORD_TOL)
    return 1 if (lat_match and lon_match) else 0
    # beef up the origin from 1 (nearest) to 121 grid points (10x10, or 2.5x2.5 deg, 275x275 km)
    # for i in range(-5, 6):  # -5,-4,-3,-2,-1,0,1,2,3,4,5
    #     for j in range(-5, 6):  # -5,-4,-3,-2,-1,0,1,2,3,4,5
    #         if (row['latitude'] == grid_lat + i * grid_inc) and (row['longitude'] == grid_lon + j * grid_inc) and (
    #                 row['time'] == og_time):
    #             # print(f'time:{row["time"]}, lat:{row["latitude"]}, lon:{row["longitude"]}')
    #             return 1
    # return 0


def get_back_dates(_dt: datetime, _hrs_back: int = 54, _hr_dec: int = 6) -> (list, list, list, list, list):
    # Calculate the end time (48 hours back)
    end_time = _dt - timedelta(hours=_hrs_back)

    # Generate all prior date/times in 6-hour increments
    times = []
    current_time = _dt
    while current_time > end_time:
        # ERA5 only stores data on whole hours. Snap any sub-hour offset
        # (e.g. IBTrACS records at HH:30) to the nearest whole hour so that
        # CDS requests don't fail with MarsNoDataError.
        if current_time.minute != 0:
            current_time = current_time + timedelta(minutes=(60 - current_time.minute)) \
                if current_time.minute >= 30 \
                else current_time - timedelta(minutes=current_time.minute)
            current_time = current_time.replace(second=0, microsecond=0)
        times.append(current_time)
        current_time = _dt - timedelta(hours=_hr_dec * (len(times)))

    # the times in a list of strings:
    times_str = [t.strftime("%Y-%m-%d %H:%M") for t in times]
    # print(f'DEBUG: {times_str}')
    # Extract and print year, month, day, hour, minute as separate strings for each time
    for t in times:
        _year = t.strftime("%Y")
        _mo = t.strftime("%m")
        _day = t.strftime("%d")
        _hr = t.strftime("%H")
        _mi = t.strftime("%M")
        # print(f"Year: {_year}, Month: {_mo}, Day: {_day}, Hour: {_hr}, Minute: {_mi}")

    # store these values in lists:
    years = [t.strftime("%Y") for t in times]
    months = [t.strftime("%m") for t in times]
    days = [t.strftime("%d") for t in times]
    hours = [t.strftime("%H") for t in times]
    minutes = [t.strftime("%M") for t in times]

    return years, months, days, hours, minutes


def _resolve_storm_csv(storm_dir: str, _dt: datetime, _nmbr: str,
                       _sname: str) -> str | None:
    """
    Return the path to an existing per-storm CSV for this storm, or None.

    The on-disk filename is built from the OLDEST timestep (index -1 of
    get_back_dates), not the genesis time, and storm-name separators vary
    between the origins CSV ("JOAN-MIRIAM") and the saved file
    ("JOAN_MIRIAM"). This tries the canonical name and common separator
    variants so existence checks are reliable. Mirrors the filename logic in
    repair_origins().
    """
    years, months, days, hours, _ = get_back_dates(_dt, 54, 6)
    stamp = f'{years[-1]}{months[-1]}{days[-1]}{hours[-1]}'
    for nv in (_sname, _sname.replace('-', '_'), _sname.replace('_', '-'),
               _sname.replace('-', ''), _sname.replace(' ', '_')):
        cand = os.path.join(storm_dir, f'{stamp}-{_nmbr}-{nv}.csv')
        if os.path.exists(cand):
            return cand
    return None


def _backfill_pi_from_cache(chunk: pd.DataFrame, stem: str,
                            pi_dir: str, round_dp: int = 2):
    """
    Fill a per-storm chunk's 'pi' column from the pi/ cache when it is missing
    or all-NaN. The positive per-storm CSVs are written without the PI join
    (PI is computed separately and cached in pi/<stem>.csv as
    [latitude, longitude, time, pi]); this backfills it so the stacked dataset
    carries real PI for positives, matching the negatives.

    Joins on (latitude, longitude, time). Returns (chunk, status_str) where
    status is one of: 'ok' (filled), 'present' (already had PI), 'no-cache',
    'partial:<n_miss>' (some rows unmatched — left NaN), 'no-pi-col'.

    No-op when PI is already populated, so it is safe to run on every stack.
    """
    if 'pi' not in chunk.columns:
        return chunk, 'no-pi-col'
    # already populated? leave it.
    if chunk['pi'].notna().any():
        return chunk, 'present'

    pi_path = os.path.join(pi_dir, stem)
    if not os.path.exists(pi_path):
        return chunk, 'no-cache'
    try:
        pic = pd.read_csv(pi_path)
    except Exception:
        return chunk, 'no-cache'
    if not {'latitude', 'longitude', 'time', 'pi'}.issubset(pic.columns):
        return chunk, 'no-cache'

    # build match keys without mutating original coordinate columns
    lat = chunk['latitude'].round(round_dp)
    t   = pd.to_datetime(chunk['time'])
    # Normalize longitude to a canonical 0-360 frame on BOTH sides before
    # matching, so the join works regardless of whether either file stores
    # -180/+180 or 0-360. (The per-storm CSVs are -180/+180; the PI cache is
    # 0-360 from the PI calc — without this they never match and the join
    # silently fills nothing.)
    def _to360(s):
        s = s.astype(float) % 360.0
        return s.round(round_dp)

    key = pd.DataFrame({'_lat': lat, '_lon': _to360(chunk['longitude']), '_t': t})

    pic_key = pd.DataFrame({
        '_lat': pic['latitude'].round(round_dp),
        '_lon': _to360(pic['longitude']),
        '_t':   pd.to_datetime(pic['time']),
        '_pi':  pic['pi'].values,
    }).drop_duplicates(subset=['_lat', '_lon', '_t'])

    merged = key.merge(pic_key, on=['_lat', '_lon', '_t'], how='left')
    n_miss = int(merged['_pi'].isna().sum())
    chunk = chunk.copy()
    chunk['pi'] = merged['_pi'].values
    if n_miss:
        return chunk, f'partial:{n_miss}'
    return chunk, 'ok'


def _stack_per_storm_csvs(stack_path: str, final_csv: str) -> None:
    """
    Stack all per-storm CSVs in `stack_path` into `final_csv`.

    Written incrementally (one storm at a time) to avoid loading the entire
    dataset into RAM at once — the naive whole-list pd.concat needs ~3 GB for
    the full positive set. Skips files that fail to read or look transposed.

    PI backfill: each per-storm chunk whose 'pi' column is missing/all-NaN is
    filled from the pi/ cache (sibling of stack_path) before writing, so the
    stacked dataset carries real PI for positives as well as negatives.
    """
    storm_files = sorted(fn for fn in os.listdir(stack_path)
                         if fn.endswith(".csv"))
    # pi/ cache lives as a sibling of the per-storm stack directory.
    # stack_path is e.g. ...\data\NACSGM\  -> pi dir is ...\data\NACSGM\pi
    _pi_dir = os.path.join(stack_path.rstrip('\\/'), 'pi')
    _pi_stats = {'ok': 0, 'present': 0, 'no-cache': 0,
                 'partial': 0, 'no-pi-col': 0}
    try:
        if not storm_files:
            raise ValueError(f"No storm CSVs found in {stack_path}")
        total_rows = 0
        for _fi, _fn in enumerate(storm_files):
            _fp = os.path.join(stack_path, _fn)
            try:
                _chunk = pd.read_csv(_fp)
            except Exception as _e:
                print(f"  Warning: skipping corrupt CSV {_fn}: {_e}")
                continue
            # Sanity-check shape — a transposed CSV would have 1 row, many cols
            if len(_chunk) <= 1 and len(_chunk.columns) > 20:
                print(f"  Warning: skipping likely-transposed CSV {_fn} "
                      f"(shape {_chunk.shape})")
                continue
            # Backfill PI from the pi/ cache when missing/all-NaN.
            _chunk, _pi_status = _backfill_pi_from_cache(_chunk, _fn, _pi_dir)
            if _pi_status.startswith('partial'):
                _n_miss = _pi_status.split(':', 1)[1]
                print(f"  ⚠ PI backfill {_fn}: {_n_miss} rows unmatched "
                      f"(left NaN)")
                _pi_stats['partial'] += 1
            else:
                _pi_stats[_pi_status] = _pi_stats.get(_pi_status, 0) + 1
            _chunk.to_csv(final_csv,
                          mode='w' if _fi == 0 else 'a',
                          header=(_fi == 0),
                          index=False)
            total_rows += len(_chunk)
            _chunk = None  # free memory immediately
        print(f"  Stacked {len(storm_files)} per-storm CSVs -> "
              f"{final_csv} ({total_rows:,} rows)")
        _filled = _pi_stats['ok']
        if _filled or _pi_stats['partial']:
            print(f"  PI backfill: filled {_filled} file(s) from pi/ cache"
                  + (f", {_pi_stats['partial']} partial"
                     if _pi_stats['partial'] else "")
                  + (f", {_pi_stats['no-cache']} without cache"
                     if _pi_stats['no-cache'] else ""))
    except ValueError as e:
        print(f"No storm CSVs to stack in {stack_path} - {e}")
    except Exception as e:
        print(f"Stacking per-storm CSVs in {stack_path} failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def quarantine_malformed() -> None:
    """
    Move (not delete) per-storm CSVs with malformed IBTrACS SIDs out of
    data/NACSGM/ into data/NACSGM/quarantine/, then re-stack final/NACSGM.csv
    from the cleaned directory.

    A "malformed" file is a positive-style per-storm CSV — name pattern
    `{10-digit stamp}-{number}-{NAME}.csv` — whose `ID` column contains any SID
    whose length != 13. A valid IBTrACS SID is 13 chars (YYYYDDDN + 5-char grid
    code, e.g. 2021222N14301). The bad ones are 16 chars with a coordinate pair
    injected (YYYYDDDN[lat*100][lon*10], e.g. 2006251N02192999 -> lat 2.19°N,
    lon 299.9°=-60.1°): deep-tropics genesis points outside NA-CSGM produced by
    a SID-construction bug.

    Reversible: files are relocated, not removed. Negative files (CLM_*, AL_*)
    and anything not matching the positive-style name pattern are left in place.
    """
    import re
    import shutil

    storm_dir  = f'{PATH}{_BASIN}{_SUBBASIN}\\'
    quarantine = os.path.join(storm_dir, 'quarantine')
    final_path = f'{PATH}{_BASIN}{_SUBBASIN}\\final\\'
    os.makedirs(quarantine, exist_ok=True)
    os.makedirs(final_path, exist_ok=True)

    print("=" * 60)
    print("QUARANTINE-MALFORMED MODE — relocating bad-SID files, no downloads")
    print(f"  Source:     {storm_dir}")
    print(f"  Quarantine: {quarantine}")
    print("=" * 60)

    pos_name = re.compile(r'^\d{10}-\d+-')  # stamp-number-NAME
    moved = 0
    left_other = 0
    for f in sorted(glob.glob(os.path.join(storm_dir, '*.csv'))):
        base = os.path.basename(f)
        # Only positive-style files are candidates; never touch CLM_/AL_/etc.
        if not pos_name.match(base):
            left_other += 1
            continue
        try:
            ids = pd.read_csv(f, usecols=['ID'])['ID'].astype(str).unique()
        except Exception as e:
            print(f"  (skip unreadable {base}: {e})")
            continue
        if any(len(s) != 13 for s in ids):
            shutil.move(f, os.path.join(quarantine, base))
            moved += 1
            print(f"  moved {base}")

    print("-" * 60)
    print(f"  Quarantined malformed files: {moved}")
    print(f"  Left in place (negatives / non-positive-style): {left_other}")

    # Re-stack the cleaned directory.
    print("\n  Re-stacking per-storm CSVs from cleaned directory...")
    _stack_per_storm_csvs(storm_dir, f'{final_path}{_BASIN}{_SUBBASIN}.csv')


def repair_origins() -> None:
    """
    Re-mark the `origin` column on the per-storm CSVs already on disk,
    WITHOUT re-downloading any ERA5 data, then re-stack into final/.

    Why this exists
    ---------------
    An earlier run wrote every per-storm CSV with origin=0 everywhere because
    of two bugs (both now fixed in this file):
      A) longitude-convention mismatch — grid_lon was force-converted to 0-360
         while the data stores longitude in -180/+180, so no genesis cell ever
         matched; and
      B) snapped-genesis-time mismatch — og_time used the raw sub-hour IBTrACS
         time while the data row carries the whole-hour snapped time.

    Only the `origin` column was wrong; the downloaded lat/lon/time/variable
    data is correct. So this re-applies the FIXED matching logic (same
    grid_lat/grid_lon computation, snapped og_time, and beefup_origin with
    _lon_close) to each existing per-storm CSV and rewrites it in place.

    Iterates the same origins_df as a normal run, so respects
    BOUNDARY_REDOWNLOAD_MODE (True → just the 19 boundary storms; False → all).
    """
    global grid_lat, grid_lon

    final_path = f'{PATH}{_BASIN}{_SUBBASIN}\\final\\'
    os.makedirs(final_path, exist_ok=True)
    storm_dir = f'{PATH}{_BASIN}{_SUBBASIN}\\'

    print("=" * 60)
    print("ORIGIN-REPAIR MODE — no downloads, re-marking origin only")
    print(f"  Storms queued: {len(origins_df)} "
          f"({'boundary subset' if BOUNDARY_REDOWNLOAD_MODE else 'full set'})")
    print("=" * 60)

    n_repaired = 0
    n_missing  = 0
    n_no_match = 0

    n_nearest = 0  # storms recovered via nearest-surviving-cell fallback

    for i in range(len(origins_df)):
        _ID:    str = origins_df.iloc[i]['SID']
        _nmbr:  str = str(int(origins_df.iloc[i]['NUMBER']))
        _sname: str = origins_df.iloc[i]['NAME']
        _dt: datetime = pd.to_datetime(origins_df.iloc[i]['ISO_TIME'])
        _lat: float = origins_df.iloc[i]['LAT']
        _lon: float = origins_df.iloc[i]['LON']

        # Reproduce the SAME snapped genesis time the data row was written with.
        # og_time is the GENESIS timestep = newest = index 0.
        years, months, days, hours, minutes = get_back_dates(_dt, 54, 6)
        og_time = (f'{years[0]}-{months[0]:02}-{days[0]:02} '
                   f'{hours[0]:02}:{minutes[0]:02}')

        # Per-storm CSV filename is built from the OLDEST timestep (index -1),
        # NOT the genesis time. The download loop overwrites `_name` on every
        # back-step and the file is saved after the loop ends, so the persisted
        # name carries the last (oldest) timestep's stamp. e.g. EMILY genesis
        # 1987-09-21 06:00 is saved as 1987091906-79-EMILY (oldest = 09-19 06).
        _stamp = f'{years[-1]}{months[-1]}{days[-1]}{hours[-1]}'

        # Storm names with separators are stored inconsistently between the
        # origins CSV (e.g. "JOAN-MIRIAM") and the on-disk filename (e.g.
        # "JOAN_MIRIAM"). Try the CSV spelling first, then common separator
        # variants, before giving up.
        _name_variants = [_sname,
                          _sname.replace('-', '_'),
                          _sname.replace('_', '-'),
                          _sname.replace('-', ''),
                          _sname.replace(' ', '_')]
        _csv = None
        _t0_name = f'{_stamp}-{_nmbr}-{_sname}'
        for _nv in _name_variants:
            _cand = os.path.join(storm_dir, f'{_stamp}-{_nmbr}-{_nv}.csv')
            if os.path.exists(_cand):
                _csv = _cand
                _t0_name = f'{_stamp}-{_nmbr}-{_nv}'
                break

        if _csv is None:
            print(f'{i + 1:03}. MISSING {_ID} {_sname} -> {_t0_name}.csv')
            n_missing += 1
            continue

        df = pd.read_csv(_csv)

        # Same nearest-grid-point computation as the download path.
        grid_lat = round(round(_lat / grid_inc) * grid_inc, 4)
        grid_lon = round(round(_lon / grid_inc) * grid_inc, 4)
        # grid_lon stays in native -180/+180; beefup_origin uses _lon_close().

        df['origin'] = df.apply(lambda row: beefup_origin(row, og_time), axis=1)
        n_ones = int((df['origin'] == 1).sum())
        _used_fallback = False

        # ── Nearest-surviving-cell fallback ───────────────────────────────────
        # Some storms have their genesis point OUTSIDE the clean ocean grid:
        # south of the domain's 11°N edge, west of its -97° edge, or on a
        # coastline where SST=NaN dropped that cell. The exact genesis cell
        # then doesn't exist in the data, so the strict match marks nothing.
        # For these we attribute genesis to the nearest SURVIVING cell at the
        # genesis timestep (smallest great-circle-ish distance), which is the
        # physically sensible label for an edge/coastal genesis.
        if n_ones == 0:
            gen = df[df['time'] == og_time]
            if len(gen) > 0:
                # squared distance in degrees; longitude via circular delta
                dlat = gen['latitude'].to_numpy() - _lat
                raw = (gen['longitude'].to_numpy() - _lon) % 360.0
                dlon = np.minimum(raw, 360.0 - raw)
                # weight lon by cos(lat) so the metric is roughly isotropic
                import math
                w = math.cos(math.radians(_lat))
                d2 = dlat * dlat + (dlon * w) ** 2
                j = int(d2.argmin())
                nearest_idx = gen.index[j]
                df.loc[nearest_idx, 'origin'] = 1
                n_ones = 1
                n_nearest += 1
                _used_fallback = True
                near_lat = df.loc[nearest_idx, 'latitude']
                near_lon = df.loc[nearest_idx, 'longitude']
                print(f'{i + 1:03}. {_ID} {_sname}: genesis ({_lat},{_lon}) '
                      f'outside clean grid -> nearest cell '
                      f'({near_lat},{near_lon}) [fallback]')

        df.to_csv(_csv, index=False)
        n_repaired += 1
        if n_ones == 0:
            n_no_match += 1
            print(f'{i + 1:03}. {_ID} {_sname}: 0 genesis cells matched '
                  f'(og_time={og_time}, grid=({grid_lat},{grid_lon})) - '
                  f'genesis timestep absent from file ⚠')
        elif not _used_fallback:
            print(f'{i + 1:03}. {_ID} {_sname}: origin=1 set on {n_ones} cell(s)')

    print("-" * 60)
    print(f"  Repaired files:        {n_repaired}")
    print(f"    via exact cell:      {n_repaired - n_nearest}")
    print(f"    via nearest cell:    {n_nearest} (edge/coastal genesis)")
    print(f"  Missing files:         {n_missing} (no CSV on disk - need download)")
    print(f"  Files with 0 matches:  {n_no_match} (genesis timestep absent ⚠)")

    # Re-stack into final/NACSGM.csv
    print("\n  Re-stacking per-storm CSVs...")
    _stack_per_storm_csvs(storm_dir, f'{final_path}{_BASIN}{_SUBBASIN}.csv')


def main() -> None:
    global grid_lat, grid_lon  #, og_time

    if MISSING_REDOWNLOAD_MODE:
        if BOUNDARY_REDOWNLOAD_MODE:
            raise SystemExit(
                "MISSING_REDOWNLOAD_MODE requires BOUNDARY_REDOWNLOAD_MODE=False "
                "so origins_df is the full positive set, not the 19-storm subset.")
        # Report which storms will be fetched before doing any work.
        _sd = f'{PATH}{_BASIN}{_SUBBASIN}\\'
        _missing = [
            (origins_df.iloc[i]['SID'], origins_df.iloc[i]['NAME'])
            for i in range(len(origins_df))
            if _resolve_storm_csv(_sd,
                                  pd.to_datetime(origins_df.iloc[i]['ISO_TIME']),
                                  str(int(origins_df.iloc[i]['NUMBER'])),
                                  origins_df.iloc[i]['NAME']) is None
        ]
        print("=" * 60)
        print(f"MISSING-REDOWNLOAD MODE — fetching {len(_missing)} storm(s) "
              "with no per-storm CSV on disk")
        for _sid, _nm in _missing:
            print(f"   {_sid}  {_nm}")
        print("=" * 60)

    # Compute final_path here (before the storm loop) so it is always in scope
    final_path = f'{PATH}{_BASIN}{_SUBBASIN}\\final\\'
    os.makedirs(final_path, exist_ok=True)

    try:
        # loop over all storms in a subbasin
        for i in range(len(origins_df)):
            _ID: str = origins_df.iloc[i]['SID']
            _seasn: str = origins_df.iloc[i]['SEASON']
            _nmbr: str = str(int(origins_df.iloc[i]['NUMBER']))
            _basin: str = origins_df.iloc[i]['BASIN']
            _subbasin: str = origins_df.iloc[i]['SUBBASIN']
            _sname: str = origins_df.iloc[i]['NAME']
            _dt: datetime = pd.to_datetime(origins_df.iloc[i]['ISO_TIME'])
            _year: str = _dt.strftime("%Y")  # origins_df.iloc[i]['year']
            _mo: str = _dt.strftime("%m")  #origins_df.iloc[i]['month']
            _day: str = _dt.strftime("%d")  #origins_df.iloc[i]['day']
            _hr: str = _dt.strftime("%H")  #origins_df.iloc[i]['hour']
            _mi: str = _dt.strftime("%M")  #origins_df.iloc[i]['min']
            _lat: float = origins_df.iloc[i]['LAT']
            _lon: float = origins_df.iloc[i]['LON']
            # _name:  str = f'{_year}-{_nmbr}-{_sname}'
            og_time: str = f'{_year}-{_mo:02}-{_day:02} {_hr:02}:{_mi:02}'
            # print(f'OG time:\t{og_time}\nISO_TIME:\t{_dt}')

            # ── Skip storms already downloaded ────────────────────────────────
            # The saved filename is built from the OLDEST timestep (index -1),
            # not the genesis time, and name separators vary, so resolve the
            # real on-disk path via _resolve_storm_csv (which tries variants).
            # In MISSING_REDOWNLOAD_MODE this is the core filter: only storms
            # with NO existing file are downloaded; everything else is skipped.
            _existing = _resolve_storm_csv(
                f'{PATH}{_BASIN}{_SUBBASIN}\\', _dt, _nmbr, _sname)
            if _existing is not None:
                if MISSING_REDOWNLOAD_MODE:
                    # quiet skip — only the missing ones are of interest
                    continue
                print(f'{i + 1:03}. SKIP (exists) {_ID} {_sname}')
                continue
            print(f'{i + 1:03}. PROCESSING {_ID} {_sname}  og_time={og_time}')
            sys.stdout.flush()   # force print before first blocking CDS call

            # get _back // _inc long 5 lists of years, months, days, hours, minutes back from the original TCG date/time
            years, months, days, hours, minutes = get_back_dates(_dt, 54, 6) # 48hr lead time # LSTM sequence
            print(f'{years}\n{months}\n{days}\n{hours}\n{minutes}')

            # ── Genesis-time snapping fix ────────────────────────────────────
            # get_back_dates() snaps sub-hour genesis times (e.g. IBTrACS HH:30)
            # to the nearest whole hour, because ERA5 only stores whole hours.
            # The first returned timestep (index 0) is the snapped genesis time
            # that the data rows are actually written with. Rebuild og_time from
            # it so beefup_origin's time match succeeds for snapped storms
            # (e.g. DORIAN 2019-08-27 01:30 -> 02:00). Without this, og_time would
            # keep the raw sub-hour value and never match any data row.
            og_time = f'{years[0]}-{months[0]:02}-{days[0]:02} {hours[0]:02}:{minutes[0]:02}'

            # Calculate Potential Intensity PI
            _pi_ = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _var: str = 'pi'
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                print(f'Calculating PI for {_sname} {_name}')

                _pi = pi2csv(_spath, _var, _year, _mo, _day, _time, _north, _east, _south, _west)
                _pi_ = pd.concat([_pi_, _pi], axis=0, ignore_index=True)
                print(f'PI completed for {_sname} {_name}')

            _pi_.to_csv(_spath+'.csv', index=False)

            _lat = origins_df.iloc[i]['LAT']
            _lon = origins_df.iloc[i]['LON']

            # Reformat time: "20201031 18:00" → "2020-10-31 18:00"
            _pi_['time'] = pd.to_datetime(_pi_['time'], format='%Y%m%d %H:%M').dt.strftime('%Y-%m-%d %H:%M')

            # Sea Surface Temperature hi
            sst = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "sea_surface_temperature"
                _var = "sst"
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _sst = ncs2csv(_spath, _atmvar, _var, _year, _mo, _day, _time, _north, _east, _south, _west, C, False)
                sst = pd.concat([sst, _sst], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # Barometric Pressure at Mean Sea Level lo
            prmsl = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "mean_sea_level_pressure"
                _var = "msl"
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(PATH + f'{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _prmsl = ncs2csv(_spath, _atmvar, _var, _year, _mo, _day, _time, _north, _east, _south, _west)
                prmsl = pd.concat([prmsl, _prmsl], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # CAPE (Convective Available Potential Energy) hi
            cape = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "convective_available_potential_energy"
                _var = "cape"
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _cape = ncs2csv(_spath, _atmvar, _var, _year, _mo, _day, _time, _north, _east, _south, _west)
                cape = pd.concat([cape, _cape], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # Relative Humidity hi
            rhum = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "relative_humidity"
                _var = "r"
                _level = "850"
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(PATH + f'{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _rhum = (ncp2csv(_spath, _atmvar, _var, _level, _year, _mo, _day, _time, _north, _east, _south, _west))
                rhum = pd.concat([rhum, _rhum], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # Vorticity hi
            vo = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "Vorticity"
                _var = "vo"
                _level = "850"
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(PATH + f'{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _vo = (ncp2csv(_spath, _atmvar, _var, _level, _year, _mo, _day, _time, _north, _east, _south, _west))
                vo = pd.concat([vo, _vo], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # Divergence hi
            divrg = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _atmvar = "Divergence"
                _var = "d"
                _level = "850" # @850 low level convergence; upper leve divergence @200? u200, v200?
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(PATH + f'{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _divrg = (ncp2csv(_spath, _atmvar, _var, _level, _year, _mo, _day, _time, _north, _east, _south, _west))
                divrg = pd.concat([divrg, _divrg], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # Vertical Windshear lo
            vwsh = pd.DataFrame()
            for back in range(len(years)):
                _year = years[back]
                _mo = months[back]
                _day = days[back]
                _hr = hours[back]
                _time = f'{_hr:02}:{minutes[back]:02}'
                _name = f'{_year}{_mo}{_day}{_hr}-{_nmbr}-{_sname}'

                _var: str = 'vwsh'
                _level: list = ["850", "200"]
                print(f'Getting {_var} for {_sname} {_name}')
                _spath: str = f'{PATH}{_BASIN}{_SUBBASIN}\\{_var}\\{_name}'
                os.makedirs(PATH + f'{_BASIN}{_SUBBASIN}\\{_var}\\', exist_ok=True)
                _vwsh = (ncvwsh2csv(_spath, _var, _level, _year, _mo, _day, _time, _north, _east, _south, _west))
                vwsh = pd.concat([vwsh, _vwsh], axis=0, ignore_index=True)
                # print(f'Finished {_var} for {_sname} {_name}')

            # DEBUG
            sst = sst.dropna(subset=['sst'])
            vwsh = vwsh.dropna(subset=['vwsh'])
            cape = cape.dropna(subset=['cape'])
            df1 = sst.merge(vwsh, on=["latitude", "longitude", "time"], how="inner")
            df2 = cape.merge(rhum, on=["latitude", "longitude", "time"], how="inner")
            df3 = prmsl.merge(vo, on=["latitude", "longitude", "time"], how="inner")
            df4 = df1.merge(divrg, on=["latitude", "longitude", "time"], how="inner")
            df5 = df2.merge(df3, on=["latitude", "longitude", "time"], how="inner")
            df6 = df4.merge(df5, on=["latitude", "longitude", "time"], how="inner")
            df  = df6.merge(_pi_, on=["latitude", "longitude", "time"], how="left")

            # Calculate the nearest grid point
            grid_lat = round(_lat / grid_inc) * grid_inc
            grid_lon = round(_lon / grid_inc) * grid_inc

            # Round to 4 decimal places to eliminate IEEE 754 dust
            # e.g.  round(13 * 0.25, 4) = 3.25  not  3.2500000000000004
            grid_lat = round(grid_lat, 4)
            grid_lon = round(grid_lon, 4)

            # ── Longitude convention ──────────────────────────────────────────
            # The per-storm CSV data stores longitude in -180/+180 (verified:
            # genesis-timestep lon range -97.0 to -55.0). grid_lon is left in
            # its native -180/+180 convention to match. beefup_origin() uses
            # _lon_close() (minimum circular distance), so matching is robust to
            # either convention regardless — but keeping grid_lon native avoids
            # writing a value in a convention the data never uses.

            # Assign 1 to the nearest grid point to a given origin point
            df['origin'] = 0
            # condition on the OG date/time only!
            df['origin'] = df.apply(lambda row: beefup_origin(row, og_time), axis=1)
            # NOTE: this strict match does NOT include the nearest-surviving-cell
            # fallback used by repair_origins() for coastal / out-of-domain
            # genesis. A storm whose genesis cell was dropped (SST=NaN over land)
            # or lies outside the 11°N / -97°E edges will get origin all-zero
            # here. The canonical workflow finalizes labels with a follow-up
            # ORIGIN_REPAIR_MODE pass (which has the fallback), so after a
            # MISSING_REDOWNLOAD run, re-run with ORIGIN_REPAIR_MODE=True to
            # guarantee every storm — including edge cases — is labeled.

            # Add storm ID column...
            df['ID'] = _ID
            # ...and move it left to be 1st
            cols = df.columns.tolist()
            cols = ['ID'] + [col for col in cols if col != 'ID']
            df = df[cols]
            print(f'{i + 1:03}. {_ID} {_sname}')

            # Save the storm grid .csv
            df.to_csv(f'{PATH}{_BASIN}{_SUBBASIN}\\{_name}.csv', index=False)
            # df6.to_csv(f'{PATH}{_BASIN}{_SUBBASIN}\\{_name}_pi.csv', index=False)
        # Stack all per-storm CSVs into the final combined file
        _stack_per_storm_csvs(f'{PATH}{_BASIN}{_SUBBASIN}\\',
                              f'{final_path}{_BASIN}{_SUBBASIN}.csv')

    except KeyboardInterrupt:
        print("\n  Interrupted by user - partial outputs preserved on disk.")
        raise
    except Exception as e:
        print(f"\n  main() failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    if QUARANTINE_MALFORMED_MODE:
        quarantine_malformed()
    elif ORIGIN_REPAIR_MODE:
        repair_origins()
    else:
        # Normal full run, or MISSING_REDOWNLOAD_MODE (main() detects the flag
        # and downloads only storms whose per-storm CSV is absent).
        main()
