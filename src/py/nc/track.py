"""
1. track.py
=============
Downloads the current IBTrACS and extracts storms for the specified basin

Run once:
    python track.py

Input:  <NONE>
Output: ibtracs.since1980.list.v04r01.csv current
1. 1979-ibtracs.since1980.list.v04r01.csv
       filter: SEASON > 1979
2. 1979-NA-CSGM-ibtracs.since1980.list.v04r01.csv
       filter: BASIN, SUBBASIN AND TRACK_TYPE='main'
       (drops PROVISIONAL, US-PROVISIONAL, spur-merge/split/other)
3. 1979-NA-CSGM-origins-ibtracs.since1980.list.v04r01.csv
       filter: TCG point (1st date/time row) AND
               NATURE in {TS, SS, MX} (POSITIVES)
4. 1979-NA-CSGM-negatives-origins-ibtracs.since1980.list.v04r01.csv
       filter: TCG point AND NATURE in {DS, NR} (NEGATIVES)
5. 1979-NA-CSGM-cascade-{YYYYMMDD-HHMMSSZ}.png
       cascade funnel diagram showing row counts at each filter step

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
PATH = '..\\..\\..\\data\\'

# Define the grid parameters
# _basin:    str = 'NA' # assumed, hardcoded for now
# _subbasin: str = 'GM' # assumed, hardcoded for now
# _south: float = 18 # NAGM Starting latitude (degrees)
# _north: float = 30 # NAGM Ending latitude (degrees)
# _west: float = 263 # NAGM Starting longitude (degrees)
# _east: float = 279 # NAGM Ending longitude (degrees)

_from_year: int = 1979
_basin: str = 'NA'                      # assumed, hardcoded for now
_subbasin: str = 'CSGM'                 # combined NACS + NAGM label (used for file naming)
_valid_subbasins: set = {'CS', 'GM'}    # IBTrACS SUBBASIN codes to accept
_south: float = 11.0   # combined MBR south (degrees)
_north: float = 32.0   # combined MBR north (degrees)
_west: float  = 263.0  # combined MBR west  (degrees, 0–360)
_east: float  = 305.0  # combined MBR east  (degrees, 0–360)

# Combined NACS + NAGM buffered MBR
# Raw union:  south=10, north=31, west=262, east=301
# 1° inset on sensitive edges to avoid boundary storm artefacts:
#   south: 10→11  (removes sparse southern genesis fringe)
#   north: 31→30  (clips US Gulf coast landfall/re-curve zone)
#   west:  262→263 (stays inside Gulf of Campeche genesis region)
#   east:  301→300 (trims eastern Caribbean edge noise)

# _basin:    str = 'WP' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _valid_subbasins: set = {'MM'}
# _south: float =  5 # WPMM Starting latitude (degrees)
# _north: float = 45 # WPMM Ending latitude (degrees)
# _west: float = 105 # WPMM Starting longitude (degrees)
# _east: float = 180 # WPMM Ending longitude (degrees)

# _basin:    str = 'MD' # assumed, hardcoded for now
# _subbasin: str = 'MD' # assumed, hardcoded for now
# _south: float = 30 # MDMD Starting latitude (degrees)
# _north: float = 46 # MDMD Ending latitude (degrees)
# _west: float = 5   # MDMD Starting longitude (degrees)
# _east: float = 36  # MDMD Ending longitude (degrees)

# _basin:    str = 'EP' # assumed, hardcoded for now
# # _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =  0 #  Starting latitude (degrees)
# _north: float = 30 #  Ending latitude (degrees)
# _west: float = 180 #  Starting longitude (degrees)
# _east: float = 270 #  Ending longitude (degrees)

# _basin:    str = 'SP' # assumed, hardcoded for now
# # _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float = -50 #  Starting latitude (degrees)
# _north: float = 0 #  Ending latitude (degrees)
# _west: float = 150 #  Starting longitude (degrees)
# _east: float = 270 #  Ending longitude (degrees)

# _basin:    str = 'SA' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =-30 #  Starting latitude (degrees)
# _north: float =  0 #  Ending latitude (degrees)
# _west: float = 317 #  Starting longitude (degrees)
# _east: float = 360 #  Ending longitude (degrees)

# _basin:    str = 'SI' # assumed, hardcoded for now
# _subbasin: str = 'WA' # assumed, hardcoded for now
# _south: float = -19.5 #  Starting latitude (degrees)
# _north: float = -3 #  Ending latitude (degrees)
# _west: float =  90 #  Starting longitude (degrees)
# _east: float = 135 #  Ending longitude (degrees)

# _basin:    str = 'SI' # assumed, hardcoded for now
# _subbasin: str = 'MM' # assumed, hardcoded for now
# _south: float =-35 #  Starting latitude (degrees)
# _north: float = -2 #  Ending latitude (degrees)
# _west: float =  33 #  Starting longitude (degrees)
# _east: float =  90 #  Ending longitude (degrees)

# _basin:    str = 'NI' # assumed, hardcoded for now
# _subbasin: str = 'AS' # AS, BB
# _south: float =  0 #  Starting latitude (degrees)
# _north: float = 25 #  Ending latitude (degrees)
# _west: float =  50 #  Starting longitude (degrees)
# _east: float =  77.5 #  Ending longitude (degrees)

# _basin:    str = 'NI' # assumed, hardcoded for now
# _subbasin: str = 'BB' # AS, BB
# _south: float =  0 #  Starting latitude (degrees)
# _north: float = 25 #  Ending latitude (degrees)
# _west: float =  77.5 #  Starting longitude (degrees)
# _east: float = 101 #  Ending longitude (degrees)

# round up to the nearest multiple of 3 hours
def round_up_to_nearest_3_hours(dt):
    if dt.minute != 0 or dt.second != 0 or dt.hour % 3 != 0:
        # Calculate hours to add
        hours_to_add = 3 - (dt.hour % 3)
        # Reset minutes and seconds to zero
        dt = dt.replace(minute=0, second=0, microsecond=0)
        # Add the hours
        dt += timedelta(hours=hours_to_add)
    return dt

def basin_grid(_basin: str, _subbasin: str) -> pd.DataFrame:
    # all storms origin file
    PATH = '..\\..\\..\\data\\'
    if _basin == 'MD':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-MD-MDorigins-test.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'NA':
        if _subbasin == 'CS':
            print()
            df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
            # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.ALL.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
            # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.ALL.list.v04r01-test2.csv", sep=',', header='infer', keep_default_na=False)
            # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.ALL.list.v04r01-test.csv", sep=',', header='infer', keep_default_na=False)
            # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.ALL.list.v04r01-2020.csv", sep=',', header='infer', keep_default_na=False)
            # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CS-origins-ibtracs.ALL.list.v04r01-2024.csv", sep=',', header='infer', keep_default_na=False)
        elif _subbasin == 'GM':
            df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-GM-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
        elif _subbasin == 'CSGM':
            # Combined NACS + NAGM: load the pre-merged origins file produced by pipeline when run with _subbasin='CSGM'.
            print (f'{PATH}1979-NA-CSGM-origins-ibtracs.since1980.list.v04r01.csv')
            df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NA-CSGM-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
        else:
            raise NotImplementedError(f'No such subbasin "{_subbasin}" in North Atlantic.')
    elif _basin == 'NI':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-NI-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'SI':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-SI-MM-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'WP':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-WP-MM-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'EP':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-EP-MM-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'SP':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-SP-MM-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    elif _basin == 'SA':
        df = pd.read_csv(filepath_or_buffer=f"{PATH}1979-SA-origins-ibtracs.since1980.list.v04r01.csv", sep=',', header='infer', keep_default_na=False)
    else:
        # df = pd.read_csv(filepath_or_buffer=f"{PATH}1979origins-lon-corrected.csv", sep=',', header='infer', keep_default_na=False)
        raise NotImplemented(f'None such {_basin} basin on planet Earth.')

    # For single subbasins keep the original equality check; for combined use isin
    _sub_filter = list(_valid_subbasins) if _subbasin == 'CSGM' else [_subbasin]
    df = df[(df['BASIN'] == _basin) & (df['SUBBASIN'].isin(_sub_filter))]
    # correct for some positive 180W+ longitudes
    df['LON'] = df['LON'].apply(lambda x: x + 360 if x < 0 else x)
    # round time up to 3hr multiple
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['ISO_TIME'] = df['ISO_TIME'].apply(round_up_to_nearest_3_hours)
    df['ISO_TIME'] = df['ISO_TIME'].dt.strftime('%Y-%m-%d %H:%M:%S') # IF NEEDED

    # print(f'df[{_basin}{_subbasin}]: {df.head()}')

    return df

def plot_cascade_funnel(cascade_log: list, run_ts: datetime,
                        basin: str, subbasin: str, out_path: str) -> None:
    """Render the filter cascade as a vertical funnel + branch chart.

    Trunk steps (level=0) stack vertically with width proportional to
    log(count) -- raw count would make the 700k first row dwarf the 410-row
    last step.  Branch steps (level=1) fan out below the last trunk step.
    """
    trunk    = [s for s in cascade_log if s.get('level', 0) == 0]
    branches = [s for s in cascade_log if s.get('level', 0) == 1]

    if not trunk:
        return

    # Use log-scaled widths so 700k and 410 are both visible
    max_count = max(s['count'] for s in cascade_log)
    min_count = max(1, min(s['count'] for s in cascade_log if s['count'] > 0))

    def width_for(count):
        # Map [log(min_count), log(max_count)] -> [2.5, 9.0]
        if count <= 0:
            return 0.6
        log_min = np.log10(max(1, min_count))
        log_max = np.log10(max_count)
        log_v   = np.log10(max(1, count))
        if log_max == log_min:
            return 6.0
        frac = (log_v - log_min) / (log_max - log_min)
        return 2.5 + 6.5 * frac

    # Layout config
    n_trunk      = len(trunk)
    n_branches   = max(1, len(branches))
    trunk_height = 0.55
    trunk_gap    = 0.30
    branch_gap   = 0.80
    branch_h     = 1.10  # taller to fit 3-line labels (NAME + NATURE + count)

    fig_h = max(7.5, 0.95 * n_trunk + 3.5)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    branch_colors = {
        'positive': '#2ca02c',  # green
        'negative': '#d62728',  # red
        'excluded': '#7f7f7f',  # grey
    }
    trunk_color = '#1f77b4'  # blue

    # Draw trunk steps top-to-bottom
    y_centers = []
    for i, step in enumerate(trunk):
        y = -i * (trunk_height + trunk_gap)
        y_centers.append(y)

        w = width_for(step['count'])
        x = -w / 2.0

        rect = mpatches.FancyBboxPatch(
            (x, y - trunk_height / 2.0), w, trunk_height,
            boxstyle='round,pad=0.02,rounding_size=0.05',
            linewidth=1.0, facecolor=trunk_color,
            edgecolor='black', alpha=0.85,
        )
        ax.add_patch(rect)

        # Label inside the bar
        label = step['label']
        count_str = f'{step["count"]:,}'
        text = f'{label}\n{count_str} rows'
        if step.get('dropped', 0) > 0:
            text += f'  (-{step["dropped"]:,})'

        ax.text(0, y, text, ha='center', va='center',
                fontsize=9, color='white', weight='bold')

        # Connector arrow to next trunk step
        if i + 1 < n_trunk:
            y_next = -(i + 1) * (trunk_height + trunk_gap)
            ax.annotate('', xy=(0, y_next + trunk_height / 2.0),
                        xytext=(0, y - trunk_height / 2.0),
                        arrowprops=dict(arrowstyle='-|>', color='black',
                                        lw=1.5))

    # Branches at the bottom -- positioned BELOW the last trunk step
    last_y = y_centers[-1] - trunk_height / 2.0
    branch_y_center = last_y - branch_gap - branch_h / 2.0

    if branches:
        # Spread horizontally; with 2 branches use wider spacing, 3 narrower
        if n_branches == 1:
            branch_x_centers = np.array([0.0])
        elif n_branches == 2:
            branch_x_centers = np.array([-3.0, 3.0])
        else:
            branch_x_centers = np.linspace(-4.5, 4.5, n_branches)

        for j, b in enumerate(branches):
            cx     = branch_x_centers[j]
            color  = branch_colors.get(b.get('branch', ''), '#7f7f7f')
            # Wider boxes so labels fit without clipping
            w_b    = max(3.5, width_for(b['count']) * 0.7)
            x_b    = cx - w_b / 2.0
            y_b    = branch_y_center
            rect_b = mpatches.FancyBboxPatch(
                (x_b, y_b - branch_h / 2.0), w_b, branch_h,
                boxstyle='round,pad=0.02,rounding_size=0.05',
                linewidth=1.0, facecolor=color,
                edgecolor='black', alpha=0.9,
            )
            ax.add_patch(rect_b)

            # Two-line label: short tag + NATURE codes on second line
            short_label = b['label'].split(' (')[0]   # 'POSITIVES'
            nature_part = ''
            if '(' in b['label']:
                nature_part = '(' + b['label'].split('(', 1)[1]
            label_text = (f'{short_label}\n{nature_part}\n{b["count"]:,} storms'
                          if nature_part else
                          f'{short_label}\n{b["count"]:,} storms')

            ax.text(cx, y_b, label_text, ha='center', va='center',
                    fontsize=8.5, color='white', weight='bold')

            # Connector from last trunk step to this branch
            ax.annotate('', xy=(cx, y_b + branch_h / 2.0),
                        xytext=(0, last_y),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=1.5, alpha=0.8))

    # Title with run metadata
    title = (f'IBTrACS Filter Cascade — basin={basin}, subbasin={subbasin}\n'
             f'Run: {run_ts.strftime("%Y-%m-%d %H:%M:%S UTC")}'
             f'  (widths log-scaled by row count)')
    ax.set_title(title, fontsize=11, weight='bold', pad=15)

    # Axes off, set limits to comfortably contain everything
    x_max_extent = 6.0
    ax.set_xlim(-x_max_extent, x_max_extent)
    y_min = branch_y_center - branch_h / 2.0 - 0.3
    y_max = trunk_height / 2.0 + 0.3
    ax.set_ylim(y_min, y_max)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    # Directory containing storm CSV files
    _path = f'{PATH}{_basin}{_subbasin}\\'

    # Run timestamp + cascade log for the funnel diagram at the end.
    # Each entry: {label, count, dropped (optional), notes (optional)}.
    run_started_utc = datetime.now(timezone.utc)
    cascade_log = []

    # URL, ibtracs file name and download path to save it
    url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/'
    # ibtracks: str = 'ibtracs.ALL.list.v04r01.csv'
    ibtracks: str = 'ibtracs.since1980.list.v04r01.csv'

    dl_ibtracs = f'{PATH}{ibtracks}'

    # Download the file
    response = requests.get(f'{url}{ibtracks}')
    with open(dl_ibtracs, 'wb') as file:
        file.write(response.content)

    # Check if the file exists
    if not os.path.exists(dl_ibtracs):
        # Download the file
        response = requests.get(f'{url}{ibtracks}')
        with open(dl_ibtracs, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded the file to {dl_ibtracs}")
    else:
        print(f"File already exists at {dl_ibtracs}, skipped download.")

    # Read the CSV file into a DataFrame to inspect columns
    df = pd.read_csv(dl_ibtracs, keep_default_na=False, na_values=None)

    # Read the CSV file again with dtype for columns 2 and 3 set to int
    # Print column names for debugging
    #print(f"Column names: {df.columns.tolist()}")
    # ['SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME', 'ISO_TIME', 'NATURE', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES',
    #  'WMO_AGENCY', 'TRACK_TYPE', 'DIST2LAND', 'LANDFALL', 'IFLAG', 'USA_AGENCY', 'USA_ATCF_ID', 'USA_LAT', 'USA_LON',
    #  'USA_RECORD', 'USA_STATUS', 'USA_WIND', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW',
    #  'USA_R34_NW', 'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW', 'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW',
    #  'USA_R64_NW', 'USA_POCI', 'USA_ROCI', 'USA_RMW', 'USA_EYE', 'TOKYO_LAT', 'TOKYO_LON', 'TOKYO_GRADE', 'TOKYO_WIND',
    #  'TOKYO_PRES', 'TOKYO_R50_DIR', 'TOKYO_R50_LONG', 'TOKYO_R50_SHORT', 'TOKYO_R30_DIR', 'TOKYO_R30_LONG',
    #  'TOKYO_R30_SHORT', 'TOKYO_LAND', 'CMA_LAT', 'CMA_LON', 'CMA_CAT', 'CMA_WIND', 'CMA_PRES', 'HKO_LAT', 'HKO_LON',
    #  'HKO_CAT', 'HKO_WIND', 'HKO_PRES', 'KMA_LAT', 'KMA_LON', 'KMA_CAT', 'KMA_WIND', 'KMA_PRES', 'KMA_R50_DIR',
    #  'KMA_R50_LONG', 'KMA_R50_SHORT', 'KMA_R30_DIR', 'KMA_R30_LONG', 'KMA_R30_SHORT', 'NEWDELHI_LAT', 'NEWDELHI_LON',
    #  'NEWDELHI_GRADE', 'NEWDELHI_WIND', 'NEWDELHI_PRES', 'NEWDELHI_CI', 'NEWDELHI_DP', 'NEWDELHI_POCI', 'REUNION_LAT',
    #  'REUNION_LON', 'REUNION_TYPE', 'REUNION_WIND', 'REUNION_PRES', 'REUNION_TNUM', 'REUNION_CI', 'REUNION_RMW',
    #  'REUNION_R34_NE', 'REUNION_R34_SE', 'REUNION_R34_SW', 'REUNION_R34_NW', 'REUNION_R50_NE', 'REUNION_R50_SE',
    #  'REUNION_R50_SW', 'REUNION_R50_NW', 'REUNION_R64_NE', 'REUNION_R64_SE', 'REUNION_R64_SW', 'REUNION_R64_NW',
    #  'BOM_LAT', 'BOM_LON', 'BOM_TYPE', 'BOM_WIND', 'BOM_PRES', 'BOM_TNUM', 'BOM_CI', 'BOM_RMW', 'BOM_R34_NE',
    #  'BOM_R34_SE', 'BOM_R34_SW', 'BOM_R34_NW', 'BOM_R50_NE', 'BOM_R50_SE', 'BOM_R50_SW', 'BOM_R50_NW', 'BOM_R64_NE',
    #  'BOM_R64_SE', 'BOM_R64_SW', 'BOM_R64_NW', 'BOM_ROCI', 'BOM_POCI', 'BOM_EYE', 'BOM_POS_METHOD', 'BOM_PRES_METHOD',
    #  'NADI_LAT', 'NADI_LON', 'NADI_CAT', 'NADI_WIND', 'NADI_PRES', 'WELLINGTON_LAT', 'WELLINGTON_LON',
    #  'WELLINGTON_WIND', 'WELLINGTON_PRES', 'DS824_LAT', 'DS824_LON', 'DS824_STAGE', 'DS824_WIND', 'DS824_PRES',
    #  'TD9636_LAT', 'TD9636_LON', 'TD9636_STAGE', 'TD9636_WIND', 'TD9636_PRES', 'TD9635_LAT', 'TD9635_LON',
    #  'TD9635_WIND', 'TD9635_PRES', 'TD9635_ROCI', 'NEUMANN_LAT', 'NEUMANN_LON', 'NEUMANN_CLASS', 'NEUMANN_WIND',
    #  'NEUMANN_PRES', 'MLC_LAT', 'MLC_LON', 'MLC_CLASS', 'MLC_WIND', 'MLC_PRES', 'USA_GUST', 'BOM_GUST', 'BOM_GUST_PER',
    #  'REUNION_GUST', 'REUNION_GUST_PER', 'USA_SEAHGT', 'USA_SEARAD_NE', 'USA_SEARAD_SE', 'USA_SEARAD_SW',
    #  'USA_SEARAD_NW', 'STORM_SPEED', 'STORM_DIR']
    # Convert columns 2 and 3 to numeric, coercing errors to NaN
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df['NUMBER'] = pd.to_numeric(df['NUMBER'], errors='coerce')
    print(f'1st [ALL] df: {len(df)}')
    print(f"\t...saved to {dl_ibtracs}")
    cascade_log.append({'label': 'Raw IBTrACS download',
                        'count': len(df), 'level': 0})

    df = df[(df['SEASON'] > _from_year)]
    print(f'2nd [>{_from_year}] df: {len(df)}')
    _csv = f'{PATH}{_from_year}-{ibtracks}'
    df.to_csv(_csv, index=False)
    cascade_log.append({'label': f'SEASON > {_from_year}',
                        'count': len(df), 'level': 0})
    print(f"\t...saved to {_csv}")

    # Filter the DataFrame based on SUBBASIN columns
    # For combined CSGM, accept both 'CS' and 'GM'; otherwise single equality check

    df = df[(df['BASIN'] == _basin)]
    df = df[df['SUBBASIN'].isin(_valid_subbasins)]
    print(f'3rd [{_basin}-{_subbasin}] df: {len(df)}')
    cascade_log.append({
        'label': f'BASIN={_basin}, SUBBASIN in {sorted(_valid_subbasins)}',
        'count': len(df), 'level': 0})

    # ----------------------------------------------------------------------
    # TRACK_TYPE filter: drop everything except 'main'.
    # IBTrACS v04 TRACK_TYPE values:
    #   'main'             -> reanalyzed canonical track     (KEEP)
    #   'PROVISIONAL'      -> real-time, may be revised      (DROP)
    #   'US-PROVISIONAL'   -> US-only data is provisional    (DROP)
    #   'spur-merge', 'spur-split', 'spur-other'
    #                       -> artifacts of storm splits/merges; duplicates
    #                          of `main` records              (DROP)
    # Filtering here, before saving the basin CSV, ensures every downstream
    # file (basin, origins, negatives-origins) is clean.
    # ----------------------------------------------------------------------
    ALLOWED_TRACK_TYPE = 'main'
    n_before_track = len(df)
    excluded_tt = df[df['TRACK_TYPE'] != ALLOWED_TRACK_TYPE]
    tt_breakdown = {}
    if len(excluded_tt) > 0:
        tt_breakdown = excluded_tt['TRACK_TYPE'].value_counts().to_dict()
        print(f'  TRACK_TYPE filter: dropped {len(excluded_tt)} non-main rows '
              f'({tt_breakdown})')
    df = df[df['TRACK_TYPE'] == ALLOWED_TRACK_TYPE]
    print(f'3b [TRACK_TYPE=main] df: {len(df)} (was {n_before_track})')
    cascade_log.append({
        'label': "TRACK_TYPE = 'main'",
        'count': len(df), 'level': 0,
        'dropped': len(excluded_tt),
        'notes':   tt_breakdown if tt_breakdown else None,
    })

    _csv = f'{PATH}{_from_year}-{_basin}-{_subbasin}-{ibtracks}'
    df.to_csv(_csv, index=False)
    print(f"\t...saved to {_csv}")

    # Keep only the first occurrence of each unique SID
    # df = df.drop_duplicates(subset='SID', keep='first')
    # Convert ISO_TIME to datetime
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    # Sort the filtered DataFrame by ISO_TIME in ascending order
    df = df.sort_values(by='ISO_TIME')

    # Drop duplicates based on SID, keeping the earliest ISO_TIME (first occurrence after sorting)
    df = df.drop_duplicates(subset='SID', keep='first')
    print(f'4th [TCG] df: {len(df)}')
    cascade_log.append({
        'label': 'Genesis points (one per storm)',
        'count': len(df), 'level': 0,
        'notes': 'drop_duplicates(SID, keep first)',
    })

    # ----------------------------------------------------------------------
    # NATURE-based split for ncio.py / ncio-negatives.py.
    # (TRACK_TYPE='main' filter already applied at step 3b above.)
    #
    # Per IBTrACS v04 NATURE definitions:
    #   TS = tropical, SS = subtropical, MX = mixed (inter-agency disagreement)
    #         -> POSITIVES (real TC genesis events)
    #   DS = disturbance, NR = not reported
    #         -> NEGATIVES (tracked but never reached TS intensity)
    #   ET = extratropical
    #         -> EXCLUDED (not a tropical genesis event)
    #
    # Outputs two origin CSVs:
    #   1979-{basin}-{subbasin}-origins-...csv               (positives)
    #   1979-{basin}-{subbasin}-negatives-origins-...csv     (negatives)
    # ----------------------------------------------------------------------
    POS_NATURES = {'TS', 'SS', 'MX'}
    NEG_NATURES = {'DS', 'NR'}

    df_pos   = df[df['NATURE'].isin(POS_NATURES)]
    df_neg   = df[df['NATURE'].isin(NEG_NATURES)]
    df_other = df[~df['NATURE'].isin(POS_NATURES | NEG_NATURES)]

    print(f'  Positives (NATURE in {sorted(POS_NATURES)}): {len(df_pos)} storms')
    print(f'  Negatives (NATURE in {sorted(NEG_NATURES)}): {len(df_neg)} storms')
    if len(df_other) > 0:
        other_natures = sorted(df_other['NATURE'].unique())
        print(f'  Excluded (NATURE in {other_natures}): {len(df_other)} storms')

    # Log the three branches at level=1 (so the funnel chart can fan them out)
    cascade_log.append({
        'label': f'POSITIVES (NATURE in {sorted(POS_NATURES)})',
        'count': len(df_pos), 'level': 1, 'branch': 'positive',
    })
    cascade_log.append({
        'label': f'NEGATIVES (NATURE in {sorted(NEG_NATURES)})',
        'count': len(df_neg), 'level': 1, 'branch': 'negative',
    })
    if len(df_other) > 0:
        other_natures = sorted(df_other['NATURE'].unique())
        cascade_log.append({
            'label': f'EXCLUDED (NATURE in {other_natures})',
            'count': len(df_other), 'level': 1, 'branch': 'excluded',
        })

    # Positives CSV (read by ncio.py)
    _csv_pos = f'{PATH}{_from_year}-{_basin}-{_subbasin}-origins-{ibtracks}'
    df_pos.to_csv(_csv_pos, index=False)
    print(f'\t...saved POSITIVES to {_csv_pos}')

    # Negatives CSV (read by ncio-negatives.py as IBTrACS DS/NR source)
    _csv_neg = f'{PATH}{_from_year}-{_basin}-{_subbasin}-negatives-origins-{ibtracks}'
    df_neg.to_csv(_csv_neg, index=False)
    print(f'\t...saved NEGATIVES to {_csv_neg}')

    # For backward compatibility with code that still inspects `df`:
    df = df_pos

    # ----------------------------------------------------------------------
    # Cascade visual: matplotlib funnel saved to PNG.
    # The PNG is dropped next to the CSVs in the data root and serves as a
    # reproducible methods-section figure.
    # ----------------------------------------------------------------------
    fig_path = (f'{PATH}{_from_year}-{_basin}-{_subbasin}-cascade-'
                f'{run_started_utc.strftime("%Y%m%d-%H%M%SZ")}.png')
    plot_cascade_funnel(cascade_log, run_started_utc, _basin, _subbasin,
                        out_path=fig_path)
    print(f'\t...saved cascade figure to {fig_path}')

    # Define the polygons
    # polygons = {
    #     'nasta': Polygon([(20,-60), (20,20), (45,20), (45,-60)]),
    #     'nata': Polygon([(0,-60), (0,20), (20,20), (20,-60)]),
    #     'naec': Polygon([(20, -80), (20, -60), (45, -60), (45, -85), (30, -85), (30, -80)]),
    #     'nacs': Polygon([(0,-75), (0,-60), (20,-60), (20,-90), (15,-90), (15,-85), (10,-85), (10,-75)]),
    #     'nagm': Polygon([(20,-100), (20,-95), (15,-95), (15,-90), (20,-90), (20,-80), (30,-80), (30,-85), (45,-85), (45,-100)])
    # }
    # # Load the dataset
    # DS = xr.open_dataset(path + file + ".nc", engine='netcdf4', drop_atmvars=['sea_ice_fraction', 'analysis_error', 'mask', 'crs'])
    #
    # # Apply the mask to the dataset using NumPy functions
    # mask = (20 < DS.lat) & (DS.lat < 45) & (-60 < DS.lon) & (DS.lon < 20)
    # DS = DS.where(mask, drop=True)

if __name__ == '__main__': main()