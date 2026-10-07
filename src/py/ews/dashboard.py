"""
ews/dashboard.py
================
Streamlit early warning dashboard (NACSGM + WPMM).

Run from anywhere:
    streamlit run src/py/ews/dashboard.py
or as a module from src/py:
    python -m streamlit run ews/dashboard.py

The bootstrap below puts src/py on sys.path so `from common...` / `from ews...`
resolve whether launched by streamlit (script mode) or as a module.

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

# --- path bootstrap: parents[1] of this file is src/py -----------------------
_SRC_PY = Path(__file__).resolve().parents[1]
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from basins import BASINS, LEAD_HOURS
from ews import ews_core as core
from provenance import format_provenance_caption
import xarray as xr
import datetime as dt
from paths import ews_output_dir

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

CMAP = "viridis"     # dark blue/purple = 0.0 (low), yellow = 1.0 (high)


def _wrap180(x):
    """0..360 -> -180..180 for cartopy rendering (data stays 0..360 on disk)."""
    return np.where(np.asarray(x) > 180.0, np.asarray(x) - 360.0, np.asarray(x))

def list_runs(basin):
    """All EWS output files for a basin, newest first (by mtime)."""
    d = ews_output_dir(basin.folder)
    if not d.exists():
        return []
    return sorted(d.glob(f"{basin.key}_*.nc"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _run_label(path):
    """Human label from {key}_{source}_{YYYYMMDD}T{HH}Z[_variant...].nc.
    Handles post-processing variants (…Z_taper, …Z_taper_fill_halo6_component)."""
    parts = path.stem.split("_")
    ti = next((i for i, tok in enumerate(parts)
               if re.match(r"^\d{8}T\d{2}Z$", tok)), None)
    if ti is None:
        return path.stem
    source = "_".join(parts[1:ti]) if ti > 1 else "—"
    variant = " · ".join(parts[ti + 1:]) if len(parts) > ti + 1 else "raw"
    try:
        d = dt.datetime.strptime(parts[ti].replace("Z", ""), "%Y%m%dT%H")
        tstr = d.strftime("%Y-%m-%d %H:00Z")
    except ValueError:
        tstr = parts[ti]
    return f"{tstr} · {source} · {variant}"


def load_run(path):
    """Eager-load a chosen run (closes the file handle; Streamlit-safe)."""
    with xr.open_dataset(path, decode_timedelta=False) as _d:
        return _d.load()


def make_map(ds, basin, lead, theta, watches, truth=None, vmin=None, vmax=None):
    lon = _wrap180(ds["lon"].values)
    lat = ds["lat"].values
    prob = ds["prob"].sel(lead=lead).values
    if vmin is None:
        vmin = float(np.nanmin(prob))
    if vmax is None:
        vmax = float(np.nanmax(prob))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    ext_lon = _wrap180([basin.lon_min, basin.lon_max])
    extent = [float(ext_lon[0]), float(ext_lon[1]), basin.lat_min, basin.lat_max]
    h = 7.4 * (np.ptp(lat) / max(np.ptp(lon), 1e-6))

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(7.4, h))
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=proj)
        mesh = ax.pcolormesh(lon, lat, prob, transform=proj, cmap=CMAP,
                             vmin=vmin, vmax=vmax, shading="auto")
        ax.add_feature(cfeature.LAND, facecolor="#e9e7df", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#999", zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
        tf = dict(transform=proj)
    else:
        fig, ax = plt.subplots(figsize=(7.4, h))
        mesh = ax.pcolormesh(lon, lat, prob, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(basin.lat_min, basin.lat_max)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        tf = {}

    for w in watches:
        wlon = float(_wrap180(w.peak_lon))
        # Watch marker: circle area ∝ flagged-region area (km²).
        w_s = float(np.clip(w.area_km2 * 2.5e-4, 50.0, 4000.0))
        ax.scatter([wlon], [w.peak_lat], s=w_s, facecolors="none",
                   edgecolors="#e24b4a", linewidths=2.0, **tf)
        ax.text(wlon, w.peak_lat + 0.4, f"{w.peak_prob:.4f}",
                color="#a32d2d", fontsize=9, ha="center", **tf)

    if truth is not None:
        t_lat, t_lon = truth
        t_lon = float(_wrap180(t_lon))
        ax.scatter([t_lon], [t_lat], s=500, facecolors="none",
                   edgecolors=(0.486, 0.988, 0.0, 0.9), linewidths=2.5, zorder=5, **tf)
        # ax.text(t_lon, t_lat, "  obs", color="#00268f", fontsize=9,
        #         va="center", ha="left", zorder=6, **tf)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"P(genesis within {lead} h)")
    ax.set_title(f"{basin.name} — analysis {ds.attrs.get('analysis_time','?')} — +{lead} h")
    fig.tight_layout()
    return fig


def make_sparkline(ds, theta):
    leads, peaks = core.peak_curve(ds)
    fig, ax = plt.subplots(figsize=(4.2, 1.6))
    ax.plot(leads, peaks, "-o", color="#185fa5", ms=3, lw=1.4)
    ax.axhline(theta, color="#e24b4a", ls="--", lw=1)
    ax.set_ylim(0, 1); ax.set_xlim(0, 48)
    ax.set_xticks(list(LEAD_HOURS))               # 0,6,12,…,48 — the real leads
    ax.set_xlabel("lead (h)", fontsize=8); ax.set_ylabel("peak p", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="tcgews", layout="wide")
    st.title("tcgews: Tropical Cyclogenesis Early Warning System")

    with st.sidebar:
        st.header("Controls")
        basin_key = st.selectbox("Basin", list(BASINS.keys()),
                                 format_func=lambda k: f"{k} — {BASINS[k].name}")
        basin = BASINS[basin_key]
        runs = list_runs(basin)
        if runs:
            _ridx = st.selectbox(
                "Run", range(len(runs)),
                format_func=lambda i: _run_label(runs[i]) + (" · latest" if i == 0 else ""))
            _chosen = runs[_ridx]
        else:
            _chosen = None
        truth_str = st.text_input(
            "Ground-truth lat,lon (QA)", value="",
            help="Draw a circle at an observed genesis point; overrides any stored "
                 "truth in the file. e.g. 15.0,-78.0")
        lead = st.select_slider("Lead time (h)", options=LEAD_HOURS, value=0)

        # Load the chosen run now so the threshold and colour-scale sliders can
        # track the actual (sub-0.001) probability range instead of a fixed 0–1.
        if _chosen is not None:
            ds, synthetic = load_run(_chosen), False
        else:
            ds, synthetic = core.load_latest(basin)
        _pf = ds["prob"].sel(lead=lead).values
        fmin = float(np.nanmin(_pf)); fmax = float(np.nanmax(_pf))
        if not np.isfinite(fmax) or fmax <= fmin:
            fmax = fmin + 1e-6
        _step = max((fmax - fmin) / 200.0, 1e-7)

        # Threshold bounds span the whole run (max over all leads) so they do not
        # change with the lead time; Streamlit then preserves the chosen θ across
        # lead changes, exactly like the min-area slider.
        pmax_all = float(np.nanmax(ds["prob"].values))
        if not np.isfinite(pmax_all) or pmax_all <= 0.0:
            pmax_all = fmax
        _tstep = max(pmax_all / 200.0, 1e-7)
        theta = st.slider("Alert threshold θ", 0.0, pmax_all,
                          float(np.clip(basin.theta, 0.0, pmax_all)), _tstep, format="%.5f",
                          help=f"Run-wide probability max: {pmax_all:.5f} "
                               f"(per-basin default θ={basin.theta:.4f}); held "
                               f"constant across lead times.")
        vmin, vmax = st.slider("Colour scale (probability range)", fmin, fmax,
                               (fmin, fmax), _step, format="%.5f",
                               help="Both handles span the current field’s min–max.")
        min_area = st.slider("Min disturbance area (10³ km²)", 0, 200,
                             int(basin.min_area_km2 / 1000), 5) * 1000

    if synthetic:
        st.info(f"No output file found under data\\{basin.folder}\\ews for {basin_key} — "
                "showing synthetic demonstration data.")
    else:
        st.caption(format_provenance_caption(ds, synthetic=synthetic))

    # Ground-truth marker: from file attrs unless overridden in the sidebar.
    truth = None
    if not synthetic and "truth_lat" in ds.attrs and "truth_lon" in ds.attrs:
        truth = (float(ds.attrs["truth_lat"]), float(ds.attrs["truth_lon"]))
    if truth_str.strip():
        try:
            _la, _lo = (float(x) for x in truth_str.split(","))
            truth = (_la, _lo)
        except ValueError:
            st.warning("Ground-truth must be 'lat,lon', e.g. 15.0,-78.0 — ignoring.")
    watches = core.detect_watches(ds, lead, theta, min_area)
    overall_peak = max((w.peak_prob for w in watches),
                       default=float(ds["prob"].sel(lead=lead).values.max()))

    if watches:
        lines = "\n".join(
            f"- watch {i+1}: peak {w.peak_prob:.4f} near "
            f"{abs(w.peak_lat):.1f}°{'N' if w.peak_lat >= 0 else 'S'}, "
            f"{w.peak_lon % 360:.1f}°E "
            f"(~{w.area_km2/1000:.0f}k km²)"
            for i, w in enumerate(watches))
        st.error(f"**TCG watch — {len(watches)} active** at +{lead} h (θ={theta:.4f})\n\n{lines}")
    else:
        st.success(f"No alert at +{lead} h — peak probability "
                   f"{overall_peak:.4f} is below θ={theta:.4f}.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active watches", len(watches))
    c2.metric("Peak probability", f"{overall_peak:.4f}")
    c3.metric("Analysis time (UTC)", ds.attrs.get("analysis_time", "?"))

    left, right = st.columns([3, 1])
    with left:
        st.pyplot(make_map(ds, basin, lead, theta, watches, truth=truth, vmin=vmin, vmax=vmax))
        st.caption("© 2026 Nikolay Todorov")
    with right:
        st.caption("Peak probability vs lead time")
        st.pyplot(make_sparkline(ds, theta))
        st.caption(f"Grid {basin.nx}×{basin.ny} · model {ds.attrs.get('model','?')}")


if __name__ == "__main__":
    main()
