"""
ews/ews_opendata.py
===================
Near-real-time input adapter for the TCG EWS: a drop-in replacement for the
ERA5T fetch (`fetch_ncp` + `fetch_ncs`) that pulls ECMWF **IFS Open Data**
instead. IFS analyses are available within a couple of hours (vs ERA5T's ~5-day
lag), so t0 moves close to "now" and the 0–48 h forecast finally covers *future*
time — the whole point of an early-warning system.

Design
------
This adapter does NOT touch the model or the tensor-assembly code. It downloads
IFS Open Data and writes two NetCDFs in the **exact ERA5-CDS schema** that the
existing `build_input_tensor` / `ncvwsh2csv` / `pi2csv` already read:

    ncp (pressure levels):  vars vo,d,r,t,q,u,v   dims (time, level, lat, lon)
    ncs (single level):     vars sst,msl,cape      dims (time, lat, lon)

So `run_ews` only swaps which fetch function it calls; everything downstream —
vwsh, tcpyPI, min-max normalisation with the ERA5 extremes.csv — is unchanged.

Three honest caveats (documented; see README notes at the bottom):
  1. SST -> skt proxy. IFS Open Data ships no SST, so we use skin temperature
     (skt) over ocean as a stand-in and mask land with NaN via the land-sea mask
     (lsm), which preserves the `land_mask = isfinite(sst)` logic. Over ocean
     skt ~= SST to a few tenths K, but it is a different field from ERA5 SST.
  2. Potential-intensity fidelity. tcpyPI integrates the T/q column; IFS Open
     Data has 9 pressure levels (1000,925,850,700,500,300,250,200,50) vs ERA5's
     27, so the `pi` channel is coarser. It still computes; it is not identical.
  3. Domain shift. The model was trained on ERA5; IFS operational analysis has
     different biases. This adapter is exactly the tool to *quantify* that — run
     both sources over the same past date and compare (see the A/B recipe in the
     module notes / the accompanying write-up).

Requires: ecmwf-opendata, cfgrib (eccodes), xarray, numpy.

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# Make basins.py importable (find the dir that holds it), for the crop box.
import sys
for _up in Path(__file__).resolve().parents:
    if (_up / "basins.py").exists():
        if str(_up) not in sys.path:
            sys.path.insert(0, str(_up))
        break
from basins import BASINS
from paths import shared_cache_dir

log = logging.getLogger("ews.opendata")

# ── What IFS Open Data provides (verified against ecmwf-opendata param list) ──
IFS_PLEVELS   = [1000, 925, 850, 700, 500, 300, 250, 200, 50]
IFS_PL_PARAMS = ["t", "q", "u", "v", "vo", "d", "r"]     # short names match ERA5
IFS_SFC_PARAMS = ["msl", "skt", "mucape", "lsm"]         # skt->sst, lsm->land mask,
#                                                          mucape->cape (see note below)

LEAD_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48]          # 9 snapshots, t0-48h..t0


def _box_for_basin(basin_key: str) -> tuple[float, float, float, float]:
    """(lat_north, lat_south, lon_west, lon_east) in 0..360, from basins.py."""
    b = BASINS[basin_key]
    return (b.lat_max, b.lat_min, b.lon_min, b.lon_max)


# --------------------------------------------------------------------------- #
#  Time / cycle helpers
# --------------------------------------------------------------------------- #
def _window_times(t0: datetime) -> list[datetime]:
    """The 9 analysis times t0-48h .. t0 (ascending), tz-naive UTC."""
    t0 = t0.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    return sorted(t0 - timedelta(hours=h) for h in LEAD_HOURS)


def _oper_cycle_step(vt: datetime) -> tuple[datetime, int]:
    """Map a 6-hourly valid time to (oper base cycle 00/12 UTC, step in {0,6}).

    06z/18z frames are taken as the preceding 00z/12z run at +6h, so we only ever
    touch the well-retained 00/12z 'oper' runs and never the short-lived 'scda'
    06/18z runs (which the portal prunes early → 404). Cost: the 06/18z frames are
    +6 h forecasts rather than analyses — a small extra inconsistency vs ERA5,
    bounded and documented."""
    base_hour = 12 if vt.hour >= 12 else 0
    base = vt.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    step = (vt - base).seconds // 3600      # 0 for 00/12z, 6 for 06/18z
    return base, step


def latest_ifs_analysis(now: Optional[datetime] = None, safety_hours: int = 8) -> datetime:
    """Most recent 6-hourly cycle whose step-0 analysis should be published.
    IFS dissemination latency is a few hours; `safety_hours` backs off to stay
    safely inside the published window."""
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    t = now - timedelta(hours=safety_hours)
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)


# --------------------------------------------------------------------------- #
#  Download — always from the 00/12z 'oper' run, step 0 or 6
# --------------------------------------------------------------------------- #
def _retrieve(client, vt: datetime, levtype: str, params, levelist, target: Path):
    """Retrieve fields valid at `vt` from the appropriate oper cycle/step."""
    cycle, step = _oper_cycle_step(vt)
    kwargs = dict(
        date=cycle.strftime("%Y%m%d"), time=cycle.hour, step=step,
        type="fc", stream="oper",
        levtype=levtype, param=list(params), target=str(target),
    )
    if levelist is not None:
        kwargs["levelist"] = list(levelist)
    log.info("IFS oper %s %s %02dZ +%dh (valid %s) -> %s", levtype,
             cycle.strftime("%Y-%m-%d"), cycle.hour, step,
             vt.strftime("%Y-%m-%d %H:%MZ"), target.name)
    client.retrieve(**kwargs)


# --------------------------------------------------------------------------- #
#  GRIB -> ERA5-schema remap (the part that must be exactly right)
# --------------------------------------------------------------------------- #
def _subset_box(ds: xr.Dataset, box: tuple) -> xr.Dataset:
    """Crop to `box` = (lat_n, lat_s, lon_w, lon_e) in 0..360, robust to longitude
    convention. IFS Open Data delivers longitude as -180..180; basins.py uses
    0..360. We normalise to 0..360 and sort ascending first, so the lon slice
    selects the region whichever convention the GRIB used. Latitude descends."""
    lat_n, lat_s, lon_w, lon_e = box
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.sel({lat: slice(lat_n, lat_s)})                  # descending
    ds = ds.assign_coords({lon: ds[lon] % 360.0}).sortby(lon)
    ds = ds.sel({lon: slice(lon_w, lon_e)})                  # 0..360
    return ds


def _pl_to_era5(raw: xr.Dataset, valid_time: np.datetime64, box: tuple) -> xr.Dataset:
    """Pressure-level GRIB (typeOfLevel=isobaricInhPa) -> ERA5 ncp schema."""
    ds = _subset_box(raw, box)
    if "isobaricInhPa" in ds.coords:
        ds = ds.rename({"isobaricInhPa": "level"})
    ds = ds[[v for v in IFS_PL_PARAMS if v in ds.data_vars]]
    ds = ds.expand_dims(time=[valid_time])
    # drop scalar grib bookkeeping coords that would clash on concat
    return ds.drop_vars([c for c in ("step", "valid_time", "number", "surface")
                         if c in ds.coords], errors="ignore")


def _sfc_to_era5(var_das: dict, valid_time: np.datetime64, box: tuple) -> xr.Dataset:
    """Single-level fields (already picked by name across typeOfLevel groups) ->
    ERA5 ncs schema, deriving sst from skt with land masked to NaN."""
    for req in ("msl", "skt", "lsm"):
        if req not in var_das:
            raise ValueError(f"IFS single-level fetch missing '{req}'")
    # IFS Open Data ships CAPE as 'mucape' (most-unstable CAPE), not 'cape'.
    # We rename it to 'cape' so extremes.csv / build_input_tensor are unchanged.
    # NOTE: mucape (most-unstable parcel) differs from ERA5's surface/mixed CAPE —
    # a third proxy substitution alongside skt->sst and the 9-level pi.
    cape_key = "mucape" if "mucape" in var_das else ("cape" if "cape" in var_das else None)
    if cape_key is None:
        raise ValueError(
            "IFS single-level fetch has no CAPE field (looked for 'mucape'/'cape'). "
            "Check the ECMWF parameter DB or adjust IFS_SFC_PARAMS before running.")

    skt = _subset_box(var_das["skt"].to_dataset(name="skt"), box)["skt"]
    lsm = _subset_box(var_das["lsm"].to_dataset(name="lsm"), box)["lsm"]
    msl = _subset_box(var_das["msl"].to_dataset(name="msl"), box)["msl"]
    cape = _subset_box(var_das[cape_key].to_dataset(name="cape"), box)["cape"]

    # sst proxy: skin temperature over ocean (lsm<0.5), NaN over land — this keeps
    # build_input_tensor's `land_mask = isfinite(sst)` working unchanged. Kelvin
    # is preserved (the pipeline converts >200K to degC itself).
    sst = skt.where(lsm < 0.5)

    ds = xr.Dataset(dict(sst=sst, msl=msl, cape=cape))
    ds = ds.expand_dims(time=[valid_time])
    return ds.drop_vars([c for c in ("step", "valid_time", "number", "surface")
                         if c in ds.coords], errors="ignore")


def _open_pl(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})


def _open_sfc_vars(path: Path) -> dict:
    """cfgrib splits single-level GRIB by typeOfLevel; collect wanted vars across
    the returned datasets into {name: DataArray}."""
    import cfgrib
    dsets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    out: dict = {}
    for ds in dsets:
        for v in IFS_SFC_PARAMS:
            if v in ds.data_vars and v not in out:
                out[v] = ds[v]
    return out


def _pick_var(path: Path, name: str) -> xr.DataArray:
    """Pull a single named variable (e.g. the static 'lsm') from a GRIB file."""
    import cfgrib
    for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
        if name in ds.data_vars:
            return ds[name]
    raise ValueError(f"'{name}' not found in {path.name}")


# Per-frame single-level fields (time-varying). lsm is static and fetched once.
IFS_SFC_DYNAMIC = ["msl", "skt", "mucape"]


# --------------------------------------------------------------------------- #
#  Public entry point — drop-in for fetch_ncp + fetch_ncs
# --------------------------------------------------------------------------- #
def fetch_ifs_opendata(
    t0: datetime,
    cache_dir: Path,
    basin_key: str = "NACSGM",
    overwrite: bool = False,
    source: str = "ecmwf",
    grib_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Fetch the t0-48h..t0 window from IFS Open Data for `basin_key` and write two
    NetCDFs in ERA5-CDS schema. Returns (ncp_path, ncs_path), same as the ERA5
    fetch pair. The crop box comes from basins.py via _box_for_basin().

    GRIBs are whole-world downloads, so they cache in a shared, basin-agnostic
    `grib_dir` (default: data/_cache/ifs) and are reused across basins. Only the
    small basin-cropped NetCDFs are written per-basin into `cache_dir`."""
    from ecmwf.opendata import Client

    box = _box_for_basin(basin_key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    grib_dir = Path(grib_dir) if grib_dir is not None else shared_cache_dir("ifs", create=True)
    grib_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{basin_key}_{t0.strftime('%Y%m%d_%H')}"
    ncp_path = cache_dir / f"ifs_ncp_{tag}.nc"
    ncs_path = cache_dir / f"ifs_ncs_{tag}.nc"
    if ncp_path.exists() and ncs_path.exists() and not overwrite:
        log.info("IFS Open Data cache hit: %s / %s", ncp_path.name, ncs_path.name)
        return ncp_path, ncs_path

    client = Client(source=source)
    window = _window_times(t0)

    # Land-sea mask is static — fetch once (step 0 of t0's base cycle) and reuse.
    glsm = grib_dir / "_ifs_lsm.grib2"
    base0, _ = _oper_cycle_step(window[-1])
    if overwrite or not glsm.exists():
        _retrieve(client, base0, "sfc", ["lsm"], None, glsm)
    lsm_raw = _pick_var(glsm, "lsm")

    pl_frames, sfc_frames = [], []
    for when in window:
        vt = np.datetime64(when, "ns")
        gpl = grib_dir / f"_ifs_pl_{when:%Y%m%d_%H}.grib2"
        gsf = grib_dir / f"_ifs_sf_{when:%Y%m%d_%H}.grib2"
        if overwrite or not gpl.exists():
            _retrieve(client, when, "pl", IFS_PL_PARAMS, IFS_PLEVELS, gpl)
        if overwrite or not gsf.exists():
            _retrieve(client, when, "sfc", IFS_SFC_DYNAMIC, None, gsf)
        sfc_vars = _open_sfc_vars(gsf)
        sfc_vars.setdefault("lsm", lsm_raw)          # inject the static mask
        pl_frames.append(_pl_to_era5(_open_pl(gpl), vt, box))
        sfc_frames.append(_sfc_to_era5(sfc_vars, vt, box))

    ds_plev = xr.concat(pl_frames, dim="time").sortby("time")
    ds_sfc  = xr.concat(sfc_frames, dim="time").sortby("time")

    # Provenance so the dashboard/attrs can tell this apart from ERA5.
    for ds in (ds_plev, ds_sfc):
        ds.attrs["input_source"] = "ECMWF IFS Open Data (oper 00/12z, step 0/6)"
        ds.attrs["sst_note"] = "sst = skin temperature over ocean (proxy)"

    ds_plev.to_netcdf(ncp_path)
    ds_sfc.to_netcdf(ncs_path)
    log.info("IFS Open Data written: %s (%s) / %s",
             ncp_path.name, list(ds_plev.data_vars), ncs_path.name)
    return ncp_path, ncs_path
