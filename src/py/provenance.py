"""
provenance.py
=============
ERA5 vs ERA5T provenance for the TCG early-warning system.

Why this exists
---------------
The dashboard reads the *output* NetCDF (the model's genesis heatmaps), which by
itself doesn't know whether the *input* fields came from final ERA5 or preliminary
ERA5T. That fact has to be captured at fetch time and travel with the file. This
module does three small jobs:

  1. detect_era5_stream(ds)   -> classify a freshly-fetched ERA5 dataset as
                                 'ERA5' (final, expver 1), 'ERA5T' (preliminary,
                                 expver 5), 'mixed', or 'unknown'.
  2. stamp_provenance(...)    -> write analysis_time + era5_stream into the output
                                 dataset's global attrs before to_netcdf().
  3. format_provenance_caption(...) -> a one-line human string for the Streamlit
                                 dashboard (with a graceful fallback that infers
                                 the stream from file age for older outputs that
                                 predate this change).

No dependency on the rest of the EWS code — just numpy + (optionally) xarray-like
objects that expose .coords/.variables/.attrs.

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np

# ERA5 expver encoding: 1 = final/consolidated ERA5, 5 = preliminary ERA5T.
_FINAL = 1
_ERA5T = 5
# ERA5T is overwritten by final ERA5 ~2-3 months later; 92 days is the cutoff we
# use to *infer* the stream for old files that were written without a stamp.
_ERA5T_MAX_AGE_DAYS = 92


# --------------------------------------------------------------------------- #
#  1. Detect stream from a freshly-fetched ERA5 dataset
# --------------------------------------------------------------------------- #
def _expver_codes(ds) -> set:
    """Return the set of integer expver codes present in a fetched dataset.

    Handles expver as a dimension, a scalar coordinate, or a per-time array, and
    normalises the value whether it comes back as int (1), str ('1'), zero-padded
    str ('0001'), or bytes (b'0005') across old CDS and new CDS-Beta outputs.
    """
    holder = None
    if "expver" in getattr(ds, "coords", {}):
        holder = ds.coords["expver"]
    elif "expver" in getattr(ds, "variables", {}):
        holder = ds["expver"]
    elif "expver" in getattr(ds, "attrs", {}):
        holder = ds.attrs["expver"]
    if holder is None:
        return set()

    raw = getattr(holder, "values", holder)
    codes = set()
    for v in np.atleast_1d(np.asarray(raw).ravel()):
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode()
        if isinstance(v, str):
            v = v.strip()
        try:
            codes.add(int(v))
        except (ValueError, TypeError):
            pass  # unrecognised token -> ignore rather than misclassify
    return codes


def detect_era5_stream(ds) -> str:
    """Classify a fetched ERA5 dataset: 'ERA5' | 'ERA5T' | 'mixed' | 'unknown'.

    IMPORTANT: call this BEFORE collapse the expver dimension. Once we do
    `ds.sel(expver=5).combine_first(ds.sel(expver=1))` the provenance is gone.

    No expver present at all is treated as final 'ERA5' — consolidated archive
    data is served with the dimension already dropped.
    """
    codes = _expver_codes(ds)
    if not codes:
        return "ERA5"
    has_final, has_t = _FINAL in codes, _ERA5T in codes
    if has_final and has_t:
        return "mixed"
    if has_t:
        return "ERA5T"
    if has_final:
        return "ERA5"
    return "unknown"


# --------------------------------------------------------------------------- #
#  2. Stamp provenance into the output dataset (before to_netcdf)
# --------------------------------------------------------------------------- #
def stamp_provenance(out_ds, analysis_time: dt.datetime, era5_stream: str):
    """Write analysis_time + era5_stream into out_ds.attrs (in place) and return it."""
    if analysis_time.tzinfo is None:
        analysis_time = analysis_time.replace(tzinfo=dt.timezone.utc)
    out_ds.attrs["analysis_time"] = analysis_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_ds.attrs["era5_stream"] = era5_stream  # 'ERA5' | 'ERA5T' | 'mixed'
    out_ds.attrs["provenance_written"] = dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out_ds


# --------------------------------------------------------------------------- #
#  3. Read back + format for the dashboard
# --------------------------------------------------------------------------- #
def _parse_time(s) -> Optional[dt.datetime]:
    if not s:
        return None
    s = str(s)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_provenance(ds) -> dict:
    """Pull provenance from an output dataset's attrs, inferring stream if absent."""
    attrs = getattr(ds, "attrs", {}) or {}
    at = _parse_time(attrs.get("analysis_time"))
    stream = attrs.get("era5_stream")
    inferred = False
    if stream is None and at is not None:
        age = (dt.datetime.now(dt.timezone.utc) - at).days
        stream = "ERA5T" if age <= _ERA5T_MAX_AGE_DAYS else "ERA5"
        inferred = True
    return {"analysis_time": at, "era5_stream": stream, "inferred": inferred}


_LABELS = {
    "ERA5": "ERA5 (final, quality-controlled)",
    "ERA5T": "ERA5T (near-real-time, preliminary)",
    "mixed": "ERA5 + ERA5T (mixed window)",
}


def format_provenance_caption(ds=None, *, synthetic: bool = False) -> str:
    """One-line caption for the dashboard. Pass synthetic=True for the fallback."""
    if synthetic or ds is None:
        return "\u26a0\ufe0f Synthetic demonstration data \u2014 no ERA5 output file loaded."
    p = read_provenance(ds)
    label = _LABELS.get(p["era5_stream"], p["era5_stream"] or "unknown source")
    if p["inferred"]:
        label += " (inferred from file age)"
    at = p["analysis_time"]
    if at is None:
        return f"Source: {label}."
    age = (dt.datetime.now(dt.timezone.utc) - at).days
    return (
        f"Source: {label} \u00b7 analysis time "
        f"{at.strftime('%Y-%m-%d %H:%MZ')} \u00b7 {age} days behind real time."
    )
