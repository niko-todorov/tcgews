"""
ews/infer.py
============
Live inference against the ACTUAL trained model (TemporalTCG). It uses the
prototype (tcg_nacsgm_ews.py) only as the ERA5 download-and-assemble engine, and
does everything correctness-critical here:

  1. build the (1, 9, 8, H, W) window via proto.build_input_tensor
  2. REORDER channels from the prototype's order -> the training order
  3. per lead L, keep the newest (9 - L/6) frames ending at t0
     (matches training: fields = storms[:, :input_timestep+1])
  4. TemporalTCG -> logits -> mask land -> spatial softmax -> probability map
  5. write via ews_core.write_ews_netcdf (one schema, 0..360)

Setup: rename the prototype to ews/tcg_nacsgm_ews.py, put the model in
nc/model.py, and the all-years checkpoint at data/NACSGM/final/convlstm_nacsgm_best.pt.

Run:
    python -m ews.infer                       # newest ERA5 hour, NACSGM
    python -m ews.infer --time 2026-07-19T12  # explicit analysis time (UTC)

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# --- path bootstrap: parents[1] of this file is src/py -----------------------
_SRC_PY = Path(__file__).resolve().parents[1]
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))
# ---------------------------------------------------------------------------

import numpy as np

from basins import BASINS, LEAD_HOURS
from paths import final_dir
from ews import ews_core

# The model was trained on channels in THIS order (VARIABLES in the training
# script). The prototype's build_input_tensor stacks them in a DIFFERENT order.
TRAIN_VARS = ["sst", "msl", "cape", "r", "vo", "d", "vwsh", "pi"]
PROTO_VARS = ["sst", "vwsh", "d", "cape", "r", "msl", "vo", "pi"]
CHANNEL_PERM = [PROTO_VARS.index(v) for v in TRAIN_VARS]   # -> [0,5,3,4,6,2,1,7]


def _load_state_dict(ckpt: Path, torch):
    """Accept a raw state_dict or a checkpoint dict wrapping one."""
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "model_state_dict"):
            if key in obj:
                return obj[key]
    return obj


def run(
    basin_key: str = "NACSGM",
    analysis_time: datetime | None = None,
    ckpt: Path | None = None,
    device: str | None = None,
    base_filters: int = 32,
) -> Path:
    """Fetch ERA5 -> preprocess -> TemporalTCG -> write NetCDF into data/{basin}/ews."""
    if basin_key not in BASINS:
        raise KeyError(f"unknown basin {basin_key!r}; known: {list(BASINS)}")

    import torch
    import torch.nn.functional as F
    from ews import tcg_nacsgm_ews as proto
    from nc.model import TemporalTCG

    basin = BASINS[basin_key]

    # basins.py is the single source of truth for the domain -> push into proto.
    # The engine is basin-agnostic: same 8 channels/levels, only the box changes.
    proto.NACSGM.update(
        lat_north=basin.lat_max, lat_south=basin.lat_min,
        lon_west=basin.lon_min,  lon_east=basin.lon_max,
    )

    t0 = analysis_time or proto._latest_era5_time()
    extremes = proto.load_extremes(final_dir(basin.folder) / "extremes.csv")
    ckpt = (Path(ckpt) if ckpt
            else final_dir(basin.folder) / f"convlstm_{basin.folder.lower()}_best.pt")

    # 1: full 9-frame window (1, 9, 8, H, W), prototype channel order, True=ocean
    ncp = proto.fetch_ncp(t0)
    ncs = proto.fetch_ncs(t0)
    X, ocean = proto.build_input_tensor(ncp, ncs, t0, extremes)

    # 2: reorder channels prototype-order -> training-order
    X = X[:, :, CHANNEL_PERM, :, :]
    H, W = X.shape[-2], X.shape[-1]

    # model + checkpoint
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TemporalTCG(input_channels=len(TRAIN_VARS), base_filters=base_filters).to(dev)
    model.load_state_dict(_load_state_dict(ckpt, torch))
    model.eval()

    ocean_t = torch.as_tensor(np.asarray(ocean), dtype=torch.bool, device=dev)
    NEG = torch.tensor(-1e9, device=dev)

    prob = np.zeros((len(LEAD_HOURS), H, W), dtype="float32")
    with torch.no_grad():
        for li, L in enumerate(LEAD_HOURS):
            # 3: newest (9 - L/6) frames ending at t0
            x_in = X[:, L // 6:].to(dev)                    # (1, 9-L/6, 8, H, W)
            logits = model(x_in).squeeze(1).squeeze(0)      # (H, W)
            # 4: mask land, spatial softmax over the whole grid
            logits = torch.where(ocean_t, logits, NEG)
            p = F.softmax(logits.reshape(-1), dim=0).reshape(H, W)
            prob[li] = p.cpu().numpy()

    land = (~np.asarray(ocean)).astype("int8")              # writer wants 1=land

    # 5: single shared schema (0..360) into data/{basin}/ews
    out = ews_core.write_ews_netcdf(prob, land, basin, t0)
    print(f"wrote {out}  (leads={len(LEAD_HOURS)}, grid {W}x{H}, "
          f"analysis {t0:%Y-%m-%d %H:%M}Z)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TCG EWS (TemporalTCG) and write a NetCDF.")
    ap.add_argument("--basin", default="NACSGM")
    ap.add_argument("--time", default=None,
                    help="analysis time UTC, e.g. 2026-07-19T12 (default: newest ERA5)")
    ap.add_argument("--ckpt", default=None, help="path to the .pt checkpoint")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--base-filters", type=int, default=32)
    a = ap.parse_args()
    t0 = datetime.fromisoformat(a.time) if a.time else None
    run(a.basin, t0, Path(a.ckpt) if a.ckpt else None, a.device, a.base_filters)


if __name__ == "__main__":
    main()
