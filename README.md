# Predictive Tropical Cyclogenesis  Using Machine Learning:  A Novel Approach to Early Warning Systems

Source code accompanying the dissertation **_Predictive Tropical Cyclogenesis Using Machine Learning: A Novel Approach to Early Warning Systems_** (Nikolay Todorov, Chapman University).

This repository is a **curated release**: it contains the source code required to reproduce the results reported in the dissertation. It is not the full working tree — one-off data-repair utilities and exploratory scripts are intentionally excluded.

> If you use this code, please cite the dissertation and the archived release (see
> [Citation](#citation)).

---

## Overview

The system predicts a **calibrated genesis-location probability field** for tropical cyclogenesis
(TCG) across nine 6-hourly lead times (0–48 h). The model, `TemporalTCG`, is a ConvLSTM encoder with
temporal-attention pooling and a spatial-softmax head, trained with a Kullback–Leibler-divergence
objective on lead-time-varying Gaussian targets (positives) and a peakedness-suppression term
(negatives), under **leave-one-year-out (LOYO) cross-validation**.

Two basins are supported with a single architecture trained independently per basin:

| Basin  | Domain                                   | Grid      |
|--------|------------------------------------------|-----------|
| NACSGM | North Atlantic / Caribbean / Gulf — 11–32°N, 263–305°E | 85 × 169 @ 0.25° |
| WPMM   | Western Pacific — 5–45°N, 105–180°E      | 161 × 301 @ 0.25° |

Eight input channels: `sst, msl, cape, r, vo, d, vwsh, pi` (potential intensity via
[tcpyPI](https://github.com/dgilford/tcpyPI)). A trained model is delivered through an operational
early-warning demonstrator (`tcg_ews.py`).

---

## Repository structure

| Stage | File(s) | Role |
|-------|---------|------|
| Best-track assembly | `track.py` | Download IBTrACS; storm-catalog cascade (season, basin/sub-basin, main-track); genesis origins; positive/negative split |
| Positive features | `ncio.py`, `nci.py` | Extract the 8 ERA5 channels at each genesis point over 9 lead times |
| Negative sampling | `ncio-negatives.py`, `ncio-neg-wp.py` | Download ERA5 for non-developing systems; derive VWSH and PI |
| Normalization | `extremes.py`, `extremes-wp.py` | Per-channel min/max from the assembled dataset |
| Model, loss, training | `model.py`, `TCG-multilead-KL-NACSGM.py`, `TCG-multilead-KL-WP.py` | `TemporalTCG` architecture; KL + suppression loss; LOYO training/evaluation |
| Early-warning system | `tcg_ews.py`, `ews_opendata.py`, `ews_core.py`, `infer.py` | Real-time fetch, inference, post-processing, warning tiers |
| Visualization | `dashboard.py` | Interactive genesis-probability fields and warning overlays |

---

## Installation

```bash
git clone <REPOSITORY_URL>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python = 3.10 is recommended. A CUDA-capable GPU is recommended for training; inference runs on CPU.

---

## Data

No data are committed to this repository (licensing and size). You will need:

- **ERA5 reanalysis** — via the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).
  Create a free account and place your API key in `~/.cdsapirc`. ERA5 is the source used for all
  reproducible (historical) cases.
- **IBTrACS** — downloaded automatically by `track.py`.
- **HURDAT2** — the Atlantic best-track archive (public domain), used for negative-sample construction.
- **ECMWF open-data** — real-time forecasts only, via the `ecmwf-opendata` client. Note: the open-data
  portal retains only the **most recent ~2–3 days** of forecasts, so historical cases must use ERA5.

Model checkpoints are **not** in the git history; they are attached to the archived release
(see [Model weights](#model-weights)).

---

## Usage

**1. Build the storm catalog**
```bash
python track.py --basin NACSGM
```

**2. Extract features (positives and negatives)**
```bash
python ncio.py --basin NACSGM
python nci.py  --basin NACSGM
python ncio-negatives.py
python extremes.py
```

**3. Train (leave-one-year-out)**
```bash
python TCG-multilead-KL-NACSGM.py     # North Atlantic / Caribbean / Gulf
python TCG-multilead-KL-WP.py         # Western Pacific
```

**4. Run the operational early-warning system for one analysis time**
```bash
python tcg_ews.py --basin NACSGM --source era5 --time 2009-09-02T18:00 \
    --truth 16.3,299.0 --taper --ocean-fill --halo 6 --argmax component
```

### Reproduce the dissertation case studies (Figures 7.1–7.8)

All eight operational case studies are reproducible from ERA5 with:
```bash
python tcg_ews.py --basin <NACSGM|WPMM> --source era5 --time <time> \
    --truth <lat,lon> --taper --ocean-fill --halo 6 --argmax component
```

| Fig | Storm (year)   | Basin  | `--time` | `--truth` (lat, lon °E) |
|-----|----------------|--------|----------|---------------------|
| 7.1 | Klaus (1984)   | NACSGM | 1984-11-05T18:00 | 14.7, 291.2 |
| 7.2 | Fabian (1991)  | NACSGM | 1991-10-15T00:00 | 18.9, 274.3 |
| 7.3 | Erika (2009)   | NACSGM | 2009-09-02T18:00 | 16.3, 299.0 |
| 7.4 | Allison (1989) | NACSGM | 1989-06-24T18:00 | 27.0, 264.0 |
| 7.5 | Keith (1997)   | WPMM   | 1997-10-27T06:00 | 6.7, 168.7  |
| 7.6 | Rananim (2004) | WPMM   | 2004-08-06T00:00 | 15.3, 136.8 |
| 7.7 | Dujuan (2009)  | WPMM   | 2009-09-02T18:00 | 17.6, 130.1 |
| 7.8 | Kujira (2015)  | WPMM   | 2015-06-19T18:00 | 15.0, 112.2 |

> NACSGM longitudes are given in °E (0–360) to match the model grid. If `tcg_ews.py` expects signed
> longitudes, use the °W equivalents (-68.8, -85.7, -61.0, -96.0).

---

## Model weights

Trained LOYO checkpoints are archived with the release at **Zenodo: `<ZENODO_DOI>`**. Download and
place them where `tcg_ews.py` expects them (see the script's `--help`).

---

## Citation

```bibtex
@phdthesis{Todorov_TCG,
  author = {Todorov, Nikolay},
  title  = {Predictive Tropical Cyclogenesis Using Machine Learning:
            A Novel Approach to Early Warning Systems},
  school = {Chapman University},
  year   = {2026}
}
```
Software archive: **Zenodo DOI `<ZENODO_DOI>`**.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Acknowledgements

ERA5 data: Copernicus Climate Change Service / ECMWF. Best-track data: IBTrACS (NOAA NCEI) and HURDAT2 (NOAA NHC). Potential intensity computed with tcpyPI.
