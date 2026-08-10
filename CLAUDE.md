# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARPP is a climate ML research project for predicting **Snow Water Equivalent (SWE)** and **Arctic Sea Ice Concentration (AICE)** 4+ weeks ahead using ensemble methods. Predictors come from the INM-CM numerical weather model; targets are ERA5 reanalysis fields.

## Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `lightgbm`, `xarray`, `netCDF4`, `cartopy`, `cdsapi`, `pygrib`.

No formal test suite or lint config exists. Run notebooks interactively via `jupyter notebook` from `/scripts/`.

## Data Pipeline

The pipeline runs in order:

1. **Download** — `scripts/download/` fetches raw ERA5, GlobSnow, NOAA CDR, and GHCN data into `data/raw/`
2. **Prepare** — `scripts/prepare/` harmonizes grids, computes anomalies, applies bias correction; outputs `data/prepare/` and `data/train/`
3. **Standardize** — `prepare_std.py` computes per-variable mean/std into `data/train/{variant}/std.nc`
4. **Train/Evaluate** — Jupyter notebooks in `scripts/` orchestrate experiments; results land in `data/experiments/`

Training data lives under `data/train/{variant}/` (variant = `swe` or `aice`):
- `anom/{year}{period}.nc` — anomaly files indexed by year + period
- `clim/` — climatological reference
- `std/` — normalization parameters

## Core Modules (`scripts/`)

**`models.py`** — All model classes:
- `ClimateModel` / `BaseModel`: climatology and persistence baselines
- `BoostingModel`: LightGBM wrapper (150 rounds, lr=0.06, 15 leaves)
- `NeuralNetwork`: base PyTorch wrapper with latitude-weighted RMSE/ACC loss, GPU support, and checkpointing
- Architecture subclasses: `MLPModel`, `LSTMModel`, `TCNModel`, `HybridTCNModel`, `CNNModel`, `UNetModel`, `UNet3DModel`
- `create_nn()` / `load_nn()`: factory functions for model creation and checkpoint loading

**`train.py`** — Dataset and training utilities:
- `ClimateDataset`: iterable PyTorch dataset with four loader modes (`point`, `map`, `sequence`, `sequence_map`), multi-lead-time support, optional time-invariant features, and two-level caching
- `load_dataset()` / `make_train_test()`: load pre-configured JSON datasets from `data/datasets/`
- `evaluate()`: computes ACC metric
- Loss is latitude-weighted (cosine of latitude)

**`experiment.py`** — Evaluation and visualization:
- `save_experiment()`: persists model, metrics, and map figures
- `draw_ice_map()` / `draw_swe_map()`: cartopy projections (polar stereographic for ice, Lambert conformal for SWE)

**`utils.py`** — Russian-language variable name/unit constants for 20 climate elements (used for plot labels).

## Domain Specifics

| Variable | Region | Temporal resolution | Periods/year | Lead times |
|----------|--------|--------------------|--------------| -----------|
| SWE | 20–160°E, 46–80°N | Weekly | 52 | 0–52+ weeks |
| AICE | 40–90°N (global) | Monthly | 12 | 0–12+ months |

Grid resolution: 0.25°. Training years: 1991–2024.

Anomalies are bias-corrected against ERA5 before training. Values are clipped: ice to [0, 1], SWE to [0, 50 mm].

## Notebooks

14 numbered notebooks (`001_` … `014_`) in `scripts/` cover the full research workflow — data exploration, baseline accuracy, feature selection, architecture experiments, and figure generation for the paper. Start with notebook `007` for model comparison and `012` for experiment tracking.

## Paths

All scripts use paths relative to the repo root. Run them from the repo root or adjust `sys.path` / working directory accordingly. No centralized config file exists; paths are hardcoded per script.
