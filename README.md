# ARPP

Predicting **Snow Water Equivalent (SWE)** and **Arctic Sea Ice Concentration (AICE)** up to 4 month ahead using DL models. Predictors come from the INM-CM numerical atmosphere model; targets are ERA5 reanalysis fields.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `lightgbm`, `xarray`, `netCDF4`, `cartopy`, `cdsapi`, `pygrib`.

---

## 1. Download — `scripts/download/`

Each script writes to a subfolder under `data/raw/`. All scripts skip files that already exist (idempotent re-runs).

### download_globsnow_archive_v3.py

**Source:** GlobSnow v3.0 — `https://www.globsnow.info/swe/archive_v3.0/L3A_daily_SWE/NetCDF4/`

**What it is:** Historical satellite-derived daily SWE product for the Northern Hemisphere. Level 3A gridded NetCDF4 files, 0.25° grid.

**How retrieved:** Scrapes the index page with BeautifulSoup, collects all `.nc` links, downloads each file via HTTP streaming.

**Output:** `data/raw/globsnow_archive_v3/` — one `.nc` file per day, 1 Jan 1979 – 21 May 2018.

---

### download_globsnow_nrt.py

**Source:** GlobSnow NRT — `https://www.globsnow.info/swe/nrt/{year}/data/`

**What it is:** Near-real-time daily SWE product continuing the archive, 2016–2025.

**How retrieved:** Same scrape-and-download approach, iterating over year subdirectories. Files are gzip-compressed (`.nc.gz`).

**Output:** `data/raw/globsnow_nrt/` — one `.nc.gz` file per day, 2016–2025.

---

### download_swe_era5.py

**Source:** Copernicus CDS — dataset `derived-era5-single-levels-daily-statistics`

**What it is:** ERA5 reanalysis daily mean **snow depth**. Ground-truth target for SWE prediction. Covers 1991–2025, global (90°N–0°), 0.25° grid.

**How retrieved:** `cdsapi` client, one request per year×month, each returning a NetCDF with daily values for that month.

**Output:** `data/raw/swe_era5/era5_snow_depth_{year}_{month}.nc`

---

### download_ice_era5.py

**Source:** Copernicus CDS — dataset `derived-era5-single-levels-daily-statistics`

**What it is:** ERA5 reanalysis daily mean **sea ice cover**. Ground-truth target for AICE prediction. Covers 2018–2025, global (90°N–0°), 0.25° grid.

**How retrieved:** Same `cdsapi` approach as above, one request per year×month.

**Output:** `data/raw/ice_era5/era5_ice_cover_{year}_{month}.nc`

---

### download_noaa_cdr_ice.py

**Source:** NOAA NSIDC — `https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/daily/`

**What it is:** NOAA CDR of passive microwave sea ice concentration v6. Daily Northern Hemisphere files, 25 km grid. Covers 1991–2025. Independent observational source alongside ERA5.

**How retrieved:** Scrapes the NSIDC HTTP directory year-by-year, stream-downloads each `.nc` file.

**Output:** `data/raw/noaa_cdr_ice/` — one `.nc` file per day.

---

### download_ghcn.py

**Source:** NOAA NCDC — `https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/`

**What it is:** GHCN Daily — in-situ station observations (temperature, precipitation, snow depth). One gzip-compressed CSV per year from 1991, plus `ghcnd-stations.csv`.

**How retrieved:** Scrapes the NCDC directory index, filters by year ≥ 1991 and `.csv.gz` extension, downloads each file plus the station list.

**Output:** `data/raw/ghcn/` — `{year}.csv.gz` files + `ghcnd-stations.csv`.

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">Не используется?<span>
</div>

---

### download_time_invariant.py

**Sources:** Copernicus CDS (ERA5 static fields) + Natural Earth shapefile

**What it is:** Two static datasets:
- **ERA5 static fields** (single request for 2025-01-01): `lake_cover`, `soil_type`, `high_vegetation_cover`, `low_vegetation_cover`, `type_of_high_vegetation`, `type_of_low_vegetation`, `geopotential`, `land_sea_mask`, `standard_deviation_of_orography`. Used as auxiliary spatial features in models.
- **Natural Earth boundaries** (Russian admin-0 shapefile): used for map visualizations.

**How retrieved:** `cdsapi` for ERA5; direct HTTP download for Natural Earth zip.

**Output:** `data/raw/time_invariant/era5_{variable}.nc` × 9 and `ne_10m_admin_0_countries_rus.zip`.

---

### download_seas_ecmwf.py

**Source:** Copernicus CDS — dataset `seasonal-original-single-levels`

**What it is:** ECMWF SEAS5 (system 51) seasonal forecast ensemble for **snow depth**, 2024–2025. Lead times 24 h–125 days (every 24 h). Domain: Eastern Europe / Western Russia (50–65°N, 27–56°E), 1° grid. Alternative forecast input alongside INM-CM.

**How retrieved:** `cdsapi`, one request per year×month. Each file contains the full ensemble across all lead times for that initialization month.

**Output:** `data/raw/swe_seas/seas_snow_depth_{year}_{month}.nc`

---

## 2. Prepare — `scripts/prepare/`

Transforms raw downloads into model-ready training data in four stages. All outputs use a common 0.25° lat/lon grid (0–90°N, −180–180°E), zlib-compressed NetCDF via `h5netcdf`. Scripts are idempotent — existing files are skipped.

### Stage 1 — Regrid & Harmonize

#### prepare_era5_swe.py
Splits monthly ERA5 files into one file per day. Flips the latitude axis (ERA5 is stored N→S, output is S→N). Converts snow depth from metres to millimetres (×1000). Renames variable to `swe`.

**Output:** `data/prepare/era5_swe/{YYYYMMDD}.nc` — variable `swe` [mm].

#### prepare_era5_ice.py
Same split-and-flip for sea ice cover (`siconc`). No unit conversion — values stay in [0, 1]. Renames variable to `ice`.

**Output:** `data/prepare/era5_ice/{YYYYMMDD}.nc` — variable `ice` [fraction].

#### prepare_globsnow_swe.py обработанный спутник
Regrids GlobSnow from its native irregular/polar grid to 0.25° lat/lon using **Delaunay triangulation + barycentric interpolation**. Regrid weights computed once and cached at `data/tmp/globsnow_weights.npz`. Applies ERA5 land/sea mask. Handles archive (pre-2018) and NRT (v1.0/v2.0) format differences.

**Output:** `data/prepare/globsnow/{YYYYMMDD}.nc` — variable `swe`.

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">Не используется?<span>
</div>

#### prepare_noaa_ice.py
Regrids NOAA CDR from its native **25 km polar stereographic (EPSG:3411)** to lat/lon using Delaunay triangulation. Handles sensor transitions (F08→F11→F13→F17→am2). Fills small NaN gaps (≤15 cells) via nearest-neighbour distance transform. Applies ERA5 ocean mask. Uses a producer–consumer pattern: worker pool processes files, two saver processes write outputs.

**Output:** `data/prepare/noaa_ice/{YYYYMMDD}.nc` — variable `ice`.

#### prepare_seas_swe.py
Averages over SEAS5 ensemble members, converts m→mm, regrids to 0.25° by linear interpolation, aggregates daily values to **weekly means** (up to 3-month lead time). Assigns lead time in months relative to the forecast initialization date.

**Output:** `data/prepare/swe_seas/all.nc` — single file indexed by `week`.

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">Не используется?<span>
</div>

#### prepare_time_invariant.py
Loads 9 time-invariant ERA5 fields, flips lat axis, crops to 40–90°N, and merges them into a single dataset.

**Output:** `data/train/time_invariant.nc`

---

### Stage 2 — Climatology

Climatology = per-period spatial **mean** and **linear trend** computed over the 1991–2020 reference period. Each output file stores `{var}_mean` (the normal) and `{var}_trend` (slope in units/year) per grid cell.

#### prepare_swe_climate.py
Groups daily files by ISO week number (52 weeks/year). For each week, stacks all years 1991–2020 and computes spatial mean and linear trend.

**Output:** `data/climate/week/era5_swe/{01..52}.nc` and `data/climate/week/globsnow/{01..52}.nc`

#### prepare_ice_climate.py
Same logic, grouped by calendar month.

**Output:** `data/climate/month/era5_ice/{01..12}.nc`

#### prepare_inmcm_climate.py
Computes weekly and monthly climatologies for all INM-CM forecast variables (16 weekly + 18 monthly variables). Handles 0–360° longitude convention (rolls to −180–180°).

**Output:** `data/climate/week/{element}/{01..52}.nc` and `data/climate/month/{element}/{01..12}.nc`

---

### Stage 3 — Anomalies

#### prepare_inmcm_merge.py
For each year×period, loads all INM-CM forecast files, subtracts the climatology mean, and assembles a file with a `lead_time` dimension (forecast horizon from initialization). Produces four streams:
- `inmcm_week` — weekly atmospheric variables
- `inmcm_swe` — weekly SWE
- `inmcm_month` — monthly atmospheric variables
- `inmcm_ice` — monthly ice/ocean variables (aice, hice, sst)

**Output:** `data/merge/inmcm_{week,swe,month,ice}/{YYYY}{PP}.nc`

#### prepare_merge.py
Aggregates daily ERA5 files to period means (weekly for SWE, monthly for ice), then subtracts the climatological normal to produce ERA5 anomalies used as ground-truth targets.

**Output:** `data/merge/era5_swe/{YYYY}{WW}.nc` and `data/merge/era5_ice/{YYYY}{MM}.nc`

---

### Stage 4 — Final Assembly

#### prepare_train.py
Central assembly script. For each variable and year:
1. **Bias correction** — computes mean systematic offset between INM-CM and ERA5 anomalies over training years → `bias/{period}.nc`
2. **Interpolation** — regrids INM-CM to the ERA5 reference grid
3. **Bias removal** — subtracts the stored bias from INM-CM anomalies.<span style="color: red;">*То есть в модель подаются уже скорректированные biasом аномалии? Уточнить подход*?<span>
4. **Clipping** — clips values: ice to [0, 1], SWE to [0, ∞]
5. **Merging** — joins INM-CM predictors, the target variable anomaly, and ERA5 ground truth into one dataset with a `lead_time` dimension
6. **Climate normals** — writes combined per-period climatology files

**Output:**
- `data/train/{variable}/anom/{YYYY}{PP}.nc` — all predictors + ERA5 target
- `data/train/{variable}/clim/{PP}.nc` — combined climatology
- `data/train/{variable}/bias/{PP}.nc` — bias correction map

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">Почему в train/anom разные сетки (интерполировано только swe)? Интерполяция в ClimateDataset.unify()?<span></div>

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">В train/anom только INMCM и ERA5?<span></div>

<div style="background-color: #f04506; padding: 10px; border-radius: 5px;box-shadow: 3px 3px 5px #888888">
<span style="color: black;">В train/climate только INMCM? Не вижу swe_era5 <span></div>

#### prepare_std.py
Computes per-variable, per-grid-cell **RMS** normalization statistics over 1991–2020. Categorical variables (`slt`, `tvh`, `tvl`) are one-hot encoded; geopotential is log-transformed; orography is min-max scaled.

**Output:** `data/train/aice/std.nc`, `data/train/swe/std.nc`, `data/train/time_invariant_norm.nc`

---

## 3. Dataset loading — `scripts/train.py`

### ClimateDataset

PyTorch `IterableDataset` that reads `data/train/` and yields `(X, y, lat)` batches. On construction it:
1. Builds the list of `anom/{year}{period}.nc` files for the requested years/periods
2. Loads `time_invariant.nc` (or `_norm` if `normed=True`)
3. Loads all `clim/{period}.nc` files and builds per-period validity masks
4. Loads `std.nc` and collapses per-cell RMS to a scalar per variable

Variables are split into three categories assembled differently:

| Category | Source | Notes |
|---|---|---|
| `anom_variables` | `anom/*.nc` | INM-CM predictors + ERA5 target; divided by RMS if `normed=True` |
| `time_invariant_variables` | `time_invariant.nc` | Loaded once, reused for every sample |
| `extra_variables` | Computed on the fly | See table below |

**Extra variables computed per sample:**

| Variable | What it is |
|---|---|
| `cos_lat` / `sin_lon` / `cos_lon` | Grid position without discontinuities |
| `sin_period` / `cos_period` | Seasonality as a continuous cycle |
| `year` | `(year − 2005) / 10` — normalized for trend awareness |
| `lead_time` | Forecast horizon as a scalar feature |
| `climate` | Climatological mean for the current period |
| `era` | ERA5 observation from the period just before the forecast window |

Default feature set includes `cos_lat`, `sin_lon`, `cos_lon`, `sin_period`, `cos_period`, `era`.

### Loader modes

| Mode | Output shape `(X, y, lat)` | Best for |
|---|---|---|
| `point` | `(batch, features)`, `(batch,)`, `(batch,)` | LightGBM, MLP |
| `map` | `(batch, lat, lon, features)`, `(batch, lat, lon)`, same | CNN, U-Net |
| `sequence` | `(seq, batch, features)`, `(seq, batch)`, same | LSTM, TCN |
| `sequence_map` | `(seq, batch, lat, lon, features)`, `(seq, batch, lat, lon)`, same | 3D U-Net |

### Caching

- **Level 0** — no cache; files opened fresh every epoch
- **Level 1** — caches each `(file, lead_time)` tensor pair after first load
- **Level 2** — caches fully assembled batches; from the second epoch the loader reads only from memory. Cache persists to `data/tmp/{name}.pkl` via `save_cache()` / `load_cache()`

### Loss functions

Both are **latitude-weighted** (`cos(lat)` per sample so polar cells contribute less):

- `loss_acc` — Anomaly Correlation Coefficient (ACC), primary metric. Cosine similarity between predicted and true anomaly fields. 1.0 = perfect, 0 = no skill.
- `loss_rmse` — weighted RMSE

### Dataset factory functions

```python
load_dataset(name)          # deserialize data/datasets/{name}.json → ClimateDataset
load_datasets(name)         # load {name}_train + {name}_test
make_train_test(variant, firstYear, separateYear, lastYear, args)
```

---

## 4. Models — `scripts/models.py`

All models share the interface: `fit(ds)`, `predict(X)`, `save(filepath)`, `load(filepath)`.

### Baselines

| Model | What it predicts |
|---|---|
| `ClimateModel` | Zero anomaly — "expect the climatological normal". Trivial floor: any useful model must beat this. |
| `BaseModel` | INM-CM's own forecast of the target variable, un-normalized. Persistence baseline. |
| `LinearRegression` | Closed-form OLS via `torch.linalg.solve`. Point loader only. |

### BoostingModel (LightGBM)

Wraps `lgb.train()` with defaults: 150 rounds, lr=0.06, 15 leaves, gbdt, feature_fraction=0.9, bagging_fraction=0.7. Point loader only. Saved with `joblib`.

### NeuralNetwork (training wrapper)

Wraps any `nn.Module` with:
- **Optimizer:** Adam
- **Loss:** latitude-weighted RMSE or ACC (`loss_type='rmse'|'acc'`)
- **Checkpointing:** saves every epoch to `{filepath}.model`; history snapshots every `history` epochs
- **`create_nn()`** — loads existing checkpoint if present, otherwise creates fresh (idempotent notebook cells)
- **`load_nn()`** — fully reconstructs model from checkpoint (stores `model_class` + `model_args`)

### Architecture classes

| Class | Loader | Input → Output |
|---|---|---|
| `MLPModel` | `point` | `(batch, features)` → `(batch, 1)` |
| `RNNModel` | `sequence` | `(seq, batch, features)` → `(batch, 1)` — bidirectional, last step |
| `LSTMModel` | `sequence` | `(seq, batch, features)` → `(batch, 1)` — multi-layer bidirectional, last step |
| `TCNModel` | `sequence` | `(seq, batch, features)` → `(batch, 1)` — 1D conv stack + AdaptiveMaxPool |
| `HybridTCNModel` | `sequence` | `(seq, batch, features)` → `(batch, 1)` — TCN encoder + MLP on last step |
| `CNNModel` | `map` | `(batch, lat, lon, features)` → `(batch, lat, lon)` — 2D conv stack, same padding |
| `CNN3DModel` | `sequence_map` | `(seq, batch, lat, lon, features)` → `(batch, lat, lon)` — 3D conv stack, last step |
| `UNetModel` | `map` | `(batch, lat, lon, features)` → `(batch, lat, lon)` — 2D U-Net, 3 levels, skip connections |
| `UNet3DModel` | `sequence_map` | `(seq, batch, lat, lon, features)` → `(batch, lat, lon)` — 3D U-Net, spatial pooling only |

**UNetModel** — 3 encoder/decoder levels, channels `b→2b→4b→8b` (bottleneck), skip connections via concatenation. Input padded to multiple of 8 with reflect padding.

**UNet3DModel** — same as 2D U-Net but with `Conv3d` and pooling only in spatial dims `(1,2,2)`, preserving the time axis through all levels. Returns last time step.

---

## 5. Evaluation — `scripts/experiment.py`

### save_experiment()

```python
save_experiment(name, model, ds, base=None)
```

Runs the model over the dataset, computes spatial metrics, and writes everything to `data/experiments/`:

| File | Contents |
|---|---|
| `{name}.model` | Model checkpoint |
| `{name}_ds.json` | Dataset config snapshot |
| `{name}_rmse.csv` | RMSE per grid cell (lat × lon) |
| `{name}_corr.csv` | Correlation per grid cell |
| `{name}_rmse.png` | Spatial RMSE map |
| `{name}_corr.png` | Spatial correlation map |
| `{name}_improve.png` | RMSE improvement over `base` (if provided) |
| `{name}_corr_improve.png` | Correlation improvement (SWE only) |
| `{name}_rmse_improve.png` | RMSE improvement (SWE only) |

Metric computation path:
- `map` / `sequence_map` loaders → `calculate_spatial_metrics()` (vectorized, full grid)
- `point` / `sequence` loaders → `gen_models_df()` + `calculate_metrics()` (group by lat/lon)

### Map projections

| Function | Projection | Domain | Used for |
|---|---|---|---|
| `draw_ice_map` | North Polar Stereographic | 65–90°N | Sea ice results |
| `draw_swe_map` | Lambert Conformal (100°E, 60°N) | 20–160°E, 46–80°N | Full SWE domain |
| `draw_swe_map_etr` | Plate Carrée | 27–56°E, 50–65°N | Eastern Europe / Russia subregion |

**Colormaps and ranges:**

| Shortcut | Metric | Colormap | Range |
|---|---|---|---|
| `draw_ice_map_rmse` | RMSE | RdBu_r | 0 – 0.5 |
| `draw_ice_map_corr` | Correlation | RdBu_r | −1 – 1 |
| `draw_ice_map_improve` | RMSE improvement | PiYG | −0.3 – 0.3 |
| `draw_swe_map_rmse` | RMSE | YlOrRd | 0 – 50 mm |
| `draw_swe_map_corr` | Correlation | RdBu_r | −1 – 1 |
| `draw_swe_map_rmse_improve` | RMSE improvement | PiYG | −15 – 15 mm |
| `draw_swe_map_corr_improve` | Correlation improvement | PiYG | −0.5 – 0.5 |
