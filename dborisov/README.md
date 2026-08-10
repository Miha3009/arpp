# ARPP v2

## patcher

C++ library (Python bindings via pybind11) for efficient storage and retrieval of geospatial field patches with ZSTD compression. Requires C++17, LibTorch, and libzstd. Build with `./patcher/build.sh`.

Data is stored in a flat binary file (`{element}.bin`) alongside an index file (`{element}.idx`). The index maps `PatchKey{x0, y0, t0, xyStep, tStep, tag}` to a byte offset in the binary. After `aggregate()`, the database holds anomalies, climatology, and multi-resolution downsampled versions all in the same file.

---

### Classes

**`patcher.Context(directory: str, num_workers: int)`**

Loads `config.bin` and all `.idx` files from `directory`, and starts a thread pool of `num_workers` for parallel data loading. One `Context` is reused for the lifetime of the training process.

Default config values (written to `config.bin` on first run):

| Parameter | Default | Meaning |
|---|---|---|
| `patch_size_h` | 32 | Spatial patch height in grid cells |
| `patch_size_w` | 32 | Spatial patch width in grid cells |
| `patch_size_t` | 4 | Temporal patch size in days |
| `dict_size` | 32768 | ZSTD dictionary size in bytes |
| `compression_level` | 5 | ZSTD compression level |
| `grid_sizes` | 20 | Bitmask of xyStep scales to build (20 = 0b10100 → scales 4 and 16) |

**`patcher.Request(element, x0, y0, t0, xSize, ySize, tSize, xyStep, tStep [, tag=0])`**

Describes a single data fetch. Coordinates are in grid pixels at scale 1 (i.e. 0.25° for ERA5). The library internally maps to the appropriate downsampled patch.

| Parameter | Type | Description |
|---|---|---|
| `element` | str | Variable name, e.g. `"t2m"`, `"inm_t2m"` |
| `x0`, `y0` | int | Top-left corner in full-resolution pixels |
| `t0` | str | Start date `YYYYMMDD` |
| `xSize`, `ySize` | int | Output patch dimensions in downsampled cells |
| `tSize` | int | Number of time steps to return |
| `xyStep` | int | Spatial downsampling factor (1 = full res) |
| `tStep` | int | Temporal stride in days |
| `tag` | int | Forecast start month for INM-CM (default 0) |

---

### Functions

**`patcher.train_dict(context, data, element, precision, timeInvariant=False)`**

Trains a ZSTD compression dictionary from a list of sample tensors. Must be called once before the first `save()`. Stores the dictionary and initialises the element's config entries (`map_size_h`, `map_size_w`, `precision`, etc.) in `{element}.idx`.

- `data` — list of `[H, W]`, `[T, H, W]`, or `[E, T, H, W]` float32 tensors
- `precision` — quantization step size (e.g. 0.05 K for temperature). Controls storage size vs. accuracy trade-off.
- `timeInvariant` — if `True`, data is stored as a single snapshot with no time dimension (for static fields like `lsm`)

**`patcher.save(context, data, dates, element, tag=0)`**

Appends tensors to `{element}.bin`. Each tensor is split into `patch_size_h × patch_size_w` spatial tiles, quantized to uint16 (delta-encoded, min stored as header), ZSTD-compressed with the element's dictionary, and appended to the binary file. The index is updated with byte offsets.

- `data` — list of tensors, one per date
- `dates` — list of `YYYYMMDD` strings, same length as `data`
- `tag` — used to separate INM-CM forecasts by start month

**`patcher.aggregate(context, element, climateStart, climateEnd)`**

One-shot post-processing step called after all data is saved. Does three things in sequence:

1. **Climate** — averages patches over `climateStart`–`climateEnd` by calendar day (MMDD). Climate entries stored with `tStep=0` key.
2. **Anomalies** — rewrites every time-step patch as `(raw − climate[MMDD])`. If `climateStart == climateEnd == ""`, skips climate computation (for time-invariant fields).
3. **Multi-scale pyramid** — pools the full-resolution map spatially at each scale indicated by the `grid_sizes` bitmask, and temporally averages across `patchT`-day blocks. Pre-built downsampled patches enable efficient multi-resolution data loading at training time.

Pass `climateStart=""`, `climateEnd=""` for time-invariant fields (no anomaly computation, only spatial pyramid).

**`patcher.load(context, requests) → list[Tensor]`**

Loads anomaly data for a list of `Request` objects in parallel using the thread pool. Returns one tensor per request:
- ERA5 variable → `[tSize, ySize, xSize]`
- INM-CM variable → `[E, tSize, ySize, xSize]` (E = ensemble size at that date)

Handles longitude wrap-around and latitude boundary extrapolation automatically. Returns `NaN` where no data exists for a given date.

**`patcher.load_climate(context, requests) → list[Tensor]`**

Same interface as `load()` but returns the climatological mean `[tSize, ySize, xSize]` for each request, averaged over the configured climate period. Returns zeros for time-invariant fields.

---

## scripts/download/

### download_era5.py

Downloads ERA5 daily reanalysis fields from the CDS API into `era5/{variable}/{YM}.nc`.

**Variables downloaded:**

| Short name | CDS variable | Dataset |
|---|---|---|
| `t2m` | `2m_temperature` | single-levels |
| `tp` | `total_precipitation` | single-levels |
| `sd` | `snow_depth` | single-levels |
| `sden` | `snow_density` | single-levels |
| `t2m_min` | `minimum_2m_temperature_since_previous_post_processing` | single-levels |
| `t2m_max` | `maximum_2m_temperature_since_previous_post_processing` | single-levels |
| `pt` | `precipitation_type` | single-levels |
| `h500` | `geopotential` @ 500 hPa | pressure-levels |
| `t850` | `temperature` @ 850 hPa | pressure-levels |

**Parameters:**
- Period: 1980–2026 (daily mean, UTC+0, 6-hourly source frequency)
- Domain: global (90°N – 0°N, 180°W – 180°E), 0.25° grid
- Concurrency: up to 5 parallel CDS requests

**Output:** `era5/{variable}/{YYYYMM}.nc` — one file per variable per month

---

### download_time_invariant.py

Downloads static (time-invariant) fields used as spatial features.

**Downloads:**

| Source | Output | Contents |
|---|---|---|
| CDS ERA5 | `data/era5_time_invariant.nc` | geopotential · land_sea_mask · soil_type · high/low vegetation cover · type of high/low vegetation · standard deviation of orography · lake cover |
| Natural Earth | `data/ne_10m_admin_0_countries_rus.zip` | Country borders shapefile (used for map visualization) |

**Parameters:**
- ERA5 request: single snapshot (2025-01-01), global domain, same 0.25° grid as ERA5 dynamic fields

---

## scripts/prepare/

Both scripts ingest raw NetCDF files into the patcher binary database (`db/`). The pipeline per variable is: **train_dict → save → aggregate**, where `aggregate` computes the climatology, subtracts it to produce anomalies, and pre-computes downsampled spatial scales.

---

### prepare_db_era5.py

Populates the database with ERA5 reanalysis fields and static features.

**Variables processed:**

| DB name | Source variable | Type | Precision |
|---|---|---|---|
| `t2m` | ERA5 daily `t2m` NetCDF files | dynamic (daily) | 0.05 K |
| `lsm` | ERA5 `land_sea_mask` static file | time-invariant | 0.01 |

**Processing steps for `t2m`:**
- Reads monthly NetCDF files from `era5/t2m/`
- Flips latitude axis (north-up)
- Saves all daily frames into the database
- Calls `aggregate("19910101", "20201231")` — computes 1991–2020 climatology, stores anomalies, and builds downsampled scales

**Processing steps for `lsm`:**
- Reads a single static NetCDF, flips latitude
- Saved as time-invariant (no climatology subtraction)
- Calls `aggregate("", "")` — only builds downsampled scales, no climate

**Output:** `db/t2m.bin` + `db/t2m.idx`, `db/lsm.bin` + `db/lsm.idx`

---

### prepare_db_inm.py

Populates the database with INM-CM ensemble forecast fields.

**Variables processed:**

| DB name | Source variable | Precision |
|---|---|---|
| `inm_t2m` | INM-CM `T2` (2 m temperature) | 0.05 K |

**Processing steps:**
- Reads monthly INM-CM NetCDF files (ensemble × time × lat × lon)
- Rolls longitude axis by 180° to convert from 0–360° to −180–180° convention
- Each file corresponds to one forecast start month; `tag = start_month` is stored in the database to keep forecasts from different start months separate during aggregation
- Saves all ensemble members and forecast days into the database
- Calls `aggregate("19910101", "20210430")` — computes climatology, stores anomalies, builds downsampled scales

**Output:** `db/inm_t2m.bin` + `db/inm_t2m.idx`

---

## scripts/database.py — PatchDataset

PyTorch `Dataset` that randomly samples spatial patches from the patcher database at configurable spatial and temporal scales.

---

### Constructor

```python
PatchDataset(
    variables,                   # list[str]
    time_range,                  # (str, str)  e.g. ('19910101', '20201231')
    epoch_size,                  # int
    era_scales    = [],          # list[dict]
    inm_scales    = [],          # list[dict]
    x_range       = (0, 1440),   # (int, int)  longitude range in full-res pixels
    y_range       = (100, 300),  # (int, int)  latitude range in full-res pixels
    batch_size    = 8,           # int
    lead_time_range = (0, 3),    # (int, int)  INM-CM lead time range in months
    num_workers   = 4            # int         patcher thread pool size
)
```

**`variables`** — list of variable names to load. ERA5 variables are plain names (`t2m`, `lsm`, etc.); INM-CM variables must start with `inm_` (`inm_t2m`). Special computed variables are also accepted:

| Name | Type | Description |
|---|---|---|
| `year`, `day`, `cos_day`, `sin_day` | ERA5 time | Temporal encodings for ERA5 time axis |
| `lat`, `lon`, `cos_lat`, `cos_lon`, `sin_lon` | ERA5 spatial | Grid coordinate encodings |
| `inm_year`, `inm_day`, `inm_cos_day`, `inm_sin_day` | INM time | Temporal encodings for INM-CM time axis |
| `inm_lat`, `inm_lon`, `inm_cos_lat`, `inm_cos_lon`, `inm_sin_lon` | INM spatial | INM-CM grid coordinate encodings |
| `inm_lead_time` | INM | Lead time in days for each INM-CM time step |

**`time_range`** — `(start, end)` as `YYYYMMDD` strings. Random target dates `t` are sampled from this range. The actual ERA5 history window starts earlier (`t - era_time_width`) to ensure enough context is available.

**`era_scales` / `inm_scales`** — each entry is a dict describing one spatial/temporal scale to load. Every variable is loaded at every configured scale. Dict fields:

| Key | Description |
|---|---|
| `id` | String label, used as the key suffix in the output dict |
| `xSize` | Patch width in downsampled cells |
| `ySize` | Patch height in downsampled cells |
| `tSize` | Number of time steps |
| `xyStep` | Spatial downsampling factor (1 = 0.25° for ERA5, 4 = 1°) |
| `tStep` | Temporal stride in days |

Example — two ERA5 scales (local daily + regional weekly):
```python
era_scales = [
    {'id': 'local',    'xSize': 32, 'ySize': 32, 'tSize': 14, 'xyStep': 1,  'tStep': 1},
    {'id': 'regional', 'xSize': 32, 'ySize': 32, 'tSize': 12, 'xyStep': 8,  'tStep': 7},
]
```

**`x_range` / `y_range`** — bounds for random patch center sampling in full-resolution ERA5 pixels (1 pixel = 0.25°). Default `y_range=(100, 300)` corresponds roughly to 25°N–75°N.

**`lead_time_range`** — `(min, max)` months. A random lead time is drawn per sample to determine which INM-CM forecast run to use. If the drawn lead time does not cover the full `inm_time_width`, it is incremented automatically.

---

### Methods

**`set_seed(seed) → self`**

Re-generates the full set of random `(x, y, t)` coordinates and lead times for the epoch. Call at the start of each epoch to shuffle samples while keeping training reproducible. Returns `self` for chaining.

```python
for epoch in range(n_epochs):
    ds.set_seed(epoch)
    for batch in ds.loader:
        ...
```

**`__len__()`** — returns `epoch_size` (number of batches per epoch).

**`__getitem__(idx)`** — builds and dispatches all patcher requests for `batch_size` samples at once, returns a dict of stacked tensors (see Output below).

---

### Output

`__getitem__` returns a `dict[str, Tensor]`. Each key is `f"{variable}_{scale_id}"` and the tensor has shape `[batch_size, ...]`:

| Variable type | Tensor shape |
|---|---|
| ERA5 dynamic (`t2m`, `sd`, …) | `[batch_size, tSize, ySize, xSize]` |
| INM-CM ensemble (`inm_t2m`, …) | `[batch_size, E, tSize, ySize, xSize]` |
| Time encodings (`year`, `cos_day`, …) | `[batch_size, tSize]` |
| Spatial encodings (`lat`, `cos_lon`, …) | `[batch_size, xSize]` |
| `inm_lead_time` | `[batch_size, tSize]` |

Values are **anomalies** (climatology already subtracted by patcher). `NaN` indicates missing data for that date.

Example output keys for `variables=['t2m', 'inm_t2m', 'cos_lat']` with two ERA5 scales (`local`, `regional`) and one INM scale (`global`):
```
't2m_local'      [8, 14, 32, 32]
't2m_regional'   [8, 12, 32, 32]
'inm_t2m_global' [8, E, 10, 32, 32]
'cos_lat_local'  [8, 32]
'cos_lat_regional' [8, 32]
```
