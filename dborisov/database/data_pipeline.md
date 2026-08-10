# v2 Data Pipeline

════════════════════════════════════════════════════════════════════════════════
 STEP 1 · DOWNLOAD
════════════════════════════════════════════════════════════════════════════════

  scripts/download/download_era5.py
    CDS API  →  era5/{var}/{YYYYMM}.nc
    vars: t2m · tp · sd · sden · t2m_min · t2m_max · pt · h500 · t850
    daily mean · 0.25° global · 1980–2026

  scripts/download/download_time_invariant.py
    CDS ERA5  →  data/era5_time_invariant.nc
    vars: geopotential · lsm · soil_type · veg_cover×4 · sdor · lake_cover
    single snapshot · 0.25° global

  INM-CM model  →  raw NetCDF files  (path configured per installation)
    vars: T2 (t2m), and others to be added
    ensemble × time × lat × lon · 1° global


════════════════════════════════════════════════════════════════════════════════
 STEP 2 · INGEST INTO DATABASE  (scripts/prepare/)
════════════════════════════════════════════════════════════════════════════════

  For each variable the prepare script runs three stages:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  patcher.train_dict(context, sample_data, element, precision)        │
  │    Trains a ZSTD compression dictionary from the first batch         │
  │    of data. Stored inside element's .idx file.                       │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  patcher.save(context, data, dates, element, tag)                    │
  │    Appends daily tensors to db/{element}.bin, one at a time.         │
  │    Each spatial 32×32 patch is quantized (uint16, delta-encoded)     │
  │    and ZSTD-compressed into the file. The .idx is updated with       │
  │    byte offsets for each patch key.                                  │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  patcher.aggregate(context, element, climateStart, climateEnd)       │
  │    Two operations in one pass:                                       │
  │    1. Climate: averages patches over the specified period            │
  │       (e.g. 1991–2020) by calendar day (MMDD key). Stored as        │
  │       separate patches with tStep=0 in the same .bin file.          │
  │    2. Anomalies: rewrites every patch as (raw - climate mean).       │
  │    3. Multi-scale pyramid: builds spatially downsampled versions     │
  │       at xyStep = 2, 4, ... (controlled by config grid_sizes         │
  │       bitmask). Also builds temporally aggregated versions.          │
  └─────────────────────────────────────────────────────────────────────┘

  Current database contents:
    db/t2m.bin + t2m.idx        ERA5 2m temperature anomalies   (daily, 0.25°)
    db/lsm.bin + lsm.idx        ERA5 land-sea mask              (static)
    db/inm_t2m.bin + inm_t2m.idx   INM-CM t2m anomalies        (daily, 1°, ensemble)
    db/config.bin               Global config (patch size, compression params)


════════════════════════════════════════════════════════════════════════════════
 WHAT IS STORED IN .bin FILES
════════════════════════════════════════════════════════════════════════════════

  The database is a flat binary file of ZSTD-compressed patches.
  Each patch is a block of the full map at a given location, time, and scale.

  Patch dimensions:  [E, patch_size_t, patch_size_h, patch_size_w]
                     = [ensemble, 4 days, 32 cells, 32 cells]  (defaults)

  Encoding per patch:
    1. Subtract min value → store min as float32 header
    2. Divide by precision → quantize to uint16  (65535 = NaN)
    3. Delta-encode along the flat array
    4. ZSTD-compress using the element's trained dictionary

  Index (.idx) maps each patch by its key:
    PatchKey { x0, y0, t0, xyStep, tStep, tag }  →  byte offset in .bin

    x0, y0   : top-left corner in grid pixels
    t0        : start day (days since 1970-01-01) for anomaly patches
                MMDD integer for climate patches (tStep=0)
    xyStep    : spatial downsampling factor (1 = full resolution)
    tStep     : temporal averaging window in days (1 = daily, 0 = climate)
    tag       : forecast start month for INM-CM (separates ensemble runs)

  What's stored for a variable after aggregate():
    tStep=0, xyStep=1   climate mean by calendar day  (MMDD key)
    tStep=1, xyStep=1   daily anomalies at full resolution
    tStep=1, xyStep=N   spatially downsampled anomalies at scale N
    tStep=N, xyStep=1   temporally averaged anomalies over N days


════════════════════════════════════════════════════════════════════════════════
 STEP 3 · TRAINING  (scripts/database.py → PatchDataset)
════════════════════════════════════════════════════════════════════════════════

  PatchDataset pre-generates a fixed set of random (x, y, t) coordinates
  for the epoch (controlled by epoch_size × batch_size and a seed).

  For each sample __getitem__ builds a list of patcher.Request objects —
  one per variable × scale combination — and calls patcher.load() once
  to fetch all of them in parallel via the thread pool.

  Each request specifies:
    element   variable name in the database
    x0, y0    patch center (in grid pixels)
    t0        date string YYYYMMDD
    xSize     patch width  in downsampled cells
    ySize     patch height in downsampled cells
    tSize     number of time steps
    xyStep    spatial scale (1 = 0.25° for ERA5, 4 = 1° for INM-CM)
    tStep     temporal stride in days
    tag       forecast start month (INM-CM only)

  patcher.load() returns anomaly tensors:
    ERA5 variable  →  [T, H, W]          float32
    INM-CM variable →  [E, T, H, W]      float32  (E = ensemble size)

  patcher.load_climate() returns the climatological mean for the same
  request, useful for converting anomalies back to absolute values.

  The dataset returns a dict  { key: tensor }  where key = f"{variable}_{scale_id}",
  batched across all batch_size samples in the item.

  ┌─────────────────────────────────────────────────────────────┐
  │  Model receives per sample:                                  │
  │    ERA5 variables  at each configured era_scale              │
  │      → anomaly patches  [T, H, W]                           │
  │    INM-CM variables at each configured inm_scale             │
  │      → ensemble anomaly patches  [E, T, H, W]               │
  │    Computed on-the-fly: lat, lon, cos_lat, sin_lon,          │
  │      cos_lon, year, day, lead_time                           │
  └─────────────────────────────────────────────────────────────┘
