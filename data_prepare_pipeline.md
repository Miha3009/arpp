# SWE Data Preparation Pipeline
# ⚠ = prepared/downloaded but not used in training

────────────────────────────────────────────────────────────────────────────────
 A. ERA5 SWE  (observation, becomes prediction target)
────────────────────────────────────────────────────────────────────────────────

  CDS API  ·  ERA5 snow_depth  (daily, 1991–2025)
    │  download_swe_era5.py
    ▼
  raw/swe_era5/{Y}_{M}.nc  [var: sd]
    │  prepare_era5_swe.py  ·  flip lat  ·  sd × 1000 → swe  ·  split to daily files
    ▼
  prepare/era5_swe/{YYYYMMDD}.nc  [var: swe, 0.25°, daily]
    │
    ├─► prepare_swe_climate.py  ·  weekly agg  ·  mean & linear trend  (1991–2020)
    │     ▼
    │   climate/week/era5_swe/{ww}.nc  [swe_mean · swe_trend]
    │     │
    │     └─► used in: bias computation · anomaly reference · merged into train/swe/clim/
    │
    └─► prepare_merge.py  ·  weekly mean  ·  subtract era5 climate
          ▼
        merge/era5_swe/{Y}{ww}.nc  [ERA5 SWE weekly anomaly]
          └─► stored as "era5" variable (prediction target) in train/swe/anom/


────────────────────────────────────────────────────────────────────────────────
 B. GlobSnow SWE  ⚠
────────────────────────────────────────────────────────────────────────────────

  globsnow.info  ·  archive v3 + NRT  (polar stereographic, daily)
    │  download_globsnow_{archive_v3,nrt}.py
    ▼
  raw/globsnow_{archive_v3,nrt}/  [var: swe / SWE]
    │  prepare_globsnow_swe.py  ·  Delaunay barycentric regrid  ·  apply ERA5 land mask
    ▼
  prepare/globsnow/{YYYYMMDD}.nc  [var: swe, 0.25°, daily]
    │  prepare_swe_climate.py  ·  weekly agg  ·  mean & trend  (1991–2020)
    ▼
  climate/week/globsnow/{ww}.nc  [swe_mean · swe_trend]
    ✗  not used further  (prepare_merge.py has globsnow commented out)


────────────────────────────────────────────────────────────────────────────────
 C. SEAS5  ⚠
────────────────────────────────────────────────────────────────────────────────

  CDS API  ·  SEAS5 ECMWF SWE ensemble  (2024–2025)
    │  download_seas_ecmwf.py
    ▼
  raw/swe_seas/{Y}_{M}.nc  [var: sd, ensemble dim]
    │  prepare_seas_swe.py  ·  ensemble mean  ·  × 1000  ·  linear interp  ·  weekly agg
    ▼
  prepare/swe_seas/all.nc  [var: swe, week + lead_time dims]
    ✗  not used further  (not referenced in prepare_train.py)


────────────────────────────────────────────────────────────────────────────────
 D. INM-CM model  (forecast, becomes features)
────────────────────────────────────────────────────────────────────────────────

  INM-CM  ·  weekly ensemble forecast  ·  16 variables:
    swe  ·  cld  ·  h500  ·  olr  ·  prec  ·  ps  ·  rq2
    t2   ·  t2max  ·  t2min  ·  t850  ·  u850  ·  v850  ·  uv10  ·  ws  ·  ww
    │  (raw model output, own grid)
    ▼
  raw/inmcm/{var}/{YM}.nc
    │
    ├─► prepare_inmcm_climate.py  ·  weekly agg  ·  mean & linear trend  (1991–2020)
    │     ▼
    │   climate/week/{var}/{ww}.nc  [{var}_mean · {var}_trend]  (all 16 vars)
    │     │
    │     └─► used in: bias computation · anomaly subtraction
    │         all vars (except swe) → merged into train/swe/clim/
    │
    └─► prepare_inmcm_merge.py  ·  weekly agg  ·  subtract INM-CM climate mean
          ▼
        merge/inmcm_swe/{Y}{ww}.nc    [INM-CM SWE anomaly,          multi lead-time]
        merge/inmcm_week/{Y}{ww}.nc   [15 weather var anomalies,    multi lead-time]


────────────────────────────────────────────────────────────────────────────────
 E. ERA5 time-invariant static fields
────────────────────────────────────────────────────────────────────────────────

  CDS API  ·  geopotential · land_sea_mask · soil_type
             veg_cover × 4 · sdor · lake_cover
    │  download_time_invariant.py
    ▼
  raw/time_invariant/era5_time_invariant.nc
    │  prepare_time_invariant.py  ·  interp → 0.25°  ·  crop lat ≥ 40°  ·  merge vars
    ▼
  train/time_invariant.nc  [9 vars]
    │  prepare_std.py  ·  log(z)  ·  min-max scale  ·  one-hot encode (slt · tvh · tvl)
    ▼
  train/time_invariant_norm.nc


────────────────────────────────────────────────────────────────────────────────
 F. GHCN  ⚠
────────────────────────────────────────────────────────────────────────────────

  NOAA GHCN  ·  station observations
    │  download_ghcn.py
    ▼
  raw/ghcn/{year}.csv.gz
    ✗  no prepare script  (not used in training)


════════════════════════════════════════════════════════════════════════════════
 BIAS CORRECTION + ASSEMBLY  ─  prepare_train.py
════════════════════════════════════════════════════════════════════════════════

  Inputs
    merge/inmcm_swe/{Y}{ww}.nc          INM-CM SWE anomaly
    merge/inmcm_week/{Y}{ww}.nc         15 weather var anomalies
    merge/era5_swe/{Y}{ww}.nc           ERA5 SWE anomaly
    climate/week/swe/{ww}.nc            INM-CM SWE climate mean
    climate/week/era5_swe/{ww}.nc       ERA5 SWE climate mean

  Steps
    1.  bias = mean( INM-CM_swe + INM-CM_clim − ERA5_swe − ERA5_clim )  over years
               → train/swe/bias/{ww}.nc
    2.  corrected_swe = INM-CM_swe_anom + INM-CM_clim − bias − [clamp to ≥ 0] - ERA5_clim
    
    3.  merge: corrected SWE  +  15 weather var anomalies  +  ERA5 SWE (target)
               → train/swe/anom/{Y}{ww}.nc
    4.  merge all INM-CM climates + era5_swe climate into per-week file
               → train/swe/clim/{ww}.nc

════════════════════════════════════════════════════════════════════════════════
 NORMALIZE  ─  prepare_std.py
════════════════════════════════════════════════════════════════════════════════

  train/swe/anom/*.nc  ─► RMS std per variable  ─►  train/swe/std.nc

════════════════════════════════════════════════════════════════════════════════
 ClimateDataset  reads:
    train/swe/anom/{Y}{ww}.nc       features (INM-CM SWE + 15 vars) + era5 target
    train/swe/clim/{ww}.nc          validity mask · climate feature · trend removal
    train/swe/std.nc                per-variable normalization
    train/time_invariant_norm.nc    static spatial features
════════════════════════════════════════════════════════════════════════════════
