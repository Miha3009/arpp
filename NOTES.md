# Notes

## ClimateDataset — files loaded

`ClimateDataset` loads from three locations under `data/`:

| File(s) | Path | Purpose |
|---|---|---|
| Anomaly files | `data/train/{variant}/anom/{year}{period:02d}.nc` | Main features + ERA5 target (one file per year×period) |
| Climatology files | `data/train/{variant}/clim/{period:02d}.nc` | Climate mean/trend per period (loaded at init) |
| Normalization | `data/train/{variant}/std.nc` | Per-variable mean/std for normalization |
| Time-invariant | `data/train/time_invariant[_norm].nc` | Static fields (e.g. land mask, topography) |

A few notes:
- The `era` extra variable causes it to also open a **neighbouring period's anomaly file** (the previous week/month relative to the lead time) — see `load_file:400–408`
- `time_invariant` is only added to `variables` for the `swe` variant (line 106)
- If `normed=False`, it opens `time_invariant.nc`; otherwise `time_invariant_norm.nc` (line 88)

## Climatology files — what they are used for

The anomaly files already have the climate mean subtracted. The clim files are used for three things, none of which is recomputing anomalies:

**1. Building validity masks** (`train.py:127`) — the mask marks grid points where the climate mean is non-NaN and non-zero. This filters out ocean points for SWE and land points for AICE:
```python
self.masks[p] = torch.as_tensor((~np.isnan(self.climate[p])) & (self.climate[p] != 0) & (self.mask.ravel()))
```

**2. `climate` extra variable** (`train.py:393–394`) — when `'climate'` is listed in `variables`, the raw climatological mean is appended as a feature channel so the model knows the background state:
```python
elif variable == 'climate':
    features.append(self.climate[period])
```

**3. Trend detrending** (`train.py:129–132, 377–378, 417`) — when `climate_trend=True`, a linear trend is loaded from the clim file and subtracted from both features and the target `y` to remove long-term drift:
```python
if self.climate_trend:
    values -= self.trends[variable] * (year - self.trends_mean_year[variable])
```

## Bias correction approach — thoughts

The pipeline uses a two-stage correction: statistical bias removal, then a learned DL correction. This is classic **Model Output Statistics (MOS)**.

**What's good about it:**
- Bias correction removes the systematic offset so the DL model only needs to learn residual corrections — a much easier task than learning both the mean and the variability
- Using INM-CM SWE as a feature gives the model the model's own estimate as a starting point — a strong prior
- Anomaly space is cleaner for learning: climatological cycles are removed

**Potential concerns:**
- ERA5 SWE is itself a model product (land surface model + data assimilation), not pure observations — correcting one model toward another rather than toward ground truth. GlobSnow is actual satellite-derived SWE, which makes it interesting that it's prepared but unused
- The bias is computed over 1991–2020 and assumed stationary — if INM-CM's systematic errors drift over time, the correction degrades
- The DL model might partially re-learn what the bias correction already did, or the bias correction may remove signal the model could have exploited

**The unused GlobSnow is the most interesting loose end** — using it as an alternative or additional target (or even an extra input feature) would make the pipeline less circular, since it's observational rather than modeled.

## Bias-correcting all meteo variables — is it worth it?

Currently weather vars are anomalies w.r.t. **INM-CM's own climatology**, while the target is an anomaly w.r.t. **ERA5 climatology** — different reference frames. The model has to implicitly compensate for INM-CM's biases in each variable on top of learning the SWE relationship.

Correcting all variables toward ERA5 would:
- Put features and target in the same reference frame
- Let the model focus purely on learning the SWE response
- Match standard MOS practice

**The catch:** requires ERA5 equivalents for all 15 vars (h500, t2, t850, u850, v850, prec, etc.) — feasible since ERA5 has all of them, but means more downloads and a more complex pipeline. The SWE case in `prepare_train.py` is already the template.

**Where it likely matters most:** precipitation and t2 tend to have the largest and most spatially structured biases in climate models. H500 and wind vars at pressure levels are usually better constrained. The benefit scales with how large and structured the biases are.
