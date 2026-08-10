# Variable transforms

Per-variable preprocessing for the DL pipeline. Pipeline style: `stepA --> stepB`.
Notes in *italic* capture options/reasoning from the design session — come back for
the full argument, this is the digest.

---

## Global conventions

- **Pipeline order:** `[transform] --> anomaly --> normalize --> [boundary fill/mask]`
- **Anomaly** is already stored in the `.bin` (linear, per cell × calendar-day, vs
  1991–2021 climatology). `load()` returns it; `load_climate()` returns the mean.
- **Structural NaN — geography decides, weather never.** sea (`lsm=0`) and glacier
  (`glacier=1`) --> NaN. Keep every real value on snow-capable land, *including
  genuine in-season 0s* (snow-free land is signal, not a mask).
- **Normalize** = standardize `(x - mean) / max(std, eps)` with `compute_norm` stats:
  one scalar per **(variable, scale_id)** channel, over **train years only**, then
  **frozen**. NOT cos-lat weighted (matches the pixel-uniform sampler, not the
  physical domain). Use `mad_std` instead of `std` for skewed channels.
- **Input boundary:** `normalize --> NaN->0 fill (+ validity channel)`. Fill *after*
  normalize so 0 = climatology = neutral.
- **Target boundary (`sd`):** keep NaN (it *is* the loss mask). Never fill. Train in
  transformed/anomaly space, **report metrics in physical mm**.
- **Masks:**
  - `loss_mask  = isfinite(target) & snow_possible & cos_lat`
  - `metric_mask = snow_possible` (tight)
  - `snow_possible` = static `[period, H, W]` bool, **EVER-snow** threshold
    (max over years, not mean), built once from ERA5 snow climatology.

---

## ERA5

- **t2m** — normal, defined everywhere. `normalize`
- **tp** — right-skew. `log1p --> anomaly --> normalize (mad_std)` · no spatial mask
- **sd** *(TARGET + input)* — right-skew; sentinels 5000=glacier, 0=sea/no-snow.
  `structural NaN (sea, glacier) --> log1p --> anomaly --> normalize`
  - as **input:** `--> NaN->0 fill + validity channel`
  - as **target:** `--> keep NaN` (loss mask); metrics inverted to mm
- **sden** — sentinels 100=no-snow, 300=glacier; normal excl. masked.
  `sentinels->NaN --> normalize`. *Secondary predictor: only defined where snow is,
  co-moves with target.*
- **pt** — categorical (precip type). *One-hot or drop for exp1; not a headline var.*
- **snow_cover** — `{0-1}` fraction, near-binary/seasonal. `glacier->NaN --> normalize`.
  *Keep as fraction (more info); binarize w/ threshold only if EDA shows it's noise.*
- **sst** — land masked with near-zeros. `<=0.1 -> NaN (set, don't clip) --> normalize`.
  *Bimodal (freeze ~271K vs tropics) is physical — do NOT gaussianize.*

## INM-CM  *(all `[E, T, H, W]`; E handling is separate — see `details.md`)*

- **h500** — normal. `normalize`
- **hlt** — latent heat flux, normal. `normalize`
- **olr** — normal. `normalize`
- **tp** — right-skew. `log1p --> anomaly --> normalize (mad_std)`
- **mslp** — normal. `normalize`
- **ts** — surface temp, normal. `normalize`
- **swe** — right-skew; outliers ~1000mm. `clip outliers (don't drop) --> log1p -->
  anomaly --> normalize`. *structural NaN if it carries sea/glacier sentinels too.*
- **snow_cover** — near-binary. `glacier->NaN --> normalize` · *see ERA5 snow_cover*
- **ww** — 100cm soil water, normal-bimodal. `<0.1 -> NaN --> normalize`
- **u850 / v850** — wind, normal. `normalize`
- **t2m** — normal. `normalize`

## Time-invariant

- **sdor** — std orography (large). `normalize`
- **z** — surface height (~1e3). `normalize`
- **glacier** — `{0,1}` mask. *keep as-is; used to build structural + loss masks*
- **lsm** — `{0-1}` mask. *keep as-is; used to build structural masks*

---

## Notes (session digest)

- ***Exp 1 = full resolution only.*** *No spatial pooling, so ignore
  invalid-value-contaminating-averaged-patches entirely (that was the only reason to
  bake NaN into the `.bin`). All masking/transforms can live at the **Dataset level**
  with precomputed stats — no `.bin` rewrite.*
- ***`compute_norm` must share the Dataset's sentinel->NaN masking***, else finite
  sea/glacier sentinels sail through its `isfinite` filter and collapse the std.*
- ***log1p is the one transform coupled to the `.bin`*** *(patcher anomaly is linear;
  `log1p(anom) != anom`). Options:*
  - *(a) targeted re-ingest of `sd`/`tp`/`swe` with log1p before `aggregate`*
  - *(b) Dataset-level & exact: `raw = anom + load_climate` --> `log1p` --> subtract a
    precomputed log-climatology file. No rewrite, per-sample `load_climate` cost.*
  - *(c) skip log1p for exp1 (linear anomalies).*
  - ***--> (c) now, (b) when committing to log.***
- ***Loss family drives the mask:***
  - *plain RMSE: keeps the melt-line "collar" for free, numerically safe — but easy
    zeros dilute gradient (1:99) and it's scale-unbalanced across leads/seasons.*
  - *ACC / per-period std-norm RMSE (Issue-1 fix): balanced, but zero-variance tier-2
    cells put σ≈0 in a denominator --> explode --> **forces the tight mask**.*
  - ***--> exp1: tight `snow_possible` mask for BOTH loss and metric.***
- ***Metric on tight mask is non-negotiable*** *(broad-mask RMSE hides zero-skill: a
  climatology model can score ~2.5mm vs perfect 0 because 99% of cells are trivial
  zeros).*
- ***Seasonal `snow_possible` already contains the collar*** *(cells that CAN snow this
  window but are bare today). Only the thin tier-2 ring at the boundary is dropped ->
  recover with a **1–2 cell dilation** of the loss mask if snow bleeds past the edge.*
- ***Masked-cells-to-climatology regularizer (liked):***
  `L = RMSE(snow_mask) + λ · mean_masked[(pred - 0)²]`, *λ~0.01. Keeps the full field
  sane (no India-in-July snow) without the zeros re-dominating. Optional.*
- ***Model learns snow<-inputs, so it outputs ~0 off-mask for free*** *(India's inputs
  are far outside the snow regime); first-epoch hallucination is transient and doesn't
  corrupt scored cells (spatial output independence). Caveat: bottleneck attention /
  global ops are a weak coupling channel.*
- ***Always post-mask the output at inference*** *(clamp masked cells to climatology).*
- ***Past `sd` as input is not leakage — it's the strongest predictor*** *(snow memory);
  history window is strictly before the valid date by construction.*
