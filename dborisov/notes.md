# Notes

1. xyStep, tStep - просто осреднение вдоль осей? 
2. Snow climate mask надо тоже загрузить в .bin
3. E dimension varies in INMCM versions
4. inmcm данные откуда скачивать исходники

 Great! Thanks for your help! That's it for today, next step is the model class. I'm thinking 
  about U-net, but can't yet understand how to cover time axis, just different channels?       
  
⏺ Good question to sit with before the next session. A few quick thoughts:

  The simplest approach — time as channels. Flatten the time axis into the channel dimension:
  [B, T*C, H, W]. U-Net sees it all at once, but there's no inductive bias for temporal
  ordering — the network has to learn "step 3 matters more than step 1" purely from weights.
  
  The more principled approaches:                      
                                                            
  - 3D convolutions [B, C, T, H, W] — capture local spatio-temporal patterns jointly. Natural
  extension of U-Net into 3D (U-Net3D is already in your v1 models.py). Works well when the
  temporal patterns are local (last few days), gets expensive for long histories.
  - ConvLSTM / ConvGRU in the encoder — process the time sequence spatially, then pass the
  final hidden state into a standard spatial U-Net decoder. Good for longer histories but
  sequential so harder to parallelize.
  - Temporal attention before spatial U-Net — first reduce the time axis via attention (like a
  small transformer over T), then feed the result into U-Net. Flexible, parallelizable, handles
  variable-length histories well.

  The natural fit for your setup: since you have multi-scale patches with different tSize per
  scale, flattening time into channels per scale and concatenating is the pragmatic starting
  point — simple, fast, and your v1 models already do something similar. You can always add a
  temporal attention module later as an improvement.

  Worth deciding before next session: do you want the model to output a single map (one lead
  time at a time) or multiple lead times at once?

✻ Sautéed for 13s

❯ I'll think about that. Attention sounds good, I was thinkink about it from the start,        
  spatial attention as well.                                                                   
  
⏺ Good instinct — spatial attention is particularly well-suited here because snow and ice
  prediction has strong spatial selectivity: the model should learn to attend to specific
  teleconnection regions (e.g. North Atlantic SST anomalies for European snow, Arctic
  Oscillation patterns for Siberia) rather than weighting all locations equally.

  A natural architecture sketch:

  Multi-scale patches
         │
         ▼
  Temporal attention       ← collapse T axis per scale
         │
         ▼
  Scale fusion / embed     ← align all scales to same spatial size
         │
         ▼
  U-Net encoder
    + spatial attention    ← at bottleneck and/or skip connections
         │
         ▼
  U-Net decoder
         │
         ▼
  SWE anomaly map

  Spatial attention at the bottleneck is the most common placement — the feature map is small
  there so attention is cheap, and it's where global context matters most. Adding it on skip 
  connections too (like in Attention U-Net, Oktay et al. 2018) lets the decoder selectively
  pull spatial detail from the encoder only where relevant.

  One thing to think about for next time: since your patches are randomly sampled and not
  always global, spatial attention within a patch captures local dependencies well — but
  cross-region teleconnections only show up if your coarse-scale inputs are large enough to
  span both regions simultaneously. That ties back to the pyramid scale design.

  Save that thought for next session — it'll directly inform how you configure era_scales.


  ### On the architecture (from Claude)

The real risk is sample size, not architecture

  This is the thing I'd flag hardest, and it's the opposite of where the energy usually goes:

  - 1991–2016 is ~25 winters. For a seasonal target, your effective independent sample size is closer to 25–100, not "thousands of daily patches."* Daily samples within a season
  are heavily autocorrelated — SWE barely moves day to day. The patch sampling makes it look like you have huge data, but the information content is small.
  - Consequence: overfitting is the default outcome, and a big attention U-Net will memorize 25 winters effortlessly. The danger isn't that the model won't fit — it's that it'll
  fit beautifully and generalize poorly to held-out years.

  This reframes the whole project priority order:

  1. Validation discipline > architecture. Split by year/block, never randomly. Random splits leak (a January patch and February patch of the same winter are nearly the same
  sample). Use leave-N-years-out / rolling-origin CV. Get this wrong and every number you report is inflated.
  2. Start small and earn complexity. A modest CNN/U-Net that beats baselines on held-out years is worth more than a sophisticated attention model you can't trust. The baselines
  (ClimateModel, BaseModel) aren't formalities — with this little data they're genuinely hard to beat at long leads.
  3. Strong regularization and inductive bias matter more than capacity. The anomaly framing, latitude weighting, and SWE masking are doing real work.


### Precipitation: special processing (log1p)

Context: INMCM February precip distribution is extremely right-skewed and zero-inflated
(percentiles: 1% ≈ 7.7e-11, 10% ≈ 0.021, 25% ≈ 0.158, 50% ≈ 0.636 mm/day; long thin tail to
~200 mm/day). The dynamical model likely **underestimates** precip — most cells are pinned near
zero — and bias-correcting that toward ERA5 is exactly what the DL model should learn.

Decisions / reasoning:

- **Storage precision `inm_pr = 0.01` mm/day.** The mass sits below 1 mm/day, so coarser steps
  collapse the signal: at `0.1` the bottom ~10–15% of wet cells round to 0 and the lower half
  collapses to ~6 levels — erasing the light-snowfall accumulation that drives SWE. At `0.01`
  the bulk gets 16–64 levels (shape preserved), span = `0.01 × 65535 = 655 mm/day` covers the
  ~200 mm/day tail with no clipping (per-patch min ≈ 0). Rule of thumb confirmed: representable
  span = precision × 65535, offset = per-patch min stored as float32. **Always look at the
  histogram before picking precision for a new variable** — skew can flip the answer by an order
  of magnitude.

- **The bias-correction signal lives in the low end.** The difference between the model putting a
  cell at 0.05 vs 0.16 vs 0.6 mm/day is the discriminative information the correction keys off.
  Coarse low-end quantization destroys those gradients before the network sees them.

- **Use a `log1p` transform on precip before the model (and before anomalizing).** Reasons:
  1. A multiplicative / scale bias (model says 0.2×, truth 1×) becomes an *additive shift* in
     log space — much easier for the network to learn and regularize.
  2. Anomalies of a zero-bounded heavy-skew field are lopsided (small negative floor, long
     positive tail); `log1p` makes the climatology subtraction happen in a more symmetric space.
  Storage precision and transform are independent decisions but point the same direction.
  NB: anomalies are computed relative to **INMCM's own climatology** (patcher `aggregate`,
  per day-of-year, ensemble-mean, climate window 1991–2021), so the mean underestimation largely
  cancels in anomaly space; what remains for the model to fix is the variance/scale bias.

- **Risk to watch:** if INMCM low-end precip is *noise* rather than *attenuated signal*, there's
  nothing to correct and the model falls back to climatology. Spatial pattern usually still
  carries info even when amplitude is off. Sanity check: compare ERA5 vs INMCM February precip
  means over the domain to tell a (correctable) scale bias from a (harder) structural one.

## Log-transform of right-skewed inputs — the full "why" (conceptual)

Companion to the precip note above: *why* `log1p` helps the model at all, and the
distinctions that are easy to conflate. Applies to any right-skewed positive field
(precip, SWE, snow depth).

### Three shapes people lump together (they need three different fixes)

- **Right-skewed** — the *bulk* sits near the low end and the density trails off in a
  long, smooth, *populated* tail to the right. Asymmetry is intrinsic to the whole
  shape (mean ≫ median), a continuum of ever-rarer large values following a rule.
  Precip is the textbook case. → fix with a **log / power transform** (reshapes the
  populated tail).
- **A body with right outliers** — the body may be symmetric (even Gaussian) but a
  *handful* of points sit far out, usually with a gap between them and the body.
  Difference from skew: continuity and mass. Skew = a populated tail following a rule;
  outliers = a sparse anomaly (often error or a distinct process). → fix with
  **clipping / winsorizing** (a log just drags them in, wrong tool).
- **Zero-inflated** — a *spike of probability mass exactly at 0* on top of whatever the
  nonzero values do. A mixture of two processes: "did it happen?" (rain/no-rain) and
  "if so, how much?". The zeros are NOT the small tail of a continuum — a separate
  point mass. → fix with **`log1p`** (log(0)=−∞ otherwise) and/or a **two-part model**.

Precip and SWE are **all three at once**: a big bar at 0 (dry / snow-free cells) + a
right-skewed continuum for wet/snowy cells + a thin outlier-ish extreme tail. That's
why `log(x)` alone fails and you use `log1p(x) = log(1+x)`, and why the robust scale
(mad_std) goes degenerate on them (see the mad_std note / compute_norm).

### Why shrinking the skew helps — four mechanisms (roughly by importance)

1. **Fixes the loss geometry (the big one).** With squared error on a skewed *target*,
   a few large values carry huge squared errors, so loss and gradient are dominated by
   the rare heavy cells while millions of ordinary small cells contribute ~nothing →
   the model chases the tail and underfits the body. `log1p` compresses the tail so a
   "10 vs 12 mm" miss and a "0.1 vs 0.5 mm" miss weigh more comparably. You optimize a
   balanced objective instead of one hijacked by extremes.

2. **Turns multiplicative structure into additive structure.** Skewed physical
   variables vary *multiplicatively* — the meaningful "distance" between two values is
   the **ratio**, not the absolute gap:

   | pair | absolute Δ | ratio |
   |---|---|---|
   | 0.1 → 1 mm | 0.9 | ×10 |
   | 10 → 100 mm | 90 | ×10 |

   By absolute Δ these look 100× apart; physically they're the same *step* (drizzle→rain,
   heavy→extreme, each "a category up"). The ratio is the meaningful yardstick for three
   independent reasons: (a) **physical response** — snowpack/soil-moisture/runoff respond
   to relative changes; (b) **statistical spread** — heteroscedastic, local variability
   ∝ level, so a fixed ratio = a fixed number of std = equally surprising; (c)
   **measurement** — precision is relative (% not fixed mm). Contrast temperature, which
   is *additive* (−20→−18 and 10→12 both mean "2° warmer") — that's exactly why you do
   **not** log temperature; its meaningful metric already is the absolute difference.

   Weights connection: if the true relation is "a 10% increase in precip raises SWE by a
   fixed amount," then in **raw space** the network must add 0.01 near base 0.1 but add
   10 near base 100 to represent the same 10% step — the needed offset is a function of
   local magnitude, i.e. a magnitude-dependent gain (a multiplicative interaction) that
   costs nonlinearity, capacity, and data. In **log space** a 10% step is always
   `+log(1.1) ≈ +0.095` everywhere → one constant shift, one weight, generalizes across
   regimes. In a small-sample regime that's a real data saving.

3. **Stabilizes variance across the range (variance homogenization).** In raw space
   spread grows with magnitude (heavy cells noisy, light cells tight). That
   heteroscedasticity means one global `(x−mean)/std` can't fit both regimes — std is
   set by the tail and the body gets crammed into a sliver near zero. After `log1p`
   variance is far more uniform, so standardization actually spreads values across a
   usable range.

4. **Conditions the inputs numerically.** As a *predictor*, a raw skewed channel enters
   the conv stem with most mass near 0 and a few values orders of magnitude larger.
   Those large activations dominate BatchNorm/LayerNorm stats, blow through the network,
   and produce large erratic gradients early — same pathology as an unnormalized
   channel, self-inflicted by the shape.

### The through-line

A neural net + squared loss + linear normalization implicitly *want* roughly symmetric,
additive, homoscedastic, comparably-scaled variables. A right-skewed field violates all
of that. `log1p` is the cheap monotonic reshaping that restores those assumptions
**without discarding information** — it's invertible (predict in log space, `expm1`
back), monotonic (preserves ordering, so rank-based ACC transforms sensibly).

### "But isn't it good the network reacts strongly to heavy precip?" — salience vs magnitude

Yes — and `log1p` does **not** stop it. Separate two things:

- **Salience** (good): should heavy precip strongly drive the prediction? Yes, it's
  important signal — and after `log1p` the model *still* learns to respond strongly via
  its **weights**. Heavy precip is still the max, still monotonically ordered, still
  distinguishable (rank preserved).
- **Raw numerical magnitude** (the accident): should the input *number* be 1000× bigger
  than a typical cell before any learning happens? That's a unit artifact, not
  information, and it's the *only* thing mechanism 4 objects to — it dominates
  normalization stats and throws huge early activations/gradients regardless of how
  important the event is.

So the reaction still happens; it now comes from *learned weights deciding heavy precip
matters*, not from the raw magnitude bulldozing the numerics. Importance becomes a
semantic thing the model learns, not something dictated by the unit scale. There's even
a functional-form argument: a raw-magnitude "1000× bigger activation" bakes in a
**linear** response, but physical responses to precip usually *saturate* or act
multiplicatively — log space sits closer to the true shape, making the model's job
easier, not just cleaner.

Honest caveat: log *compresses* numerical distance between extremes, so if the goal were
**detecting rare extreme events** (classification-flavored, tail is the point), keep the
raw magnitude too — feed both `log1p(x)` and a clipped-raw / indicator. For smooth SWE
regression, `log1p` is right and costs no sensitivity the model would actually want.

### Practical order (consistent train & inference)

`log1p → anomaly → normalize`, identically applied everywhere (this is caveat 4 in
`compute_norm`). Consequences already noted elsewhere: after `log1p` the field is
genuinely **not** zero-mean → subtract the mean for these channels (don't trust `x/std`);
and mad_std is degenerate on the zero spike → scale these channels by a positive-cell
stat or an upper percentile (`q99`), not by MAD. The climatology used for the anomaly
must also be built in log space or the composition is inconsistent.

## Variables normalization options

1. What granularity of σ?

┌────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                   option                   │                                                        effect                                                         │
├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ one scalar per variable (over domain +     │ preserves the spatial pattern of variance — that high-latitude t2m anomalies are genuinely larger than mid-latitude   │
│ training years)                            │ ones stays visible to the model. Simple, robust.                                                                      │
├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ per-grid-cell σ                            │ equalizes every cell; makes low-variance regions (tropics, snow-free desert, ocean under a land variable) explode     │
│                                            │ into noise. Needs a σ floor.                                                                                          │
├────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ per-cell × per-day-of-year σ               │ most aggressive; also the most leak-prone and noisiest to estimate from ~30 samples per calendar day.                 │
└────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

I'd default to one scalar per variable for inputs. The regional variance structure is physical signal, not nuisance — you don't want to erase it on the input side. Per-cell normalization is the kind of thing to try as an ablation, not as the default.

2. Inputs and target are separate decisions. Input standardization is about conditioning the optimizer. Target standardization is about the loss being balanced across leads and seasons — that's Issue 1 in model_sketch.md, and there the argument does favor dividing by per-cell/per-period climatological σ (or training on ACC). Don't conflate them: you can standardize inputs globally and still normalize the target per-period for the loss.

3. ERA5 t2m and INMCM t2m should get their own separate stats, not shared ones. Sharing keeps the amplitude mismatch visible to the model; separate stats hand the model an already-scale-matched pair so it can spend capacity on the pattern correction. Since the output rescaling is a recorded affine transform either way, I'd go separate — but flag it, because for a bias-correction task it's a genuine design choice rather than an obvious one.

4. Things that will bite you:
- Compute stats on training years only. With a year-blocked CV scheme, σ computed over all years is a real leak — small, but it's exactly the kind of thing a reviewer asks about.
- Freeze and version the stats into the experiment JSON / a norm.nc artifact. Inference in 2026 must use the 1991–2021 stats, not recomputed ones.
- Mask before reducing. σ over ocean cells for a land variable, or over NaN fill, is meaningless. Use lsm / valid mask.
- Precip goes through log1p first, per your own note — and the anomaly should be computed in log space, so the order is log1p → anomaly → standardize. That means the patcher aggregate climatology for tp also has to be built in log space, or the composition is inconsistent. Worth checking what's currently stored.
- Static fields need it too: z and sdor are in the thousands. lsm/glacier are already 0–1, leave them.
- Ensemble subsampling changes the effective σ of the pooled std channel. If you feed mean+std and you're randomly subsampling members, the std channel's scale depends on k. Normalize it with a fixed reference, and be aware the train/eval distributions differ if eval uses full E.

5. Does a norm layer in the stem make this redundant? Partly — GroupNorm right after the first conv absorbs a lot of scale mismatch. But it doesn't fix initialization (the first conv's output is already skewed before any norm sees it), and it doesn't help LightGBM-style baselines or diagnostics. Explicit input standardization is cheap and makes everything downstream easier to reason about. Do both.

Practically: this is one more artifact next to the climatology — a per-variable {mean, std} computed over training years and valid cells, written once, referenced by name from the experiment config.

## About snow masking

Hey, claude! Could you tell me something. I have a snow density field for some date. This field should be filtered with glaicer mask (snow all-year long, arctic, mountains) and with sea mask. Actually we are not trying to predict SWE there.
  Is there some method to exclude those zones from training the model? I understand that we can exclude them from loss, but can we exclude them from forward pass as well? Like make them -inf or something.
  Just wondering.

Good question, and there's a conceptual trap worth separating out first. Let me ground this in your actual sketch.

  Ran 1 shell command

Grounded in your sketch — you already have both hooks: lsm in the static stream, and the "snow-climate mask" on the loss.

First, the reframe that matters

There are two different exclusions, and they shouldn't be conflated:

- "Don't score me here" — glacier/sea cells shouldn't contribute to RMSE/ACC. Yes, absolutely. Loss mask. You've got this.
- "Don't look here" — remove those cells from the input. You almost certainly don't want this. Ocean and glacier cells carry real predictive signal for continental SWE: SST and sea-ice state are classic sources of subseasonal predictability, and your bottleneck attention exists precisely to pick up those teleconnections. Blanking the ocean would delete the thing that layer is there to find.

So the honest answer is: the loss mask is the main mechanism, and that's not a workaround — it's the correct one.

Why -inf specifically breaks

-inf is an attention idiom. It works because it's fed into a softmax, which maps exp(-inf) → 0 — a clean, exact zero weight.

A convolution has no softmax. -inf in a conv goes straight into a weighted sum, and the first time it meets a zero or opposite-signed weight you get inf - inf = NaN or 0 * inf = NaN. One masked pixel then spreads NaN across its receptive field, and after a few layers your whole map is NaN. This is the same failure you just hit with the pt climatology, but propagating.

For dense conv inputs the neutral fill is 0 after standardization — which is exactly "this cell sits at its climatological mean," i.e. contributes no anomaly signal. Never NaN, never -inf.

Where -inf is legitimate in your model

Your bottleneck spatial self-attention — that's a real softmax over H/8 × W/8 tokens. You can genuinely exclude tokens there:

attn_mask = valid_tokens[:, None, None, :]      # [B,1,1,N]
scores = scores.masked_fill(~attn_mask, float('-inf'))

Caveat: at H/8 the mask is heavily downsampled, so it's coarse — pool the mask with the same stride as the features (max_pool if "any valid pixel counts", avg_pool + threshold if you want mostly-valid tokens). And guard against a fully-masked row, which produces all--inf → NaN after softmax.

Real forward-pass exclusion, if you want it

The technique you're reaching for is partial convolution (Liu et al., NVIDIA, originally for inpainting). Each conv window is renormalized by the fraction of valid pixels inside it:

$$x' = W^\top(X \odot M)\cdot\frac{\text{sum}(1)}{\text{sum}(M)} + b$$

Invalid pixels contribute exactly nothing, and the mask propagates through the network as a first-class tensor. Gated convolution is the softer, learned variant. This is the legitimate "exclude from the forward pass" answer — but it roughly doubles conv bookkeeping and, given your sample-size constraint, I'd treat it as a later ablation rather than a default.

What I'd actually do

1. Fill masked input values with 0 post-standardization (never NaN/-inf).
2. Add the glacier and sea masks as binary channels in your static stream, alongside lsm/sdor. This is the cheap, high-value move — the model learns those regions are different and can gate its own behavior, without you hard-coding it.
3. Keep the loss mask as the real enforcement.
4. Consider cropping the domain to your region of interest — pure compute savings, no modeling downside.

One subtle gotcha worth flagging: normalization layers. If a large fraction of your grid is masked, BatchNorm computes its statistics over those dead cells and skews them. GroupNorm/LayerNorm doesn't fix the spatial pollution either. If masked area is a big share of the domain, that's a genuine source of drift — worth checking before you chase architecture changes.

For the snow density field you started with: that one's a fair case for masking the values (density over permanent ice is physically degenerate), but do it by filling with 0/climatology and flagging via the mask channel — not by removing the cells.