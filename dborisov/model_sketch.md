# SWE U-Net — model class sketch (for discussion)

Draft architecture for the SWE bias-correction / forecast model. Goal of the
sketch: pin down the **input contract** and the **forward-pass shapes** so the
topology (full-domain vs multi-scale) and the ensemble handling stay
**config-swappable experiments**, not hardcoded decisions.

Nothing here is final — it's a template to argue over.

Conventions:
- `B` batch, `E` ensemble members (**varies 10..30**, unknown at inference),
- `C` channels (= stacked variables), `T` time steps (weekly),
- `H, W` spatial. Example full-domain numbers in parentheses are illustrative.

---

## Dataset options (experiment ladder)

Regime A everywhere = **analysis time (issuance `t0`) = INM init**; `t0` is the seam
(ERA history ends at `t0`, INM forecast starts at `t0`, no input overlap). Target =
ERA SWE at `t0 + lead`; forecast valid weeks `w1..w16`.

- **exp1 — per-frame bias correction.** ERA5 history up to `w0` + INM's `w_k` frame →
  correct SWE at `w_k`. No forecast trajectory. History = short right-aligned temporal
  pyramid (recent weekly + long monthly), flattened to channels; `sd(w0)` carries the
  long memory. *Temporal pooling of history (attention / linear / TCN) deferred — try
  later only if flatten plateaus.*
- **exp2 — causal window.** History + INM's forecast trajectory `w1..w_k` (causal
  temporal module on the forecast axis) → correct `w_k`. Adds the forecasted forcing
  *path* (biased is fine — the model learns it) beyond what `inm_swe` already integrates.
- **exp3 — rolling analysis (Regime B).** Analysis time `t0` rolls forward within a
  run's life (weekly re-correction as new ERA arrives); INM run fixed. Multiplies the
  dataset but samples are heavily correlated — an augmentation/operational arm, measured
  against exp1, not a starting point.
- **exp4 — autoregression.** ML-corrected SWE fed back as an input channel for the next
  lead. Effectively replaces INM's snow scheme → data-hungry, drift-prone; keep `inm_swe`
  as an anchor. Highest ceiling, highest risk — last.

Independent axes: exp2 (trajectory) ⟂ exp3 (rolling); combine once both work.

---

## Input contract

The model consumes a **dict of streams** (exactly what `PatchDataset` returns).
A stream is one source at one scale, with its variables stacked into channels.
Three stream *types*:

```
  ERA5 history     [B,        C_e, T_e, H, W]      reanalysis, has time, no ensemble
  INMCM forecast   [B,  E,    C_i, T_i, H, W]      dynamical model, has ensemble + time
  time-invariant   [B,        C_s,      H, W]      lsm, orography, ... no time, no ensemble
```

- **Full-domain experiment**  -> the dict has ONE stream per type, each already
  at the full domain `H x W`. No fusing of scales.
- **Multi-scale experiment**  -> the dict has SEVERAL streams per type at
  different `xyStep`; the front-end resizes each to a common working grid before
  fusion. *Same code path, more streams.*

---

## Forward pass — per-stream front-end

Shown for the INMCM stream (the only one with `E`). ERA5 skips E-POOL;
time-invariant skips both E-POOL and T-COLLAPSE.

```
  INMCM stream                         [B, E, C_i, T_i, H, W]     E varies 10..30
        |
        |   E-POOL          (swappable: mean_std | quantiles | deepsets | pma)
        |                   collapses the ensemble, permutation-invariant,
        |                   independent of E  -> see NOTE
        v
                                       [B, C_i*k, T_i, H, W]      k = #stats (mean,std -> k=2)
        |
        |   T-COLLAPSE      (swappable: flatten | temporal_attn | conv3d)
        |                   default = flatten time into channels
        v
                                       [B, C_i*k*T_i, H, W]
        |
        |   STEM  conv 3x3 + norm + act
        v
                                       [B, F, H, W]               F = per-stream stem width


  ERA5 stream      [B, C_e, T_e, H, W]  --(no E-POOL)--> T-COLLAPSE --> STEM --> [B, F, H, W]
  static stream    [B, C_s,      H, W]  --(no E-POOL, no T-COLLAPSE)--> STEM --> [B, F, H, W]
```

If multi-scale: each stream's STEM output is resized (interp / strided conv) to
the common working grid `H0 x W0`. Full-domain: everyone is already `H x W`.

---

## Forward pass — fusion + U-Net body

```
  concat all stream features on channel dim
        v
                                       [B, F_total, H, W]         (H=144, W=576 example)
        |
   == ENCODER ==                                                  depth L is config (3 shown)
   enc1  conv block        -> [B,  b, H,    W   ]  --------------- skip1 ----------------+
   down  /2                -> [B,  b, H/2,  W/2 ]   (72 x 288)                           |
   enc2  conv block        -> [B, 2b, H/2,  W/2 ]  ------------ skip2 --------------+     |
   down  /2                -> [B, 2b, H/4,  W/4 ]   (36 x 144)                      |     |
   enc3  conv block        -> [B, 4b, H/4,  W/4 ]  --------- skip3 ----------+      |     |
   down  /2                -> [B, 4b, H/8,  W/8 ]   (18 x 72)                |      |     |
        |                                                                   |      |     |
   == BOTTLENECK ==                                                         |      |     |
   conv block              -> [B, 8b, H/8,  W/8 ]                           |      |     |
   SPATIAL SELF-ATTENTION  -> [B, 8b, H/8,  W/8 ]   tokens = H/8 * W/8       |      |     |
        |                                            (= 18*72 = 1296)       |      |     |
        |   <-- this is where global context / teleconnections get mixed   |      |     |
        v                                                                   |      |     |
   == DECODER ==                                                            |      |     |
   up /2  + concat skip3   -> [B, 8b+4b, H/4, W/4] <-------------------------+      |     |
   dec3  conv block        -> [B, 4b,    H/4, W/4]                                  |     |
   up /2  + concat skip2   -> [B, 4b+2b, H/2, W/2] <--------------------------------+     |
   dec2  conv block        -> [B, 2b,    H/2, W/2]                                        |
   up /2  + concat skip1   -> [B, 2b+b,  H,   W  ] <--------------------------------------+
   dec1  conv block        -> [B,  b,    H,   W  ]
        |
   HEAD  conv 1x1          -> [B, 1, H, W]
   squeeze                 -> [B, H, W]              SWE anomaly (vs climatology)
```

Notes on the body:
- **Depth `L` is the lever** for receptive field. Patch-scale inputs need ~2-3
  levels; **full-domain (144x576) needs ~5-6** so the bottleneck actually spans
  the domain. Drawn with 3 for readability.
- **Padding** to a multiple of `2^L` (reflect) before the encoder; crop back at
  the end.
- Output is an **anomaly**; absolute SWE = anomaly + climatology (from patcher).
- Loss (outside the model): latitude-weighted RMSE / ACC, applied under the
  **snow-climate mask** so snow-free cells don't dominate.

---

## Class skeleton (config-driven)

```python
class SWEUNet(nn.Module):
    """Everything below is selected by the experiment JSON."""

    def __init__(self, cfg):
        super().__init__()
        # one front-end per declared stream
        self.fronts = nn.ModuleDict()
        for s in cfg.streams:                       # s.name, s.source, s.channels, s.t, s.scale
            self.fronts[s.name] = StreamFrontEnd(
                has_ensemble = (s.source == "inmcm"),
                has_time     = (s.t is not None),
                e_pool       = build_e_pool(cfg.e_pool),        # mean_std | quantile | deepsets | pma
                t_collapse   = build_t_collapse(cfg.t_collapse),# flatten   | temporal_attn | conv3d
                in_ch=s.channels, t=s.t, stem_ch=cfg.stem_ch,
                work_grid=cfg.work_grid,            # None for full-domain; (H0,W0) for multi-scale
            )
        self.unet = UNet(
            in_ch = cfg.stem_ch * len(cfg.streams),
            base  = cfg.base, depth = cfg.depth,
            bottleneck_attention = cfg.attention,   # heads, dim (e.g. dim=32)
        )
        self.head = nn.Conv2d(cfg.base, 1, kernel_size=1)

    def forward(self, batch):                       # batch: dict[str, Tensor]
        feats = [self.fronts[name](batch[name]) for name in self.fronts]  # each -> [B, F, H, W]
        x = torch.cat(feats, dim=1)                 # [B, F_total, H, W]
        x = self.unet(x)                            # [B, base,   H, W]
        return self.head(x).squeeze(1)              # [B, H, W]  SWE anomaly


class StreamFrontEnd(nn.Module):
    def forward(self, x):
        if self.has_ensemble:                       # [B, E, C, T, H, W]
            x = self.e_pool(x)                      # -> [B, C*k, T, H, W]   (E gone)
        if self.has_time:                           # [B, C', T, H, W]
            x = self.t_collapse(x)                  # -> [B, C'',     H, W]
        x = self.stem(x)                            # -> [B, F,       H, W]
        if self.work_grid is not None:
            x = resize(x, self.work_grid)           # multi-scale only
        return x
```

The **only thing that changes** between the full-domain and multi-scale
experiments is `cfg.streams` (and `cfg.work_grid` / `cfg.depth`). The E-handling
experiment is just `cfg.e_pool`. That's the whole point.

---

## NOTE — ensemble handling options (the `e_pool` module)

All operate on the INMCM stream `[B, E, C, T, H, W]` and return `[B, C*k, T, H, W]`
with **E removed**. Every option below is **permutation-invariant** over members
and **independent of E** (handles 10 vs 30 vs future counts). "Members as
channels" is deliberately absent — it breaks both properties.

All are special/general cases of the same set-function form:
`out = rho( pool_over_E( phi(member) ) )`.

```python
# 1 — MEAN + STD            (k=2, no params)  WORKHORSE BASELINE
def mean_std(x):                                  # x: [B, E, C, T, H, W]
    return cat([x.mean(1), x.std(1)], dim=1)      # -> [B, 2C, T, H, W]

# 1.5 — MOMENT / QUANTILE   (k=3..5, no params)  cheap upgrade, captures tails
def quantile(x, qs=(0.1, 0.5, 0.9)):
    return cat([x.mean(1), x.std(1),
                *[x.quantile(q, dim=1) for q in qs]], dim=1)   # -> [B, kC, T, H, W]

# 2 — DEEPSETS             (learned phi, mean-pool)  Zaheer 2017
def deepsets(x):
    h = phi(x)                                    # shared 1x1 conv over each member
    return rho(h.mean(1))                         # mean-pool -> [B, C', T, H, W]

# 3 — PMA / attention pool (learnable query attends over E)  Set Transformer, Lee 2019
def pma(x):                                       # q: [n_seeds, d] learnable
    return multihead_attn(query=q, key=x, value=x)# down-weights bad members per cell

# 4 — ISAB: members self-attend, then pool        most expressive, most sample-hungry
```

| option | params | captures | cost | when |
|---|---|---|---|---|
| 1 mean+std | none | center + spread | trivial | **start here** |
| 1.5 +quantiles | none | + tails / skew | trivial | cheap arm |
| 2 DeepSets | small | learned per-member features | low | if 1.x plateaus |
| 3 PMA attn | small | data-dependent member weighting | med | if 2 helps |
| 4 ISAB | med | inter-member structure | high | probably skip |

**Practical:**
- **Pool early** (before the spatial encoder, as drawn) so the U-Net runs once,
  not E times. Cheap and keeps INMCM a drop-in stream.
- **Variable-E batching:** samples in a batch have different E -> pad to max-E +
  carry a **mask** (mean/std/attention all respect the mask), or bucket by E.
- **Prior:** at 4-month lead, members are ~independent draws — *mean is the
  signal, spread is mostly irreducible uncertainty*. Expect 1.x to be hard to
  beat; learned poolers mainly help if some members are *systematically* better,
  which is itself hard to learn from ~25 winters. The ablation is cheap and the
  null result ("simple moments won") is publishable.

---

## Open knobs to decide

- **Stream topology:** full-domain (no aggregation, sample-poor) vs multi-scale
  (global context cheap, sample-rich) — launch as experiment.
- **T-COLLAPSE:** flatten (simple, already tried via 3DCNN) vs temporal attention
  (handles long ERA5 history, parallel) vs conv3d.
- **Output:** one lead time per forward pass, or multiple lead times at once
  (extra output channels / a lead-time conditioning input).
- **U-Net depth `L`:** tie to chosen domain size so bottleneck spans it.
- **Attention placement:** bottleneck only (cheap) vs also on skips (Attention
  U-Net, Oktay 2018).
-  One thing I intentionally left abstract: the STEM and resize for multi-scale (how exactly streams at different xyStep align to a common grid). For the full-domain experiment it's a no-op, so it doesn't block your first run — but it's the part that needs the most thought if/when you launch the multi-scale arm. Worth a dedicated discussion before that experiment.

---
# Notes 

## Handling lead times

Decision: predict **all lead times in one output** -> head emits `[B, L, H, W]`.
The doubt to keep in mind is not the joint output (that's the right fit) — it's
the **loss** and a **structural wrinkle about which time axis we collapse**.

### All-leads output: keep it — it's the natural fit

One INMCM forecast run from a single init date already contains the *whole* lead
trajectory (its forecast time axis = lead times), and the ERA5 history is the
same regardless of lead. So one sample -> all leads is the natural unit: one
forward pass predicts the full forecast trajectory. On top of that, joint output
is a **regularizer we want** in a sample-poor regime — the hard long leads borrow
representation from the easy short leads through the shared backbone.

### Issue 1: an averaged RMSE loss is unbalanced across leads

Leads differ enormously in both **error magnitude** and **difficulty**:
- short leads -> high skill, different anomaly variance,
- long leads  -> low skill,
- and **lead is coupled with season** (a lead-16 forecast from October targets
  deep winter; from March it targets melt), so each lead channel also mixes very
  different physical regimes and magnitudes.

A plain `mean(RMSE over leads)` is dominated by whichever leads/seasons have the
largest anomaly variance; the gradient chases those and neglects the rest.

**Fixes (any one works, in order of preference):**
1. **Train on ACC**, not raw RMSE. ACC is correlation — normalized per field —
   so it's scale-invariant across leads/seasons, and it's the headline metric.
2. **Standardize each target by its per-period climatological std** (already in
   `std.nc`) before RMSE. Loss becomes "error in units of local climatological
   variability," neutralizing both lead and season magnitude differences at once.
3. Explicit **per-lead weights** — crudest, needs tuning, avoid unless 1-2 fail.

Regardless of training loss: **always report metrics per lead** (the
skill-vs-lead curve). A single averaged number hides exactly what the paper is
about.

### Issue 2 (structural): two time axes with different roles

There are two "T" axes, and they must not be collapsed the same way:

- **ERA5 history T** (past weeks) -> genuine context -> *collapse it*
  (flatten / temporal-attn). Correct as drawn.
- **INMCM forecast T** (= the lead horizon) -> this **is** the output lead axis.
  `T-COLLAPSE`-ing it would destroy the per-lead signal we need to emit per lead.

Clean structure for the all-leads design:

```
  encode history + static  ONCE          -> shared context embedding
  broadcast context across leads, combine with each INMCM forecast frame
        (the lead axis L rides along like a batch dim)
  decode                                  -> [B, L, H, W]
```

i.e. "correct each lead's INMCM forecast, conditioned on a shared history
embedding." Sharing the (expensive) history encoding and respecting the lead
axis. The current `T-COLLAPSE` is fine for a single-lead version but conflates
the two roles for multi-lead.

**Summary:** keep all-leads output; train on ACC or std-normalized RMSE; split
the two time axes in the front-end (history-T collapsed, forecast-T preserved as
the lead axis); head -> `[B, L, H, W]`.

---

## Why stem? (why not just concat streams over channels?)

You can — for a first full-domain version it's genuinely fine. "Concat then the
U-Net's first conv" *is* a stem; the only question is whether that first
projection is **shared across all streams** (concat-first) or **per-stream**
(stem-then-concat). Both are one conv. So the real question is when the
per-stream version earns its extra params. Three reasons it does:

**1. Channel-count imbalance.** After T-collapse the streams are wildly lopsided.
Rough example:
- ERA5 history: 6 vars x 12 weeks = **72 ch**
- INMCM (mean+std): 6 vars x 2 x 16 leads = **192 ch**
- static: **4 ch**

If you concat raw -> `[B, 268, H, W]` and hit it with one shared conv, the
4-channel static input is numerically drowned by the 192-channel INMCM block —
the network has to *learn* to rebalance from a very skewed input. Per-stream
stems project each to a comparable width `F` first, so every stream enters fusion
with an equal "voice."

**2. Heterogeneous statistics.** ERA5 anomalies, bias-corrected INMCM anomalies,
and static fields (one-hot soil type, log-geopotential, min-max orography) have
nothing like the same distributions. A shared first conv has to handle all of
them with one set of filters; per-stream stems let each source be
projected/normalized appropriately before they're mixed.

**3. Spatial alignment (the multi-scale case).** You literally *can't* concat over
channels until streams share an `H x W` grid. In multi-scale, streams arrive at
different `xyStep`; the stem (+ resize) is where they're brought to a common
grid. In full-domain everyone's already aligned, so this reason disappears —
which is exactly why concat-first is fine there.

**The kicker for our design:** we already decided the front-end treats the two
time axes differently — ERA5 **history-T collapsed**, INMCM **forecast-T
preserved as the lead axis**. That processing is *per-stream by construction*. So
a per-stream front-end isn't extra structure we're adding for the stem's sake —
it's already there, and the stem is just its output projection. Concat-first only
stays simple if you *don't* do the two-time-axes split, and we do.

**Bottom line:** if all streams were homogeneous, same grid, same time semantics
— concat-first, let the U-Net's first conv be the stem, done. Given imbalanced
channels + heterogeneous sources + the history-vs-forecast time split, per-stream
stems are the cleaner default. If you want maximum simplicity for the very first
full-domain run, concat-first is a legitimate shortcut to start with — just
expect to add per-stream projection once static fields or multi-scale come in.
