# Model details — axis handling

Companion to `model_sketch.md`. This file drills into the two aggregation
problems the front-end has to solve: collapsing the **time axis `T`** and
handling the **ensemble axis `E`**. Both are config-swappable knobs
(`t_collapse` / `e_pool` in the experiment JSON).

Conventions (same as the sketch): `B` batch, `E` ensemble members (varies
10..30, unknown at inference), `C` channels (stacked variables), `T` time steps
(weekly), `H, W` spatial.

---

## Handling T (time axis)

A stream with time arrives as `[B, C, T, H, W]` and must leave the front-end as
`[B, C', H, W]` — the `T` axis collapsed into the feature representation. Three
options, increasing in expressiveness and cost.

> **Scope reminder (from `model_sketch.md`, Issue 2).** There are *two* time
> axes with different roles. **ERA5 history T** (past weeks of context) is the
> one we collapse here. **INMCM forecast T** is the *lead-time output axis* and
> must **not** be collapsed — it rides through to the head as `[B, L, H, W]`.
> So everything below applies to **history streams**, not the forecast lead
> axis.

### Option 1 — flatten into channels (default / baseline)

Reshape the `T` weeks into the channel dimension:

```
[B, C, T, H, W]  ->  [B, C*T, H, W]
```

Simple, no parameters of its own, already tried via the 3D-CNN line. The first
conv of the stem then mixes the flattened time-channels. Downside: every week
gets a fixed set of filters regardless of content, and `C*T` can get wide.

### Option 2 — conv3D

Run 3D convolutions over `(T, H, W)`, then reduce `T` (stride / pool / a final
`T`-spanning kernel) down to 1:

```
[B, C, T, H, W]  --conv3d(s)-->  [B, C', 1, H, W]  ->  [B, C', H, W]
```

Gives a local temporal inductive bias (smoothness across adjacent weeks) that
flatten lacks. Costs more compute and assumes a fixed `T` (kernel sizes are
baked in).

### Option 3 — PMA (attention pooling over T)

Collapse `T` with **attention pooling using a learnable query** — the same
set-function primitive used for the E axis (Set Transformer, Lee 2019). This is
the option worth expanding on, because it's the most flexible and its mechanics
are subtle.

**The core idea.** A plain `flatten` or `mean` over `T` treats every week the
same. Attention instead computes, *per spatial cell*, **how much each week should
contribute**, with weights that depend on the actual content. `mean` is the
degenerate case where all weights are forced to `1/T`.

**Shape bookkeeping.**

```
x:  [B, C, T, H, W]

1. Make T the token axis, fold B,H,W into a batch:
       permute/reshape  ->  [B*H*W, T, C]
   Now (B·H·W) independent sequences, each of length T, each token a C-vector.
   Every spatial cell gets its own little attention over its T weeks.

2. (optional) add positional encoding over T  -> tokens know week 0 vs week 11.

3. Attention pool with ONE learnable query q : [1, d]:
       Q = q                      # [1, d]          ONE query -> output length 1
       K = x W_k                  # [B*H*W, T, d]
       V = x W_v                  # [B*H*W, T, d]
       attn = softmax(Q Kᵀ / √d)  # [B*H*W, 1, T]   <- the per-week weights
       out  = attn @ V            # [B*H*W, 1, d]   <- T is GONE

4. Unfold spatial back:
       [B*H*W, 1, d]  ->  [B, d, H, W]
```

**Why one query? (vs LLM self-attention).** The general rule for *any* attention
block is: **output length = number of queries**.

```
Q : [Lq, d]      attn = softmax(Q Kᵀ/√d) : [Lq, Lk]      out = attn @ V : [Lq, d]
K : [Lk, d]                                               ^^^ output has Lq rows
V : [Lk, d]
```

In an LLM, `Q`, `K`, `V` are all projections of the *same* T-token sequence, so
`Lq = Lk = T` → the matrix is the square `[T, T]` and length is preserved
(T tokens in, T tokens out). Self-attention is a sequence-**to-sequence**
transform; it fundamentally **cannot reduce length**, because every input token
contributes its own query → its own output row.

To collapse `T → 1` you need `Lq = 1`. But there's no "summary token" in the
input to derive that query from — so you **invent** it as a free learned
parameter `q : [1, d]` (`nn.Parameter`). Then `softmax(QKᵀ)` is a single `[1, T]`
row — one probability distribution over the weeks — and `attn @ V` is one
weighted average. T is gone.

**What the query *means*.** It's a fixed *question* the model asks of every
sequence: *"across these T weeks, which ones — and which features in them —
should I pull out to summarize this cell?"* The query itself is the same for
every cell and sample; what's data-dependent is the **answer** (the `[1, T]`
weights, since `K` depends on the actual weeks). Same pattern as BERT's `[CLS]`
token, DETR/Perceiver learned queries, Set Transformer's PMA seed.

**Length-invariance (the free lunch).** A query of shape `[1, d]` reads a
sequence of *any* length down to one vector — `Lk` can be 8 or 16 weeks and `q`
doesn't change shape. Different lengths in one batch → **pad to max length +
carry a mask**, and apply the mask *inside the softmax* so padded positions get
exactly zero weight:

```
logits = Q @ Kᵀ / √d
logits = logits.masked_fill(~mask[:, None, :], float('-inf'))
attn   = softmax(logits, dim=-1)   # softmax(-inf) = 0  -> pad contributes nothing
```

(For history streams `T` is usually fixed-length, so T-padding may be
unnecessary — variable length is really the E problem below. Keep positional
encoding on the *real* time index.)

**Want `n` summaries instead of a full collapse?** Make `q : [n, d]` → output
`[n, d]` (e.g. "early / mid / late winter" slots). The output-length-equals-
query-count rule is the one lever.

**Going further — self-attention first, then pool.** If you want the weeks to
exchange information *before* pooling (full LLM-style self-attention), prepend a
**SAB** = `MAB(X, X)` (the square `[T, T]` matrix, `T → T`), then PMA to collapse.
For large sets there's **ISAB** = the `O(T·m)` inducing-point *approximation* of
SAB. For history (`T ~ 12`) the `O(T²)` cost is nothing, so **SAB → PMA** is the
direct choice; ISAB only earns its keep when the set is big.

| t_collapse | params | captures | cost | when |
|---|---|---|---|---|
| flatten | none | nothing temporal, just stacks | trivial | **baseline** |
| conv3d | small | local temporal smoothness | med | local inductive bias |
| PMA | small | content-dependent week weighting | med | long/variable history |
| SAB → PMA | med | weeks interact, then pool | higher | if PMA plateaus |

---

## Handling E (ensemble axis)

The INMCM stream arrives as `[B, E, C, T, H, W]` with `E` varying 10..30 (unknown
at inference). Two fundamentally different strategies: **collapse E** (pool it
away in the front-end) or **don't collapse E** (use the members as augmentation).

### Strategy A — collapse E (pooling)

All poolers take `[B, E, C, T, H, W]` and return `[B, C*k, T, H, W]` with **E
removed**. Every option is **permutation-invariant** over members and
**independent of E** (handles 10 vs 30 vs future counts). "Members as channels"
is deliberately absent — it breaks both properties. All are special/general cases
of the same set-function form: `out = rho( pool_over_E( phi(member) ) )`.

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

**PMA over E is mechanically identical to PMA over T** (see the T section above):
a learnable query of shape `[1, d]` reads the variable-length member set down to
one vector, length-invariant by construction. The only difference is the axis —
and that members are an **unordered set**, so **no positional encoding** (unlike
T). Variable-E batching uses the same pad-to-max-E + masked-softmax trick.

**Practical notes:**
- **Pool early** (before the spatial encoder) so the U-Net runs once, not E
  times. Keeps INMCM a drop-in stream.
- **Variable-E batching:** pad to max-E + carry a mask (mean/std/attention all
  respect it — attention masks the softmax, moments mask the reduction: divide by
  the *real* count, not max-E), or bucket by E.
- **Prior:** at 4-month lead, members are ~independent draws — *mean is the
  signal, spread is mostly irreducible uncertainty*. Expect 1.x to be hard to
  beat; learned poolers mainly help if some members are *systematically* better,
  hard to learn from ~25 winters. The ablation is cheap and the null result
  ("simple moments won") is publishable.

### Strategy B — don't collapse E: members as augmentation

A different framing discovered in discussion: instead of (or in addition to)
pooling E away, treat the ensemble members as **on-manifold data augmentation**.

**The idea.** Each time a sample is drawn, feed a *random subset* of its members
(sample `k`, `k` itself random, drop the rest). One winter's forecast becomes
*many* distinct training inputs across epochs. This is essentially **bagging /
dropout on the set axis**.

**Why it's principled — on-manifold vs off-manifold.** Contrast with classic
"add weak Gaussian noise to input fields":

- **Gaussian noise is off-manifold.** It perturbs each pixel independently,
  breaking spatial coherence, inter-variable balance (temperature vs pressure),
  and temporal continuity. The result is a state that *could never physically
  occur*. You're guessing the noise distribution and teaching invariance to
  corruption that never appears at inference.
- **Ensemble members are on-manifold.** Each member is the INMCM model integrated
  forward with full physics from a perturbed initial state — mass-conserving,
  balanced, spatially and inter-variable consistent. Subsampling moves around the
  manifold of *physically valid* states, never off it. **The perturbation varies;
  the physics stays constant.**

The deeper point: **the ensemble *is* the model's own estimate of the input
uncertainty distribution**, with the correct covariance structure baked in.
Gaussian noise is you hand-injecting guessed uncertainty; member subsampling
*resamples the uncertainty nature already produced.* Strictly more honest.

**What it buys (esp. in our sample-poor regime, ~25 winters):**
- Multiplies effective sample count — the thing we're most starved for.
- Regularizes the pooler hard and teaches **E-invariance**, which we need because
  inference-time E may differ from training E.
- Nearly free given the masking already wired for variable-E.

**Caveats — don't fool yourself:**
- **It's regularization, not information.** Subsampling can't add signal that
  wasn't there; it makes the model robust and harder to overfit. The skill
  ceiling is still set by the 25 winters.
- **Doesn't fix model bias.** Members share INMCM's *systematic* biases (same
  physics, perturbed ICs), so subsampling captures sampled internal variability,
  not structural error. Model bias is what the ERA5 target + bias-correction
  handle, not the augmentation.
- **Train-only.** Augment at train time *only*. At eval, use a fixed protocol
  (full/real member count, full history) so skill numbers are comparable across
  experiments.
- **Keep the target fixed.** Vary the input *view* (which members); the SWE
  target for that init date never changes.

**Config sketch:** a train-time `e_pool` knob, e.g. `train_subsample: [k_min,
k_max]`, applied *before* whatever pooler is selected (the two compose — you can
subsample members *and* mean/std-pool the survivors). Eval forces the full set.

**Slogan:** the ensemble is *"natural noise with the physics held fixed,"* and
subsampling it is augmentation that stays on the data manifold — precisely the
property hand-injected Gaussian noise can't give you.
