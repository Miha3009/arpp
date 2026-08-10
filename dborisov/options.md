# Options

---

## T handling — collapse `[B,C,T,H,W] -> [B,C',H,W]`

> only **ERA5 history T**. **INMCM forecast T = lead axis -> NOT collapsed** (-> `[B,L,H,W]`)

- **Linear** — `[T,1] or [C,T]`

- **flatten** — `[B,C,T,H,W] -> [B,C*T,H,W]`
  - baseline · no params · fixed filters per week · `C*T` wide. params = C·T·C' at next layer (conv1d case).

- **conv3D** — `[B,C,T,H,W] --conv3d--> [B,C',1,H,W] -> [B,C',H,W]`
  - local temporal smoothness · more compute · fixed T

- **PMA** (attention pool, 1 learnable query) — `[B,C,T,H,W] -> [B,d,H,W]`
  - per-cell content-dependent week weighting (`mean` = uniform weights)
  - `q:[1,d]` · `softmax(qKᵀ):[1,T]` · weighted sum over T -> T gone
  - length-invariant (pad + masked softmax) · `+pos-enc` on real time
  - `q:[n,d]` -> keep n summary slots instead of full collapse

- **SAB → PMA** — weeks self-attend (`[T,T]`) then pool · if PMA plateaus
  - (ISAB = `O(T·m)` approx of SAB, only for big sets)

| | params | captures | cost |
|---|---|---|---|
| flatten | none | stacks weeks | trivial |
| conv3D | small | local temporal | med |
| PMA | small | week weighting | med |
| SAB→PMA | med | weeks interact | high |


+ RNN-style (Conv-LSTM)
---

## E handling — `[B,E,C,T,H,W]`, E varies

### A) collapse E (pooling) → `[B,C*k,T,H,W]`, E removed

- **Linear** — `[T,1] or [C,T]` ! sensitive to permutations, static form (E members)

> all: permutation-invariant + E-independent · form `rho(pool_E(phi(member)))`

- **mean+std** (k=2, no params) — **baseline**, center + spread
- **+quantiles** (k=3..5, no params) — + tails / skew
- **DeepSets** (small) — learned per-member features · if mean+std plateaus
- **PMA** (small) — data-dependent member weighting
  - = same as PMA-over-T, **no pos-enc** (members unordered)
- **ISAB** (med) — members self-attend then pool · most expressive, probably skip

> pool **early** (U-Net runs once) · variable-E: pad-to-maxE + mask, or bucket by E
> prior: mean = signal, spread ≈ irreducible → mean+std hard to beat

### B) don't collapse E → members as augmentation

- random-subsample `k` members per draw → 1 winter = many inputs (bagging/dropout on set axis)
- **on-manifold** noise: physics constant, perturbation varies
  - vs Gaussian noise = off-manifold (breaks spatial / inter-var / temporal consistency)
  - ensemble = model's own input-uncertainty distribution, real covariance
- buys: ×effective samples · regularizes pooler · teaches E-invariance · ~free
- caveats: regularization not info (ceiling = 25 winters) · doesn't fix model bias · **train-only, fixed eval** · target unchanged
- config: `train_subsample:[k_min,k_max]` before pooler (composes with A)
