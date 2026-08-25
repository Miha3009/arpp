"""Training + evaluation for SWEUNet (exp1).

Data contract (the collated dict from ContiguousDataset, everything already
normalized):
    model input   : the whole batch dict (SWEUNet.forward consumes it)
    prediction    : model(batch) -> [B, L, Hw, Ww]   (normalized SWE log-anomaly)
    target        : batch[target_key]        [B, L, He, We]
    loss mask     : batch[mask_key]          [B, L, He, We]  bool, True = EXCLUDE
                    (structural + seasonally-dry cells; ~mask = the cells we score)

All metrics are computed on the masked, latitude-weighted cells. Because the
model predicts ANOMALIES, the *climatology forecast is the zero field*, so:
    RMSE_clim = sqrt(<w * y^2>)              (y = target anomaly)
    ACC_clim  = 0                            (no anomaly pattern to correlate)
which makes the skill scores baseline-free:
    RMSESS = 1 - RMSE_model / RMSE_clim      (>0 beats climatology)
    ACC    = <w * p*y> / sqrt(<w*p^2><w*y^2>)  (uncentered; anomalies ~ zero-mean)
    PSS    = 1 - RMSE_model / RMSE_persist   (persistence = t0 anomaly held over L)

Metrics are returned at three aggregation levels (all masked+weighted):
    1. global  : one scalar over all cells and leads
    2. per_lead: [L]  over all cells, per lead time
    3. spatial : [He,We] over the date+lead axis, per cell (skill maps)

Note RMSE is in *normalized* units; ACC / RMSESS / PSS are scale-invariant, so
they read the same in physical space. Denormalize RMSE with the target's std if
you want mm-scale numbers.
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm


# =====================================================================================
# helpers
# =====================================================================================
def cos_lat_weights(lat_min_deg, lat_max_deg, n):
    """Area weights cos(lat) for a grid of n rows spanning [lat_min, lat_max] degrees.
    Orientation-agnostic (cos is symmetric). Pass the OUTPUT grid extent (work_hw[0]).
    Returns [n]; feed to train()/evaluate() as `lat_weights`."""
    lat = torch.linspace(lat_min_deg, lat_max_deg, n)
    return torch.cos(torch.deg2rad(lat)).clamp_min(0.0)


def _match_grid(pred, target, valid):
    """Bring target + valid mask onto the prediction grid (no-op at full res)."""
    hw = pred.shape[-2:]
    if target.shape[-2:] != hw:
        print('Interpolating target to match pred grid...')
        target = F.interpolate(target, size=hw, mode='bilinear', align_corners=False)
        valid = F.interpolate(valid.float(), size=hw, mode='nearest') > 0.5
    return target, valid


def _weight_field(valid, lat_weights):
    """W = (included cells) * cos(lat), broadcast to [B,L,H,W]. lat_weights: [H] or None."""
    W = valid.float()
    if lat_weights is not None:
        W = W * lat_weights.to(W.device).view(1, 1, -1, 1)
    return W


def masked_weighted_mse(pred, target, valid, lat_weights, per_lead_balanced=True):
    """Latitude-weighted MSE over the unmasked cells.

    per_lead_balanced: normalize each lead by its own valid-weight then average the
    leads equally, so a lead with more valid cells (or a larger-variance season)
    can't dominate the gradient. This is the per-lead-balanced loss.
    """
    W = _weight_field(valid, lat_weights)
    se = (pred - target) ** 2 * W
    if per_lead_balanced:
        num = se.sum(dim=(0, 2, 3))                 # [L]
        den = W.sum(dim=(0, 2, 3)).clamp_min(1e-8)  # [L]
        return (num / den).mean()
    return se.sum() / W.sum().clamp_min(1e-8)


def _finalize(se, tt, pt, pp, w, ep2=None):
    """Turn accumulated sums into metrics; nan where a cell/lead was never scored."""
    ok = w > 0
    inv = torch.where(ok, 1.0 / w.clamp_min(1e-12), torch.zeros_like(w))
    rmse = (se * inv).sqrt()
    rmse_clim = (tt * inv).sqrt()
    acc = pt / (pp * tt).clamp_min(1e-12).sqrt()
    out = {
        'rmse': torch.where(ok, rmse, torch.nan),
        'acc': torch.where(ok, acc, torch.nan),
        'rmsess': torch.where(ok, 1.0 - rmse / rmse_clim.clamp_min(1e-12), torch.nan),
    }
    if ep2 is not None:
        rmse_p = (ep2 * inv).sqrt()
        out['rmse_persist'] = torch.where(ok, rmse_p, torch.nan)
        out['pss'] = torch.where(ok, 1.0 - rmse / rmse_p.clamp_min(1e-12), torch.nan)
    return out


def _persistence(batch, persist_key, target_key, pred, norm_stats):
    """t0 observed anomaly held constant over all leads, expressed in the TARGET's
    normalized units. Source = last frame of a history feature [B,Ts,H,W]. History and
    target are normalized by different std, so rescale by std_hist/std_target."""
    hist = batch[persist_key]                       # [B, Ts, He, We]
    last = hist[:, -1:]                             # [B, 1, He, We]  (window ends at t0)
    if last.shape[-2:] != pred.shape[-2:]:
        last = F.interpolate(last, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    if norm_stats is not None:
        factor = norm_stats[persist_key]['std'] / norm_stats[target_key]['std']
        last = last * factor
    return last.expand_as(pred)                     # [B, L, H, W]


# =====================================================================================
# evaluate
# =====================================================================================
@torch.no_grad()
def evaluate(model, loader, *, device, target_key, mask_key=None,
             lat_weights=None, persist_key=None, norm_stats=None):
    """Stream the test set and return RMSE / ACC / RMSESS (+ PSS if persist_key) at
    the global, per-lead and spatial levels.

    mask_key defaults to f'{target_key}_mask' (the climate loss mask); pass mask_key=''
    to score every cell. Sufficient stats are accumulated per lead ([L]) and per cell
    ([H,W]); the global scalars are just the per-lead sums summed over L (no third
    pass). Reductions run in fp32 on-device (mps has no fp64), accumulated in fp64 cpu.
    """
    model.eval()
    if mask_key is None:
        mask_key = f'{target_key}_mask'
    mask_key = mask_key or None
    lat_weights = None if lat_weights is None else lat_weights.to(device)

    # lazily-sized fp64 cpu accumulators
    pl = None   # per-lead   sums: [L]  for each of se,tt,pt,pp,w(,ep2)
    sp = None   # per-cell   sums: [H,W]

    for batch in tqdm(loader, desc='eval', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch)                                  # [B, L, H, W]
        target = batch[target_key]
        valid = torch.ones_like(pred, dtype=torch.bool) if mask_key is None else ~batch[mask_key]
        target, valid = _match_grid(pred, target, valid)
        W = _weight_field(valid, lat_weights)                # [B, L, H, W]

        prod = {
            'se': (pred - target) ** 2 * W,
            'tt': target * target * W,
            'pt': pred * target * W,
            'pp': pred * pred * W,
            'w':  W,
        }
        if persist_key is not None:
            persist = _persistence(batch, persist_key, target_key, pred, norm_stats)
            prod['ep2'] = (persist - target) ** 2 * W

        # reduce this batch, move the small results to the fp64 cpu accumulators
        pl_b = {k: v.sum(dim=(0, 2, 3)).float().cpu() for k, v in prod.items()}   # [L]
        sp_b = {k: v.sum(dim=(0, 1)).float().cpu() for k, v in prod.items()}      # [H,W]
        if pl is None:
            pl, sp = pl_b, sp_b
        else:
            for k in pl_b:
                pl[k] += pl_b[k]
                sp[k] += sp_b[k]

    ep_pl = pl.get('ep2')
    ep_gl = None if ep_pl is None else ep_pl.sum()
    per_lead = _finalize(pl['se'], pl['tt'], pl['pt'], pl['pp'], pl['w'], ep_pl)
    global_ = _finalize(*(pl[k].sum() for k in ('se', 'tt', 'pt', 'pp', 'w')), ep_gl)
    spatial = _finalize(sp['se'], sp['tt'], sp['pt'], sp['pp'], sp['w'])  # persist maps skipped

    return {
        'global': {k: float(v) for k, v in global_.items()},
        'per_lead': {k: v for k, v in per_lead.items()},     # each [L]
        'spatial': {k: v for k, v in spatial.items()},       # each [H,W]
    }


# =====================================================================================
# lr schedule
# =====================================================================================
def _build_scheduler(name, opt, epochs, steps_per_epoch, warmup_epochs, min_lr):
    """Returns (scheduler, step_per_batch). step_per_batch=True means .step() after
    every optimizer step (OneCycle); otherwise it steps once per epoch.
        cosine  : cosine anneal to min_lr, optional linear warmup (epoch-stepped)
        plateau : halve lr when test RMSE stalls (epoch-stepped, fed the metric)
        onecycle: one-cycle super-convergence (batch-stepped)
        none    : constant lr
    """
    name = (name or 'none').lower()
    if name == 'none':
        return None, False
    if name == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=max(1, epochs // 10), min_lr=min_lr)
        return sched, False
    if name == 'onecycle':
        base_lr = opt.param_groups[0]['lr']
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=base_lr, epochs=epochs, steps_per_epoch=max(1, steps_per_epoch))
        return sched, True
    if name == 'cosine':
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
        if warmup_epochs > 0:
            warm = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=warmup_epochs)
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, [warm, cos], milestones=[warmup_epochs])
        else:
            sched = cos
        return sched, False
    raise ValueError(f"unknown scheduler '{name}'")


# =====================================================================================
# history persistence
# =====================================================================================
def _atomic_save(obj, path):
    """torch.save via a .tmp rename so a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.rename(path)


def _save_epoch(out_dir, row, scalars_accum):
    """Persist one epoch's full history row (global + per_lead + spatial tensors, NaNs and
    all) as epoch_XXX.pt -- lossless. Also rewrite a slim scalars.json (globals + per_lead
    as short lists) for quick pandas/plotting without loading the big spatial maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(row, out_dir / f"epoch_{row['epoch']:03d}.pt")

    g, pl = row['metrics']['global'], row['metrics']['per_lead']
    scalars_accum.append({
        'epoch': row['epoch'], 'train_loss': row['train_loss'], 'lr': row['lr'],
        **{f'global_{k}': v for k, v in g.items()},
        **{f'perlead_{k}': [round(float(x), 6) for x in pl[k].tolist()] for k in pl},
    })
    tmp = out_dir / 'scalars.json.tmp'
    tmp.write_text(json.dumps(scalars_accum, indent=2))
    tmp.rename(out_dir / 'scalars.json')


def load_history(out_dir):
    """Reload the per-epoch history rows written by train(out_dir=...), in epoch order."""
    out_dir = Path(out_dir)
    return [torch.load(p, map_location='cpu', weights_only=False)
            for p in sorted(out_dir.glob('epoch_*.pt'))]


# =====================================================================================
# train
# =====================================================================================
def train(model, train_loader, test_loader, *, epochs=50, lr=3e-4, weight_decay=1e-4,
          device=None, target_key='target_sd_log_17weeks', mask_key=None,
          lat_weights=None, grad_clip=1.0, persist_key=None, norm_stats=None,
          scheduler='cosine', warmup_epochs=0, min_lr=1e-6, out_dir=None,
          save_weights='all', log_fn=print):
    """Train SWEUNet with the mask-aware, latitude-weighted per-lead-balanced loss.
    Logs mean train loss + the global eval metrics + current lr every epoch.

    mask_key defaults to f'{target_key}_mask' (set mask_key='' to disable masking).
    scheduler: 'cosine' (default) | 'plateau' | 'onecycle' | 'none' (see _build_scheduler).
    out_dir: if set, persist each epoch's full history row (global+per_lead+spatial) to
        out_dir/epoch_XXX.pt plus a slim out_dir/scalars.json (reload via load_history).
    save_weights (needs out_dir): 'all' -> model_epoch_XXX.pt every epoch (+ model_best.pt
        by lowest test RMSE); 'last' -> overwrite model_last.pt each epoch (+ best);
        'best' -> only model_best.pt; 'none' -> no weights.
    Returns the per-epoch history list.
    """
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    if mask_key is None:
        mask_key = f'{target_key}_mask'
    mask_key = mask_key or None
    model.to(device)
    lw = None if lat_weights is None else lat_weights.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else 1
    sched, step_per_batch = _build_scheduler(scheduler, opt, epochs, steps_per_epoch,
                                             warmup_epochs, min_lr)
    is_plateau = isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    out_dir = Path(out_dir) if out_dir else None
    scalars_accum = []
    best_rmse = float('inf')

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, nb = 0.0, 0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{epochs}', leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)                              # [B, L, H, W]
            target = batch[target_key]
            valid = torch.ones_like(pred, dtype=torch.bool) if mask_key is None else ~batch[mask_key]
            target, valid = _match_grid(pred, target, valid)
            loss = masked_weighted_mse(pred, target, valid, lw)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if step_per_batch and sched is not None:
                sched.step()
            running += loss.item()
            nb += 1
            pbar.set_postfix(mse=f'{running / nb:.4f}')

        train_loss = running / max(nb, 1)
        metrics = evaluate(model, test_loader, device=device, target_key=target_key,
                           mask_key=mask_key, lat_weights=lat_weights,
                           persist_key=persist_key, norm_stats=norm_stats)
        g = metrics['global']

        if sched is not None and not step_per_batch:      # epoch-stepped schedulers
            sched.step(g['rmse']) if is_plateau else sched.step()
        cur_lr = opt.param_groups[0]['lr']

        row = {'epoch': epoch, 'train_loss': train_loss, 'lr': cur_lr,
               **{f'test_{k}': v for k, v in g.items()}, 'metrics': metrics}
        history.append(row)
        if out_dir is not None:
            _save_epoch(out_dir, row, scalars_accum)
            if save_weights != 'none':
                sd = model.state_dict()
                if save_weights == 'all':
                    _atomic_save(sd, out_dir / f'model_epoch_{epoch:03d}.pt')
                elif save_weights == 'last':
                    _atomic_save(sd, out_dir / 'model_last.pt')
                if g['rmse'] == g['rmse'] and g['rmse'] < best_rmse:   # finite & improved
                    best_rmse = g['rmse']
                    _atomic_save(sd, out_dir / 'model_best.pt')
        msg = (f"epoch {epoch:3d} | lr {cur_lr:.2e} | train_mse {train_loss:.4f} | "
               f"RMSE {g['rmse']:.4f}  ACC {g['acc']:.3f}  RMSESS {g['rmsess']:+.3f}")
        if 'pss' in g:
            msg += f"  PSS {g['pss']:+.3f}"
        log_fn(msg)

    return history  
