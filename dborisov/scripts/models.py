import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================================================
# SWEUNet -- exp1 model: per-frame bias correction, all leads in one pass.
# =====================================================================================
#
# Input contract (the collated dict from ContiguousDataset). Suffix {sid} = inm scale id:
#   forecast (INM):   inm_<var>_{sid}          [B, E, L, Hi, Wi]   (E = ensemble, L = lead axis)
#   history  (ERA):   era_<var>_<scale>        [B, Ts, He, We]     (no E; Ts = history window)
#   statics/coords:   z, sdor, lsm, glacier    [B, 1, He, We]
#                     lat_era, lon_era         [B, He] / [B, We]   (1D linspaces)
#   conditioning:     inm_lead_time_{sid}      [B, L]  (days)
#                     valid_doy_sin_{sid}      [B, L]
#                     valid_doy_cos_{sid}      [B, L]
#
# Data flow:
#   forecast  -> e_pool(E) -> fold L into batch -> resize -> [B*L, 2C, Hw, Ww]
#   history   -> t_collapse(Ts) -> resize -> broadcast over L -> [B*L, Chist, Hw, Ww]
#   statics   -> resize -> broadcast over L -> [B*L, Cstat, Hw, Ww]
#   concat -> shared UNet (FiLM'd by lead+season, 2D-PE attention at bottleneck)
#          -> head -> unfold L -> [B, L, Hw, Ww]  (SWE anomaly, one map per lead)
#
# NB: E handling. Either keep E and pool at runtime (default, e_pooled=False; needs a
#     collate that stacks a fixed E), OR let the dataset pre-pool E to (mean,std) and store
#     each inm field as [.,2,L,H,W] -> set cfg['e_pooled']=True to skip the runtime pool.
#     Pre-pooling shrinks the cache (E -> 2) and sidesteps ragged E; it fixes mean/std pooling.


class SWEUNet(nn.Module):
    '''
    Here's the full path, with concrete numbers plugged in. Assumptions (so the shapes are real):

- B=4 (batch), E=10 (ensemble members), L=17 (lead frames, the 17weeks scale)
- INM grid Hi×Wi = 91×360; ERA grid He×We = 361×1440
- work_hw = (176, 704); depth=3, base=32, cond_dim=32
- forecast = 2 INM vars (inm_tp_log, inm_ww_log); history = 2 ERA vars sharing one scale (Ts=6); statics = 4 surf fields + 2 coord planes; t_collapse='flatten', e_pool='mean_std'

---
0. Collate (DataLoader). Stacks each per-sample dict into a batched dict. INM field keys → [B,E,L,Hi,Wi], history → [B,Ts,He,We], statics → [B,1,He,We], lead/season → [B,L]. (Default collate works because E is fixed at 10 here; ragged E would need collate_ragged_ensemble.)

1. _group (models.py:52). Splits the flat dict into 5 role buckets — forecast, history, statics, coords, cond — by key prefix. Deliberately drops target_* and *_mask (those are labels/loss-masks, never model inputs). No shape change.

2. Forecast — stack vars. torch.stack the 2 INM vars along a new channel dim → [B, E, C, L, Hi, Wi] = [4, 10, 2, 17, 91, 360].

3. Forecast — e_pool (MeanStdPool). Mean+std over the E axis → doubles channels, removes E → [B, 2C, L, Hi, Wi] = [4, 4, 17, 91, 360]. This is where the ragged/variable ensemble becomes a fixed-size summary.

4. Forecast — fold L into batch. permute+reshape moves the lead axis into the batch → [B·L, 2C, Hi, Wi] = [68, 4, 91, 360]. Key move: every lead frame now looks like an independent sample to the conv trunk.

5. Forecast — _resize. Bilinear-interpolate the coarse INM grid up to the work grid → [68, 4, 176, 704]. (x_fore)

6. History — t_collapse (FlattenT). Stack the 2 ERA vars → [B, C, Ts, He, We] = [4, 2, 6, 361, 1440], then flatten the Ts history window into channels → [B, C·Ts, He, We] = [4, 12, 361, 1440], resize → [4, 12, 176, 704]. (x_hist) — note this is per-sample, no L axis: history is the same shared context for all leads.

7. Statics + coords. Resize each of the 4 surf fields to [4,1,176,704]; _coord_planes broadcasts the 1D lat/lon linspaces into 2 planes [4,2,176,704]; concat → x_stat [4, 6, 176, 704]. Also per-sample.

8. Broadcast context over leads. x_hist and x_stat are repeat_interleave(L) along batch so they align with the folded forecast frames ([4,…]→[68,…]), then everything is concatenated on channels → x = [B·L, F, Hw, Ww] = [68, 22, 176, 704] (F = 4 forecast + 12 history + 6 static).

9. Build conditioning vector. lead, doy_sin, doy_cos reshaped [B,L]→[B·L]=[68], fed to LeadSeasonConditioner → c = [68, 32]. One condition row per folded frame (its lead + valid-season).

10. _pad. Reflect-pad H,W up to a multiple of 2**depth=8 so the down/up path lines up. Here 176,704 already divide 8 → no-op (this is what saves you on the odd 361-style grids).
11. UNetFiLM(x, c) (models.py:302). Shared trunk over all 68 frames: stem LazyConv2d→32 ch; 3× (FiLMConvBlock + strided down) contracting to [68,256,22,88]; bottleneck           FiLMConvBlock + 2D-PE spatial self-attention (global receptive field); 3× (ncat + FiLMConvBlock) back to [68, 32, 176, 704]. FiLM injects c at everyblock so the one trunk specialises per lead/season.                                                                                                                              
12. head. 1×1 conv 32→1 → [68, 1, 176, 704]. Projects features to a single SWE-anomaly value per cell.                                                                           
13. _crop + reshape. Crop off any padding, then unfold the batch back into (B, L) → out = [B, L, Hw, Ww] = [4, 17, 176, 704] — one SWE-anomaly map per lead week, for each sample in the batch.

    '''
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.work_hw = tuple(cfg['work_hw'])            # common grid, e.g. (176, 704)
        # e_pooled=True  -> the DATASET already pooled E to (mean,std) and stores each inm
        # field as [.,2,L,H,W]; skip the runtime pool. False -> fields are [.,E,L,H,W] and
        # we pool here. Either way the forecast enters the trunk as [B,2C,L,H,W], same
        # channel order (means..., stds...), so the two are weight-compatible.
        self.e_pooled = cfg.get('e_pooled', False)
        self.e_pool = None if self.e_pooled else build_e_pool(cfg.get('e_pool', 'mean_std'))
        self.t_collapse = build_t_collapse(cfg.get('t_collapse', 'flatten'),
                                           dim=cfg.get('t_collapse_dim'),
                                           heads=cfg.get('t_collapse_heads', 4),
                                           k=cfg.get('t_collapse_k', 1))
        self.cond = LeadSeasonConditioner(cond_dim=cfg.get('cond_dim', 32),
                                          n_freqs=cfg.get('n_freqs', 6),
                                          max_lead_days=cfg.get('max_lead_days', 120.0))
        self.unet = UNetFiLM(base=cfg.get('base', 32),
                             depth=cfg.get('depth', 3),
                             cond_dim=cfg.get('cond_dim', 32),
                             attn_heads=cfg.get('attn_heads', 4))
        self.head = nn.Conv2d(self.unet.out_ch, 1, kernel_size=1)

    # -- stream grouping -------------------------------------------------------------
    @staticmethod
    def _group(batch):
        cond = {k: v for k, v in batch.items()
                if k.startswith(('inm_lead_time', 'valid_doy'))}
        coords = {k: v for k, v in batch.items() if k.startswith(('lat_', 'lon_'))}
        forecast = {k: v for k, v in batch.items()
                    if k.startswith('inm_') and not k.startswith('inm_lead_time')}
        history = {k: v for k, v in batch.items() if k.startswith('hist_')}
        statics = {k: v for k, v in batch.items()
                   if k not in cond and k not in coords
                   and k not in forecast and k not in history
                   and not k.startswith(('target'))
                   and not k.endswith(('mask'))}
        return forecast, history, statics, coords, cond

    def _resize(self, x):                              # [N, C, H, W] -> work grid
        if x.shape[-2:] == self.work_hw:
            return x
        # align_corners=True: node-registered grids. With work_hw = (4*Hi-3, 4*Wi-3) every INM
        # node lands EXACTLY on its ERA node (INM[i] == ERA[4i]); the 3 nodes per gap are blended.
        return F.interpolate(x, size=self.work_hw, mode='bilinear', align_corners=True)

    def forward(self, batch):
        forecast, history, statics, coords, cond = self._group(batch)

        # ---- conditioning: find the forecast scale, read its lead/season, fold L ----
        lead_key = next(k for k in cond if k.startswith('inm_lead_time'))
        sid = lead_key[len('inm_lead_time_'):]
        lead = cond[lead_key]                          # [B, L]
        doy_sin = cond[f'valid_doy_sin_{sid}']         # [B, L]
        doy_cos = cond[f'valid_doy_cos_{sid}']
        B, L = lead.shape

        # ---- forecast stream: per var -> [B,2C,L,Hi,Wi] -> fold L into batch -------
        fkeys = [k for k in forecast if k.endswith(sid)]
        if self.e_pooled:                                          # each var already [B,2,L,Hi,Wi]
            fstack = torch.stack([forecast[k] for k in fkeys], dim=2)   # [B, 2, C, L, Hi, Wi]
            fstack = fstack.reshape(B, 2 * len(fkeys), *fstack.shape[3:])  # [B, 2C, L, Hi, Wi]
        else:                                                     # each var [B,E,L,Hi,Wi]
            fstack = torch.stack([forecast[k] for k in fkeys], dim=2)   # [B, E, C, L, Hi, Wi]
            fstack = self.e_pool(fstack)                                # [B, 2C, L, Hi, Wi]
        C2, Hi, Wi = fstack.shape[1], fstack.shape[-2], fstack.shape[-1]
        fstack = fstack.permute(0, 2, 1, 3, 4).reshape(B * L, C2, Hi, Wi)  # fold L into batch
        x_fore = self._resize(fstack)                               # [B*L, 2C, Hw, Ww]

        # ---- history stream: per era scale, t_collapse over Ts, resize, concat -----
        hist_scales = {}
        for k, v in history.items():                   # v: [B, Ts, He, We]
            scale = k.rsplit('_', 1)[-1]
            hist_scales.setdefault(scale, []).append(v)
        hist_feats = []
        for scale, vs in hist_scales.items():
            hs = torch.stack(vs, dim=1)                # [B, C, Ts, He, We]
            hs = self.t_collapse(hs)                   # [B, C', He, We]
            hist_feats.append(self._resize(hs))        # [B, C', Hw, Ww]
        x_hist = torch.cat(hist_feats, dim=1) if hist_feats else None

        # ---- statics + coordinate planes -------------------------------------------
        stat_feats = [self._resize(v.float()) for v in statics.values()]  # each [B,1,He,We]
        stat_feats.append(self._resize(self._coord_planes(coords, statics)))
        x_stat = torch.cat(stat_feats, dim=1)          # [B, S, Hw, Ww]

        # ---- broadcast the per-sample (history, statics) across the L folded frames -
        ctx = [x_fore]
        if x_hist is not None:
            ctx.append(x_hist.repeat_interleave(L, dim=0))
        ctx.append(x_stat.repeat_interleave(L, dim=0))
        x = torch.cat(ctx, dim=1)                      # [B*L, F, Hw, Ww]

        # ---- conditioning vector per folded frame, then shared UNet ----------------
        c = self.cond(lead.reshape(B * L), doy_sin.reshape(B * L), doy_cos.reshape(B * L))
        x = self._pad(x)
        x = self.unet(x, c)
        out = self.head(x)                             # [B*L, 1, Hp, Wp]
        out = self._crop(out).reshape(B, L, *self.work_hw)  # [B, L, Hw, Ww]
        return out

    # -- coordinate planes from 1D lat/lon linspaces --------------------------------
    def _coord_planes(self, coords, statics):
        ref = next(iter(statics.values()))             # [B, 1, He, We] -- gives H, W, device
        B, _, He, We = ref.shape
        lat = coords.get('lat_era'); lon = coords.get('lon_era')
        planes = []
        if lat is not None:
            planes.append(lat[:, None, :, None].expand(B, 1, He, We))
        if lon is not None:
            planes.append(lon[:, None, None, :].expand(B, 1, He, We))
        return torch.cat(planes, dim=1) if planes else ref.new_zeros(B, 0, He, We)

    # -- pad/crop so H,W divide 2**depth (361 is odd, etc.) -------------------------
    def _pad(self, x):
        m = 2 ** self.unet.depth
        H, W = x.shape[-2:]
        self._ph, self._pw = (m - H % m) % m, (m - W % m) % m
        return F.pad(x, (0, self._pw, 0, self._ph), mode='reflect')

    def _crop(self, x):
        H, W = self.work_hw
        return x[:, :, :H, :W]


# =====================================================================================
# Ensemble pooling (E axis)  --  swappable; exp1 = MeanStd
# =====================================================================================
class MeanStdPool(nn.Module):
    """Pool the ensemble axis to mean+std.  [B, E, C, T, H, W] -> [B, 2C, T, H, W].
    Parameter-free (unbiased=False so E=1 gives std 0, not NaN)."""
    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return torch.cat([mean, std], dim=1)


def build_e_pool(name):
    return {'mean_std': MeanStdPool}[name]()


# =====================================================================================
# Temporal collapse (T axis)  --  swappable; exp1 = flatten, exp2 = pma
# =====================================================================================
class FlattenT(nn.Module):
    """Fold the T window into channels.  [B, C, T, H, W] -> [B, C*T, H, W].
    No temporal inductive bias; the net learns which step matters. exp1 history collapse."""
    def forward(self, x):
        B, C, T, H, W = x.shape
        return x.reshape(B, C * T, H, W)


class PMAT(nn.Module):
    """Pooling by Multihead Attention over T (Set Transformer, Lee et al. 2019).
    Treat the T frames at each cell as a SET of C-dim tokens; pool into k learned seeds
    by attention.  [B, C, T, H, W] -> [B, k*C, H, W]. Order-agnostic, learned, any T.
    The exp2 temporal collapse. Needs dim = C at construction."""
    def __init__(self, dim, heads=4, k=1):
        super().__init__()
        self.k = k
        self.seed = nn.Parameter(torch.randn(1, k, dim) * dim ** -0.5)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)      # [B*H*W, T, C]
        seeds = self.seed.expand(B * H * W, -1, -1)                     # [B*H*W, k, C]
        pooled, _ = self.attn(seeds, tokens, tokens)                   # [B*H*W, k, C]
        pooled = self.norm(pooled)
        return pooled.reshape(B, H, W, self.k * C).permute(0, 3, 1, 2)  # [B, k*C, H, W]


def build_t_collapse(name, dim=None, heads=4, k=1):
    if name == 'flatten':
        return FlattenT()
    if name == 'pma':
        assert dim is not None, 'PMAT needs t_collapse_dim (= per-frame channel count)'
        return PMAT(dim=dim, heads=heads, k=k)
    raise ValueError(name)


# =====================================================================================
# Lead / season conditioning (FiLM)
# =====================================================================================
# Two kinds of side-feature, handled differently:
#   * lat/lon  -> per-CELL, vary across the grid -> concat as input channels (CoordConv).
#   * lead, valid-time season -> per-FRAME GLOBAL scalars, constant across the grid
#     -> inject via FiLM (fresh at every block), NOT as flat channels.
# FiLM is what lets the trunk SHARED across leads specialise its output per lead.

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: feat * (1 + gamma) + beta from a per-sample cond
    vector. Zero-initialised -> starts as identity; conditioning learns to deviate."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * n_channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, feat, cond):                     # feat [N,C,H,W]  cond [N,cond_dim]
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        return feat * (1 + gamma[..., None, None]) + beta[..., None, None]


class LeadSeasonConditioner(nn.Module):
    """Build FiLM's conditioning vector from inm_lead_time (days) + valid_doy_sin/cos.
    Lead is multi-frequency sinusoidally encoded; season arrives already cyclic.
    Inputs each [N], N = B*L (one row per folded frame)."""
    def __init__(self, cond_dim=32, n_freqs=6, max_lead_days=120.0):
        super().__init__()
        self.max_lead_days = max_lead_days
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs + 2, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )
        self.register_buffer('freqs', (2 ** torch.arange(n_freqs)) * math.pi)

    def forward(self, lead_days, doy_sin, doy_cos):    # each [N]
        ang = (lead_days / self.max_lead_days)[:, None] * self.freqs[None, :]
        feats = torch.cat([ang.sin(), ang.cos(), doy_sin[:, None], doy_cos[:, None]], dim=-1)
        return self.mlp(feats)                         # [N, cond_dim]


# =====================================================================================
# U-Net with FiLM blocks + bottleneck spatial attention (2D positional encoding)
# =====================================================================================
class FiLMConvBlock(nn.Module):
    """Two 3x3 convs, GroupNorm (batch-agnostic; robust to masked domains), FiLM after
    each norm, SiLU activation."""
    def __init__(self, in_ch, out_ch, cond_dim, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.film1 = FiLM(cond_dim, out_ch)
        self.film2 = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        x = self.act(self.film1(self.norm1(self.conv1(x)), cond))
        x = self.act(self.film2(self.norm2(self.conv2(x)), cond))
        return x


def sincos_2d(H, W, dim, device, dtype):
    """2D sinusoidal positional encoding: half the channels encode row y, half col x.
    dim must be divisible by 4. Returns [H*W, dim]."""
    assert dim % 4 == 0, 'attn dim must be divisible by 4 for 2D sincos PE'
    d = dim // 2
    omega = 1.0 / (10000 ** (torch.arange(d // 2, device=device, dtype=dtype) / (d // 2)))
    y = torch.arange(H, device=device, dtype=dtype)[:, None] * omega[None, :]   # [H, d/2]
    x = torch.arange(W, device=device, dtype=dtype)[:, None] * omega[None, :]   # [W, d/2]
    pe_y = torch.cat([y.sin(), y.cos()], dim=1)        # [H, d]
    pe_x = torch.cat([x.sin(), x.cos()], dim=1)        # [W, d]
    pe = torch.cat([pe_y[:, None, :].expand(H, W, d),
                    pe_x[None, :, :].expand(H, W, d)], dim=2)   # [H, W, dim]
    return pe.reshape(H * W, dim)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over the bottleneck map with additive 2D sinusoidal PE.
    Absolute position is re-injected here because the conv stack has washed the input
    coordinates out by the bottleneck -- and geography is causal in this domain."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):                              # [N, C, H, W]
        N, C, H, W = x.shape
        pe = sincos_2d(H, W, C, x.device, x.dtype)     # [H*W, C]
        t = x.flatten(2).transpose(1, 2) + pe[None]    # [N, H*W, C]
        out, _ = self.attn(self.norm(t), self.norm(t), self.norm(t))
        t = t + out
        return t.transpose(1, 2).reshape(N, C, H, W)


class UNetFiLM(nn.Module):
    def __init__(self, base=32, depth=3, cond_dim=32, attn_heads=4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]     # [b, 2b, 4b, 8b] (depth=3)
        self.stem = nn.LazyConv2d(chs[0], 3, padding=1)       # lazy -> infers in_ch
        self.enc = nn.ModuleList([FiLMConvBlock(chs[i], chs[i], cond_dim) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 2, stride=2) for i in range(depth)])
        self.bottleneck = FiLMConvBlock(chs[depth], chs[depth], cond_dim)
        assert chs[depth] % 4 == 0, 'bottleneck channels must be divisible by 4 (2D PE)'
        self.attn = SpatialSelfAttention(chs[depth], attn_heads)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([FiLMConvBlock(chs[i] * 2, chs[i], cond_dim)
                                  for i in reversed(range(depth))])
        self.out_ch = chs[0]

    def forward(self, x, cond):
        x = self.stem(x)
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x, cond)
            skips.append(x)
            x = self.down[i](x)
        x = self.attn(self.bottleneck(x, cond))
        for j, i in enumerate(reversed(range(self.depth))):
            x = self.up[j](x)
            x = torch.cat([x, skips[i]], dim=1)
            x = self.dec[j](x, cond)
        return x                                        # [N, base, H, W]


# =====================================================================================
# Shape smoke test
# =====================================================================================
if __name__ == '__main__':
    B, E, L = 2, 10, 17
    He, We = 91, 360        # tiny fake grids
    Hh, Wh = 181, 360
    cfg = {'work_hw': (64, 128), 'base': 16, 'depth': 3, 'cond_dim': 32, 'attn_heads': 4}
    batch = {
        'inm_swe_17weeks':  torch.randn(B, E, L, He, We),
        'inm_t2m_17weeks':  torch.randn(B, E, L, He, We),
        'hist_t2m_6weeks':  torch.randn(B, 6, Hh, Wh),
        'hist_sd_6months':  torch.randn(B, 6, Hh, Wh),
        'z': torch.randn(B, 1, Hh, Wh), 'sdor': torch.randn(B, 1, Hh, Wh),
        'lsm': torch.randn(B, 1, Hh, Wh), 'glacier': torch.randn(B, 1, Hh, Wh),
        'lat_era': torch.linspace(-1, 1, Hh).expand(B, Hh),
        'lon_era': torch.linspace(-1, 1, Wh).expand(B, Wh),
        'inm_lead_time_17weeks': torch.arange(L).float().expand(B, L) * 7,
        'valid_doy_sin_17weeks': torch.randn(B, L),
        'valid_doy_cos_17weeks': torch.randn(B, L),
    }
    model = SWEUNet(cfg)
    out = model(batch)
    print('output:', out.shape)        # expect [B, L, 64, 128]
    assert out.shape == (B, L, 64, 128)
    print('params:', sum(p.numel() for p in model.parameters()))

    # -- e_pooled path: dataset pre-pooled E to (mean,std) -> each inm field [B,2,L,H,W] --
    pooled_batch = dict(batch)
    for k in ('inm_swe_17weeks', 'inm_t2m_17weeks'):
        v = batch[k]                                     # [B, E, L, He, We]
        pooled_batch[k] = torch.stack([v.mean(1), v.std(1, unbiased=False)], dim=1)  # [B,2,L,He,We]
    cfg_p = {**cfg, 'e_pooled': True}
    out_p = SWEUNet(cfg_p)(pooled_batch)
    print('e_pooled output:', out_p.shape)
    assert out_p.shape == (B, L, 64, 128)

    # -- channel-order equivalence: runtime MeanStdPool == pre-pool + reshape ------------
    fkeys = ['inm_swe_17weeks', 'inm_t2m_17weeks']
    runtime = MeanStdPool()(torch.stack([batch[k] for k in fkeys], dim=2))       # [B,2C,L,H,W]
    prep = torch.stack([pooled_batch[k] for k in fkeys], dim=2)                  # [B,2,C,L,H,W]
    prep = prep.reshape(B, 2 * len(fkeys), *prep.shape[3:])                      # [B,2C,L,H,W]
    assert torch.allclose(runtime, prep, atol=1e-6), 'e_pooled channel order mismatch'
    print('e_pool layouts match:', runtime.shape)
