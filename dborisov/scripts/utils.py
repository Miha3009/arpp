import json
import torch
import random
from datetime import datetime, timedelta
from collections import defaultdict
import patcher
import numpy as np
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
import math
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class ContiguousDataset(Dataset):
    """
    Dataset that resolves contigous set of dates and returns full-resolution fields

    Usage example:

    `features` is a dict {id: {var, tStep, tSize}} -- the id is the key AND the key the
    tensor lands under in the returned dict, so name it descriptively. The id prefix
    selects the request branch:
        hist   -> ERA history  : ERA grid, window ENDS at t0, no tag.
        inm    -> INM forecast : INM grid, forward from t0, tag = init month.
        target -> ERA target   : ERA grid, forward from t0 (future valid times), no tag.
        surf   -> static field : no tStep/tSize, loaded ONCE (date-invariant), merged in.
    The last '_'-token is the "scale" suffix; features sharing it share one lead axis.

    train_ds = ContiguousDataset(
        features={
            'hist_t2m_6weeks':          {'var': 't2m',        'tStep': 7,  'tSize': 6},
            'hist_sd_log_6months':      {'var': 'sd_log',     'tStep': 30, 'tSize': 6},
            'inm_tp_log_17weeks':       {'var': 'inm_tp_log', 'tStep': 7,  'tSize': 17},
            'target_sd_log_17weeks':    {'var': 'sd_log',     'tStep': 7,  'tSize': 17},
            'surf_z':                   {'var': 'z'},
            'surf_lsm':                 {'var': 'lsm'},
            'surf_glaicer':             {'var': 'glaicer'},
        },
        dates=train_dates,
        load_climate=('hist_sd', 'target_sd'),   # tuple of id-prefixes to load climate for (masking)
    )

    """
    def __init__(self,
                 features: Dict[str, Dict],
                 dates: List[str],
                 load_climate: tuple,
                 name, 
                 num_workers=16):
        self.name = name
        self.era_nlon, self.era_nlat = 1440, 361
        self.inm_nlon, self.inm_nlat = 360, 91
        self.context = patcher.Context('/Volumes/portable/arpp/data', num_workers)
        self.features = features                 # {id: {var, tStep, tSize}} -- id is the dict key
        self.dates = dates
        self.load_climate = load_climate
        self.SHIFTED = ('inm_ww_log', 'inm_tp_log', 'inm_swe_log')  # stored 180° offset in lon
        # feature ids whose climate we load for masking (static: ids and prefixes are fixed)
        self.clim_keys = [fid for fid in features if fid.startswith(tuple(load_climate))]

        print(f'\n load climate keys: {self.clim_keys} \n')
        try:
            with open(f'database/norms/{self.name}/{self.name}.json', 'r', encoding='utf-8') as file:
                self.norm_stats = json.load(file)['channels']
                print(f'\n\n NORMED DATASET:{file}\n\n')
        except FileNotFoundError:
            print('\n\n NO NORM STATS FOR THIS DATASET \n\n')
            self.norm_stats = {}

        # surf/static features are date-invariant -> load ONCE here, merge into every item.
        surf_features = {fid: f for fid, f in features.items() if fid.startswith('surf')}
        reqs = [patcher.Request(f['var'], f['x0'], f['y0'], '19930101', f['xSize'], f['ySize'], 1, 1, 1) for f in surf_features.values()]
            #                              era_bbox crop (merged into each surf feature)         tSize,xyStep,tStep -- 9-arg, no tag
        tensors = patcher.load(self.context, reqs)
        self.surf_vars   = {fid: t for fid, t in zip(surf_features, tensors)}                 # keyed by id -> merged into instnc
        self.surf_by_var = {f['var']: t for f, t in zip(surf_features.values(), tensors)}     # keyed by var -> used by structural mask

        # structural mask: cells where SWE is meaningless (permanent glacier, sea, high orography).
        # Static -> compute ONCE. ERA grid [1, H, W]; broadcasts over a field's T axis. None if
        # the surf fields it needs weren't declared (only ERA snow vars use it).
        self.structural = None
        if {'glaicer', 'lsm', 'z'} <= set(self.surf_by_var):
            self.structural = ((self.surf_by_var['glaicer'] != 0.0)
                               | (self.surf_by_var['lsm'] == 0.0)
                               | (self.surf_by_var['z'] >= 2999.0))

        if ('surf_z' in self.surf_vars) and ('surf_sdor' in self.surf_vars):
            high = self.surf_vars['surf_z'] >= 2999.0
            self.surf_vars['surf_z'] = torch.where(high, 0.0, self.surf_vars['surf_z'])
            self.surf_vars['surf_z'] = self.surf_vars['surf_z'] / 1000.0

            self.surf_vars['surf_sdor'] = torch.where(high, 0.0, self.surf_vars['surf_sdor'])
            self.surf_vars['surf_sdor'] = self.surf_vars['surf_sdor'] / torch.quantile(self.surf_vars['surf_sdor'], 0.99).item()
        else: 
            print(f"No z and sdor features in the dataset")

        # climate is seasonal (per day-of-year) -> identical across years for a given
        # (feature, init-month). Cache by (fid, month): load each <=12x, not once per date
        # (avoids ~30x redundant reads on the throttling external SSD).
        self._clim_cache = {}

        # static grid-coordinate channels (CoordConv), normalized to [-1, 1]. Kept as 1D
        # linspaces (cheap); the model broadcasts lat over H and lon over W. NB direction:
        # these run first-row -> last-row of the grid; flip if the model needs true N->S / W->E.
        # Sizes come from the CROP (bbox), not the globe: ERA grid <- target feature (era_bbox),
        # INM grid <- an inm feature (inm_bbox).
        tgt = next(f for fid, f in features.items() if fid.startswith('target'))
        #inm = next(f for fid, f in features.items() if fid.startswith('inm'))
        self.lat_era = torch.linspace(-1.0, 1.0, tgt['ySize'], dtype=torch.float32)
        self.lon_era = torch.linspace(-1.0, 1.0, tgt['xSize'], dtype=torch.float32)
        #self.lat_inm = torch.linspace(-1.0, 1.0, inm['ySize'], dtype=torch.float32)
        #self.lon_inm = torch.linspace(-1.0, 1.0, inm['xSize'], dtype=torch.float32)

    def __len__(self): 
        return len(self.dates)

    def mask(self, key, clim=None):
        # per-VARIABLE masking; var looked up from the feature id (surf/coord keys have no feature)
        feat = self.features.get(key)
        var = feat['var'] if feat else None
        if var in ('sd', 'sd_log', 'sden', 'inm_swe', 'inm_swe_log', 'inm_ww', 'inm_ww_log'):
            assert clim is not None, f'{key}: masked var needs climate — add its id to load_climate'
        if var in ('sd', 'sd_log'):
            return True, self.structural | (clim == 0.0)
        elif var == 'sden':
            return True, self.structural | (clim == 100.0) | (clim == 300.0)
        elif var in ('inm_swe', 'inm_swe_log', 'inm_ww', 'inm_ww_log'):
            return True, (clim == 0.0)
        elif var == 'snow_cover':
            return True, self.structural
        else:
            return False, False
        
    def __getitem__(self, idx):

        date0 = self.dates[idx]
        date0_stmp = pd.Timestamp(date0)
        assert date0_stmp.day == 1, f'not init time for {date0}'
        tag = date0_stmp.month

        # Gather every request for this date, keep parallel key/meta lists, then
        # issue ONE patcher.load so the Context's worker pool reads them concurrently.
        # patcher.load returns tensors in request order, so the three lists stay aligned.
        requests  = {}

        ### REMINDER log inm_vars shift 180
        for fid, f in self.features.items():
            if fid.startswith('surf'):          # static, pre-loaded in __init__ -> skip
                continue
            var, tStep, tSize = f['var'], f['tStep'], f['tSize']
            x0, y0, xSize, ySize = f['x0'], f['y0'], f['xSize'], f['ySize']  # bbox crop
            if fid.startswith('hist'):          # ERA history: era_bbox, window ends at t0, no tag
                start = (date0_stmp - pd.Timedelta(tSize * tStep, 'd')).strftime('%Y%m%d')  # patcher searches forward from here, so history ends at date0
                requests[fid] = patcher.Request(var, x0, y0, start, xSize, ySize, tSize, 1, tStep)
            elif fid.startswith('inm'):         # INM forecast: inm_bbox, forward from t0, tag = init month
                if fid.startswith(self.SHIFTED):  # need FULL width for the 180° roll -> crop after load
                    requests[fid] = patcher.Request(var, 0, 0, date0, self.inm_nlon, self.inm_nlat, tSize, 1, tStep, tag)
                else:
                    requests[fid] = patcher.Request(var, x0, y0, date0, xSize, ySize, tSize, 1, tStep, tag)
            elif fid.startswith('target'):      # ERA target: era_bbox, forward from t0 (valid times), no tag
                requests[fid] = patcher.Request(var, x0, y0, date0, xSize, ySize, tSize, 1, tStep)
            else:
                raise ValueError(f"feature id '{fid}' must start with 'hist', 'inm', 'target' or 'surf'")

        tensors = patcher.load(self.context, list(requests.values()))
        instnc = {k:t for k,t in zip(requests.keys(), tensors)}

        # SHIFTED inm_*_log are stored 180° offset in lon and were requested at FULL width ->
        # roll the whole field, THEN crop to the inm bbox (roll-by-180 is only valid full-width).
        for k in instnc:
            if k.startswith(self.SHIFTED):
                f = self.features[k]
                rolled = torch.roll(instnc[k], 180, dims=-1)
                instnc[k] = rolled[..., f['y0']:f['y0'] + f['ySize'], f['x0']:f['x0'] + f['xSize']]
            assert not torch.isnan(instnc[k]).any(), f"{k} for {date0} contains nans: {torch.isnan(instnc[k]).any(dim=(-2,-1))}"

        # climate tensors for masking -- cached by (fid, init-month); seasonal so identical across years
        miss = [(k, requests[k]) for k in self.clim_keys if (k, tag) not in self._clim_cache]
        if miss:
            for (k, _), t in zip(miss, patcher.load_climate(self.context, [r for _, r in miss])):
                if k.startswith(self.SHIFTED):        # same full-roll-then-crop as the field
                    f = self.features[k]
                    t = torch.roll(t, 180, dims=-1)
                    t = t[..., f['y0']:f['y0'] + f['ySize'], f['x0']:f['x0'] + f['xSize']]
                assert not torch.isnan(t).any(), f"{k}_climate for {date0} contains nans: {torch.isnan(t).any(dim=(-2,-1))}"
                self._clim_cache[(k, tag)] = t
        clim_tensors = {k: self._clim_cache[(k, tag)] for k in self.clim_keys}

        # normalization -- keyed on each feature's VARIABLE (mask/skip rules are per-variable)
        if self.norm_stats:
            for key in instnc:
                feat = self.features.get(key)
                var = feat['var'] if feat else key       # surf keys: the key IS the var name
                clim = clim_tensors[key] if key in clim_tensors.keys() else None 
                to_mask, m = self.mask(key, clim)
                if key.startswith('target_sd'):
                    continue
                if var in ('snow_cover', 'inm_snow_cover'):
                    if to_mask:                          # 0-1 fraction: structural mask only, no normalize
                        instnc[key] = torch.where(m, torch.zeros((), dtype=instnc[key].dtype), instnc[key])
                    continue
                if to_mask:                              # zero-inflated: scale-only, 0-fill, no recenter
                    instnc[key] = torch.where(m, torch.zeros((), dtype=instnc[key].dtype),
                                              instnc[key] / self.norm_stats[key]['std'])
                else:                                    # standard: (x - mean) / std
                    instnc[key] = (instnc[key] - self.norm_stats[key]['mean']) / self.norm_stats[key]['std']

        instnc = instnc | self.surf_vars # surf_vars already normalized

        # Per-lead-frame time features on the INM forecast axis, one set per distinct
        # forecast layout (deduped by the id's scale suffix, e.g. '17weeks'):
        #   inm_lead_time_{sid}       -> lead in days from t0                (error-growth axis)
        #   valid_doy_{sin,cos}_{sid} -> day-of-year of the VALID time (t0+lead), cyclic so
        #                                Dec 31 -> Jan 1 is continuous       (season / base-state axis)
        # Keep both: lead conveys forecast uncertainty, season the climatological base state;
        # independent axes. Consumed as FiLM conditioning (see models.py), not spatial planes.
        seen = set()
        for fid, f in self.features.items():
            if not fid.startswith('inm'):
                continue
            sid = fid.rsplit('_', 1)[-1]
            if sid in seen:
                continue
            seen.add(sid)
            lead_days = torch.arange(f['tSize'], dtype=torch.float32) * f['tStep']
            instnc[f"inm_lead_time_{sid}"] = lead_days
            valid_times = date0_stmp + pd.to_timedelta(lead_days.numpy(), unit='D')
            doy = torch.tensor(valid_times.dayofyear.to_numpy(), dtype=torch.float32)
            angle = 2 * math.pi * doy / 365.25
            instnc[f"valid_doy_sin_{sid}"] = torch.sin(angle)
            instnc[f"valid_doy_cos_{sid}"] = torch.cos(angle)

        # static coordinate channels, per grid (1D; model broadcasts lat over H, lon over W)
        instnc['lat_era'] = self.lat_era
        instnc['lon_era'] = self.lon_era
        #instnc['lat_inm'] = self.lat_inm
        #instnc['lon_inm'] = self.lon_inm

        # target loss masks -- {id}_mask, True = EXCLUDED (structural / climate-dry).
        # For sd the mask is per-lead [T,H,W] (climate dryness is seasonal); apply ~mask in the loss.
        for fid in self.features:
            if not fid.startswith('target'):
                continue
            to_mask, m = self.mask(fid, clim_tensors.get(fid))
            if to_mask:
                instnc[f'{fid}_mask'] = m

        for key in [k for k in instnc if k.startswith('hist_sst')]:
            instnc[key] = torch.clamp(instnc[key], 
                                      min = torch.quantile(instnc[key], 0.001), 
                                      max = torch.quantile(instnc[key], 0.999))

        if self.norm_stats:
            return instnc
        else: 
            clim_tensors = {f'{k}_clim':v for k,v in clim_tensors.items()}
            instnc = instnc | clim_tensors
            return instnc

def pool_inm_ensemble(sample):
    """Collapse each INM forecast field [E,T,H,W] to its (mean,std) over the ensemble,
    stored as [2,T,H,W] (mean = channel 0, std = channel 1). Uses unbiased=False to match
    MeanStdPool (E=1 -> std 0, not NaN), so a model with cfg['e_pooled']=True consumes it
    identically to runtime pooling. Shrinks the INM footprint from E (10/30) to 2 and
    sidesteps ragged E. Leaves inm_lead_time_* and every non-inm key untouched."""
    out = {}
    for k, v in sample.items():
        if k.startswith('inm_') and not k.startswith('inm_lead_time'):
            out[k] = torch.stack([v.mean(dim=0), v.std(dim=0, unbiased=False)], dim=0)  # [2,T,H,W]
        else:
            out[k] = v
    return out


def materialize_dataset(dataset, out_dir, dtype=torch.float16, pool_ensemble=True, overwrite=False):
    """One slow pass over the source Dataset (external SSD) -> per-sample .pt files on
    fast internal storage. Floats are downcast to `dtype` to save disk; bool masks and
    int/coord tensors are left as-is. Resumable: skips files already written.

    pool_ensemble=True pre-pools the INM ensemble to (mean,std) (see pool_inm_ensemble),
    the biggest size lever -- pair with SWEUNet(cfg={..., 'e_pooled': True}).

    NB: the cache is tied to THIS feature config + norm_stats (+ pool_ensemble). Rebuild
    if any of them changes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(dataset)), desc='materialize'):
        p = out_dir / f'{i:05d}.pt'
        if p.exists() and not overwrite:
            continue
        sample = dataset[i]                                   # <- the throttling read
        if pool_ensemble:
            sample = pool_inm_ensemble(sample)
        sample = {k: (v.to(dtype) if v.is_floating_point() else v)
                  for k, v in sample.items()}
        torch.save(sample, p.with_suffix('.tmp'))            # atomic: write tmp then rename
        p.with_suffix('.tmp').rename(p)                       # a crash never leaves a half file
    return out_dir


class CachedDataset(torch.utils.data.Dataset):
    """Reads materialized .pt samples from fast local storage. Fork-safe (no patcher
    Context), so DataLoader(num_workers>0) works here — unlike the source Dataset."""
    def __init__(self, cache_dir, upcast=torch.float32):
        self.files = sorted(Path(cache_dir).glob('[0-9]*.pt'))
        assert self.files, f'no cached samples in {cache_dir}'
        self.upcast = upcast

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        s = torch.load(self.files[i], map_location='cpu')
        if self.upcast is not None:
            s = {k: (v.to(self.upcast) if v.is_floating_point() else v)
                 for k, v in s.items()}
        return s
    
def compute_norm(dataset,
                #vars, dates, scales, 
                 reservoir, sampl_sz, seed=1234, out_path=None):

    assert not bool(dataset.norm_stats), 'norm stats exist for this dataset' 
    #dataset = ContiguousDataset(vars=vars, dates=dates, scales=scales, num_workers=num_workers)
    rng = torch.Generator().manual_seed(seed)

    acc = {}
    pool = {}

    for i in tqdm(range(len(dataset))):
        instance = dataset[i]
        clim_tensors = {k:v for k,v in instance.items() if k.endswith('_clim')}
        anom_tensors = {k:v for k,v in instance.items() if not k.endswith('_clim')}
        for key, value in anom_tensors.items():
            if key.startswith(('inm_lead_time', 'valid_doy', 'lat_', 'lon_', 'surf_')) or key.endswith('_mask'):
                continue
            clim = clim_tensors.get(f'{key}_clim')
            to_mask, mask = dataset.mask(key, clim)
            if to_mask: 
                value = torch.where(mask, float('nan'), value)
            value = value.to(torch.float64).flatten()
            value = value[torch.isfinite(value)]
            if value.numel() == 0:
                continue
            a = acc.setdefault(key, torch.zeros(3, dtype=torch.float64))
            a[0] += value.numel()
            a[1] += value.sum()
            a[2] += (value * value).sum()
            p = pool.setdefault(key, [])
            if sum(t.numel() for t in p) < reservoir:
                idx = torch.randperm(value.numel(), generator=rng)[:sampl_sz]
                p.append(value[idx].clone())

    # --- stats first (cheap, must not be lost) ---
    channels = {}
    for key, (n, total, total_sq) in tqdm(acc.items()):
        mean = total / n
        var = (total_sq / n - mean * mean).clamp_min(0)
        values = torch.cat(pool[key])
        q = torch.quantile(values, torch.tensor([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999], dtype=torch.float64))
        channels[key] = {
            'n': int(n),
            'mean': float(mean),
            'std': float(var.sqrt()),
            'mad_std': float(1.4826 * (values - values.median()).abs().median()),
            'q001': float(q[0]), 'q01': float(q[1]), 'q25': float(q[2]), 'q50': float(q[3]),
            'q75': float(q[4]), 'q99': float(q[5]), 'q999': float(q[6]),
        }

    stats = {
        'parent_ds': dataset.name,
        'created': datetime.now().strftime('%Y-%m-%d'),
        'features': dataset.features,
        'dates_range': [dataset.dates[0], dataset.dates[-1]],
        'n_dates': len(dataset.dates),
        'seed': seed,
        'reservoir': reservoir,
        'channels': channels,
    }

    # persist BEFORE plotting -> a plot failure can never discard the long sweep
    os.makedirs(out_path, exist_ok=True)
    with open(f'{out_path}/{dataset.name}.json', 'w') as f:
        json.dump(stats, f, default=str, indent=2)

    # --- diagnostics plots (best-effort; never fatal) ---
    plot_dir = f'database/norms/{dataset.name}'
    os.makedirs(plot_dir, exist_ok=True)
    for key in channels:
        try:
            values = torch.cat(pool[key]).numpy()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
            ax1.hist(values, bins=100)
            ax1.axvline(np.median(values), color='r', linewidth=0.5)
            ax1.axvline(values.mean(), color='g', linewidth=0.5)
            ax2.boxplot(values, showmeans=True, meanline=True, showfliers=False)
            desc = pd.Series(values).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
            ax3.text(0, 0.5, str(desc), va='center', fontsize=12, family='monospace')
            ax3.axis('off')
            fig.savefig(f'{plot_dir}/{key}.png', dpi=150)
        except Exception as e:
            print(f'plot failed for {key}: {e}')
        finally:
            plt.close('all')

    return stats

def collate_ragged_ensemble(samples):
    """Custom DataLoader collate that keeps the INM ensemble axis ragged.

    `samples` is the list the DataLoader hands us: one entry per item in the batch,
    each entry being exactly what ContiguousDataset.__getitem__ returns (a dict). So
    `samples` is List[dict] of length batch_size, and `samples[0]` is the first item's
    dict (used only to read the shared key set).

    default_collate torch.stacks every key -> needs identical shapes -> breaks on the
    INM ensemble E (10 hindcast vs 30 operational members). Here the inm FIELD keys are
    left as a Python list (ragged E preserved) for a LEARNED ensemble pool to consume
    per-sample; everything else (already fixed-shape) is stacked normally.

    NB: pairs with a model that pools E itself from the list. The current SWEUNet
    expects inm already pooled to fixed shape (do mean/std per-sample in __getitem__),
    for which the DEFAULT collate is enough -- keep this for the later learned-pool
    experiment. Usage: DataLoader(ds, batch_size=B, collate_fn=collate_ragged_ensemble).
    """
    out = {}
    for k in samples[0]:
        if k.startswith('inm_') and not k.startswith('inm_lead_time'):
            out[k] = [s[k] for s in samples]        # list of [E_i, T, H, W], ragged E preserved
        else:
            out[k] = torch.stack([s[k] for s in samples])
    return out


def compute_log_climate(element, climate_start, climate_end, out_path, h=361, w=1440,
                        t_step=1, tag=0, scale=1.0, db_path='../db', num_workers=4):
    """Seasonal log1p climatology for a variable, derived from the existing (linear) .bin.

    Purpose: the `.bin` stores linear anomalies, but for a log-transformed variable
    (precip) we want log-anomalies. To center in log space we need the *log*
    climatology `c_log[mmdd, cell] = mean_years(log1p(scale * raw))`. This CANNOT be
    obtained from the stored linear climatology because `mean(log) != log(mean)`
    (Jensen) -- it has to be recomputed from the daily raw fields. This function
    reconstructs `raw = load() + load_climate()` for every day in the climate window,
    applies `log1p`, accumulates by calendar day (MMDD), and writes the per-MMDD mean
    to a netCDF. At train time the Dataset subtracts this to form the log-anomaly:

        log_anom = log1p(scale * raw) - c_log[mmdd]         # raw from load_absolute / load+load_climate

    tStep-specific -- READ THIS
    ---------------------------
    A log climatology is only valid for the `t_step` it was computed at. log1p does
    NOT commute with temporal averaging (`log1p(avg_N) != avg_N(log1p)`), so an N-day
    climate cannot be derived from the daily one -- compute a separate file per tStep
    you request. This function loads the absolute field AT `t_step` (a linear N-day
    average) and then logs it ("average-then-log"), which matches a Dataset that does
    `log1p(load_absolute(tStep=N))`. Keep the two in lockstep; if you ever re-ingest
    as `tp_log` (patcher averages already-logged values = "log-then-average") this
    file would no longer match.

    Units
    -----
    log1p is ~identity for small values, so `scale` must bring `raw` into a range that
    spans across ~1 for the transform to do anything. If the `.bin` stores precip in
    metres (~1e-3), pass `scale=1000` (-> mm). If it is already mm, `scale=1.0`. Check
    first: this is the same units gotcha as the ingest path.

    Storage
    -------
    Written as `{element: (mmdd, y, x)}` in the *same pixel grid / orientation* patcher
    returns (the array is built straight from `patcher.load` output, so orientation is
    automatically consistent with what the Dataset will index). `mmdd` = month*100+day.
    Leap day (0229) simply has fewer contributing years.

    Cost: one-time, heavy -- ~ (years * 365) full-field load pairs. Run once per
    (element, t_step), then load the netCDF at train time.

    Parameters
    ----------
    element : variable name in the database (e.g. 'tp').
    climate_start, climate_end : 'YYYYMMDD', same window used for the linear climate
        in patcher (e.g. '19910101'..'20201231').
    out_path : netCDF path to write.
    h, w : full field size in cells (ERA5 = 361 x 1440). Must match how the Dataset
        will slice this climatology.
    t_step : temporal averaging window in days. One file per value you use.
    tag : INM-CM forecast-init month; 0 / omit for ERA5.
    scale : multiplier applied before log1p to reach mm (see Units).
    db_path, num_workers : patcher.Context args.

    Returns
    -------
    xarray.Dataset with the per-MMDD log climatology, also written to out_path.
    """
    import patcher
    import xarray as xr

    context = patcher.Context(db_path, num_workers)
    start = datetime.strptime(climate_start, '%Y%m%d')
    end = datetime.strptime(climate_end, '%Y%m%d')

    acc = {}                       # mmdd -> float64 [h, w] running sum of log1p(scale*raw)
    count = defaultdict(int)       # mmdd -> number of contributing years

    date = start
    while date <= end:
        t0 = date.strftime('%Y%m%d')
        args = (element, 0, 0, t0, w, h, 1, 1, t_step)
        req = patcher.Request(*args, tag) if tag else patcher.Request(*args)
        anom = patcher.load(context, [req])[0]              # [1, h, w] linear anomaly
        clim = patcher.load_climate(context, [req])[0]      # [1, h, w] linear climate
        raw = (anom + clim)[0].to(torch.float64)            # [h, w] absolute, linear units
        logged = torch.log1p(torch.clamp(raw, min=0) * scale)
        logged = torch.nan_to_num(logged, nan=0.0)          # tp has no mask; guard anyway

        mmdd = date.month * 100 + date.day
        acc[mmdd] = logged if mmdd not in acc else acc[mmdd] + logged
        count[mmdd] += 1
        date += timedelta(days=1)

    mmdds = sorted(acc.keys())
    stack = torch.stack([acc[m] / count[m] for m in mmdds], dim=0)   # [D, h, w]

    ds = xr.Dataset(
        {element: (('mmdd', 'y', 'x'), stack.to(torch.float32).numpy())},
        coords={'mmdd': mmdds},
    )
    ds.attrs.update(
        element=element, climate_start=climate_start, climate_end=climate_end,
        t_step=t_step, tag=tag, scale=scale,
        note='seasonal log climatology = mean_years(log1p(scale*raw)); average-then-log; tStep-specific',
    )
    ds.to_netcdf(out_path)
    return ds

