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

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class ContiguousDataset(Dataset):
    """
    Dataset that resolves contigous set of dates and returns full-resolution fields

    Usage example: 

    train_ds = PatchDataset(
    vars = {'era_vars': ['snow_cover', 'sd', 'sst', 't2m', 'tp', 'sden', 'pt'],
            'inm_vars': ['inm_swe', 'inm_ts', 'inm_mslp', 'inm_tp', 'inm_u850', 'inm_v850', 'inm_snow_cover', 'inm_olr', 'inm_h500', 'inm_hlt', 'inm_t2m'],  
            'surf_vars': ['z', 'sdor', 'lsm', 'glaicer']}, 
    dates = train_dates, 
    scales={'era_scales': [  
        {'id': '6weeks', 'tStep': 7, 'tSize': 6},
        {'id': '6months', 'tStep': 30, 'tSize': 6}
        ]
    , 'inm_scales':[
        {'id': '17weeks', 'tSize': 17, 'tStep': 7}, # maximum lead time = 119-122 days = 17 weeks
    ]}
    )

    """
    def __init__(self,
                 vars: Dict[str, List[str]],  
                 dates: List[str],
                 scales =  Dict[str, List[Dict]],  
                 num_workers=16): 
        self.era_nlon, self.era_nlat = 1440, 361
        self.inm_nlon, self.inm_nlat = 360, 91
        self.context = patcher.Context('/Volumes/portable/arpp/data', num_workers)
        self.vars = vars
        self.dates = dates
        self.scales = scales
        try: 
            with open(f'database/norms/{self.__class__.__name__}/{self.__class__.__name__}.json', 'r', encoding='utf-8') as file:
                self.norm_stats = json.load(file)['channels'] 
        except FileNotFoundError: 
            print('no norm stats for this dataset')
            self.norm_stats = {}
        self.zero_inflated_vars = ['era_sd', 'inm_swe', 'era_sden', 'inm_ww'] 

        reqs = [patcher.Request(var, 0, 0, '19930101', self.era_nlon, self.era_nlat, 1, 1, 1) for var in self.vars['surf_vars']]
            #                                    x0,y0                full field            tSize,xyStep,tStep -- 9-arg form, no tag
        tensors = patcher.load(self.context, reqs)
        self.surf_vars = dict(zip(self.vars['surf_vars'], tensors))

    def __len__(self): 
        return len(self.dates)

    def mask(self, key, tensor):
        if any(key.startswith(prfx) for prfx in ['era_sd', 'era_sden']): 
            return True, (self.surf_vars['glaicer'] != 0.0) | (self.surf_vars['lsm'] == 0.0) | (self.surf_vars['z'] >= 2999.0) | ((tensor>= -0.01) & (tensor <= 0.01))
        elif key.startswith('inm_swe'): 
            return True, (tensor >= -0.001) & (tensor <= 0.001)
        elif key.startswith('inm_ww'): 
            return True, (tensor >= -0.001) & (tensor <= 0.001)
        elif key.startswith('era_snow_cover'): 
            return True, (self.surf_vars['glaicer'] != 0.0) | (self.surf_vars['lsm'] == 0.0) | (self.surf_vars['z'] >= 2999.0)
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
        requests, keys, meta = [], [], []

        for var in self.vars['era_vars']:
            for scale in self.scales['era_scales']:
                history_start_date = (date0_stmp - pd.Timedelta(scale['tSize']*scale['tStep'], 'd')).strftime('%Y%m%d') # window start; patcher searches forward from here, so history ends at date0
                requests.append(patcher.Request(var, 0, 0, history_start_date, self.era_nlon, self.era_nlat, scale['tSize'], 1, scale['tStep']))
                #                                    x0,y0                     full field                                    xyStep (full res)  -- 9-arg form, no tag for ERA
                keys.append(f"era_{var}_{scale['id']}")
                meta.append(('era', var, scale))

        for var in self.vars['inm_vars']:
            for scale in self.scales['inm_scales']:
                requests.append(patcher.Request(var, 0, 0, date0, self.inm_nlon, self.inm_nlat, scale['tSize'], 1, scale['tStep'], tag))
                #                                    x0,y0        full field                                    xyStep       tag = init month
                keys.append(f"{var}_{scale['id']}")
                meta.append(('inm', var, scale))

        tensors = patcher.load(self.context, requests)

        instnc = {}
        for key, tensor, (kind, var, scale) in zip(keys, tensors, meta):
            assert not torch.isnan(tensor).any(), f'{kind} {var} for {date0}, scale {scale} contains nans: {torch.isnan(tensor).any(dim=(-2,-1))}'
            instnc[key] = tensor

        instnc = instnc | self.surf_vars

        # normalization
        if self.norm_stats:
            for key in instnc:
                if any(key.startswith(prfx) for prfx in ['lsm', 'glaicer', 'inm_snow_cover']): 
                    continue
                to_mask, mask = self.mask(key, instnc[key])
                if key.startswith('era_snow_cover'): 
                    instnc[key] = torch.where(mask,  
                                              torch.zeros((), dtype=instnc[key].dtype), 
                                              instnc[key])
                if to_mask: 
                    instnc[key] = torch.where(mask,  
                                              torch.zeros((), dtype=instnc[key].dtype), 
                                              instnc[key] / self.norm_stats[key]['std'])
                else: 
                    instnc[key] = (instnc[key] - self.norm_stats[key]['mean']) / self.norm_stats[key]['std']

        # lead time depends only on the INM scale, not on the variable: lead in days at each forward step's start
        for scale in self.scales['inm_scales']:
            instnc[f"inm_lead_time_{scale['id']}"] = torch.arange(scale['tSize'], dtype=torch.float32) * scale['tStep']

        return instnc


def compute_norm(dataset, 
                #vars, dates, scales, 
                 reservoir, sampl_sz, seed=1234, out_path=None):

    assert not bool(dataset.norm_stats), 'norm stats exist for this dataset' 
    #dataset = ContiguousDataset(vars=vars, dates=dates, scales=scales, num_workers=num_workers)
    rng = torch.Generator().manual_seed(seed)

    acc = {}
    pool = {}

    for i in tqdm(range(len(dataset))):
        for key, value in dataset[i].items():
            if key.startswith('inm_lead_time'):
                continue
            to_mask, mask = dataset.mask(key, value)
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

    channels = {}
    for key, (n, total, total_sq) in tqdm(acc.items()):
        mean = total / n
        var = (total_sq / n - mean * mean).clamp_min(0)
        values = torch.cat(pool[key])
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.hist(values.numpy(), bins=100)
        ax1.axvline(values.median().item(), color='r', linewidth = 0.5)
        ax1.axvline(values.mean().item(), color='g', linewidth = 0.5)
        ax2.boxplot(values.numpy(), showmeans=True, meanline=True, showfliers=False)
        os.makedirs(f'database/norms/{dataset.__class__.__name__}', exist_ok=True)
        fig.savefig(f'database/norms/{dataset.__class__.__name__}/{key}.jpg', dpi=300)
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
        'parent_ds': dataset.__class__.__name__,
        'created': datetime.now().strftime('%Y-%m-%d'),
        'vars': dataset.vars,
        'dates_range': [dataset.dates[0], dataset.dates[-1]],
        'n_dates': len(dataset.dates),
        'scales': dataset.scales,
        'seed': seed,
        'reservoir': reservoir,
        'channels': channels,
    }
    
    return stats


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

