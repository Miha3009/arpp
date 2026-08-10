import torch
import patcher
import xarray as xr
import os
import numpy as np
from datetime import datetime, timedelta

context = patcher.Context('../../database', 1)
precision = {'inm_t2m': 0.05, # температура, K 
             'inm_sfr': 0.01, # доля снега 0-1 
             'inm_ss': 0.5, # глубина снега, мм 
             'inm_ps': 0.1, # давление н.у.м, гПа
             'inm_ts': 0.05, # температура поверхности, K
             'inm_ww': 0.5} # влага почвы, мм

def fill_bad_cells(vals):
    bad = vals < 0
    if not bad.any():
        return vals
    land = (vals > 0).astype(np.float32)  # 0.0 - маска моря
    v = np.where(vals > 0, vals, 0.0).astype(np.float32)
    num = np.zeros_like(v)
    den = np.zeros_like(v)
    for shift, axis in [(1, 3), (-1, 3), (1, 2), (-1, 2)]:   
        rv = np.roll(v, shift, axis=axis)
        rl = np.roll(land, shift, axis=axis)
        if axis == 2:                                  
            idx = 0 if shift == 1 else -1             
            rv = rv.copy(); rv[:, :, idx, :] = 0   # обнуляются соседи на крайних широтах (нельзя интерполировать) 
            rl = rl.copy(); rl[:, :, idx, :] = 0       
        num += rv
        den += rl
    filled = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    out = vals.copy()
    out[bad] = filled[bad]
    return out

def process_element(element, element_inm):
    if os.path.exists(f'../../database/{element}.bin'):
        return

    damaged_files = []
    directory = f'../../raw_data/{element_inm}'
    files = sorted(list(os.listdir(directory)))
    files = [f for f in files if f.endswith('nc')]
    k = 0
    for i, file in enumerate(files):
        data, dates = [], []
        year = int(file[5:9])
        month = int(file[9:11])
        print(f'Read {element}/{file}')
        try:
            ds = xr.open_dataset(f'{directory}/{file}')
            vals = ds[element_inm].values
        except (RuntimeError, OSError) as e:
            print(f'Damaged file {file}: {e}')
            damaged_files.append(file)
            continue
        vals = np.roll(vals, 180, axis=3)
        if element == 'inm_ww':
            n_bad = int((vals < 0).sum())
            frac = n_bad / vals.size
            print(f"Interpolating {n_bad} damaged cells ({frac:.4%}) in {element}...")
            assert frac < 0.001, f"Too many damaged cells ({frac:.2%}) in {file} — structural problem?"
            vals = fill_bad_cells(vals)
        base_date = datetime(year, month, 1)
        for j in range(vals.shape[1]):
            data.append(torch.from_numpy(vals[:, j, :, :].copy()).float())
            date = base_date + timedelta(days=int(ds.day.values[j]))
            dates.append(date.strftime('%Y%m%d'))
        if k == 0:
            k = 1
            print(f'Train dict {element}')
            patcher.train_dict(context, data, element, precision[element])
        print(f'Save {element}')
        patcher.save(context, data, dates, element, tag=month)
    print(f'Aggregate {element}')
    patcher.aggregate(context, element, "19910101", "20210430")

    if damaged_files: 
        damaged_files = '\n'.join(damaged_files)
        print(f'Damaged files: {damaged_files}')
    else: 
        print('No damaged files')

process_element('inm_ww', 'WW')

