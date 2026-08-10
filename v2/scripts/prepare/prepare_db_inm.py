import torch
import patcher
import xarray as xr
import os
import numpy as np
from datetime import datetime, timedelta

context = patcher.Context('../../db', 1)
precision = {'inm_t2m': 0.05, 'inm_swe': 0.001, 'inm_snow_cover': 0.01, 'inm_ww': 0.001,
             'inm_tp': 0.001, 'inm_mslp': 0.1, 'inm_olr': 0.1, 'inm_hlt': 0.1,
             'inm_u850': 0.05, 'inm_v850': 0.05, 'inm_h500': 0.1, 'inm_ts': 0.05}
log_scale = ['inm_tp', 'inm_swe', 'inm_ww']
no_climate = ['inm_snow_cover']

def process_element(element, element_inm):
    if os.path.exists(f'../../db/{element}.bin'):
        return

    directory = f'../../../../{element_inm}'
    files = sorted(list(os.listdir(directory)))
    files = [f for f in files if f.endswith('nc')]
    k = 0
    for i, file in enumerate(files):
        data, dates = [], []
        year = int(file[5:9])
        month = int(file[9:11])
        print(f'Read {element}/{file}')
        ds = xr.open_dataset(f'{directory}/{file}')
        vals = ds[element_inm].values
        if element in log_scale:
            vals = np.log(1 + vals)
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
    if element in no_climate:
        patcher.aggregate(context, element, "", "")
    else:
        patcher.aggregate(context, element, "19910101", "20210430")

for element, element_inm in [
        ('inm_t2m', 'T2'),
        ('inm_swe', 'SS'),
        ('inm_snow_cover', 'SFR'),
        ('inm_tp', 'PREC'),
        ('inm_mslp', 'PS'),
        ('inm_ww', 'WW'),
        ('inm_u850', 'U850'),
        ('inm_v850', 'V850'),
        ('inm_h500', 'H500'),
        ('inm_olr', 'OLR'),
        ('inm_hlt', 'HLT'),
        ('inm_ts', 'TS')
    ]:
    process_element(element, element_inm)

