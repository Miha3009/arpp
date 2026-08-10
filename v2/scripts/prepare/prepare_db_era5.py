import torch
import patcher
import xarray as xr
import os
import numpy as np
from datetime import datetime, timedelta

directory='/home/miha3009/work/weather/patcher/era5'
context = patcher.Context('../../db', 1)
precision = {'t2m': 0.05, 'lsm': 0.01, 'sd': 0.001, 'z': 0.001, 'sdor': 0.001, 'tp': 0.001, 'sden': 0.1, 'pt': 0.25,
             'snow_cover': 0.01, 'glaicer': 1, 'sst': 0.05}
aliases = {'sden': 'rsn', 'pt': 'ptype'}
scale = {
    'sd': 1000, # m to mm
    'z': 1 / 9.8066, # geopotential to m
    'tp': 24000 # m/hr to mm/day
}
log_scale = ['sd', 'z', 'sdor', 'tp']
min_level = {'z': -129}
fill_value = {'sst': 273.15}
no_climate = ['pt']
snow_cover_limit_mm = 4

def process_element(element):
    if os.path.exists(f'../../db/{element}.bin'):
        return

    files = sorted(list(os.listdir(f'{directory}/{element}')))
    files = [f for f in files if f.endswith('nc')]
    for i, file in enumerate(files):
        year = int(file[:4])
        print(f'Read {element}/{file}')
        ds = xr.open_dataset(f'{directory}/{element}/{file}')
        vals = ds[aliases.get(element, element)].values * scale.get(element, 1)
        vals = np.nan_to_num(vals, nan=fill_value.get(element, 0.0))
        if element == 'sd':
            vals = np.clip(vals, 0, 5000)
        if element in log_scale:
            vals = np.log(1 + vals - min_level.get(element, 0))
        times = ds.valid_time.values
        data, dates = [], []
        for j in range(len(times)):
            date = np.datetime_as_string(times[j], unit='D').replace('-', '')
            tensor = torch.from_numpy(vals[j, 360::-1, :].copy()).float()
            data.append(tensor)
            dates.append(date)
        if i == 0:
            print(f'Train dict {element}')
            patcher.train_dict(context, data, element, precision[element])
            first_batch = False
        print(f'Save {element}')
        patcher.save(context, data, dates, element)
    print(f'Aggregate {element}')
    if element in no_climate:
        patcher.aggregate(context, element, "", "")
    else:
        patcher.aggregate(context, element, "19910101", "20201231")

def process_element_static(element, origin_element):
    if os.path.exists(f'../../db/{element}.bin'):
        return

    print('Prepare', element)
    ds = xr.open_dataset(f'../../data/{origin_element}_0_daily-mean.nc')
    data = torch.from_numpy(ds[element].values[0, 360::-1, :].copy()).float() * scale.get(element, 1)
    if element in log_scale:
        data = np.log(1 + data - min_level.get(element, 0))
    patcher.train_dict(context, [data], element, precision[element], True)
    patcher.save(context, [data], ['19910101'], element)
    patcher.aggregate(context, element, "", "")

def process_snow_cover_from_swe():
    element = 'snow_cover'
    if os.path.exists(f'../../db/{element}.bin'):
        return

    start_date = datetime(1980, 1, 1)
    end_date = datetime(2026, 6, 30)

    current_date = start_date
    block_size = 30
    while current_date <= end_date:
        print(f'Read {current_date}')
        req = patcher.Request('sd', 0, 0, current_date.strftime('%Y%m%d'), 1440, 361, block_size, 1, 1)
        swe = patcher.load(context, [req])[0]
        swe_climate = patcher.load_climate(context, [req])[0]
        swe += swe_climate
        swe = np.exp(swe) - 1
        is_all_nan = torch.isnan(swe).all(dim=(-2, -1), keepdim=True)
        swe = torch.where(is_all_nan, torch.zeros_like(swe), torch.nan_to_num(swe, nan=10000.0))
        snow_cover = torch.clamp(swe / snow_cover_limit_mm, 0, 1)
        data = [snow_cover[i, : :] for i in range(block_size)]
        dates = [(current_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(block_size)]
        if current_date == datetime(1980, 1, 1):
            print(f'Train dict {element}')
            patcher.train_dict(context, data, element, precision[element])
        print(f'Save {element}')
        patcher.save(context, data, dates, element)
        current_date += timedelta(days=block_size)

    print(f'Aggregate {element}')
    patcher.aggregate(context, element, "", "")

def process_glaicer_mask():
    element = 'glaicer'
    if os.path.exists(f'../../db/{element}.bin'):
        return

    reqs = []
    for year in range(1980, 2026):
        reqs.append(patcher.Request('snow_cover', 0, 0, f'{year}0601', 1440, 361, 1, 1, 92))
    data = (torch.cat(patcher.load(context, reqs)).mean(dim=0) > 0.9).float()
    patcher.train_dict(context, [data], element, precision[element], True)
    patcher.save(context, [data], ['19910101'], element)
    patcher.aggregate(context, element, "", "")

for element in ['t2m', 'sd', 'tp', 'sden', 'pt', 'sst']:
    process_element(element)

for element, origin_element in [('lsm', 'land_sea_mask'),
                                ('z', 'geopotential'),
                                ('sdor', 'standard_deviation_of_orography')]:
    process_element_static(element, origin_element)

process_snow_cover_from_swe()
process_glaicer_mask()
