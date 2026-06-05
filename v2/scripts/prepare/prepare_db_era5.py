import torch
import patcher
import xarray as xr
import os
import numpy as np

directory='/home/miha3009/work/weather/era5db/era5'
context = patcher.Context('../../db', 1)
precision = {'t2m': 0.05, 'lsm': 0.01}

def process_element(element):
    if os.path.exists(f'../../db/{element}.bin'):
        return

    files = sorted(list(os.listdir(f'{directory}/{element}')))
    files = [f for f in files if f.endswith('nc')]
    for i, file in enumerate(files):
        year = int(file[:4])
        print(f'Read {element}/{file}')
        ds = xr.open_dataset(f'{directory}/{element}/{file}')
        vals = ds[element].values
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
    patcher.aggregate(context, element, "19910101", "20201231")

def process_element_static(element, origin_element):
    if os.path.exists(f'../../db/{element}.bin'):
        return

    print('Prepare', element)
    ds = xr.open_dataset(f'../../data/{origin_element}_0_daily-mean.nc')
    data = torch.from_numpy(ds[element].values[0, 360::-1, :].copy()).float()
    patcher.train_dict(context, [data], element, precision[element], True)
    patcher.save(context, [data], ['19910101'], element)
    patcher.aggregate(context, element, "", "")

process_element('t2m')
process_element_static('lsm', 'land_sea_mask')
