import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import shutil
from sklearn.preprocessing import OneHotEncoder

output_path = "../../data/train"
years = list(range(1991, 2020))

def check_exists(files):
    for file in files:
        if not os.path.exists(file):
            print(f"Not found {file}")
            return False
    return True

def process_variable(variable, period_count):
    std_file = f"{output_path}/{variable}/std.nc"
    if os.path.exists(std_file):
        return

    print(f"Generate std for {variable}")
    
    coords = None
    dims = {}
    sums = {}
    counts = {}

    for year in years:
        for period in range(1, period_count + 1):
            input_file = f"{output_path}/{variable}/anom/{year}{period:02d}.nc"
            if not os.path.exists(input_file):
                continue

            print(f'Process {input_file}')
            ds = xr.open_dataset(input_file, engine="h5netcdf")
            if coords is None:
                coords = ds.coords
                del coords['lead_time']

            for v in ds.data_vars:
                if v == 'era5':
                    continue
                if v not in sums:
                    dims[v] = ds[v].dims[1:]
                    shape = ds[v].shape[1:]
                    sums[v] = np.zeros(shape, dtype=np.float64)
                    counts[v] = np.zeros(shape, dtype=np.int32)

                data = ds[v].values
                if v == 'h500':
                    print(data)
                valid = ~np.isnan(data)
                for lead_time in range(data.shape[0]):
                    sums[v] += np.where(valid[lead_time], data[lead_time] ** 2, 0)
                    counts[v] += valid[lead_time].astype(np.int32)
            ds.close()

    if len(sums) == 0:
        return

    result_ds = {}
    for v in sums:
        mask = counts[v] > 1
        variance = np.full_like(sums[v], np.nan)
        variance[mask] = sums[v][mask] / (counts[v][mask] - 1)
        result_ds[v] = (dims[v], np.sqrt(variance).astype(np.float32))

    result_ds = xr.Dataset(result_ds, coords=coords)
    result_ds.to_netcdf(std_file, engine="h5netcdf")

def process_time_invariant():
    std_file = f"{output_path}/time_invariant_norm.nc"
    if os.path.exists(std_file):
        return

    print(f"Generate norm time invariant")

    input_file = f"{output_path}/time_invariant.nc"
    if not os.path.exists(input_file):
        print(f'Not found {input_file}')
        return

    ds = xr.open_dataset(input_file, engine="h5netcdf")

    result_vars = {} 
    for v in ds.data_vars:
        data = ds[v].values
        
        if v == 'z':
            data = np.log(data - np.min(data) + 1)

        if v in ['cvh', 'cl', 'lsm', 'cvl']:
            result_vars[v] = (ds[v].dims, data)
        elif v in ['z', 'sdor']:
            norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            result_vars[v] = (ds[v].dims, norm_data)
        elif v in ['slt', 'tvh', 'tvl']:
            flat_data = data.flatten().reshape(-1, 1)
            encoder = OneHotEncoder(sparse_output=False)
            one_hot = encoder.fit_transform(flat_data)
            n_categories = one_hot.shape[1]
            for i in range(n_categories):
                result_vars[f"{v}_{i}"] = (ds[v].dims, one_hot[:, i].reshape(data.shape).astype(np.float32))

    result_ds = xr.Dataset(result_vars, coords=ds.coords)
    result_ds.to_netcdf(std_file, engine="h5netcdf")
    ds.close()

if __name__ == "__main__":
    process_variable('aice', 12)
    process_variable('swe', 52)
    process_time_invariant()
