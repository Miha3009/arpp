import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

input_path = "../../data/raw/time_invariant"
output_path = "../../data/train/time_invariant.nc"
target_lats = np.arange(0, 90.25, 0.25).astype("float32")
target_lons = np.arange(-180, 180, 0.25).astype("float32")

variables = ["geopotential", "high_vegetation_cover", "lake_cover_daily-mean", "land_sea_mask", "low_vegetation_cover",
             "soil_type", "standard_deviation_of_orography", "type_of_high_vegetation", "type_of_low_vegetation"]

def open_netcdf4(filename):
    ds = xr.open_dataset(filename, engine='netcdf4')
    name = list(ds.data_vars)[0]
    vals = ds[name].to_numpy()
    ds.close()
    return name, vals[0, ::-1, :].astype("float32")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        exit()

    arrs = []
    for variable in variables:
        with ProcessPoolExecutor() as executor:
            future = executor.submit(open_netcdf4, f'{input_path}/era5_{variable}.nc')
            name, vals = future.result()
        arr = xr.DataArray(vals, dims=["lat", "lon"], coords={"lat": target_lats, "lon": target_lons}, name=name)
        arrs.append(arr.sel(lat=slice(40, None)))
    ds = xr.merge(arrs)
    encoding = {v: {"zlib": True, "complevel": 9} for v in ds.variables}
    ds.to_netcdf(output_path, engine='h5netcdf', encoding=encoding)
