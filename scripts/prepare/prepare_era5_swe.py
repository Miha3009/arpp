import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

era5_input_path = "../../data/raw/swe_era5/"
era5_output_path = "../../data/prepare/era5_swe/"
target_lats = np.arange(0, 90.25, 0.25).astype("float32")
target_lons = np.arange(-180, 180, 0.25).astype("float32")

def open_netcdf4(file):
    ds = xr.open_dataset(file, engine='netcdf4')
    valid_time = ds.valid_time.to_numpy()
    vals = ds.sd.to_numpy()
    ds.close()
    return valid_time, vals

def process_file(date):
    date_str = date.strftime('%Y%m%d')
    filename = f"era5_snow_depth_{date_str[:4]}_{date_str[4:6]}.nc"
    input_filepath = os.path.join(era5_input_path, filename)

    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} not found")
        return

    print(f"Process {input_filepath}")
    try:
        with ProcessPoolExecutor() as executor:
            future = executor.submit(open_netcdf4, input_filepath)
            valid_time, vals = future.result()
        for i in range(len(valid_time)):
            date = pd.to_datetime(valid_time[i]).strftime("%Y%m%d")
            output_filepath = os.path.join(era5_output_path, f"{date}.nc")
            if os.path.exists(output_filepath):
                continue
            newValues = (vals[i, ::-1, :] * 1000).astype("float32")
            arr = xr.DataArray(
                    newValues,
                    dims=["lat", "lon"],
                    coords={"lat": target_lats, "lon": target_lons},
                    name="swe",
            )
            encoding = {"swe": {"zlib": True, "complevel": 9}}
            arr.to_netcdf(output_filepath, engine='h5netcdf', encoding=encoding)
    except Exception as e:
        print(f'Error load dataset {input_filepath}: {e}')

if __name__ == "__main__":
    os.makedirs(era5_output_path, exist_ok=True)

    dates = pd.date_range('1991-01-01', '2025-12-31', freq='ME')
    with ProcessPoolExecutor(max_tasks_per_child=1) as executor: # HDF5 has global state :(
        executor.map(process_file, dates)
