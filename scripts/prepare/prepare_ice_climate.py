import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from collections import defaultdict

era5_input_path = "../../data/prepare/era5_ice"
noaa_input_path = "../../data/prepare/noaa_ice"
climate_output_path = "../../data/climate"

def process_mean(task):
    element, input_files, output_file = task
    if os.path.exists(output_file):
        print(output_file)
        return

    print(f"Process {output_file}")

    count = 0
    for i, file in enumerate(input_files):
        if not os.path.exists(file):
            print(f"Not found {file}")
            continue
        ds = xr.open_dataset(file, engine="h5netcdf")
        values = ds.ice.values
        if i == 0:
            mean = values
            lat = ds.lat.values
            lon = ds.lon.values
        else:
            mean += values
        count += 1

    if count == 0:
        return

    mean /= count
    arr = xr.DataArray(mean, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="ice")
    arr.to_netcdf(output_file, engine="h5netcdf")

def process_element(element, input_directory):
    output_directory = f"{climate_output_path}/month/{element}"
    os.makedirs(output_directory, exist_ok=True)

    dates = pd.date_range(f"1991-01-01", f"2019-12-31", freq="D")
    dates_group = defaultdict(list)
    for date in dates:
        dates_group[f"{date.month:02d}"].append(date)

    tasks = []
    for month in dates_group:
        input_files = [f"{input_directory}/{d.year}{d.month:02d}{d.day:02d}.nc" for d in dates_group[month]]
        output_file = f"{output_directory}/{month}.nc"
        tasks.append((element, input_files, output_file))

    with ProcessPoolExecutor() as executor:
        executor.map(process_mean, tasks)

if __name__ == "__main__":
    process_element("era5_ice", era5_input_path)
    process_element("noaa_ice", noaa_input_path)
