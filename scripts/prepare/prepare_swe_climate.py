import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from collections import defaultdict
from utils import get_weeks

era5_input_path = "../../data/prepare/era5_swe"
globsnow_input_path = "../../data/prepare/globsnow"
climate_output_path = "../../data/climate"

def process_mean(task):
    element, input_files, output_file = task
    if os.path.exists(output_file):
        return

    print(f"Process {output_file}")

    counts = np.array([0])
    for file in input_files:
        if not os.path.exists(file):
            #print(f"Not found {file}")
            continue
        ds = xr.open_dataset(file, engine="h5netcdf")
        ds = ds.sel(lat=slice(40, None))
        values = ds.swe.values
        if counts.max() == 0:
            mean = np.zeros_like(values, dtype=np.float32)
            counts = np.zeros_like(values, dtype=np.int32)
            lat = ds.lat.values
            lon = ds.lon.values
        mask = ~np.isnan(values)
        mean[mask] += values[mask]
        counts[mask] += 1

    if counts.max() == 0:
        return

    mean[counts > 0] /= counts[counts > 0]
    arr = xr.DataArray(mean, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="swe")
    arr.to_netcdf(output_file, engine="h5netcdf")

def process_element(element, input_directory):
    output_directory = f"{climate_output_path}/week/{element}"
    os.makedirs(output_directory, exist_ok=True)

    weeks = get_weeks()
    dates = pd.date_range(f"1991-01-01", f"2019-12-31", freq="D")
    dates_group = defaultdict(list)
    for date in dates:
        week_key = f"{date.month:02d}{date.day:02d}"
        if not week_key in weeks:
            continue
        week = weeks[week_key]
        dates_group[f"{week:02d}"].append(date)

    tasks = []
    for week in dates_group:
        input_files = [f"{input_directory}/{d.year}{d.month:02d}{d.day:02d}.nc" for d in dates_group[week]]
        output_file = f"{output_directory}/{week}.nc"
        tasks.append((element, input_files, output_file))

    with ProcessPoolExecutor() as executor:
        executor.map(process_mean, tasks)

if __name__ == "__main__":
    process_element("era5_swe", era5_input_path)
    process_element("globsnow", globsnow_input_path)
