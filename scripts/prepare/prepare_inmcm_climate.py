import numpy as np
import pandas as pd
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from utils import get_weeks, week_elements, month_elements

inmcm_input_path = "../../data/raw/inmcm"
climate_output_path = "../../data/climate"
years = np.arange(1991, 2020)

def process_element(element, period_name, period_count, to_index):
    output_directory = f"{climate_output_path}/{period_name}/{element}"
    os.makedirs(output_directory, exist_ok=True)
    if len(os.listdir(output_directory)) == period_count:
        return

    files = sorted(glob.glob(f"{inmcm_input_path}/{element}/*.nc"))
    if not files:
        print(f"No files found for {element}")
        return

    print(f"Processing {element}, {len(files)} files")

    means = []
    counts = [0]*period_count
    shift = False
    for i, f in enumerate(files):
        ds = xr.open_dataset(f, engine="h5netcdf")
        values = ds[element].values
        if i == 0:
            lat = ds.lat.values.astype(np.float32)
            lon = ds.lon.values.astype(np.float32)
            if np.max(lon) > 180:
                lon = np.roll(((lon + 180) % 360) - 180, len(lon)//2)
                shift = True
            means = [np.zeros((len(lat), len(lon)), dtype=np.float32) for j in range(period_count)]
        for j, t in enumerate(ds.time.values):
            index = to_index(t) - 1
            if np.isnan(index):
                continue
            if shift:
                means[index] += np.roll(values[j, :, :], len(lon) // 2, axis=1)
            else:
                means[index] += values[j, :, :]
            counts[index] += 1
    for index in range(period_count):
        mean = means[index]
        mean /= counts[index]
        mean[np.abs(mean) > 1e15] = np.nan
        arr = xr.DataArray(means[index], dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name=element)
        output_filepath = f"{output_directory}/{(index+1):02d}.nc"
        arr.to_netcdf(output_filepath, engine="h5netcdf")

def process_month(element):
    process_element(element, "month", 12, lambda t: pd.to_datetime(t).month)

def process_week(element):
    weeks = get_weeks()
    process_element(element, "week", 52,
                    lambda t: weeks.get(f"{pd.to_datetime(t).month:02d}{pd.to_datetime(t).day:02d}", np.nan)
    )

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        executor.map(process_month, month_elements)
        executor.map(process_week, week_elements)
