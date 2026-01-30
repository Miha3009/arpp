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

    data = {}
    years = {}
    shift = False
    for i, f in enumerate(files):
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds = ds.sel(lat=slice(40, None))
        values = ds[element].values
        if i == 0:
            lat = ds.lat.values.astype(np.float32)
            lon = ds.lon.values.astype(np.float32)
            if np.max(lon) > 180:
                lon = np.roll(((lon + 180) % 360) - 180, len(lon)//2)
                shift = True
            for j in range(period_count):
                data[j] = []
                years[j] = []
        for j, t in enumerate(ds.time.values):
            year = int(f[-9:-5])
            index = to_index(t) - 1
            if np.isnan(index):
                continue
            if shift:
                data[index].append(np.roll(values[j, :, :], len(lon) // 2, axis=1))
            else:
                data[index].append(values[j, :, :])
            years[index].append(year)

    for index in range(period_count):
        years_ = np.array(years[index])
        years_norm = years_ - years_.min()
        x_centered = years_norm - np.mean(years_norm)
        data_array = np.array(data[index])
        mean = np.mean(data_array, axis=0)
        mean[np.abs(mean) > 1e15] = np.nan
        trend = np.tensordot(x_centered, data_array, axes=(0, 0)) / np.sum(x_centered**2)
        ds_out = xr.Dataset({
            f'{element}_mean': (['lat', 'lon'], mean),
            f'{element}_trend': (['lat', 'lon'], trend),
            f'{element}_year_mean': ([], np.mean(years_)),
        }, coords={'lat': lat, 'lon': lon})
        output_filepath = f"{output_directory}/{(index+1):02d}.nc"
        ds_out.to_netcdf(output_filepath, engine="h5netcdf")

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
