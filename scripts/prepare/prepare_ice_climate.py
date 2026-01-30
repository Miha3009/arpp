import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from collections import defaultdict

era5_input_path = "../../data/prepare/era5_ice"
climate_output_path = "../../data/climate"

def process_mean(task):
    element, input_files, output_file = task
    if os.path.exists(output_file):
        print(output_file)
        return

    print(f"Process {output_file}")

    data_list = []
    years = []
    
    for file in input_files:
        if not os.path.exists(file):
            continue
        ds = xr.open_dataset(file, engine="h5netcdf")
        ds = ds.sel(lat=slice(40, None))
        data_list.append(ds.ice.values)
        years.append(int(os.path.basename(file)[:4]))
        if len(data_list) == 1:
            lat, lon = ds.lat.values, ds.lon.values

    if len(data_list) < 2:
        return

    data_array = np.array(data_list)
    years_norm = np.array(years) - np.array(years).min()
    
    mean = np.mean(data_array, axis=0)
    x_centered = years_norm - np.mean(years_norm)
    trend = np.tensordot(x_centered, data_array, axes=(0, 0)) / np.sum(x_centered**2)

    ds_out = xr.Dataset({
        'aice_mean': (['lat', 'lon'], mean),
        'aice_trend': (['lat', 'lon'], trend),
        'aice_year_mean': ([], np.mean(years)),
    }, coords={'lat': lat, 'lon': lon})

    ds_out.to_netcdf(output_file, engine="h5netcdf")

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
