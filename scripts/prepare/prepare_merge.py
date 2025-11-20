import numpy as np
import pandas as pd
import os
import xarray as xr
from utils import get_week_dates
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

base_input_path = "../../data/prepare"
climate_input_path = "../../data/climate"
merge_output_path = "../../data/merge"

climate = defaultdict(dict)

def process_period(year, period, dates, element, name):
    date_str = f"{year}{period:02d}"
    output_file = f"{merge_output_path}/{name}/{date_str}.nc"
    if os.path.exists(output_file) or not period in climate[name]:
        return

    print(f"Process {name} {year}/{period:02d}")

    means = []
    for date in dates:
        input_file = f"{base_input_path}/{name}/{date.year}{date.month:02d}{date.day:02d}.nc"
        if not os.path.exists(input_file):
            print(f"File not found {input_file}")
            continue
        ds = xr.open_dataset(input_file, engine="h5netcdf")
        if len(means) == 0:
            lat, lon = ds.lat.values.astype(np.float32), ds.lon.values.astype(np.float32)
        means.append(ds[element].values)

    if len(means) == 0:
        return

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.array(means), axis=0)
    mean -= climate[name][period]
        
    encoding = {element: {"zlib": True, "complevel": 9}}
    arr = xr.DataArray(mean.astype(np.float32), coords=[lat, lon], dims=["lat", "lon"], name=element)
    arr.to_netcdf(output_file, engine='h5netcdf', encoding=encoding)

def process_week(task):
    year, week = task
    dates = get_week_dates(year, week)
    process_period(year, week, dates, "swe", "era5_swe")
    process_period(year, week, dates, "swe", "globsnow")

def process_month(task):
    year, month = task
    start = pd.Timestamp(year=year, month=month, day=1)
    dates = pd.date_range(start=start, periods=start.days_in_month, freq='D')
    dates = get_week_dates(year, week)
    process_period(year, month, dates, "ice", "era5_ice")
    process_period(year, month, dates, "ice", "noaa_ice")

if __name__ == "__main__":
    for name, period, element in [("era5_swe", "week", "swe"), ("globsnow", "week", "swe"),
                    ("era5_ice", "month", "ice"), ("noaa_ice", "month", "ice")]:
        for i in range(52 if period == "week" else 12):
            climate_file = f"{climate_input_path}/{period}/{name}/{(i+1):02d}.nc"
            if not os.path.exists(climate_file):
                print(f"File not found {climate_file}")
                continue
            ds = xr.open_dataset(climate_file, engine="h5netcdf")
            climate[name][i+1] = ds[element].values
        os.makedirs(f"{merge_output_path}/{name}", exist_ok=True)

    weeks, months = [], []
    for year in range(1991, 2020):
        for week in range(1, 53):
            weeks.append((year, week))
        for month in range(1, 13):
            months.append((year, month))
    with ProcessPoolExecutor() as executor:
        executor.map(process_week, weeks) 
        executor.map(process_month, months)
