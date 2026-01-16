import numpy as np
import pandas as pd
import os
import xarray as xr
from utils import get_week_dates
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dateutil.relativedelta import relativedelta

inmcm_input_path = "../../data/raw/inmcm"
climate_input_path = "../../data/climate"
merge_week_output_path = "../../data/merge/inmcm_week"
merge_swe_output_path = "../../data/merge/inmcm_swe"
merge_ice_output_path = "../../data/merge/inmcm_ice"
merge_month_output_path = "../../data/merge/inmcm_month"
main_elements = ["cld", "h500", "olr", "prec", "ps", "rq2", "t2", "t2max", "t2min", "t850",
                 "u850", "uv10", "v850", "ws", "ww"]
ice_elements = ["aice", "hice", "sst"]

climate = defaultdict(dict)
data = defaultdict(dict)
loaded = {}

def months_diff(d1, d2):
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2)
    if d1 > d2:
        d1, d2 = d2, d1
    delta = relativedelta(d2, d1)
    months = delta.years * 12 + delta.months
    if delta.days < 0:
        months -= 1
    return months

def load_file(task):
    date, element, input_file = task
    ds = xr.open_dataset(input_file, engine="h5netcdf")
    ds = ds.sel(lat=slice(40, None))
    values = ds[element].values
    times = ds.time.values
    lat, lon = ds.lat.values, ds.lon.values
    t0 = pd.to_datetime(times[0])
    out = []
    for i, t in enumerate(times):
        t_date = pd.to_datetime(t)
        t_year = date.year if t_date.year == 1991 else date.year + 1 # Bug
        date_str = f"{t_year}{t_date.month:02d}{t_date.day:02d}"
        index_str = f"{element}_{t0.month:02d}"
        out.append((date_str, index_str, (element, months_diff(t0, t), values[i, :, :])))
    return out, lat, lon

def calc_mean_arr(task):
    element, dates, period, period_name = task
    if not period in climate[element]:
        return [], []

    to_delete = []
    result = []
    mean = defaultdict(list)
    count = defaultdict(int)
    lead_times = []
    for date in dates:
        date_str = f"{date.year}{date.month:02d}{date.day:02d}"
        for idx in data[date_str]:
            if '_'.join(idx.split('_')[:-1]) != element:
                continue
            element, lead_time, values = data[date_str][idx]
            if f"shift_{period_name}" in data:
                values = np.roll(values, values.shape[1]//2, axis=1)
            if len(mean[lead_time]) == 0:
                mean[lead_time] = values
            else:
                mean[lead_time] += values
            count[lead_time] += 1
            to_delete.append((date_str, idx))
    for lead_time in sorted(mean.keys()):
        lead_times.append(lead_time)
        result.append(mean[lead_time] / count[lead_time] - climate[element][period])

    if len(lead_times) == 0:
        return [], []

    result, lead_times = np.array(result), np.array(lead_times)
    coords = [lead_times] + data[f"coords_{period_name}"]
    return xr.DataArray(result.astype(np.float32), coords=coords, dims=["lead_time", "lat", "lon"], name=element), to_delete

def process_period(year, merge_output_path, dates, elements, period, period_name):
    os.makedirs(merge_output_path, exist_ok=True)

    output_file = f"{merge_output_path}/{year}{period:02d}.nc"
    if os.path.exists(output_file):
        return

    print(f"Process {year}/{period:02d} {period_name}")

    tasks = []
    for date in dates:
        for element in elements:
            date2 = date + pd.DateOffset(months=1) if element == 'swe' and (date.day != 1 or date.year != 1991 or date.month != 1) else date # Bug
            input_file = f"{inmcm_input_path}/{element}/{date2.year}{date2.month:02d}.nc"
            if input_file in loaded:
                continue
            loaded[input_file] = True
            if not os.path.exists(input_file):
                print(f"File not found {input_file}")
                continue
            tasks.append((date, element, input_file))

    if len(tasks) > 0:
        with ProcessPoolExecutor() as executor:
            for out, lat, lon in executor.map(load_file, tasks):
                if f"coords_{period_name}" not in data:
                    if np.max(lon) > 180:
                        data[f"shift_{period_name}"] = True
                        lon = np.roll(((lon + 180) % 360) - 180, len(lon)//2)
                    data[f"coords_{period_name}"] = [lat.astype(np.float32), lon.astype(np.float32)]
                for date_str, index_str, payload in out:
                    data[date_str][index_str] = payload

    tasks = [(element, dates, period, period_name) for element in elements]
    arrs = []
    for task in tasks:
        arr, to_delete = calc_mean_arr(task)
        if len(to_delete) == 0:
            continue
        arrs.append(arr)
        for date_str, idx in to_delete:
            del data[date_str][idx]

    encoding = {}
    dataset = {}
    for arr in arrs:
        encoding[arr.name] = {"zlib": True, "complevel": 9}
        dataset[arr.name] = arr
    xr.Dataset(dataset).to_netcdf(output_file, engine='h5netcdf', encoding=encoding)

def process_week(year, week):
    dates = get_week_dates(year, week)
    process_period(year, merge_week_output_path, dates, main_elements, week, "week")

def process_swe(year, week):
    dates = get_week_dates(year, week)
    process_period(year, merge_swe_output_path, dates, ["swe"], week, "swe")

def process_month(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    dates = pd.date_range(start=start, periods=start.days_in_month, freq='D')
    process_period(year, merge_month_output_path, dates, main_elements, month, "month")

def process_ice(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    dates = pd.date_range(start=start, periods=start.days_in_month, freq='D')
    process_period(year, merge_ice_output_path, dates, ice_elements, month, "ice")

if __name__ == "__main__":
    for element in (main_elements + ["swe"]):
        for i in range(52):
            climate_file = f"{climate_input_path}/week/{element}/{(i+1):02d}.nc"
            if not os.path.exists(climate_file):
                print(f"File not found {climate_file}")
                continue
            ds = xr.open_dataset(climate_file, engine="h5netcdf")
            climate[element][i+1] = ds[element].values

    for year in range(1991, 2020):
        for week in range(1, 53):
            process_week(year, week)
            process_swe(year, week)

    climate = defaultdict(dict)
    for element in (main_elements + ice_elements):
        for i in range(12):
            climate_file = f"{climate_input_path}/month/{element}/{(i+1):02d}.nc"
            if not os.path.exists(climate_file):
                print(f"File not found {climate_file}")
                continue
            ds = xr.open_dataset(climate_file, engine="h5netcdf")
            climate[element][i+1] = ds[element].values

    data = defaultdict(dict)
    loaded = {}
    for year in range(1991, 2020):
        for month in range(1, 13):
            process_month(year, month)
            process_ice(year, month)
