import os
import calendar
import numpy as np
import pandas as pd
import zarr
import xarray as xr
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

VARIABLES = {
    "t2m":     "2m_temperature",
    "t2m_min": "2m_temperature",
    "td2m":    "2m_dewpoint_temperature",
    "ts":      "skin_temperature",
    "tp":      "total_precipitation",
    "sd":      "snow_depth",
    "sden":    "snow_density",
    "u10":     "10m_u_component_of_wind",
    "v10":     "10m_v_component_of_wind",
    "msl":     "mean_sea_level_pressure",
    "olr":     "top_net_thermal_radiation",
}

OUT_DIR = "era5"
YEARS = list(range(1980, 2027))
HOURS = [0, 6, 12, 18]

ORIGIN = pd.Timestamp("1900-01-01")

os.makedirs(OUT_DIR, exist_ok=True)

store = zarr.open(STORE, mode="r", storage_options={"token": "anon"})

t_raw = store["time"][:]
time = ORIGIN + pd.to_timedelta(t_raw, unit="h")

lat = store["latitude"][:]
lon = store["longitude"][:]

i_90 = int(np.argmin(np.abs(lat - 90)))
i_0  = int(np.argmin(np.abs(lat - 0)))
i_lo, i_hi = sorted([i_90, i_0])
lat_north = lat[i_lo:i_hi+1]

lon_new = ((lon + 180) % 360) - 180
lon_order = np.argsort(lon_new)
lon_out = lon_new[lon_order]


def save_month(daily, var_ours, start, last_day, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if lat[i_lo] < lat[i_hi]:
        daily = daily[:, ::-1, :]
        lat_out = lat_north[::-1]
    else:
        lat_out = lat_north
    daily = daily[:, :, lon_order] 

    ds = xr.Dataset(
        {var_ours: (("valid_time", "latitude", "longitude"), daily)},
        coords={
            "valid_time": pd.date_range(start, periods=last_day, freq="D"),
            "latitude": lat_out.astype("float64"),
            "longitude": lon_out.astype("float64"),
        },
    )

    ds.to_netcdf(
        out_file,
        encoding={var_ours: {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    del ds, daily

def download_month(var_theirs, var_ours, year, month):
    out_file = os.path.join(OUT_DIR, var_ours, f"{year}{month:02d}.nc")
    if os.path.exists(out_file):
        return out_file

    start = pd.Timestamp(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = pd.Timestamp(year, month, last_day, 23)

    mask = (time >= start) & (time <= end)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f'no data for {year}-{month:02d}')

    sel_hours = HOURS
    if var_ours == 't2m_min':
        sel_hours = list(range(0, 24))

    hours = time[idx].hour
    sel = idx[np.isin(hours, sel_hours)]
    if sel.size == 0:
        raise ValueError(f'no selected hours for {year}-{month:02d}')

    arr = store[var_theirs]
    data = arr[sel, i_lo:i_hi+1, :]
    data = data.astype("float32")

    data = data.reshape(last_day, len(sel_hours), data.shape[1], data.shape[2])
    if var_ours == 't2m_min':
        save_month(data.min(axis=1), var_ours, start, last_day, out_file)
        save_month(data.max(axis=1), 't2m_max', start, last_day, os.path.join(OUT_DIR, 't2m_max', f"{year}{month:02d}.nc"))
    else:
        save_month(data.mean(axis=1), var_ours, start, last_day, out_file)

    del data
    return out_file


def main():
    tasks = []
    for var_ours, var_theirs in VARIABLES.items():
        for year in YEARS:
            months = range(1, 13) if year != 2026 else range(1, 7)
            for month in months:
                tasks.append((var_theirs, var_ours, year, month))

    for t in tqdm(tasks):
        try:
            out = download_month(*t)
            print("ok:", t[1], t[2], t[3], "->", out)
        except Exception as e:
            print("fail:", t, e)

if __name__ == "__main__":
    main()
