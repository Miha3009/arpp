import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

seas_input_path = "../../data/raw/swe_seas/"
seas_output_path = "../../data/prepare/swe_seas/"

minlon, minlat, maxlon, maxlat = 27, 50, 56, 65
target_lons = np.arange(minlon, maxlon+0.1, 0.25)
target_lats = np.arange(minlat, maxlat+0.1, 0.25)

def get_fixed_week(date):
    day_of_year = date.dayofyear
    return (day_of_year - 1) // 7 + 1

def process_file(date):
    date_str = date.strftime('%Y%m%d')
    filename = f"seas_snow_depth_{date_str[:4]}_{date_str[4:6]}.nc"
    input_filepath = os.path.join(seas_input_path, filename)

    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} not found")
        return None

    print(f"Process {input_filepath}")
    try:
        ds = xr.open_dataset(input_filepath, engine='cfgrib')
        ds = ds['sd'].mean(dim='number', keep_attrs=True) * 1000
        ds = ds.rename('swe').to_dataset().rename({'latitude': 'lat', 'longitude': 'lon'})
        ds = ds.interp(lat=target_lats, lon=target_lons, method="linear", kwargs={"fill_value": "extrapolate"})
        ds = ds.drop_vars(['time', 'surface'])
        ds['month'] = ((ds['valid_time'].dt.year - date.year) * 12 +
                    ds['valid_time'].dt.month - date.month)
        ds = ds.where(ds['month'] < 4, drop=True)
        days = (ds['valid_time'].dt.dayofyear - 1).astype(int)
        ds['week'] = (days // 7 + 1).where(days // 7 + 1 <= 52, 52)
        swe_mean = ds['swe'].groupby(ds['week']).mean()
        month_min = ds['month'].groupby(ds['week']).min()
        ds['year'] = ds['valid_time'].dt.year
        year_first = ds['year'].groupby(ds['week']).first()
        ds = xr.Dataset({
            'swe': swe_mean,
            'lead_time': month_min,
            'year': year_first
        })
        return ds
    except Exception as e:
        print(f'Error load dataset {input_filepath}: {e}')

if __name__ == "__main__":
    os.makedirs(seas_output_path, exist_ok=True)

    dates = pd.date_range('1991-01-01', '2025-12-31', freq='MS')
    with ProcessPoolExecutor() as executor:
        dss = list(executor.map(process_file, dates))
        dss = [ds for ds in dss if ds is not None]
        ds = xr.concat(dss, dim='week')
        encoding = {"swe": {"zlib": True, "complevel": 9}}
        ds.to_netcdf(seas_output_path + "all.nc", engine='h5netcdf', encoding=encoding)
