import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import shutil

climate_input_path = "../../data/climate"
merge_input_path = "../../data/merge"
reference_grid = "../../data/climate/month/era5_ice/01.nc"
output_path = "../../data/train"
years = list(range(1991, 2020))

def check_exists(files):
    for file in files:
        if not os.path.exists(file):
            print(f"Not found {file}")
            return False
    return True

def process_file(period, inmcm_climate, era5_climate, inmcm_anomaly_path, inmcm_anomaly_path2, era5_anomaly_path,
                 variable, era5_variable):
    if inmcm_climate.shape[0] == 0 or era5_climate.shape[0] == 0:
        return

    os.makedirs(f"{output_path}/{variable}/anom", exist_ok=True)
    os.makedirs(f"{output_path}/{variable}/clim", exist_ok=True)
    os.makedirs(f"{output_path}/{variable}/bias", exist_ok=True)

    bias_file = f"{output_path}/{variable}/bias/{period:02d}.nc"
    if not os.path.exists(bias_file):
        print(f"Generate bias for {variable} {period}")
        anomalies = []
        era5_anomalies = []
        for year in years:
            input_file = f"{inmcm_anomaly_path}/{year}{period:02d}.nc"
            era5_input_file = f"{era5_anomaly_path}/{year}{period:02d}.nc"
            if not check_exists([input_file, era5_input_file]):
                continue
            ds = xr.open_dataset(input_file, engine="h5netcdf")
            if len(ds.variables) == 0:
                continue
            anomaly = ds[variable].fillna(0).interp(**interp_options).values
            ds.close()
            ds = xr.open_dataset(era5_input_file, engine="h5netcdf")
            era5_anomaly = ds[era5_variable].values
            ds.close()
            for i in range(anomaly.shape[0]):
                anomalies.append(anomaly[i, :, :] + inmcm_climate - (era5_anomaly + era5_climate))
        if len(anomalies) == 0:
            return
        anomalies = np.array(anomalies, dtype=np.float32)
        anomaly_bias = np.mean(anomalies, axis=0)

        arr = xr.DataArray(anomaly_bias, coords=[lat, lon], dims=["lat", "lon"], name=variable)
        encoding = {variable: {"zlib": True, "complevel": 9}}
        arr.to_netcdf(bias_file, engine="h5netcdf", encoding=encoding)
    else:
        anomaly_bias = xr.open_dataset(bias_file, engine="h5netcdf")[variable].values

    period_name = 'month' if variable == 'aice' else 'week'
    shutil.copy(f'{climate_input_path}/{period_name}/era5_{era5_variable}/{period:02d}.nc',
                f'{output_path}/{variable}/clim/{period:02d}.nc')

    for year in years:
        input_file = f"{inmcm_anomaly_path}/{year}{period:02d}.nc"
        input_file2 = f"{inmcm_anomaly_path2}/{year}{period:02d}.nc"
        era5_input_file = f"{era5_anomaly_path}/{year}{period:02d}.nc"
        output_file = f"{output_path}/{variable}/anom/{year}{period:02d}.nc"

        if os.path.exists(output_file):
            continue

        print(f'Process {output_file}')
        if not check_exists([input_file, input_file2, era5_input_file]):
            continue

        ds = xr.open_dataset(input_file, engine="h5netcdf")
        if len(ds.variables) == 0:
            continue
        var_interp = ds[variable].fillna(0).interp(**interp_options)
        ds = ds.drop_vars(variable)
        ds = ds.rename({"lat": "lat1", "lon": "lon1"})
        
        var_interp.values = var_interp.values + inmcm_climate - anomaly_bias
        if variable == 'aice':
            var_interp.values = np.minimum(1, np.maximum(0, var_interp.values))
        elif variable == 'swe':
            var_interp.values = np.maximum(0, var_interp.values)
        var_interp.values -= era5_climate
        ds = ds.assign({variable: var_interp})

        ds2 = xr.open_dataset(input_file2, engine="h5netcdf").rename({"lat": "lat2", "lon": "lon2"})
        if variable == 'aice':
            ds2 = ds2.drop_vars(['ww', 'ws'])

        ds_era5 = xr.open_dataset(era5_input_file, engine="h5netcdf").rename_vars({era5_variable: "era5"})

        lead_times = np.intersect1d(ds.coords['lead_time'].values, ds2.coords['lead_time'].values)
        ds = xr.merge([ds.sel(lead_time=lead_times), ds2.sel(lead_time=lead_times), ds_era5])
        encoding = {v: {"zlib": True, "complevel": 9} for v in list(ds.variables)}
        ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)

def read_climate(variable, period, path):
    if not os.path.exists(path):
        print(f'File {path} not found')
        return np.array([])

    ds = xr.open_dataset(path, engine="h5netcdf")
    if not 'era5' in path:
        ds = ds.fillna(0).interp(**interp_options)
    return ds[variable].values

def process_ice(month):
    inmcm_climate = read_climate('aice', month, f'{climate_input_path}/month/aice/{month:02d}.nc')
    era5_climate = read_climate('ice', month, f'{climate_input_path}/month/era5_ice/{month:02d}.nc')
    process_file(month, inmcm_climate, era5_climate, f'{merge_input_path}/inmcm_ice', f'{merge_input_path}/inmcm_month',
                 f'{merge_input_path}/era5_ice', 'aice', 'ice')

def process_swe(week):
    inmcm_climate = read_climate('swe', week, f'{climate_input_path}/week/swe/{week:02d}.nc')
    era5_climate = read_climate('swe', week, f'{climate_input_path}/week/era5_swe/{week:02d}.nc')
    process_file(week, inmcm_climate, era5_climate, f'{merge_input_path}/inmcm_swe', f'{merge_input_path}/inmcm_week',
                 f'{merge_input_path}/era5_swe', 'swe', 'swe')

if __name__ == "__main__":
    ds = xr.open_dataset(reference_grid, engine="h5netcdf")
    lat, lon = ds.lat.values, ds.lon.values
    interp_options = {
        "lat": lat,
        "lon": lon,
        "method": "linear",
        "kwargs": {"fill_value": "extrapolate"}
    }
    with ProcessPoolExecutor() as executor:
        executor.map(process_ice, [month for month in range(1, 13)])
        executor.map(process_swe, [week for week in range(1, 53)])
