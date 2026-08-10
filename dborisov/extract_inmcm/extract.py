import numpy as np
import os
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.interpolate import RegularGridInterpolator
import logging

def interpolate_to_1deg(arr):
    target_lat = np.arange(0, 91, 1)
    target_lon = np.arange(0, 360, 1)
    
    grid_lat, grid_lon = np.meshgrid(target_lat, target_lon, indexing='ij')
    points = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)

    orig_lat = arr.lat.values
    orig_lon = arr.lon.values

    ensmem_size = arr.ensmem.size
    day_size = arr.day.size
    
    result = np.zeros((ensmem_size, day_size, len(target_lat), len(target_lon)), dtype=np.float32)
    
    for e in range(ensmem_size):
        for d in range(day_size):
            data_2d = arr.values[e, d]
            interp = RegularGridInterpolator((orig_lat, orig_lon), data_2d, method='linear', bounds_error=False, fill_value=None)
            result[e, d] = interp(points).reshape(len(target_lat), len(target_lon))
    
    result[:, :, -1, :] = result[:, :, -2, :]
    
    coords = {
        'ensmem': arr.ensmem.values,
        'day': arr.day.values,
        'lat': target_lat,
        'lon': target_lon
    }
    dims = ['ensmem', 'day', 'lat', 'lon']
    return xr.DataArray(result, dims=dims, coords=coords, name=arr.name)

def day_of_year(date):
    if date.month == 2 and date.day == 29:
        date = datetime(date.year, 2, 28)
    return datetime(2022, date.month, date.day).timetuple().tm_yday 

def read_forecast(run_dirs, element, start_date, layer=0, nlayers=1):
    end_date = start_date + relativedelta(months=4) - timedelta(days=1)

    start_day = day_of_year(start_date)
    end_day = day_of_year(end_date)
    ndays = (end_day - start_day + 365) % 365 + 1

    years_to_read = [(start_date.year, start_day-1, end_day-1, 0)]
    if end_day < start_day:
        years_to_read = [
            (start_date.year, start_day-1, 364, 0),
            (start_date.year+1, 0, end_day-1, 365-start_day)
        ]

    result = np.full((len(run_dirs), ndays, M // 2, N), np.nan, dtype=np.float32)

    one_layer_bytes = N * M * 4
    one_day_bytes = nlayers * one_layer_bytes
    for i, run_dir in enumerate(run_dirs):
        for year, start_day, end_day, start_idx in years_to_read:
            file_path = os.path.join(run_dir, str(year - 1850), 'DAILY', f'D{element}.STD')
            if not os.path.exists(file_path):
                logging.error(f'ERROR file {file_path} NOT FOUND')
                continue

            logging.info('READ ' + file_path)
            with open(file_path, 'rb') as f:
                f.seek(0, 2)
                max_days = f.tell() // one_day_bytes
                for j in range(start_day, min(max_days, end_day+1)):
                    f.seek(j * one_day_bytes + one_layer_bytes * layer)
                    tmp = np.roll(np.fromfile(f, dtype=np.float32, count=N * M).reshape((M, N)), N // 2, axis=1)
                    result[i, start_idx+j-start_day, :, :] = tmp[90:, :]

    return result

def process_task(task):
    directory, date = task
    date = datetime.strptime(date, '%Y-%m-%d')
    date = date + relativedelta(months=1)
    date = datetime(date.year, date.month, 1)

    run_dirs = []
    for run_dir in os.listdir(directory):
        if not run_dir.startswith('RUN_') or not run_dir[4:].isdigit():
            continue
        run_dirs.append(os.path.join(directory, run_dir))

    for element, element_out, layer, nlayers in elements:
        element_directory = os.path.join(output_directory, element_out)
        os.makedirs(element_directory, exist_ok=True)

        output_filepath = os.path.join(element_directory, f'north{date.year}{date.month:02d}.nc')
        if os.path.exists(output_filepath):
            logging.info(f'SKIP {directory}, {element_out}')
            continue

        data = read_forecast(run_dirs, element, date, layer, nlayers)
        ensmem = np.arange(0, len(run_dirs))
        days = np.arange(0, data.shape[1])
        encoding = {f"{element_out}": {"zlib": True, "complevel": 9}}
        target_lat = np.arange(0, 91, 1)
        target_lon = np.arange(0, 360, 1)

        output_filepath = os.path.join(element_directory, f'north{date.year}{date.month:02d}.nc')

        arr = xr.DataArray(data,
            dims=["ensmem", "day", "lat", "lon"],
            coords={"ensmem": ensmem, "day": days, "lat": lat[90:], "lon": lon},
            name=element_out,)

        logging.info("INTERPOLATE " + output_filepath)
        arr = interpolate_to_1deg(arr)
        arr.to_netcdf(output_filepath, engine='h5netcdf', encoding=encoding)

def gen_tasks(first_date, last_date, directory):
    dates = pd.date_range(
        start=f"{first_date // 100}-{first_date % 100:02d}-01",
        end=f"{last_date // 100}-{last_date % 100:02d}-01",
        freq='MS'
    ).strftime('%Y-%m-22').tolist()
    return [(os.path.join(directory, date), date) for date in dates]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='extract.log', filemode='w')

N = 288
M = 180
lat = np.linspace(-89.5, 89.5, M)
lon = np.linspace(0, 360, N+1)[:-1]
levels = list(map(float,'0.2/0.3/0.5/0.7/1/2/3/5/7/10/20/30/50/70/100/150/200/250/300/400/500/600/700/850/925/1000'.split('/')))
lev_to_idx = {lev: i for i, lev in enumerate(levels)}

output_directory = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(output_directory, exist_ok=True)

elements = ['OLR', 'HLT', 'WW', 'TS']
elements = [(element, element, 0, 1) for element in elements]
#elements.append(('T', 'T850', lev_to_idx[850], len(levels)))

hindcast_tasks = gen_tasks(199101, 202012, '/RHM-Lustre3.2/users/neacc/evolodin/hindcasts_soda_full_field/clim_1991-2020.INMCM6M')
operative_tasks = gen_tasks(202408, 202604, '/RHM-Lustre3.2/users/neacc/evolodin/operative_hmc_full_field/clim_1991-2020.INMCM6M')
tasks = hindcast_tasks + operative_tasks

#for task in tasks:
#    process_task(task)

with ProcessPoolExecutor(max_workers=1) as executor:
    executor.map(process_task, tasks)
