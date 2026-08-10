import numpy as np
import os
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.interpolate import RegularGridInterpolator
import logging

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

    result = [np.full((len(run_dirs), ndays, region['M'], region['N']), np.nan, dtype=np.float32) for region in regions]

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
                    for k, region in enumerate(regions):
                        result[k][i, start_idx+j-start_day, :, :] = tmp[region['min_i']:region['max_i'], region['min_j']:region['max_j']]

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

        file_exist = True
        for i, region in enumerate(regions):
            output_filepath = os.path.join(element_directory, f'{region["name"]}{date.year}{date.month:02d}.nc')
            if not os.path.exists(output_filepath):
                file_exist = False
                break
        if file_exist:
            logging.info(f'SKIP {directory}, {element_out}')
            continue

        data = read_forecast(run_dirs, element, date, layer, nlayers)
        ensmem = np.arange(0, len(run_dirs))
        days = np.arange(0, data[0].shape[1])
        encoding = {f"{element_out}": {"zlib": True, "complevel": 9}}

        for i, region in enumerate(regions):
            output_filepath = os.path.join(element_directory, f'{region["name"]}{date.year}{date.month:02d}.nc')
            arr = xr.DataArray(data[i],
                dims=["ensmem", "day", "lat", "lon"],
                coords={"ensmem": ensmem, "day": days, "lat": region['lat'], "lon": region['lon']},
                name=element_out,)
            arr.to_netcdf(output_filepath, engine='h5netcdf', encoding=encoding)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='extract.log', filemode='w')

N = 288
M = 180
lat = np.linspace(-89.5, 89.5, M)
lon = np.linspace(0, 360, N+1)[:-1]

#data_dir = '/RHM-Lustre3.2/users/neacc/evolodin/hindcasts_soda_full_field/clim_1991-2020.INMCM6M'
data_dir = '/RHM-Lustre3.2/users/neacc/evolodin/operative_hmc_full_field/clim_1991-2020.INMCM6M'

output_directory = "/RHM-Lustre3.2/users/kompa/msmirnov/arctic/data"
#first_date, last_date = 199101, 202012
first_date, last_date = 202408, 202604

elements = ['CICE', 'CWAT', 'PRC', 'PREC', 'PS', 'RUNOFF', 'SNOW', 'SFR', 'SS', 'T2', 'T2MIN', 'T2MAX']
elements = [(element, element, 0, 1) for element in elements]

levels = list(map(float,'0.2/0.3/0.5/0.7/1/2/3/5/7/10/20/30/50/70/100/150/200/250/300/400/500/600/700/850/925/1000'.split('/')))
lev_to_idx = {lev: i for i, lev in enumerate(levels)}
elements.append(('T', 'T850', lev_to_idx[850], len(levels)))

regions = [
#    {"name": "euro", "minlat": 41, "maxlat": 72, "minlon": 19, "maxlon": 60},
#    {"name": "wsib", "minlat": 49, "maxlat": 75, "minlon": 55, "maxlon": 90},
#    {"name": "esib", "minlat": 49, "maxlat": 80, "minlon": 85, "maxlon": 130},
#    {"name": "fare", "minlat": 42, "maxlat": 75, "minlon": 125, "maxlon": 190}
    {"name": "north", "minlat": 0, "maxlat": 90, "minlon": 0, "maxlon": 360},
]
for region in regions:
    lat_mask = (lat >= region["minlat"]) & (lat <= region["maxlat"])
    lon_mask = (lon >= region['minlon']) & (lon <= region['maxlon'])
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    region['min_i'] = lat_indices[0]
    region['max_i'] = lat_indices[-1] + 1
    region['min_j'] = lon_indices[0]
    region['max_j'] = lon_indices[-1] + 1
    region['M'] = region['max_i'] - region['min_i']
    region['N'] = region['max_j'] - region['min_j']
    region['lat'] = lat[lat_indices]
    region['lon'] = lon[lon_indices]

dates = pd.date_range(
    start=f"{first_date // 100}-{first_date % 100:02d}-01",
    end=f"{last_date // 100}-{last_date % 100:02d}-01",
    freq='MS'
).strftime('%Y-%m-22').tolist()

tasks = [(os.path.join(data_dir, date), date) for date in dates]

#for task in tasks:
#    process_task(task)

#with ProcessPoolExecutor(max_workers=32) as executor:
#    executor.map(process_task, tasks)
