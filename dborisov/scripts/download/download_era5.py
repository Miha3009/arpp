import cdsapi
import calendar
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

variables = [("t2m", "2m_temperature"),
             ("tp", "total_precipitation"),
             ("sd", "snow_depth"),
             ("sden", "snow_density"),
             ("t2m_min", "minimum_2m_temperature_since_previous_post_processing"),
             ("t2m_max", "maximum_2m_temperature_since_previous_post_processing"),
             ("pt", "precipitation_type"),
             ("h500", "geopotential#500"),
             ("t850", "temperature#850")]

dataset_single = "derived-era5-single-levels-daily-statistics"
dataset_pressure = "derived-era5-pressure-levels-daily-statistics"
years = list(range(1980, 2027))

tasks = []

for variable, varname in variables:
    dataset = dataset_pressure if '#' in varname else dataset_single
    level = None
    if '#' in varname:
        tmp = varname.split('#')
        varname = tmp[0]
        level = tmp[1]

    for year in years:
        months = list(range(1, 13)) if year != 2026 else list(range(1, 5))
        for month in months:
            output_file = os.path.join('era5', variable, f'{year}{month:02d}.nc')
            if os.path.exists(output_file):
                continue
            days = list(range(1, calendar.monthrange(year, month)[1] + 1))
            request = {
                "product_type": "reanalysis",
                "variable": [varname],
                "year": [year],
                "month": [f"{month:02d}"],
                "day": [f"{day:02d}" for day in days],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "6_hourly",
                "area": [90, -180, 0, 180]
            }
            if level is not None:
                request['pressure_level'] = level
            
            tasks.append({
                'dataset': dataset,
                'request': request,
                'output_file': output_file,
                'variable': variable,
                'year': year,
                'month': month
            })

def download_task(task):
    variable = task['variable']
    year = task['year']
    month = task['month']
    output_file = task['output_file']
    
    try:
        if os.path.exists(output_file):
            return
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        client = cdsapi.Client(retry_max=100, timeout=10)
        client.retrieve(task['dataset'], task['request']).download(output_file)
        return

    except Exception as e:
        print(f"[ERROR] {variable} {year}-{month:02d}: {str(e)}")
        return

success_count = 0
fail_count = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    for task in tasks:
        executor.submit(download_task, task)
