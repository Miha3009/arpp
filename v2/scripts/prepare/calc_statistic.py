import torch
import patcher
import numpy as np
import pandas as pd
import sys
import os
sys.path.append('../../db/')
from content import ELEMENTS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from tqdm.auto import tqdm

THREADS = 4
STATS_PATH = '../../data/stats.csv'

def days_in_month(date):
    return calendar.monthrange(int(date[:4]), int(date[4:]))[1]

def generate_months(start_str, end_str):
    start = datetime.strptime(start_str, "%Y%m")
    end = datetime.strptime(end_str, "%Y%m")

    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y%m"))
        current += relativedelta(months=1)
    
    return months

def process_variable(context, variable):
    stats = pd.read_csv(STATS_PATH)
    if variable in list(stats['variable']) or variable == 'inm_snow_cover':
        return

    mn, mx, sum, sum2, cnt, nans, climate_sum = np.inf, -np.inf, 0, 0, 0, 0, 0
    requests = []
    if 'inm_' in variable:
        dates = generate_months("199102", "202101") + generate_months("202409", "202606")
        for date in dates:
            requests.append(patcher.Request(variable, 0, 0, date + "01", 360, 91, 119, 1, 1, int(date[4:])))
    else:
        dates = generate_months("198001", "202605")
        for date in dates:
            requests.append(patcher.Request(variable, 0, 0, date + "01", 1440, 361, days_in_month(date), 1, 1))

    for i in tqdm(range(0, len(requests), THREADS), desc=variable):
        result = patcher.load(context, requests[i:i+THREADS])
        result_climate = patcher.load_climate(context, requests[i:i+THREADS])
        for j in range(len(result)):
            result_climate[j] = torch.nan_to_num(result_climate[j], 0.0)
            x = result[j] + result_climate[j]
            nans += torch.sum(torch.isnan(x))
            print(requests[i+j].t0, mn, mx, nans)
            min_val, linear_idx = torch.min(x.flatten(), dim=0)
            multi_idx = np.unravel_index(linear_idx.item(), x.shape)
            x = x[~torch.isnan(x)].reshape(-1)
            if len(x) == 0:
                continue
            cnt += len(x)
            mn = min(mn, torch.min(x))
            mx = max(mx, torch.max(x))
            sum += torch.sum(x)
            sum2 += torch.sum(x**2)
            climate_sum += torch.sum(result_climate[j])

    df = pd.DataFrame([{'variable': variable, 'min': mn.item(), 'max': mx.item(), 'mean': (sum / cnt).item(),
                        'std': torch.sqrt((sum2 - sum ** 2 / cnt) / cnt).item(),
                        'has_nan': (nans > 0).item(), 'has_climate': (torch.abs(climate_sum) > 1e-3).item() }])
    if len(stats) > 0:
        stats = pd.concat([stats, df])
    else:
        stats = df
    stats.round(3).to_csv(STATS_PATH, index=False)

if not os.path.exists(STATS_PATH):
    pd.DataFrame([], columns=['variable', 'min', 'max', 'mean', 'std', 'has_nan', 'has_climate']).to_csv(STATS_PATH, index=False)

context = patcher.Context('../../db', THREADS)
for variable in ELEMENTS:
    process_variable(context, variable)

