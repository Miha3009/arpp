import numpy as np
from datetime import datetime, timedelta

regions = [
    {'name': 'esib', 'min_lat': 49.5, 'max_lat': 79.5, 'min_lon': 85.0, 'max_lon': 130.0},
    {'name': 'wsib', 'min_lat': 49.5, 'max_lat': 74.5, 'min_lon': 55.0, 'max_lon': 90.0},
    {'name': 'euro', 'min_lat': 41.5, 'max_lat': 71.5, 'min_lon': 20.0, 'max_lon': 60.0},
    {'name': 'fare', 'min_lat': 42.5, 'max_lat': 74.5, 'min_lon': 125.0, 'max_lon': 190.0}
]

def get_year_days(ds, date):
    base_date = datetime.strptime(f'{date}01', '%Y%m%d').replace(year=2001)
    return np.array([(base_date + timedelta(days=int(d))).timetuple().tm_yday - 1 for d in ds.day.values])
