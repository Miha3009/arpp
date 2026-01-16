import numpy as np
import pandas as pd
from datetime import datetime

def get_weeks():
    dates = pd.date_range(f"2001-01-01", f"2001-12-30", freq="D")
    weeks = {}

    week_num = 1
    day_count = 0
    for ts in dates:
        key = f"{ts.month:02d}{ts.day:02d}"
        weeks[key] = week_num
        day_count += 1
        if day_count == 7:
            week_num += 1
            day_count = 0
    return weeks

def get_week_dates(year, week):
    result = []
    for date, week_num in weeks.items():
        if week_num == week:
            result.append(datetime(year, int(date[:2]), int(date[2:])))
    return result

weeks = get_weeks()

month_elements = ["aice", "cld", "h500", "hice", "olr", "prec", "ps", "rq2", "sst",
                  "t2", "t2max", "t2min", "t850", "u850", "uv10", "v850", "ws", "ww"]
week_elements = ["swe", "cld", "h500", "olr", "prec", "ps", "rq2", "t2", "t2max", "t2min", "t850",
                 "u850", "uv10", "v850", "ws", "ww"]

class QuantileMapper:
    def __init__(self, n_quantiles=1000):
        self.n_quantiles = n_quantiles
        self.quantiles = None
        self.values = None

    def fit(self, arr_source):
        arr_source = np.asarray(arr_source)
        arr_source = arr_source[~np.isnan(arr_source)]
        qs = np.linspace(0, 1, self.n_quantiles)
        self.quantiles = qs
        self.values = np.quantile(arr_source, qs)

    def transform(self, arr_target, ignore_zeros=False):
        arr_target = np.asarray(arr_target)
        mapped = np.full_like(arr_target, np.nan, dtype=np.float32)
        mask = ~np.isnan(arr_target)
        if ignore_zeros:
            mask &= arr_target != 0
        valid = arr_target[mask]
        ranks = np.argsort(np.argsort(valid)) / (len(valid) - 1)
        mapped_valid = np.interp(ranks, self.quantiles, self.values)
        mapped[mask] = mapped_valid
        return mapped

    def save(self, filename):
        np.savez(str(filename), quantiles=self.quantiles, values=self.values)

    def load(self, filename):
        data = np.load(filename)
        self.quantiles = data["quantiles"]
        self.values = data["values"]

