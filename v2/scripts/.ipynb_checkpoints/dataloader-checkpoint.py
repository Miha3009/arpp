import xarray as xr
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from prepare.utils import get_year_days
from torch.utils.data import DataLoader
import random

class MyDataset():
    def __init__(self, region, inmcm_dir="../data/inmcm", era5_dir="../data/era5", 
                 climate_dir="../data/inmcm_climate", era5_climate_dir="../data/era5_climate",
                 std_path="../data/std.csv", static_dir='../data/static',
                 forecast_variables=None, static_variables=None, months_count=4, obs_len=7, num_workers=4):
        self.region = region
        self.inmcm_dir = Path(inmcm_dir)
        self.era5_dir = Path(era5_dir)
        self.climate_dir = Path(climate_dir)
        self.era5_climate_dir = Path(era5_climate_dir)
        self.static_dir = Path(static_dir)
        self.forecast_variables = [p.name for p in self.inmcm_dir.iterdir() if p.is_dir()] if forecast_variables is None else forecast_variables
        self.static_variables = ['z', 'sdor'] if static_variables is None else static_variables
        self.months_count = months_count
        self.obs_len = obs_len
        self.std_df = pd.read_csv(std_path, index_col=0)
        self._climate_cache = {}
        self._era5_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dates = [f.stem.replace(self.region, '') for f in (self.inmcm_dir / self.forecast_variables[0]).glob(f"{self.region}*.nc")]
        self.loader = DataLoader(self, batch_size=None, num_workers=num_workers)
        self.era5_lat = None
        self._read_static()
        for date in self.dates:
            self._get_era5(date)
        for var in self.forecast_variables:
            self._get_climate(var)
        random.shuffle(self.dates)

    def _read_static(self):
        static_data = xr.open_dataset(self.static_dir / f"{self.region}.nc")
        self.lat = torch.Tensor(static_data.lat.values.copy()).to(dtype=torch.float32)
        self.lon = torch.Tensor(static_data.lon.values.copy()).to(dtype=torch.float32)
        self.static_data = {}
        for var in self.static_variables:
            self.static_data[var] = torch.Tensor(static_data[var].values.copy()).to(dtype=torch.float32)

    def _get_climate(self, var):
        if var not in self._climate_cache:
            if var == "era5":
                path = self.era5_climate_dir / f"{self.region}.nc"
            else:
                path = self.climate_dir / var / f"{self.region}.nc"
            ds = xr.open_dataarray(path)
            self._climate_cache[var] = torch.Tensor(ds.values.copy()).to(dtype=torch.float32)
        return self._climate_cache[var]

    def _get_era5(self, date):
        if date not in self._era5_cache:
            path = self.era5_dir / f"{self.region}{date}.nc"
            if not path.exists():
                days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                return torch.full((days[int(date[4:6])-1], len(self.lat), len(self.lon)), torch.nan)
            data = xr.open_dataarray(path)
            self._era5_cache[date] = (torch.Tensor(data.values >= 5).float()).to(dtype=torch.float32)
        return self._era5_cache[date]

    def _shift_month(self, year, month, shift):
        return f"{year + (month+shift-1)//12}{(month+shift-1)%12+1:02d}"

    def __getitem__(self, idx):
        date = self.dates[idx]
        year, month = int(date[:4]), int(date[4:])
        start_date = np.datetime64(f"{date[:4]}-{date[4:6]}-01")
        result = {}
        days = None

        has_nan = []
        for var in self.forecast_variables:
            path = self.inmcm_dir / var / f"{self.region}{date}.nc"
            ds = xr.open_dataarray(path)
            if days is None:
                days = get_year_days(ds, date)
            data = torch.Tensor(ds.values.copy()).to(dtype=torch.float32)
            climate = self._get_climate(var)[days, :, :]
            result[var] = (data - climate) / self.std_df.loc[var, 'std']
            has_nan.append(torch.isnan(result[var]).any(dim=(0, 2, 3)))

        has_nan = torch.stack(has_nan).any(dim=0)
        for var in self.forecast_variables:
            result[var] = result[var][:, ~has_nan, :, :]
        dates = np.array([start_date + timedelta(days=lead_time) for lead_time in range(0, len(days))])[~has_nan]

        for var in self.static_variables:
            result[var] = self.static_data[var]

        months = [self._shift_month(year, month, i) for i in range(self.months_count)]
        result['y'] = torch.cat([self._get_era5(month) for month in months], dim=0)[~has_nan, :, :]
        result['obs'] = self._get_era5(self._shift_month(year, month, -1))[-self.obs_len:, :, :]
        result['lat'] = self.lat
        result['lon'] = self.lon
        result['lead_time'] = torch.arange(0, len(days)).to(dtype=torch.float32)[~has_nan]
        result['year'] = torch.Tensor([(date.year - 2005) / 10.0 for date in dates]).to(dtype=torch.float32)
        result['sin_day'] = torch.Tensor([np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25) for date in dates]).to(dtype=torch.float32)
        result['cos_day'] = torch.Tensor([np.cos(2 * np.pi * date.timetuple().tm_yday / 365.25) for date in dates]).to(dtype=torch.float32)

        return result

    def __len__(self):
        return len(self.dates)
