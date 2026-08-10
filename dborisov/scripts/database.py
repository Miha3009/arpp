import torch
import patcher
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
import math

class PatchDataset(Dataset):
    def __init__(self, variables, time_range, epoch_size, era_scales=[], inm_scales=[], x_range=(0, 1440), y_range=(100, 300),
                 batch_size=8, lead_time_range=(0, 3), num_workers=4):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.lead_time_range = lead_time_range
        self.era_time_width = max([s['tSize'] * s['tStep'] for s in era_scales], default=0)
        self.inm_time_width = max([s['tSize'] * s['tStep'] for s in inm_scales], default=0)
        self.t_min = (datetime.strptime(time_range[0], '%Y%m%d') - datetime(1970, 1, 1)).days + self.era_time_width
        self.t_max = (datetime.strptime(time_range[1], '%Y%m%d') - datetime(1970, 1, 1)).days
        self.era_scales = era_scales
        self.inm_scales = inm_scales
        self.variables = variables
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.set_seed(0)
        self.context = patcher.Context('/Volumes/portable/arpp/data', num_workers)
        self.loader = DataLoader(self, batch_size=None)

    def set_seed(self, seed):
        rng = np.random.RandomState(seed)
        n = self.epoch_size * self.batch_size
        x = rng.randint(self.x_min, self.x_max + 1, size=n)
        y = rng.randint(self.y_min, self.y_max + 1, size=n)
        t = rng.randint(self.t_min, self.t_max + 1, size=n)
        self.xyt = list(zip(x, y, t))
        self.lead_times = rng.randint(self.lead_time_range[0], self.lead_time_range[1] + 1, size=n).tolist()
        return self

    def __len__(self):
        return self.epoch_size

    def get_time_variable(self, dates, variable):
        if variable == 'year':
            return torch.tensor([d.year for d in dates], dtype=torch.float32)
        elif variable == 'day':
            return torch.tensor([d.timetuple().tm_yday for d in dates], dtype=torch.float32)
        elif variable == 'cos_day':
            return torch.tensor([math.cos(2 * math.pi * d.timetuple().tm_yday / 365) for d in dates], dtype=torch.float32)
        elif variable == 'sin_day':
            return torch.tensor([math.sin(2 * math.pi * d.timetuple().tm_yday / 365) for d in dates], dtype=torch.float32)

    def get_spatial_variable(self, lat, lon, variable):
        if variable == 'lat':
            return lat
        elif variable == 'lon':
            return lon
        elif variable == 'cos_lat':
            return torch.cos(torch.deg2rad(lat))
        elif variable == 'cos_lon':
            return torch.cos(torch.deg2rad(lon))
        elif variable == 'sin_lon':
            return torch.sin(torch.deg2rad(lon))

    def get_era_request(self, idx, variable, s):
        key = f'{variable}_{s["id"]}'
        x0, y0, t0, = self.xyt[idx]
        date = datetime(1970, 1, 1) + timedelta(days=int(t0 - s['tSize']*s['tStep']))

        if variable in ['year', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable), key

        x = (x0 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']
        y = (y0 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] 

        if variable in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = 0.25*torch.arange(y, y + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = 0.25*torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable), key

        date = date.strftime('%Y%m%d')
        req = patcher.Request(variable, x, y, date, s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'])
        return req, key

    def get_inm_request(self, idx, variable, s):
        key = f'{variable}_{s["id"]}'
        x0, y0, t0 = self.xyt[idx]
        date = datetime(1970, 1, 1) + timedelta(days=int(t0 - s['tSize']*s['tStep']))

        if variable[4:] in ['year', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable[4:]), key

        lead_time = self.lead_times[idx]
        origin_date = date - relativedelta(months=lead_time)
        origin_date.replace(day=1)
        days = (date - origin_date).days
        while days < self.inm_time_width:
            lead_time += 1
            origin_date = origin_date - relativedelta(months=1)
            days = (date - origin_date).days

        if variable[4:] == 'lead_time':
            return torch.tensor([days + i*s['tStep'] for i in range(s['tSize'])], dtype=torch.float32), key

        x = (x0 // 4 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']
        y = (y0 // 4 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']

        if variable[4:] in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = torch.arange(y, y + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable[4:]), key

        req = patcher.Request(variable, x, y, date.strftime('%Y%m%d'), s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'], origin_date.month)
        return req, key

    def __getitem__(self, idx):
        requests = []
        for i in range(self.batch_size):
            for variable in self.variables:
                if variable.startswith('inm_'):
                    requests += [self.get_inm_request(idx * self.batch_size + i, variable, s) for s in self.inm_scales]
                else:
                    requests += [self.get_era_request(idx * self.batch_size + i, variable, s) for s in self.era_scales]
        requests, keys = map(list, zip(*requests))
        result = patcher.load(self.context, [r for r in requests if isinstance(r, patcher.Request)])

        batched = defaultdict(list)
        j = 0
        for i in range(len(requests)):
            if isinstance(requests[i], patcher.Request):
                value = result[j]
                j += 1
            else:
                value = requests[i]
            batched[keys[i]].append(value)

        for key in batched:
            batched[key] = torch.stack(batched[key], dim=0)
        return batched

