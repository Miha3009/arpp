import torch
import patcher
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset
from saver import saveable

class SkipDataLoader:
    def __init__(self, dataset):
        self.skip = 0
        self.dataset = dataset

    def __iter__(self):
        for i in range(self.skip, len(self.dataset)):
            yield self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    def set_skip(self, skip):
        self.skip = skip

@saveable
class PatchDataset(Dataset):
    def __init__(self, input_variables, target_variables, target_scale, modes,
                 era_scales=[], inm_scales=[], mask=None, num_workers=4, fix_seed=None):
        self.era_time_width = max([s['tSize'] * s['tStep'] for s in era_scales], default=0)
        self.inm_time_width = max([s['tSize'] * s['tStep'] for s in inm_scales], default=0)
        self.modes = {}
        for mode in modes:
            x = modes[mode].copy()
            x.setdefault("x_min", 0)
            x.setdefault("x_max", 1440)
            x.setdefault("y_min", 0)
            x.setdefault("y_max", 361)
            x.setdefault("batch_size", 8)
            x.setdefault('lead_time_range', (0, 119))
            x['t_min'] = (datetime.strptime(x['t_min'], '%Y%m%d') - datetime(1970, 1, 1)).days + self.era_time_width
            x['t_max'] = (datetime.strptime(x['t_max'], '%Y%m%d') - datetime(1970, 1, 1)).days
            x['lead_time_range'] = (max(x['lead_time_range'][0], self.inm_time_width),
                                              max(x['lead_time_range'][1], self.inm_time_width))
            x['doy_to_years'] = self.get_doy_to_years(x['t_min'], x['t_max'])
            self.modes[mode] = x
        self.mode = 'train'
        self.mask = mask
        self.era_scales = era_scales
        self.inm_scales = inm_scales
        self.target_scale = target_scale
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.context = patcher.Context('../db', num_workers)
        self.fix_seed = fix_seed
        self.set_seed(0)
        self.loader = SkipDataLoader(self)

    def get_doy_to_years(self, t_min, t_max):
        start = datetime(1970, 1, 1) + timedelta(days=t_min)
        end = datetime(1970, 1, 1) + timedelta(days=t_max)
        doy_to_years = defaultdict(set)
        d = start
        while d <= end:
            doy_to_years[d.timetuple().tm_yday - 1].add(d.year)
            d += timedelta(days=1)
        return doy_to_years

    def set_seed(self, seed):
        if self.fix_seed is not None:
            seed = self.fix_seed
        rng = np.random.RandomState(seed)
        mode = self.modes[self.mode]
        n = mode['epoch_size'] * mode['batch_size']
        if self.mask is None:
            x = rng.randint(mode['x_min'], mode['x_max'] + 1, size=n)
            y = rng.randint(mode['y_min'], mode['y_max'] + 1, size=n)
            t = rng.randint(mode['t_min'], mode['t_max'] + 1, size=n)
            self.xyt = list(zip(x, y, t))
        elif self.mask == 'snow':
            x_size, y_size = mode['x_max'] - mode['x_min'], mode['y_max'] - mode['y_min']
            req = patcher.Request('sd', mode['x_min'], mode['y_min'], '19800101', x_size, y_size, 365, 1, 1)
            climate = patcher.load_climate(self.context, [req])[0] > 4
            idx = torch.nonzero(climate)
            chosen = idx[rng.choice(len(idx), n, replace=True)]
            self.xyt = []
            for doy, y, x in chosen:
                year = rng.choice(list(mode['doy_to_years'][int(doy)]))
                date = datetime(year, 1, 1) + timedelta(days=int(doy))
                t = (date - datetime(1970, 1, 1)).days
                self.xyt.append((mode['x_min'] + int(x), mode['y_min'] + int(y), t))
        else:
            raise ValueError(f'Mask {self.mask} not found')

        self.lead_times = rng.randint(mode['lead_time_range'][0], mode['lead_time_range'][1] + 1, size=n).tolist()
        return self

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return self.modes[self.mode]['epoch_size']

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

    def get_date(self, idx, s, is_forecast):
        _, _, t1 = self.xyt[idx]
        if is_forecast:
            lead_time = self.lead_times[idx]
            origin_date = datetime(1970, 1, 1) + timedelta(days=int(t1)) + relativedelta(months=1)
            origin_date = origin_date.replace(day=1)
            date = origin_date + timedelta(days=int(lead_time - s['tSize']*s['tStep']))
            return date, origin_date
        else:
            return datetime(1970, 1, 1) + timedelta(days=int(t1 - s['tSize']*s['tStep'])), None

    def get_era_request(self, idx, variable, s, is_target=False):
        key = f'{variable}_{s['id']}' if not is_target else variable
        x0, y0, _ = self.xyt[idx]
        date, _ = self.get_date(idx, s, is_target)

        if variable in ['year', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable), key, is_target

        x = (x0 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']
        y = (y0 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] 

        if variable in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = 0.25*torch.arange(y, y + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = 0.25*torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable), key, is_target

        date = date.strftime('%Y%m%d')
        req = patcher.Request(variable, x, y, date, s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'])
        return req, key, is_target

    def get_inm_request(self, idx, variable, s):
        is_target = 'id' not in s
        key = f'{variable}_{s['id']}' if not is_target else variable
        x0, y0, _ = self.xyt[idx]
        date, origin_date = self.get_date(idx, s, True)

        if variable[4:] in ['year', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable[4:]), key, is_target

        if variable[4:] == 'lead_time':
            days = (date - origin_date).days
            return torch.tensor([days + i*s['tStep'] for i in range(s['tSize'])], dtype=torch.float32), key, is_target

        x = (x0 // 4 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']
        y = (y0 // 4 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep']

        if variable[4:] in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = torch.arange(y, y + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable[4:]), key, is_target

        req = patcher.Request(variable, x, y, date.strftime('%Y%m%d'), s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'], origin_date.month)
        return req, key, is_target

    def __getitem__(self, idx):
        requests = []
        batch_size = self.modes[self.mode]['batch_size']
        for i in range(batch_size):
            for variable in self.input_variables:
                if variable.startswith('inm_'):
                    requests += [self.get_inm_request(idx * batch_size + i, variable, s) for s in self.inm_scales]
                else:
                    requests += [self.get_era_request(idx * batch_size + i, variable, s) for s in self.era_scales]
            for variable in self.target_variables:
                if variable.startswith('inm_'):
                    requests += [self.get_inm_request(idx * batch_size + i, variable, self.target_scale)]
                else:
                    requests += [self.get_era_request(idx * batch_size + i, variable, self.target_scale, is_target=True)]
        requests, keys, is_target = map(list, zip(*requests))
        result = patcher.load(self.context, [r for r in requests if isinstance(r, patcher.Request)])

        inputs = defaultdict(list)
        targets = defaultdict(list)
        j = 0
        for i in range(len(requests)):
            if isinstance(requests[i], patcher.Request):
                value = result[j]
                j += 1
            else:
                value = requests[i]
            if is_target[i]:
                targets[keys[i]].append(value)
            else:
                inputs[keys[i]].append(value)

        for key in inputs:
            inputs[key] = torch.stack(inputs[key], dim=0)
        for key in targets:
            targets[key] = torch.stack(targets[key], dim=0)

        return inputs, targets

