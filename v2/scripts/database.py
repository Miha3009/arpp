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
                 era_scales=[], inm_scales=[], mask=None, num_workers=4, fix_seed=None, with_climate=[]):
        self.era_time_width = max([s['tSize'] * s['tStep'] for s in era_scales], default=0)
        self.inm_time_width = max([s['tSize'] * s['tStep'] for s in inm_scales], default=0)
        self.modes = {}
        self.with_climate = with_climate
        for mode in modes:
            x = modes[mode].copy()
            x.setdefault("x_min", 0)
            x.setdefault("x_max", 1440)
            x.setdefault("y_min", 0)
            x.setdefault("y_max", 361)
            x.setdefault("batch_size", 8)
            x.setdefault('lead_time_range', (0, 119))
            x['lead_time_range'] = (max(x['lead_time_range'][0], self.inm_time_width),
                                              max(x['lead_time_range'][1], self.inm_time_width))
            x['t_min'] = (datetime.strptime(x['t_min'], '%Y%m%d') - datetime(1970, 1, 1)).days
            x['t_max'] = (datetime.strptime(x['t_max'], '%Y%m%d') - datetime(1970, 1, 1)).days
            x['doy_to_years'] = self.get_doy_to_years(x['t_min'], x['t_max'] - x['lead_time_range'][0])
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
        lead_min, lead_max = mode['lead_time_range']
        n = mode['epoch_size'] * mode['batch_size']
        if self.mask is None:
            x = rng.randint(mode['x_min'], mode['x_max'] + 1, size=n)
            y = rng.randint(mode['y_min'], mode['y_max'] + 1, size=n)
            t = rng.randint(mode['t_min'], mode['t_max'] - lead_min, size=n)
            self.xyt = list(zip(x, y, t))
        elif self.mask == 'snow':
            x_size, y_size = mode['x_max'] - mode['x_min'], mode['y_max'] - mode['y_min']
            req = patcher.Request('sd', mode['x_min'], mode['y_min'], '19800101', x_size, y_size, 365, 1, 1)
            climate = patcher.load_climate(self.context, [req])[0] > 4
            days = np.array([int(k) for k in mode['doy_to_years'].keys() if int(k) < 365], dtype=np.int32)
            climate = climate[days, :, :]
            idx = torch.nonzero(climate)
            chosen = idx[rng.choice(len(idx), n, replace=True)]
            self.xyt = []
            for doy, y, x in chosen:
                day = int(days[int(doy)])
                year = rng.choice(list(mode['doy_to_years'][day]))
                date = datetime(year, 1, 1) + timedelta(days=day)
                t = (date - datetime(1970, 1, 1)).days
                self.xyt.append((mode['x_min'] + int(x) // 4 * 4, mode['y_min'] + int(y) // 4 * 4, t))
        else:
            raise ValueError(f'Mask {self.mask} not found')

        t_values = np.array([item[2] for item in self.xyt])
        max_lead = mode['t_max'] - t_values
        max_lead = np.clip(max_lead, lead_min, lead_max)
        self.lead_times = rng.randint(lead_min, max_lead + 1).tolist()
        return self

    def set_full(self, mode):
        mode = self.modes[mode]
        lead_min, lead_max = mode['lead_time_range']
        x_size = self.target_scale['xSize']
        y_size = self.target_scale['ySize']
        t_size = self.target_scale['tSize']

        x_range = range(mode['x_min'] + x_size // 2, mode['x_max'] + 1, x_size)
        y_range = range(mode['y_min'] + y_size // 2, mode['y_max'] + 1, y_size)
        
        start_date = datetime(1970, 1, 1) + timedelta(days=mode['t_min'])
        end_date = datetime(1970, 1, 1) + timedelta(days=mode['t_max'])

        tl = []
        current = start_date.replace(day=1)
        max_allowed = end_date + relativedelta(months=1)
        while current <= end_date:
            current = current.replace(year=current.year + 1, month=1) if current.month == 12 else current.replace(month=current.month + 1)
            current2 = (current + relativedelta(months=1)) - timedelta(days=1)
            for lead_time in range(lead_min, lead_max + t_size, t_size):
                l = max(lead_min, min(lead_time, lead_max, (max_allowed - current).days))
                if len(tl) > 0 and tl[-1][1] == l:
                    break
                tl.append(((current2 - datetime(1970, 1, 1)).days, l))

        self.xy = [(x, y) for x in x_range for y in y_range]
        self.xyt = [(x, y, t) for t, _ in tl for x, y in self.xy]
        self.xyl = [(x, y, l) for _, l in tl for x, y in self.xy]
        self.lead_times = [l for _, l in tl for x, y in self.xy]
        self.t = defaultdict(int)
        for t, l in tl:
            self.t[t] += len(self.xy)

        return self

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return int(np.ceil(len(self.xyt) / self.modes[self.mode]['batch_size']))

    def get_time_variable(self, dates, variable):
        if variable == 'year':
            return torch.tensor([d.year for d in dates], dtype=torch.float32)
        elif variable == 'year_norm':
            return (torch.tensor([d.year for d in dates], dtype=torch.float32) - 2000) / 10
        elif variable == 'day':
            return torch.tensor([d.timetuple().tm_yday for d in dates], dtype=torch.float32)
        elif variable == 'cos_day':
            return torch.tensor(np.cos(2 * np.pi / 365 * np.array([d.timetuple().tm_yday for d in dates])), dtype=torch.float32)
        elif variable == 'sin_day':
            return torch.tensor(np.sin(2 * np.pi / 365 * np.array([d.timetuple().tm_yday for d in dates])), dtype=torch.float32)

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

    def get_variable_props(self, variable):
        if variable.endswith('_clim'):
            return variable[:-5], False, True
        prefix = ''
        if variable.startswith('inm_'):
            prefix = 'inm_'
        if variable[len(prefix):] in self.with_climate:
            return variable, True, True
        return variable, True, False

    def get_era_request(self, idx, variable, s, is_target=False):
        key = f'{variable}_{s['id']}' if not is_target else variable
        x0, y0, _ = self.xyt[idx]
        date, _ = self.get_date(idx, s, is_target)

        if variable in ['year', 'year_norm', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable), key, is_target, False, False

        x = (x0 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] if 'fixX' not in s else s['fixX']
        y = (y0 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] if 'fixY' not in s else s['fixY']
        if s['xyStep'] == 1:
            x = x // 4 * 4
            y = y // 4 * 4

        if variable in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = 0.25*torch.arange(y, y + s['ySize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = 0.25*torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable), key, is_target, False, False

        date = date.strftime('%Y%m%d')
        variable, var, clim = self.get_variable_props(variable)
        req = patcher.Request(variable, x, y, date, s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'])
        return req, key, is_target, var, clim

    def get_inm_request(self, idx, variable, s):
        is_target = 'id' not in s
        key = f'{variable}_{s['id']}' if not is_target else variable
        x0, y0, _ = self.xyt[idx]
        date, origin_date = self.get_date(idx, s, True)

        if variable[4:] in ['year', 'year_norm', 'day', 'cos_day', 'sin_day']:
            dates = [date + timedelta(days=int(i*s['tStep'])) for i in range(s['tSize'])]
            return self.get_time_variable(dates, variable[4:]), key, is_target, False, False

        if variable[4:] == 'lead_time':
            days = (date - origin_date).days
            return torch.tensor([days + i*s['tStep'] for i in range(s['tSize'])], dtype=torch.float32), key, is_target, False, False

        x = (x0 // 4 - s['xSize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] if 'fixX' not in s else s['fixX']
        y = (y0 // 4 - s['ySize'] * s['xyStep'] // 2) // s['xyStep'] * s['xyStep'] if 'fixY' not in s else s['fixY']

        if variable[4:] in ['lat', 'lon', 'cos_lat', 'cos_lon', 'sin_lon']:
            lat = torch.arange(y, y + s['ySize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            lon = torch.arange(x, x + s['xSize']*s['xyStep'], s['xyStep'], dtype=torch.float32)
            return self.get_spatial_variable(lat, lon, variable[4:]), key, is_target, False, False

        variable, var, clim = self.get_variable_props(variable)
        req = patcher.Request(variable, x, y, date.strftime('%Y%m%d'), s['xSize'], s['ySize'], s['tSize'], s['xyStep'], s['tStep'], origin_date.month)
        return req, key, is_target, var, clim

    def __getitem__(self, idx):
        requests = []
        batch_size = self.modes[self.mode]['batch_size']
        current_batch_size = min(batch_size, len(self.xyt) - idx * batch_size)
        for i in range(current_batch_size):
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
        requests, keys, is_target, var, clim = map(list, zip(*requests))
        result = patcher.load(self.context, [req for req, mask in zip(requests, var) if mask])
        result_clim = patcher.load_climate(self.context, [req for req, mask in zip(requests, clim) if mask])

        inputs = defaultdict(list)
        targets = defaultdict(list)
        j, k = 0, 0
        for i in range(len(requests)):
            if var[i] and clim[i]:
                value = result[j] + result_clim[k]
                j, k = j + 1, k + 1
            elif var[i] and not clim[i]:
                value = result[j]
                j += 1
            elif not var[i] and clim[i]:
                value = result_clim[k]
                k += 1
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

