import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import xarray as xr
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import json
import pickle

current_dir = Path(__file__).parent
datasets_dir = current_dir / '../data/datasets'
tmp_dir = current_dir / '../data/tmp'
default_extra_variables = ['cos_lat', 'sin_lon', 'cos_lon', 'sin_period', 'cos_period']
all_extra_variables = default_extra_variables + ['lat', 'lon', 'period', 'year', 'climate']

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__ndarray__': obj.tolist(), 'dtype': str(obj.dtype)}
        return super().default(obj)

def numpy_decoder(obj):
    if '__ndarray__' in obj:
        return np.array(obj['__ndarray__'], dtype=obj['dtype'])
    return obj

def make_train_test(variant, firstYear, separateYear, lastYear, args={}):
    dsTrain = ClimateDataset(variant, years=list(range(firstYear, separateYear)), **args)
    dsTest = ClimateDataset(variant, years=list(range(separateYear, lastYear+1)), **args)
    return dsTrain, dsTest

def load_dataset(name, filename=None):
    if filename == None:
        filename = datasets_dir / f'{name}.json'
    with open(filename, 'r') as f:
        args = json.load(f, object_hook=numpy_decoder)
        args['temporary'] = False
        ds = ClimateDataset(**args)
        ds.load_cache(name)
        return ds

def load_datasets(name):
    return load_dataset(f'{name}_train'), load_dataset(f'{name}_test')

class ClimateDataset(IterableDataset):
    def __init__(self, variant, name='', periods=None, data_path=current_dir / '../data', years=list(range(1991, 2020)),
                 variables=None, lead_times=[0], batch_size=1000, normed=True, cache_level=0, region=None,
                 loader_type="point", loader_opts={}, climate_trend=False, temporary=False):
        self.init_args = locals().copy()
        del self.init_args['self']
        del self.init_args['data_path']
        if temporary:
            return

        self.data_path = data_path
        self.train_path = f'{data_path}/train/{variant}'
        self.lead_times = np.array(lead_times)
        self.years = years
        self.variant = variant
        self.batch_size = int(batch_size)
        self.cache = {}
        self.cache_level = cache_level
        self.cache['full'] = False
        self.normed = normed
        if periods is None:
            self.periods = list(range(1, 53)) if self.variant == 'swe' else list(range(1, 13))
        else:
            self.periods = periods
        self.period_count = 52 if self.variant == 'swe' else 12
        self.loader_type = loader_type
        self.loader_opts = loader_opts
        self.clamp_value = 50 if self.variant == 'swe' else 1
        self.climate_trend = climate_trend
        self.name = name

        self.files = [f"{self.train_path}/anom/{y}{p:02d}.nc" for y in years for p in self.periods]
        self.files = [file for file in self.files if os.path.exists(file)]
        self.files_periods = {}
        for file in self.files:
            self.files_periods[file] = int(file[-5:-3])

        self.time_invariant = xr.open_dataset(f"{self.data_path}/train/time_invariant{'_norm' if self.normed else ''}.nc", engine="h5netcdf")
        sample_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        self.lat, self.lon = sample_ds.lat.values, sample_ds.lon.values
        if region is None:
            self.mask = None
        else:
            self.mask = region['mask']
            self.lat = region['lat']
            self.lon = region['lon']
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        self.lat_grid = torch.as_tensor(lat_grid, dtype=torch.float32)
        self.lon_grid = torch.as_tensor(lon_grid, dtype=torch.float32)

        if variables:
            self.variables = variables
        else:
            self.variables = list(sample_ds.data_vars)
            if variant == 'swe':
                self.variables += list(self.time_invariant.data_vars)
            self.variables += default_extra_variables
            self.variables = [v for v in self.variables if v != 'era5']
        self.anom_variables = [v for v in self.variables if v in sample_ds.data_vars]
        self.time_invariant_variables = [v for v in self.variables if v in self.time_invariant.data_vars]
        self.extra_variables = [v for v in self.variables if v in all_extra_variables]
        self.variables = self.anom_variables + self.time_invariant_variables + self.extra_variables
        sample_ds.close()

        self.climate = {}
        self.masks = {}
        self.trends = {}
        self.trends_mean_year = {}
        for p in range(1, self.period_count+1):
            climate_ds = xr.open_dataset(f"{self.train_path}/clim/{p:02d}.nc", engine="h5netcdf")
            self.climate[p] = self.unify(climate_ds, f'{self.variant}_mean', None).values.ravel()
            if self.mask is None:
                self.mask = np.zeros(self.climate[p].shape[0]) == 0
            self.masks[p] = torch.as_tensor((~np.isnan(self.climate[p])) & (self.climate[p] != 0) & (self.mask.ravel()))
            for v in climate_ds.data_vars:
                if self.climate_trend and 'trend' in v:
                    base = v[:-6]
                    self.trends_mean_year[base] = climate_ds[f'{base}_year_mean'].values
                    self.trends[base] = self.unify(climate_ds, v, None, fillna=True)
            climate_ds.close()

        self.extra_data = {
            'lat': self.lat_grid,
            'lon': self.lon_grid,
            'cos_lat': torch.cos(torch.deg2rad(self.lat_grid)),
            'sin_lon': torch.sin(torch.deg2rad(self.lon_grid)),
            'cos_lon': torch.cos(torch.deg2rad(self.lon_grid))
        }

        if self.normed:
            std_ds = xr.open_dataset(f"{self.train_path}/std.nc", engine="h5netcdf")
            self.std = {}
            for variable in std_ds.data_vars:
                self.std[variable] = np.nanmean(std_ds[variable].values)
            std_ds.close()

        self.loader = DataLoader(self, batch_size=None) #, num_workers=1 if self.loader_type == "sequence" else 4)

    def save(self, name, filename=None):
        if filename == None:
            filename = datasets_dir / f'{name}.json'
            self.init_args['name'] = name
        with open(filename, 'w') as f:
            json.dump(self.init_args, f, cls=NumpyEncoder)

    def save_cache(self, name, full=False):
        if full and not self.cache['full']:
            for _, _, _ in self.loader:
                continue
        with open(tmp_dir / f'{name}.pkl', 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self, name):
        filename = tmp_dir / f'{name}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.cache = pickle.load(f)

    def __iter__(self):
        worker_info = get_worker_info()
        if self.loader_type == 'point':
            yield from self.rebatch_loader(self.point_loader(worker_info))
        elif self.loader_type == 'map':
            yield from self.rebatch_loader(self.map_loader(worker_info))
        elif self.loader_type == 'sequence':
            yield from self.rebatch_loader(self.sequence_loader(self.loader_opts['len']), 1)
        elif self.loader_type == 'sequence_map':
            yield from self.rebatch_loader(self.sequence_map_loader(self.loader_opts['len']), 1)
        else:
            raise Exception(f"Unknown loader type {self.loader_type}")

    def point_loader(self, worker_info):
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id::worker_info.num_workers]

        for file in files:
            period = self.files_periods[file]
            for lead_time in self.lead_times:
                if self.check_lead_times([file], lead_time):
                    continue
                X, y, lat = self.load_file(file, lead_time)
                yield X, y, lat
        for file in self.files:
            ds = xr.open_dataset(file, engine="h5netcdf")
            self.cache[f"{file}_lead_times"] = ds.lead_time.values
            ds.close()

    def map_loader(self, worker_info):
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id::worker_info.num_workers]

        for file in files:
            period = self.files_periods[file]
            mask = self.masks[period]

            for lead_time in self.lead_times:
                if self.check_lead_times([file], lead_time):
                    continue
                X, y, lat = self.load_file(file, lead_time)
                flatX = X.reshape(-1, X.shape[-1])
                flatX[~mask] = np.nan
                X = flatX.reshape(X.shape)
                flaty = y.reshape(-1)
                flaty[~mask] = np.nan
                y = flaty.reshape(y.shape)
                X, y = torch.nan_to_num(X, nan=0.0), torch.nan_to_num(y, nan=0.0)
                yield X.unsqueeze(0), y.unsqueeze(0), lat.unsqueeze(0)

    def sequence_loader(self, seq_len):
        for i in range(len(self.files)):
            filename = Path(self.files[i]).name
            year, period = map(int, [filename[:4], filename[4:6]])
            dates = [(year + (period-i-1)//self.period_count, (period-i-1)%self.period_count+1) for i in range(seq_len)]
            files = [f"{self.train_path}/anom/{y}{p:02d}.nc" for y, p in dates[::-1]]

            period = int(Path(files[-1]).name[4:6])
            mask = self.masks[period]
            #for j in range(seq_len):
            #    period = int(Path(files[j]).name[4:6])
            #    if mask is None:
            #        mask = self.masks[period]
            #    else:
            #        mask = mask & self.masks[period]

            if mask.sum() == 0:
                continue

            for lead_time in self.lead_times:
                if self.check_lead_times(files, lead_time):
                    continue

                X, y, lat = [], [], []
                for j in range(seq_len):
                    X_frame, y_frame, lat_frame = self.load_file(files[j], lead_time)
                    X.append(X_frame.reshape(-1, X_frame.shape[-1])[mask])
                    y.append(y_frame.ravel()[mask])
                    lat.append(lat_frame.ravel()[mask])
                X, y, lat = torch.stack(X), torch.stack(y), torch.stack(lat)
                yield X, y, lat

    def sequence_map_loader(self, seq_len):
        for i in range(len(self.files)):
            filename = Path(self.files[i]).name
            year, period = map(int, [filename[:4], filename[4:6]])
            dates = [(year + (period-i-1)//self.period_count, (period-i-1)%self.period_count+1) for i in range(seq_len)]
            files = [f"{self.train_path}/anom/{y}{p:02d}.nc" for y, p in dates[::-1]]
            mask = self.masks[period]

            for lead_time in self.lead_times:
                if self.check_lead_times(files, lead_time):
                    continue

                X, y, lat = [], [], []
                for j in range(seq_len):
                    X_frame, y_frame, lat_frame = self.load_file(files[j], lead_time)
                    flatX = X_frame.reshape(-1, X_frame.shape[-1])
                    flatX[~mask] = np.nan
                    X_frame = flatX.reshape(X_frame.shape)
                    flaty = y_frame.reshape(-1)
                    flaty[~mask] = np.nan
                    y_frame = flaty.reshape(y_frame.shape)
                    X.append(torch.nan_to_num(X_frame, nan=0.0).unsqueeze(0))
                    y.append(torch.nan_to_num(y_frame, nan=0.0).unsqueeze(0))
                    lat.append(lat_frame.unsqueeze(0))
                X, y, lat = torch.stack(X), torch.stack(y), torch.stack(lat)
                yield X, y, lat

    def check_lead_times(self, files, lead_time):
        for file in files:
            if not f'{file}_lead_times' in self.cache:
                if not os.path.exists(file):
                    return True
                ds = xr.open_dataset(file, engine="h5netcdf")
                self.cache[f"{file}_lead_times"] = ds.lead_time.values
                ds.close()
            if not lead_time in self.cache[f'{file}_lead_times']:
                return True
        return False

    def rebatch_loader(self, loader, batch_dim=0):
        if self.cache['full']:
            for k in sorted(self.cache.keys()):
                if k.startswith('batch_'):
                    yield self.cache[k]
            return

        buf_X, buf_y, buf_lat, buf_size, batch_num = [], [], [], 0, 0
        for X, y, lat in loader:
            buf_X.append(X), buf_y.append(y), buf_lat.append(lat)
            buf_size += X.size(batch_dim)

            if buf_size >= self.batch_size:
                Xa, ya, lata = torch.cat(buf_X, batch_dim), torch.cat(buf_y, batch_dim), torch.cat(buf_lat, batch_dim)
                n_full = int(buf_size // self.batch_size)

                for i in range(n_full):
                    start = i * self.batch_size
                    end = start + self.batch_size
                    result = (Xa.narrow(batch_dim, start, self.batch_size),
                           ya.narrow(batch_dim, start, self.batch_size),
                           lata.narrow(batch_dim, start, self.batch_size))
                    if self.cache_level == 2:
                        self.cache[f'batch_{batch_num:06d}'] = result
                    yield result
                    batch_num += 1
                rem = buf_size % self.batch_size
                if rem:
                    buf_X = [Xa.narrow(batch_dim, n_full*self.batch_size, rem)]
                    buf_y = [ya.narrow(batch_dim, n_full*self.batch_size, rem)]
                    buf_lat = [lata.narrow(batch_dim, n_full*self.batch_size, rem)]
                    buf_size = rem
                else:
                    buf_X, buf_y, buf_lat, buf_size = [], [], [], 0
        if buf_X:
            result = torch.cat(buf_X, batch_dim), torch.cat(buf_y, batch_dim), torch.cat(buf_lat, batch_dim)
            if self.cache_level == 2:
                self.cache[f'batch_{batch_num:06d}'] = result
            yield result
        if self.cache_level == 2:
            clear_cache = [k for k in self.cache if 'file' in k]
            for k in clear_cache:
                del self.cache[k]
            self.cache['full'] = True

    def load_file(self, file, lead_time):
        field_id = f'file_{file}_{lead_time}'
        if self.cache_level >= 1 and field_id in self.cache:
            return self.cache[field_id]

        filename = Path(file).name
        year, period = map(int, [filename[:4], filename[4:6]])

        ds = xr.open_dataset(file, engine="h5netcdf")

        features = []
        for variable in self.anom_variables:
            values = self.unify(ds, variable, lead_time).values
            if self.climate_trend:
                values -= self.trends[variable] * (year - self.trends_mean_year[variable])
            if self.normed:
                values /= self.std[variable]
            features.append(values)
        for variable in self.time_invariant_variables:
            features.append(self.unify(self.time_invariant, variable, None).values)
        for variable in self.extra_variables:
            if variable in ['lat', 'lon', 'cos_lat', 'sin_lon', 'cos_lon']:
                features.append(self.extra_data[variable])
            elif variable == 'period':
                features.append(features[-1] * 0 + period)
            elif variable == 'sin_period':
                features.append(features[-1] * 0 + np.sin(2 * np.pi * (period - 1) / self.period_count))
            elif variable == 'cos_period':
                features.append(features[-1] * 0 + np.cos(2 * np.pi * (period - 1) / self.period_count))
            elif variable == 'climate':
                features.append(self.climate[period])
            elif variable == 'year':
                features.append(features[-1] * 0 + (year - 2005) / 10)

        X = np.stack(features, axis=-1)
        X = torch.as_tensor(X, dtype=torch.float32)

        y = self.unify(ds, "era5", None).values
        if self.climate_trend:
            y = np.asarray(y - self.trends[self.variant] * (year - self.trends_mean_year[self.variant]),
                           dtype=np.float32)

        y = torch.as_tensor(y, dtype=torch.float32)
        y = torch.clamp(y, -self.clamp_value, self.clamp_value)

        lat = self.extra_data['lat']

        if self.loader_type == 'point':
            mask = self.masks[period]
            X, y, lat = X.reshape(-1, len(self.variables))[mask], y.ravel()[mask], lat.ravel()[mask]

        if self.cache_level == 1:
            self.cache[field_id] = (X, y, lat)
        ds.close()
        return (X, y, lat)

    def add_extra(self, variable):
        if not variable in self.variables and variable in all_extra_variables:
            self.set_variables(self.variables + [variable])
        elif not variable in all_extra_variables:
            raise ValueError(f'Незвестная переменная {variable}')
        return self

    def remove_extra(self, variable):
        if variable in self.variables and variable in all_extra_variables:
            self.set_variables([v for v in self.variables if v != variable])
        elif not variable in all_extra_variables:
            raise ValueError(f'Незвестная переменная {variable}')
        return self

    def set_variables(self, variables):
        self.variables = variables
        sample_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        self.anom_variables = [v for v in self.variables if v in sample_ds.data_vars]
        self.time_invariant_variables = [v for v in self.variables if v in self.time_invariant.data_vars]
        self.extra_variables = [v for v in self.variables if v in all_extra_variables]
        self.variables = self.anom_variables + self.time_invariant_variables + self.extra_variables
        sample_ds.close()
        invalid_cache = [k for k in self.cache if k.startswith('batch_') or k.startswith('file_')]
        for k in invalid_cache:
            del self.cache[k]
        self.cache['full'] = False

    def unify(self, ds, variable, lead_time, fillna=False):
        ds = ds[variable]
        if "lead_time" in ds.coords:
            ds = ds.sel(lead_time=lead_time)
        if variable in ['sst', 'hice'] or fillna:
            ds = ds.fillna(0)

        if "lat" in ds.coords and "lon" in ds.coords and len(ds.lon.values) == len(self.lon):
            return ds

        lat_candidates = [c for c in ds.coords if "lat" in c]
        lon_candidates = [c for c in ds.coords if "lon" in c]

        if lat_candidates and lon_candidates:
            ds = ds.rename({lat_candidates[0]: "lat", lon_candidates[0]: "lon"})
            return ds.interp(lat=self.lat, lon=self.lon, method="linear", kwargs={"fill_value": "extrapolate"})
        return ds

    def load_all(self):
        batch_dim = 1 if self.loader_type == 'sequence' else 0
        X, y, lat = [], [], []

        for X_batch, y_batch, lat_batch in self.loader:
            X.append(X_batch)
            y.append(y_batch)
            lat.append(lat_batch)

        return torch.cat(X, dim=batch_dim), torch.cat(y, dim=batch_dim), torch.cat(lat, dim=batch_dim)

def evaluate(ds, model):
    variant = ds.variant

    losses = []
    y_true, y_pred, lat = [], [], []

    for X_batch, y_batch, lat_batch in ds.loader:
        y_pred.append(model.predict(X_batch))
        if ds.loader_type == 'sequence' or ds.loader_type == 'sequence_map':
            y_true.append(y_batch[-1, :])
            lat.append(lat_batch[-1, :])
        elif ds.loader_type == 'point' or ds.loader_type == 'map':
            y_true.append(y_batch)
            lat.append(lat_batch)
        if len(y_pred[-1].shape) < len(y_true[-1].shape):
            y_pred[-1] = y_pred[-1].unsqueeze(0)

    y_true, y_pred, lat = torch.cat(y_true), torch.cat(y_pred), torch.cat(lat)
    losses.append(loss(variant, y_true, y_pred, lat).item())

    return pd.DataFrame({
        'loss': losses,
        'lead_time': [-1]
    }).round(4)

def loss(variant, y_true, y_pred, lat):
    if variant == 'swe':
        return loss_acc(y_true, y_pred, lat)
    elif variant == 'aice':
        return loss_acc(y_true, y_pred, lat) #loss_acc(y_true, y_pred, lat)

def loss_rmse(y_true, y_pred, lat):
    weights = torch.cos(torch.deg2rad(lat))
    return torch.sqrt(torch.mean(((y_pred - y_true) ** 2) * weights))

def loss_acc(y_true, y_pred, lat):
    weights = torch.cos(torch.deg2rad(lat))
    cov = torch.sum(weights * y_true * y_pred)
    std_true = torch.sqrt(torch.sum(weights * y_true ** 2))
    std_pred = torch.sqrt(torch.sum(weights * y_pred ** 2))
    return cov / (std_true * std_pred + 1e-6)

def evaluate_print(model, dsTrain, dsTest):
    print('%40s, train %8.4f, test %8.4f'
          % (model.name, evaluate(dsTrain, model)['loss'].iloc[0], evaluate(dsTest, model)['loss'].iloc[0]))
