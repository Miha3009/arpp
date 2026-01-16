import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import xarray as xr
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

current_dir = Path(__file__).parent
default_extra_variables = ['cos_lat', 'sin_lon', 'cos_lon', 'sin_period', 'cos_period']
all_extra_variables = default_extra_variables + ['lat', 'lon', 'period', 'climate']

def make_train_test(variant, firstYear, separateYear, lastYear, args={}):
    dsTrain = ClimateDataset(variant, years=range(firstYear, separateYear), **args)
    dsTest = ClimateDataset(variant, years=range(separateYear, lastYear+1), **args)
    return dsTrain, dsTest

class ClimateDataset(IterableDataset):
    def __init__(self, variant, periods=None, data_path=current_dir / '../data', years=list(range(1991, 2020)),
                 variables=None, lead_times=[0], batch_size=1000, normed=False, cache_level=0,
                 loader_type="point", loader_opts={}):
        self.data_path = data_path
        self.train_path = f'{data_path}/train/{variant}'
        self.lead_times = np.array(lead_times)
        self.years = years
        self.variant = variant
        self.batch_size = batch_size
        self.normed = normed
        self.cache = {}
        self.cache_level = cache_level
        self.cache_full = False
        if periods is None:
            self.periods = list(range(1, 53)) if self.variant == 'swe' else list(range(1, 13))
        else:
            self.periods = periods
        self.period_count = 52 if self.variant == 'swe' else 12
        self.loader_type = loader_type
        self.loader_opts = loader_opts
        self.clamp_value = 50 if self.variant == 'swe' else 1

        self.files = [f"{self.train_path}/anom/{y}{p:02d}.nc" for y in years for p in self.periods]
        self.files = [file for file in self.files if os.path.exists(file)]
        self.files_periods = {}
        for file in self.files:
            self.files_periods[file] = int(file[-5:-3])

        self.time_invariant = xr.open_dataset(f"{self.data_path}/train/time_invariant{'_norm' if self.normed else ''}.nc", engine="h5netcdf")
        sample_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        self.lat, self.lon = sample_ds.lat.values, sample_ds.lon.values
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
        for p in self.periods:
            climate_ds = xr.open_dataset(f"{self.train_path}/clim/{p:02d}.nc", engine="h5netcdf")
            self.climate[p] = climate_ds[list(climate_ds.data_vars)[0]].values.ravel()
            self.masks[p] = torch.as_tensor((~np.isnan(self.climate[p])) & (self.climate[p] != 0))
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

        for file in self.files:
            ds = xr.open_dataset(file, engine="h5netcdf")
            self.cache[f"{file}_lead_times"] = ds.lead_time.values
            ds.close()

        self.loader = DataLoader(self, batch_size=None) #, num_workers=1 if self.loader_type == "sequence" else 4)

    def __iter__(self):
        worker_info = get_worker_info()
        if self.loader_type == 'point':
            yield from self.rebatch_loader(self.point_loader(worker_info))
        elif self.loader_type == 'map':
            yield from self.rebatch_loader(self.map_loader(worker_info))
        elif self.loader_type == 'sequence':
            yield from self.rebatch_loader(self.sequence_loader(self.loader_opts['len']), 1)
        else:
            raise Exception(f"Unknown loader type {self.loader_type}")

    def point_loader(self, worker_info):
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id::worker_info.num_workers]

        for file in files:
            period = self.files_periods[file]

            lead_times = np.intersect1d(self.lead_times, self.cache[f'{file}_lead_times'])
            for lead_time in lead_times:
                X, y, lat = self.load_file(file, lead_time)
                yield X, y, lat

    def map_loader(self, worker_info):
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id::worker_info.num_workers]

        for file in files:
            period = self.files_periods[file]
            mask = self.masks[period]

            lead_times = np.intersect1d(self.lead_times, self.cache[f'{file}_lead_times'])

            for lead_time in lead_times:
                X, y, lat = self.load_file(file, lead_time)
                X, y = torch.nan_to_num(X, nan=0.0), torch.nan_to_num(y, nan=0.0)
                yield X.unsqueeze(0), y.unsqueeze(0), lat.unsqueeze(0)

    def sequence_loader(self, seq_len):
        for i in range(len(self.files) - seq_len + 1):
            has_lead_time = np.array([1 in self.cache[f'{self.files[i + j]}_lead_times'] for j in range(seq_len)])
            if not np.all(has_lead_time):
                continue

            mask = None
            for j in range(seq_len):
                period = self.files_periods[self.files[i + j]]
                if mask is None:
                    mask = self.masks[period]
                else:
                    mask = mask & self.masks[period]

            if mask.sum() == 0:
                continue
            
            X, y, lat = [], [], []
            for j in range(seq_len):
                X_frame, y_frame, lat_frame = self.load_file(self.files[i + j], 1)
                X.append(X_frame.reshape(-1, X_frame.shape[-1])[mask])
                y.append(y_frame.ravel()[mask])
                lat.append(lat_frame.ravel()[mask])
            X, y, lat = torch.stack(X), torch.stack(y), torch.stack(lat)
            yield X, y, lat

    def rebatch_loader(self, loader, batch_dim=0):
        if self.cache_full:
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
                n_full = buf_size // self.batch_size

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
            self.cache_full = True

    def load_file(self, file, lead_time):
        field_id = f'{file}_{lead_time}'
        if self.cache_level == 1 and field_id in self.cache:
            return self.cache[field_id]

        period = self.files_periods[file]

        ds = xr.open_dataset(file, engine="h5netcdf")

        features = []
        for variable in self.anom_variables:
            values = self.unify(ds, variable, lead_time).values
            if self.normed:
                values /= self.std[variable]
            features.append(values)
        for variable in self.time_invariant_variables:
            features.append(self.time_invariant[variable].values)
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

        X = np.stack(features, axis=-1)
        X = torch.as_tensor(X, dtype=torch.float32)

        y = ds["era5"].values
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
            self.variables.append(variable)
            self.extra_variables.append(variable)
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

    def unify(self, ds, variable, lead_time):
        ds = ds[variable]
        if "lead_time" in ds.coords:
            ds = ds.sel(lead_time=lead_time)
        if variable in ['sst', 'hice']:
            ds = ds.fillna(0)

        if "lat" in ds.coords and "lon" in ds.coords:
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

def evaluate(ds, model, plot=False, figure_filepath=None):
    variant = ds.variant
    lead_times = ds.lead_times

    losses = []
    for lead_time in lead_times:
        ds.lead_times = [lead_time]
        y_true, y_pred, lat = [], [], []

        if plot:
            lon = []
            ds.add_extra('lon')

        for X_batch, y_batch, lat_batch in ds.loader:
            y_true.append(y_batch)
            y_pred.append(model.predict(X_batch))
            lat.append(lat_batch)
            if plot:
                lon.append(X_batch[:, ds.variables.index('lon')])

        y_true, y_pred, lat = torch.cat(y_true), torch.cat(y_pred), torch.cat(lat)

        losses.append(loss(variant, y_true, y_pred, lat).item())

        if plot:
            lon = torch.cat(lon)
            df = pd.DataFrame({
                'lat': lat.numpy(),
                'lon': lon.numpy(),
                'y_true': y_true.numpy(),
                'y_pred': y_pred.numpy()
            })

            index = pd.MultiIndex.from_product([ds.lat, ds.lon], names=['lat', 'lon'])
            loss_df = df.groupby(['lat', 'lon']).apply(
                lambda g: loss(
                    variant,
                    torch.tensor(g['y_true'].values),
                    torch.tensor(g['y_pred'].values),
                    torch.ones(g['y_pred'].values.shape)*90
                ).item(), include_groups=False
            ).reset_index().set_index(['lat', 'lon']).reindex(index).reset_index()

            loss_grid = loss_df.pivot(index='lat', columns='lon', values=0).values
            vmax = 1 if variant == 'aice' else 200
            plt.figure(figsize=(12, 4))
            im = plt.imshow(loss_grid, cmap='jet',
                vmin=0, vmax=vmax,
                extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()],
                origin='lower')
            cbar = plt.colorbar(im, orientation='horizontal', aspect=100, pad=0.1)
            cbar.set_label('Ошибка' if variant == 'aice' else 'Ошибка (мм)', fontsize=10)
            variant_name = 'Водный эквивалент снега' if variant == 'swe' else 'Доля морского льда'
            plt.title(f'{variant_name}, {model.name}, заблг. {lead_time + 1} месяц{"a" if lead_time > 0 else ""}')

            if not figure_filepath is None:
                plt.savefig(figure_filepath, dpi=200, bbox_inches='tight')
            plt.show()

    return pd.DataFrame({
        'loss': losses,
        'lead_time': lead_times
    }).round(4)

def loss(variant, y_true, y_pred, lat):
    if variant == 'swe':
        return loss_acc(y_true, y_pred, lat)
    elif variant == 'aice':
        return loss_acc(y_true, y_pred, lat)

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
