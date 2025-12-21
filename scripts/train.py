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
                 variables=None, lead_times=[0], batch_size=201*1440, normed=False, use_cache=False):
        self.data_path = data_path
        self.train_path = f'{data_path}/train/{variant}'
        self.lead_times = np.array(lead_times)
        self.years = years
        self.variant = variant
        self.batch_size = batch_size
        self.normed = normed
        self.use_cache = use_cache
        self.cache = {}
        if periods is None:
            self.periods = list(range(1, 53)) if self.variant == 'swe' else list(range(1, 13))
        else:
            self.periods = periods
        self.period_count = 52 if self.variant == 'swe' else 12

        self.files = [f"{self.train_path}/anom/{y}{p:02d}.nc" for y in years for p in self.periods]
        self.files = [file for file in self.files if os.path.exists(file)]
        self.files_periods = {}
        for file in self.files:
            self.files_periods[file] = int(file[-5:-3])

        self.time_invariant = xr.open_dataset(f"{self.data_path}/train/time_invariant{'_norm' if self.normed else ''}.nc", engine="h5netcdf")
        sample_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        self.lat, self.lon = sample_ds.lat.values, sample_ds.lon.values
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)
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
            'lat': self.lat_grid.ravel(),
            'lon': self.lon_grid.ravel(),
            'cos_lat': np.cos(np.deg2rad(self.lat_grid.ravel())),
            'sin_lon': np.sin(np.deg2rad(self.lon_grid.ravel())),
            'cos_lon': np.cos(np.deg2rad(self.lon_grid.ravel()))
        }

        if self.normed:
            std_ds = xr.open_dataset(f"{self.train_path}/std.nc", engine="h5netcdf")
            self.std = {}
            for variable in std_ds.data_vars:
                self.std[variable] = np.nanmean(std_ds[variable].values)
            std_ds.close()

        self.loader = DataLoader(self, batch_size=None, num_workers=os.cpu_count())

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files = self.files
        else:
            files = self.files[worker_info.id::worker_info.num_workers]

        for file in files:
            ds = xr.open_dataset(file, engine="h5netcdf")
            period = self.files_periods[file]
            mask = self.masks[period]

            lead_times = ds.lead_time.values
            lead_times = np.intersect1d(self.lead_times, lead_times)

            for lead_time in lead_times:
                field_id = f'{file}_{lead_time}'
                if self.use_cache and field_id in self.cache:
                    yield self.cache[field_id]
                    continue

                features = []
                for variable in self.anom_variables:
                    values = self.unify(ds, variable, lead_time).values.ravel()
                    if self.normed:
                        values /= self.std[variable]
                    features.append(values)
                for variable in self.time_invariant_variables:
                    features.append(self.time_invariant[variable].values.ravel())
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
                        features.append(self.climate[period].ravel())

                X = np.stack(features, axis=-1)[mask]
                X = torch.as_tensor(X, dtype=torch.float32)

                y = ds["era5"].values.ravel()[mask]
                y = torch.as_tensor(y, dtype=torch.float32)

                if self.use_cache:
                    self.cache[field_id] = (X, y)

                for i in range(0, len(X), self.batch_size):
                    yield X[i:i+self.batch_size], y[i:i+self.batch_size]
            ds.close()

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

def evaluate(ds, model, plot=False, figure_filepath=None):
    variant = ds.variant
    lead_times = ds.lead_times
    m = len(ds.variables)

    losses = []
    for lead_time in lead_times:
        ds.lead_times = [lead_time]
        y_true, y_pred, lat = [], [], []

        ds.add_extra('lat')
        if variant == 'aice':
            climate = []
            ds.add_extra('climate')
        if plot:
            lon = []
            ds.add_extra('lon')

        for X_batch, y_batch in ds.loader:
            y_true.append(y_batch)
            y_pred.append(model.predict(X_batch[:, :m]))
            lat.append(X_batch[:, ds.variables.index('lat')])
            if variant == 'aice':
                climate.append(X_batch[:, ds.variables.index('climate')])
            if plot:
                lon.append(X_batch[:, ds.variables.index('lon')])

        y_true, y_pred, lat = torch.cat(y_true), torch.cat(y_pred), torch.cat(lat)
        if variant == 'aice':
            climate = torch.cat(climate)
            y_true, y_pred = y_true + climate, y_pred + climate
            y_pred = torch.clamp(y_pred, 0, 1)

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

    ds.set_variables(ds.variables[:m])

    return pd.DataFrame({
        'loss': losses,
        'lead_time': lead_times
    }).round(4)

def loss(variant, y_true, y_pred, lat):
    if variant == 'swe':
        return loss_rmse(y_true, y_pred, lat)
    # Это не вероятности, так что тут тоже rmse
    elif variant == 'aice':
        return loss_rmse(y_true, y_pred, lat)

def loss_rmse(y_true, y_pred, lat):
    weights = torch.cos(torch.deg2rad(lat))
    return torch.sqrt(torch.mean(((y_pred - y_true) ** 2) * weights))

def evaluate_print(model, dsTrain, dsTest):
    print('%40s, train %8.4f, test %8.4f'
          % (model.name, evaluate(dsTrain, model)['loss'].iloc[0], evaluate(dsTest, model)['loss'].iloc[0]))
