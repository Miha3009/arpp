import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path
import torch
import geopandas as gpd

current_dir = Path(__file__).parent
experiments_dir = current_dir / '../data/experiments'

def draw_ice_map(filename, data, title, label, vmin, vmax, cmap='RdBu_r'):
    lats = np.arange(40, 90.25, 0.25)
    lons = np.arange(-180, 180, 0.25)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon_grid, lat_grid, data,
                     transform=ccrs.PlateCarree(),
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     shading='gouraud',
                     rasterized=True)
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    gl = ax.gridlines(linewidth=0.5, color='gray')
    plt.title(title)

    cbar = plt.colorbar(mesh, orientation='vertical', pad=0.04, shrink=0.5)
    cbar.set_label(label)
    if filename != None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def draw_swe_map(filename, data, title, label, vmin, vmax, cmap='RdBu_r'):
    lat_range = (46, 80)
    lon_range = (20, 160)
    lats = np.arange(40, 90.25, 0.25)
    lons = np.arange(-180, 180, 0.25)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(
        central_longitude=100,
        central_latitude=60,
        standard_parallels=(50, 70)
    ))
    
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                  ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon_grid, lat_grid, data,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         shading='gouraud',
                         rasterized=True)
    
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1, alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':', zorder=2)
    gl = ax.gridlines(linewidth=0.5, color='gray', alpha=0.5)
    plt.title(title)
    
    cbar = plt.colorbar(mesh, orientation='vertical', pad=0.04, shrink=0.5)
    cbar.set_label(label, fontsize=12)
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def draw_swe_map_etr(filename, data, title, label, vmin, vmax, cmap='RdBu_r'):
    minlon, minlat, maxlon, maxlat = 27, 50, 56, 65
    lons = np.arange(minlon, maxlon+0.1, 0.25)
    lats = np.arange(minlat, maxlat+0.1, 0.25)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon_grid, lat_grid, data,
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         shading='gouraud',
                         rasterized=True)
    
    ax.add_feature(cfeature.BORDERS, linewidth=1, zorder=2)
    gdf = gpd.read_file('../data/raw/time_invariant/ne_10m_admin_1_states_provinces.zip')
    russia_regions = gdf[gdf['admin'] == 'Russia']
    ax.add_geometries(russia_regions.geometry, crs=ccrs.PlateCarree(),
                      edgecolor='k', facecolor='none', linewidth=0.5, linestyle=':', zorder=3)
    plt.title(title)

    cbar = plt.colorbar(mesh, orientation='vertical', pad=0.04, shrink=0.5)
    cbar.set_label(label, fontsize=12)
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def draw_ice_map_rmse(filename, data, title):
    draw_ice_map(filename, data, title, 'RMSE (доля)', 0, 0.5, cmap='RdBu_r')

def draw_ice_map_corr(filename, data, title):
    data[data == 1] = np.nan
    draw_ice_map(filename, data, title, 'Корреляция аномалий', -1, 1, cmap='RdBu_r')

def draw_ice_map_improve(filename, data, title):
    draw_ice_map(filename, data, title, 'RMSE (доля)', -0.3, 0.3, cmap='PiYG')

def draw_swe_map_rmse(filename, data, title):
    draw_swe_map_etr(filename, data, title, 'RMSE (мм)', 0, 50, cmap='YlOrRd')

def draw_swe_map_corr(filename, data, title):
    draw_swe_map_etr(filename, data, title, 'Корреляция аномалий', -1, 1, cmap='RdBu_r')

def draw_swe_map_rmse_improve(filename, data, title):
    draw_swe_map_etr(filename, data, title, 'RMSE (мм)', -15, 15, cmap='PiYG')

def draw_swe_map_corr_improve(filename, data, title):
    draw_swe_map_etr(filename, data, title, 'Корреляция аномалий', -0.5, 0.5, cmap='PiYG')

def gen_models_df(model, ds):
    ds.add_extra('lon')
    y_preds, ys, lats, lons = [], [], [], []
    for X, y, lat in ds.loader:
        if ds.loader_type == 'point':
            y_preds.append(model.predict(X[:, :-1]))
            lons.append(X[:, -1])
            ys.append(y)
            lats.append(lat)
        elif ds.loader_type == 'sequence':
            y_preds.append(model.predict(X[:, :, :-1]))
            lons.append(X[-1, :, -1])
            ys.append(y[-1, :])
            lats.append(lat[-1, :])
    ds.remove_extra('lon')

    df = pd.DataFrame({
        'y': np.concatenate(ys),
        'y_pred': np.concatenate(y_preds),
        'lat': np.concatenate(lats),
        'lon': np.concatenate(lons)
    })
    return df

def calculate_metrics(group):
    y, p = group['y'].values, group['y_pred'].values
    mask = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[mask], p[mask]
    
    if len(y) < 2:
        return pd.Series([np.nan, np.nan], index=['rmse', 'corr'])
    
    rmse = np.sqrt(np.mean((y - p) ** 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        try:
            corr = np.corrcoef(y, p)[0,1]
        except:
            corr = np.nan
    
    return pd.Series([rmse, corr], index=['rmse', 'corr'])

def calculate_spatial_metrics(model, ds, lats, lons):
    y_pred, y_true = [], []
    for X_batch, y_batch, _ in ds.loader:
        y_pred.append(model.predict(X_batch))
        if ds.loader_type == 'sequence_map':
            y_true.append(y_batch[-1, :, :, :])
        else:
            y_true.append(y_batch)
        if len(y_pred[-1].shape) < len(y_true[-1].shape):
            y_pred[-1] = y_pred[-1].unsqueeze(0)

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    rmse_map = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))

    y_pred_centered = y_pred - torch.mean(y_pred, dim=0, keepdim=True)
    y_true_centered = y_true - torch.mean(y_true, dim=0, keepdim=True)
    cov = torch.mean(y_pred_centered * y_true_centered, dim=0)
    std_pred = torch.std(y_pred, dim=0)
    std_true = torch.std(y_true, dim=0)
    corr_map = cov / (std_pred * std_true + 1e-6)

    df = pd.DataFrame({
        'lat': np.repeat(lats, len(lons)),
        'lon': np.tile(lons, len(lats)),
        'rmse': rmse_map.flatten().numpy(),
        'corr': corr_map.flatten().numpy()
    })
    
    return df

def save_experiment(name, model, ds, base=None):
    lats = ds.lat
    lons = ds.lon
    if ds.loader_type == 'map' or ds.loader_type == 'sequence_map':
        df = calculate_spatial_metrics(model, ds, lats, lons)
    else:
        df = gen_models_df(model, ds)
        df = df.groupby(['lat', 'lon']).apply(calculate_metrics, include_groups=False).reset_index()
    rmse_df = df.pivot(index='lat', columns='lon', values='rmse').reindex(index=lats, columns=lons)
    corr_df = df.pivot(index='lat', columns='lon', values='corr').reindex(index=lats, columns=lons)
    if not base is None:
        rmse_base = pd.read_csv(experiments_dir / f'{base}_rmse.csv').to_numpy()[:, 1:]
        corr_base = pd.read_csv(experiments_dir / f'{base}_corr.csv').to_numpy()[:, 1:]
    if ds.variant == 'aice':
        draw_ice_map_rmse(experiments_dir / f'{name}_rmse.png', rmse_df.to_numpy(), name)
        draw_ice_map_corr(experiments_dir / f'{name}_corr.png', corr_df.to_numpy(), name)
        if base != None:
            draw_ice_map_improve(experiments_dir / f'{name}_improve.png', rmse_base - rmse_df.to_numpy(), f'{name}_improve')
    elif ds.variant == 'swe':
        draw_swe_map_rmse(experiments_dir / f'{name}_rmse.png', rmse_df.to_numpy(), name)
        draw_swe_map_corr(experiments_dir / f'{name}_corr.png', corr_df.to_numpy(), name)
        if base != None:
            draw_swe_map_corr_improve(experiments_dir / f'{name}_corr_improve.png', corr_df.to_numpy() - corr_base, f'{name}_improve')
            draw_swe_map_rmse_improve(experiments_dir / f'{name}_rmse_improve.png', rmse_base - rmse_df.to_numpy(), f'{name}_improve')
    ds.save('', experiments_dir / f'{name}_ds.json')
    model.save(str(experiments_dir / f'{name}'))
    rmse_df.round(3).to_csv(experiments_dir / f'{name}_rmse.csv')
    corr_df.round(3).to_csv(experiments_dir / f'{name}_corr.csv')
