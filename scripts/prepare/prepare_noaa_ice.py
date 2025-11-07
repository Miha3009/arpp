import numpy as np
import pandas as pd
import os
import scipy.spatial
from concurrent.futures import ProcessPoolExecutor
import pyproj
from scipy.ndimage import distance_transform_edt
import multiprocessing

noaa_ice_input_path = "../../data/raw/noaa_cdr_ice/"
noaa_ice_output_path = "../../data/prepare/noaa_ice/"
template_path = "../../data/raw/noaa_cdr_ice/sic_psn25_19910101_F08_v06r00.nc"
weights_path = "../../data/tmp/noaa_ice_weights.npz"
landsea_mask_path = "../../data/raw/time_invariant/era5_land_sea_mask.nc"
target_lats = np.arange(0, 90.25, 0.25).astype("float32")
target_lons = np.arange(-180, 180, 0.25).astype("float32")

q = multiprocessing.Queue(maxsize=100)

def open_landsea_mask():
    import xarray as xr
    ds_land_sea_mask = xr.open_dataset(landsea_mask_path, engine="h5netcdf")
    landsea_mask = ds_land_sea_mask.lsm.values[0, ::-1, :]
    landsea_mask = np.where(landsea_mask < 0.5, 1, np.nan)
    ds_land_sea_mask.close()
    return landsea_mask

def open_template():
    import xarray as xr
    ds_template = xr.open_dataset(template_path, engine="netcdf4")
    x = ds_template.x.values
    y = ds_template.y.values
    ds_template.close()
    return (x, y)

def save(q):
    import xarray as xr
    encoding = {"ice": {"zlib": True, "complevel": 9}}
    while True:
        item  = q.get()
        if item == None:
            return
        file, data = item
        ds = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": target_lats, "lon": target_lons},
            name="ice",
        )
        ds.to_netcdf(file, engine='h5netcdf', encoding=encoding)

def create_regrid_weights(x, y):
    print("Generate weights...")
    transformer = pyproj.Transformer.from_crs("EPSG:3411", "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(x, y)
    lons, lats = transformer.transform(x, y)

    points = np.column_stack([lats.ravel(), lons.ravel()])
    target_lats_grid, target_lons_grid = np.meshgrid(target_lats, target_lons)
    xi = np.column_stack([target_lats_grid.ravel(), target_lons_grid.ravel()])

    tri = scipy.spatial.Delaunay(points)
    simplex = tri.find_simplex(xi)
    mask = simplex >= 0

    weights = np.zeros((xi.shape[0], 3), dtype='float32') + np.nan
    vertices = np.zeros((xi.shape[0], 3), dtype='int')

    T = tri.transform[simplex[mask], :2, :]
    r = tri.transform[simplex[mask], 2, :]
    xi_r = xi[mask] - r
    bary = np.einsum('ijk,ik->ij', T, xi_r)
    w3 = 1.0 - bary.sum(axis=1)
    weights[mask] = np.column_stack([bary, w3])
    vertices[mask] = tri.simplices[simplex[mask]]

    return weights, vertices

def regrid(ds, landsea_mask, weights, vertices):
    values = ds.cdr_seaice_conc.isel(time=0).values.ravel()
    newValues = np.sum(values[vertices] * weights, axis=1).reshape((1440, 361)).T
    nan_mask = np.isnan(newValues)
    distances, indices = distance_transform_edt(nan_mask, return_indices=True)
    nan_mask = nan_mask & (distances <= 15)
    newValues[nan_mask] = newValues[tuple(indices[:, nan_mask])]
    newValues = newValues * landsea_mask
    return newValues.astype("float32")

def get_version(x):
    dates = [
        (pd.Timestamp(1991, 1, 1), "F08"),
        (pd.Timestamp(1991, 12, 3), "F11"),
        (pd.Timestamp(1995, 10, 1), "F13"),
        (pd.Timestamp(2008, 1, 1), "F17"),
        (pd.Timestamp(2025, 1, 1), "am2"),
    ]
    result = ""
    for date, version in dates:
        if x >= date:
            result = version
        else:
            break
    return result

def process_file(date):
    import xarray as xr

    date_str = date.strftime('%Y%m%d')
    version = get_version(date)
    filename = f"sic_psn25_{date_str}_{version}_v06r00.nc"
    input_filepath = os.path.join(noaa_ice_input_path, filename)
    output_filepath = os.path.join(noaa_ice_output_path, f"{date_str}.nc")

    if os.path.exists(output_filepath):
        return

    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} not found")
        return

    print(f"Process {input_filepath}")
    try:
        ds = xr.open_dataset(input_filepath, engine="netcdf4")
        regridded = regrid(ds, landsea_mask, weights, vertices)
        ds.close()
        q.put((output_filepath, regridded))
    except Exception as e:
        print(f'Error load dataset {input_filepath}: {e}')

if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        landsea_mask = executor.submit(open_landsea_mask).result()
    with ProcessPoolExecutor() as executor:
        x, y = executor.submit(open_template).result()

    if not os.path.exists(weights_path):
        weights, vertices = create_regrid_weights(x, y)
        np.savez(weights_path, vertices=vertices, weights=weights)
    else:
        data = np.load(weights_path)
        vertices = data["vertices"]
        weights = data["weights"]

    os.makedirs(noaa_ice_output_path, exist_ok=True)

    num_save_workers = 2
    save_processes = []
    for i in range(num_save_workers):
        p = multiprocessing.Process(target=save, args=(q,))
        p.start()
        save_processes.append(p)

    dates = pd.date_range('1991-01-01', '2025-12-31', freq='D')
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, dates)

    for i in range(num_save_workers):
        q.put(None)
    for p in save_processes:
        p.join()
