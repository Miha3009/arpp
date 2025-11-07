import xarray as xr
import numpy as np
import pandas as pd
import os
import scipy.spatial
from concurrent.futures import ProcessPoolExecutor

globsnow_archive_input_path = "../../data/raw/globsnow_archive_v3/"
globsnow_nrt_input_path = "../../data/raw/globsnow_nrt/"
globsnow_output_path = "../../data/prepare/globsnow/"
template_v1_path = "../../data/raw/globsnow_nrt/GlobSnow_SWE_L3A_20160101_v.1.0.nc.gz"
template_v2_path = "../../data/raw/globsnow_nrt/GlobSnow_SWE_L3A_20250101_v.2.0.nc.gz"
weights_path = "../../data/tmp/globsnow_weights.npz"
landsea_mask_path = "../../data/raw/time_invariant/era5_land_sea_mask.nc"
target_lats = np.arange(0, 90.25, 0.25).astype("float32")
target_lons = np.arange(-180, 180, 0.25).astype("float32")

def create_regrid_weights(lats, lons):
    print("Generate weights...")
    points = np.column_stack([lats, lons])
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

def regrid(ds, is_archive, mask, landsea_mask, weights, vertices):
    if is_archive:
        values = ds.swe.values.T[mask].ravel()
    else:
        values = ds.SWE.values[mask].ravel()
    values[values < 0] = np.nan
    newValues = np.sum(values[vertices] * weights, axis=1).reshape((1440, 361)).T
    if is_archive:
        newValues = np.roll(newValues, 360)
    newValues = newValues * landsea_mask
    return xr.DataArray(
            newValues.astype("float32"),
            dims=["lat", "lon"],
            coords={"lat": target_lats, "lon": target_lons},
            name="swe",
    )

def process_file(date):
    date_str = date.strftime('%Y%m%d')
    is_archive = date.year < 2018
    if is_archive:
        filename = f"{date_str}_northern_hemisphere_swe_0.25grid.nc"
        input_filepath = os.path.join(globsnow_archive_input_path, filename)
        engine = "h5netcdf"
    else:
        version = "1.0" #if date < pd.Timestamp(2024, 11, 24) else "2.0"
        filename = f"GlobSnow_SWE_L3A_{date_str}_v.{version}.nc.gz"
        input_filepath = os.path.join(globsnow_nrt_input_path, filename)
        engine = "scipy"

    output_filepath = os.path.join(globsnow_output_path, f"{date_str}.nc")

    if os.path.exists(output_filepath):
        return

    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} not found")
        return

    print(f"Process {input_filepath}")
    try:
        ds = xr.open_dataset(input_filepath, engine=engine)
        regridded = regrid(ds, is_archive, valid_mask_v1, landsea_mask, weights, vertices)
        ds.close()
        encoding = {"swe": {"zlib": True, "complevel": 9}}
        regridded.to_netcdf(output_filepath, engine='h5netcdf', encoding=encoding)
    except Exception as e:
        print(f'Error load dataset {input_filepath}: {e}')

if __name__ == "__main__":
    ds_template_v1 = xr.open_dataset(template_v1_path)
    lats_v1 = ds_template_v1.latitude.to_numpy()
    lons_v1 = ds_template_v1.longitude.to_numpy()
    ds_template_v1.close()
    valid_mask_v1 = (np.isfinite(lats_v1)) & (np.isfinite(lons_v1)) \
        & (lats_v1 != 1e+20) & (lons_v1 != 1e+20)
    lats_v1, lons_v1 = lats_v1[valid_mask_v1], lons_v1[valid_mask_v1]

    ds_land_sea_mask = xr.open_dataset(landsea_mask_path, engine="h5netcdf")
    landsea_mask = ds_land_sea_mask.lsm.values[0, ::-1, :]
    landsea_mask = np.where(landsea_mask > 0.5, 1, np.nan)
    ds_land_sea_mask.close()

    if not os.path.exists(weights_path):
        weights, vertices = create_regrid_weights(lats_v1, lons_v1)
        np.savez(weights_path, vertices=vertices, weights=weights)
    else:
        data = np.load(weights_path)
        vertices = data["vertices"]
        weights = data["weights"]

    os.makedirs(globsnow_output_path, exist_ok=True)

    dates = pd.date_range('1991-01-01', '2025-12-31', freq='D')
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, dates)
