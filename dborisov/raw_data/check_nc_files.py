import numpy as np
import xarray as xr
import glob
import sys

var = sys.argv[1]
files = glob.glob(f'{var}/*.nc')

damaged_files = []
files_with_nans = []
for file in files: 
    try:
        ds = xr.open_dataset(file)
        vals = ds[var].values
        if np.isnan(vals).any(): 
            print(f'{file} contain {np.isnan(vals).sum()} nans ({np.isnan(vals).sum() / vals.size * 100:.1f}%)')
            nan_idxs = np.argwhere(np.isnan(vals))
            print(f'nans from {nan_idxs[0]} to {nan_idxs[-1]}')
            files_with_nans.append(file)
        else: 
            continue
    except (RuntimeError, OSError) as e:
        print(f'Damaged file {file}: {e}')
        damaged_files.append(file)
        continue

if damaged_files: 
    ln=len(damaged_files)
    damaged_files = '\n'.join(damaged_files)
    print(f'{ln} damaged files: {damaged_files}')
else: 
    print('No damaged files')

if files_with_nans: 
    files_with_nans = '\n'.join(files_with_nans)
    print(f'{len(files_with_nans)} files with nans: {files_with_nans}')
else: 
    print('No files with nans')