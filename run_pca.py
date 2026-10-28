#!/usr/bin/env python3
"""
Authors: Maia Posternack (maiaposternack@gmail.com), Kirstin Koepnick (kirstinkoepnick@g.harvard.edu)

In this file we find the SAM using EOF analysis of Southern Hemisphere (90S-20S) MSL anomalies.
We will use this as a baseline for future analysis.

Our input data is cosine-weighted so we undo this on the EOFs at the end.

Input
$SCRATCH/sam_preprocessed_data.nc  (from run_preprocess_msl.py)

Output
$SCRATCH/sam_pca_data_all.nc  every mode (516); the PC-clustering baseline, Figs S13 and S14
$SCRATCH/sam_pca_data.nc      the leading three modes; EOF1/PC1 for everything else
    eofs                EOF patterns, unit variance, physical (unweighted) space  (mode, latitude, longitude)
    pcs                 principal components, carrying the amplitude  (time, mode)
    variance_explained  fraction of variance per mode  (mode,)
"""
import os
import time
import numpy as np
import xarray as xr

DATA_DIR = os.path.expandvars("/glade/derecho/scratch/$USER")
INPUT_FILE = f"{DATA_DIR}/sam_preprocessed_data.nc"
N_LEADING = 3
LAT_BOUNDS = (-90, -20)

t0 = time.time()
data = xr.open_dataset(INPUT_FILE, chunks={"time": 12})["removed_trend_and_climatology"]
data = data.sel(latitude=slice(*LAT_BOUNDS)).compute()
nt, nlat, nlon = data.shape

X = data.values.reshape(nt, -1)
X = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
variance_explained = S ** 2 / np.sum(X ** 2)
pcs = U * S

# remove the cosine weights from the EOFs, then give them unit variance and let the PCs carry the magnitude
cos = np.cos(np.deg2rad(data.latitude.values))
eofs = Vt.reshape(-1, nlat, nlon) / np.sqrt(cos / cos.mean())[None, :, None]
eof_std = np.nanstd(eofs, axis=(1, 2))
eofs /= eof_std[:, None, None]
pcs *= eof_std

modes = np.arange(1, len(S) + 1)
ds = xr.Dataset(
    {
        "eofs": (("mode", "latitude", "longitude"), eofs, {"long_name": "Empirical Orthogonal Functions"}),
        "pcs": (("time", "mode"), pcs, {"long_name": "Principal Components"}),
        "variance_explained": ("mode", variance_explained, {"long_name": "Fraction of variance explained"}),
    },
    coords={"mode": modes, "time": data.time, "latitude": data.latitude, "longitude": data.longitude},
    attrs={"title": "EOF analysis of Southern Hemisphere mean sea level pressure",
           "source_file": INPUT_FILE, "latitude_bounds": str(LAT_BOUNDS)},
)
ds.to_netcdf(f"{DATA_DIR}/sam_pca_data_all.nc", mode="w")
ds.isel(mode=slice(N_LEADING)).to_netcdf(f"{DATA_DIR}/sam_pca_data.nc", mode="w")
print(f"Saved {len(S)} modes in {(time.time() - t0) / 60:.1f} min")
