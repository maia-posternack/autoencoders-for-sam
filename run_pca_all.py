#!/usr/bin/env python3
"""
Compute all available EOF/PC modes for clustering the EOF baseline.

This leaves run_pca.py unchanged and writes a separate output file:
    $SCRATCH/sam_pca_data_all.nc
"""

import os
import time

import numpy as np
import xarray as xr


START_TIME = time.time()
DATA_DIR = os.path.expandvars("/glade/derecho/scratch/$USER")
INPUT_FILE = os.path.join(DATA_DIR, "sam_preprocessed_data.nc")
OUTPUT_FILE = os.path.join(DATA_DIR, "sam_pca_data_all.nc")

LAT_BOUNDS = (-90, -20)

# Load the preprocessed SAM MSL anomalies (already cosine-weighted by run_preprocess_msl.py).
data = xr.open_dataset(INPUT_FILE, chunks={"time": 12})["removed_trend_and_climatology"]
data_sh = data.sel(latitude=slice(LAT_BOUNDS[0], LAT_BOUNDS[1])).compute()

data_2d = data_sh.values.reshape(len(data_sh.time), len(data_sh.latitude) * len(data_sh.longitude))
data_centered = data_2d - data_2d.mean(axis=0)

# full_matrices=False gives all available modes: min(n_time, n_space).
U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
N_MODES = len(S)

variance_explained = (S ** 2) / np.sum(data_centered ** 2)
eofs_reshaped = Vt.reshape(N_MODES, len(data_sh.latitude), len(data_sh.longitude))
pcs = U * S[np.newaxis, :]

# Remove cosine weights from the EOFs.
cos_weights = np.cos(np.deg2rad(data_sh.latitude.values))
cos_weights = cos_weights / cos_weights.mean()
eofs_reshaped = eofs_reshaped / np.sqrt(cos_weights)[np.newaxis, :, np.newaxis]

# Normalise so EOFs have unit variance; PCs absorb the magnitude.
for i in range(N_MODES):
    eof_std = np.nanstd(eofs_reshaped[i])
    eofs_reshaped[i] = eofs_reshaped[i] / eof_std
    pcs[:, i] = pcs[:, i] * eof_std

ds_out = xr.Dataset(
    {
        "eofs": xr.DataArray(
            eofs_reshaped,
            dims=["mode", "latitude", "longitude"],
            coords={
                "mode": np.arange(1, N_MODES + 1),
                "latitude": data_sh.latitude,
                "longitude": data_sh.longitude,
            },
            attrs={
                "long_name": "Empirical Orthogonal Functions",
                "description": "All available EOF patterns of MSL variability in physical (unweighted) space",
                "units": "normalized",
            },
        ),
        "pcs": xr.DataArray(
            pcs,
            dims=["time", "mode"],
            coords={
                "time": data_sh.time,
                "mode": np.arange(1, N_MODES + 1),
            },
            attrs={
                "long_name": "Principal Components",
                "description": "Time series of EOF amplitudes",
                "units": "standardized",
            },
        ),
        "variance_explained": xr.DataArray(
            variance_explained,
            dims=["mode"],
            coords={"mode": np.arange(1, N_MODES + 1)},
            attrs={
                "long_name": "Variance Explained",
                "description": "Fraction of total variance explained by each mode",
                "units": "fraction",
            },
        ),
    },
    attrs={
        "title": "All-mode EOF Analysis of Southern Hemisphere Mean Sea Level Pressure",
        "source_file": INPUT_FILE,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude_bounds": str(LAT_BOUNDS),
        "n_modes": N_MODES,
        "svd_method": "full_svd",
        "computation_time_seconds": time.time() - START_TIME,
    },
)

print(f"Saving {N_MODES} modes to {OUTPUT_FILE}")
ds_out.to_netcdf(OUTPUT_FILE, mode="w")
print(f"Done in {(time.time() - START_TIME) / 60:.1f} min")
