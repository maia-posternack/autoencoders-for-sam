#!/usr/bin/env python3
"""
We preprocess the ERA5 monthly mean sea-level pressure (MSL) for SAM analysis in 3 steps:

1. Remove the monthly climatology
2. Remove a per-pixel linear trend

We will apply cosine weighting later in the analysis.

Input
ERA5 monthly MSL files at 0.25° resolution, one file per year:
    <ERA5_MODA_DIR>/<year>/e5.moda.an.sfc.128_151_msl.ll025sc.<year>010100_<year>120100.nc

Output
sam_preprocessed_data.nc  (written to OUT_DIR)
    Variables:
        removed_trend_and_climatology  – detrended anomaly (used for training)
        removed_climatology            – anomaly before detrending
        MSL                            – raw MSL field
        polyfit_coefficients           – linear trend coefficients per pixel
"""
# imports
import os
from pathlib import Path
import numpy as np
import xarray as xr
import dask
import time

START_TIME = time.time()

# set the start and end years of the directory
START_YEAR = 1980
END_YEAR = 2022
ERA5_MODA_DIR = "/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc"
OUT_DIR = os.path.expandvars("/glade/derecho/scratch/$USER")
OUT_NC = os.path.join(OUT_DIR, "sam_preprocessed_data.nc")

def build_file_list(start_year: int, end_year: int) -> list[str]:
    files = []
    for year in range(start_year, end_year + 1):
        fp = (
            f"{ERA5_MODA_DIR}/{year}/"
            f"e5.moda.an.sfc.128_151_msl.ll025sc."
            f"{year}010100_{year}120100.nc"
        )
        files.append(fp)
    return files

def remove_climatology(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    clim = da.groupby("time.month").mean("time")
    anom = (da.groupby("time.month") - clim).reset_coords("month", drop=True)
    return anom, clim

def remove_linear_trend(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    fit = da.polyfit(dim="time", deg=1)
    detrended = (da - xr.polyval(da.time, fit.polyfit_coefficients)).astype("float32")
    return detrended, fit.polyfit_coefficients

def apply_cosine_weighting(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da.latitude))
    weights = weights / weights.mean() # normalize the weights
    da = da * weights
    return da.astype("float32") 

os.makedirs(OUT_DIR, exist_ok=True)
file_list = build_file_list(START_YEAR, END_YEAR)
ds = xr.open_mfdataset(
    file_list,
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="minimal",
    compat="override",
    engine="h5netcdf",
    chunks={"time": 12, "latitude": 180, "longitude": 360},
).sortby(["time", "latitude", "longitude"])

# remove the monthly climatology and the linear trend
msl = ds["MSL"].astype("float32")
anom, clim = remove_climatology(msl)

# remove the linear trend
detrended, polyfit_coefficients = remove_linear_trend(anom)

# apply the cosine weighting
preprocessed = apply_cosine_weighting(detrended)

# save the preprocessed data to a NetCDF file
out = xr.Dataset(
    {
        "removed_trend_and_climatology": preprocessed,
        "MSL": msl,
        "polyfit_coefficients": polyfit_coefficients,
    },
    coords={
        "time": ds["time"],
        "latitude": ds["latitude"],
        "longitude": ds["longitude"],
    },
    attrs=ds.attrs,
).chunk({"time": 1,  "latitude": 180, "longitude": 360})

encoding = {
    "removed_trend_and_climatology": {"dtype": "float32", "zlib": True, "complevel": 1},
    "MSL": {"dtype": "float32", "zlib": True, "complevel": 1},
    "polyfit_coefficients": {"dtype": "float32", "zlib": True, "complevel": 1},
}

with dask.config.set(scheduler="single-threaded"):
    out.to_netcdf(OUT_NC, engine="h5netcdf", encoding=encoding)

print("Saving to", OUT_NC)
print(f"Done in {(time.time() - START_TIME) / 60:.1f} min")
