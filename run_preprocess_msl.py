#!/usr/bin/env python3
"""
Authors: Maia Posternack (maiaposternack@gmail.com), Kirstin Koepnick (kirstinkoepnick@g.harvard.edu)

We preprocess the ERA5 monthly mean sea-level pressure (MSL) for SAM analysis in 3 steps:

1. Remove the monthly climatology
2. Remove a per-pixel linear trend
3. Apply cosine weighting

Input
ERA5 monthly MSL files at 0.25° resolution, one file per year:
    <ERA5_MODA_DIR>/<year>/e5.moda.an.sfc.128_151_msl.ll025sc.<year>010100_<year>120100.nc

Output
$SCRATCH/sam_preprocessed_data.nc
    removed_trend_and_climatology   detrended, weighted anomaly (used for training)
    MSL                             raw MSL field (the seasonal runs start from this)
    polyfit_coefficients            linear trend coefficients per pixel
"""
import os
import time
import numpy as np
import xarray as xr
import dask

START_YEAR, END_YEAR = 1980, 2022
ERA5_MODA_DIR = "/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc"
OUT_NC = os.path.expandvars("/glade/derecho/scratch/$USER/sam_preprocessed_data.nc")


def remove_climatology(da):
    clim = da.groupby("time.month").mean("time")
    return (da.groupby("time.month") - clim).reset_coords("month", drop=True), clim


def remove_linear_trend(da):
    fit = da.polyfit(dim="time", deg=1)
    return (da - xr.polyval(da.time, fit.polyfit_coefficients)).astype("float32"), fit.polyfit_coefficients


def apply_cosine_weighting(da):
    # sqrt so that squared anomalies are area weighted; normalized so mean weight == 1
    weights = np.cos(np.deg2rad(da.latitude))
    return (da * np.sqrt(weights / weights.mean())).astype("float32")


if __name__ == "__main__":
    t0 = time.time()
    # disable HDF5 file locking (required on Lustre/GPFS)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    # a stale partial file causes HDF5 lock conflicts on resubmission
    if os.path.exists(OUT_NC):
        os.remove(OUT_NC)

    files = [f"{ERA5_MODA_DIR}/{y}/e5.moda.an.sfc.128_151_msl.ll025sc.{y}010100_{y}120100.nc"
             for y in range(START_YEAR, END_YEAR + 1)]
    ds = xr.open_mfdataset(files, combine="by_coords", parallel=True, coords="minimal",
                           data_vars="minimal", compat="override", engine="h5netcdf",
                           chunks={"time": 12, "latitude": 180, "longitude": 360},
                           ).sortby(["time", "latitude", "longitude"])

    msl = ds["MSL"].astype("float32")
    anom, _ = remove_climatology(msl)
    detrended, coeffs = remove_linear_trend(anom)

    out = xr.Dataset(
        {"removed_trend_and_climatology": apply_cosine_weighting(detrended),
         "MSL": msl, "polyfit_coefficients": coeffs},
        coords={"time": ds["time"], "latitude": ds["latitude"], "longitude": ds["longitude"]},
        attrs=ds.attrs,
    ).chunk({"time": 1, "latitude": 180, "longitude": 360})
    encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 1} for v in out.data_vars}

    with dask.config.set(scheduler="single-threaded"):
        out.to_netcdf(OUT_NC, engine="h5netcdf", encoding=encoding)
    print(f"Saved {OUT_NC} in {(time.time() - t0) / 60:.1f} min")
