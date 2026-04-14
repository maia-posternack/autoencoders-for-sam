#!/usr/bin/env python3
"""
This script trains season-specific SAM MSL convolutional autoencoders and saves all outputs.
Since we need to preprocess the data for each season individually, 
we can't use the output of run_preprocess_msl.py, so we do that here as well. 

Input
sam_preprocessed_data.nc  (from run_preprocess_msl.py; path set via --input_file
                            or defaults to $SCRATCH/sam_preprocessed_data.nc)
Note that this file contains the raw MPSL data as well as the preprocessed data. We will pull from that. 

Outputs  (all written to --save_dir, default $SCRATCH/autoencoder_models/)
autoencoder_<tag>.keras       full autoencoder model
encoder_<tag>.keras           encoder sub-model
encoded_all_<tag>.npy         latent representations for every seasonal time step
data_standardized_<tag>.npy   standardised (weighted) input used for training
lats_<tag>.npy                latitude coordinate array
lons_<tag>.npy                longitude coordinate array
times_<tag>.npy               time coordinate array
lat_weights_<tag>.npy         cosine area weights (sqrt(cos(lat)) / mean)
scaler_<tag>.pkl              StandardScaler fitted on the weighted data (needed to invert to Pa)
climatology_<tag>.nc          seasonal climatology removed during preprocessing
polyfit_coefs_<tag>.nc        linear trend coefficients removed during preprocessing
summary_<tag>.json            training metadata and performance summary

Our default settings are 64x32x16x8x4 compression, 1x coarsen, 50 epochs.

We can run with `python run_seasonal_autoencoder.py --tag <tag> --season <season> --rounds <rounds> --coarsen <coarsen> --epochs <epochs>`
"""

import time
print("starting at ", time.time())
import argparse
import gc
import json
import os
import pickle
import time
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print("everything is imported")


# this takes all our arguments and saves them locally
parser = argparse.ArgumentParser(description="Train seasonal SAM MSL autoencoder")

parser.add_argument("--tag", type=str, default="sam_ae",
                    help="Output label used for all saved filenames (default: sam_ae)")
parser.add_argument("--input_file", type=str, default=None,
                    help="Path to input .nc file. Defaults to $SCRATCH/sam_preprocessed_data.nc")
parser.add_argument("--coarsen", type=int, default=1,
                    help="Spatial subsampling factor (1=full res, 4=quarter res). Default: 1")
parser.add_argument(
    "--lat_bounds",
    type=float,
    nargs=2,
    default=[-90.0, -20.0],
    metavar=("LAT_MIN", "LAT_MAX"),
    help="Latitude bounds to subset before training (default: -90 -20). Use --lat_bounds -90 90 for global.",
)
parser.add_argument("--rounds", type=int, nargs="+", default=[64, 32, 16, 8, 4],
                    help="Conv filter sizes per encoder stage. Default: 64 32 16 8 4")
parser.add_argument("--pool_size", type=int, default=2,
                    help="Pooling factor (applied to both H and W). Default: 2")
parser.add_argument("--conv_size", type=int, default=3,
                    help="Conv kernel size (applied to both H and W). Default: 3")
parser.add_argument("--epochs", type=int, default=50,
                    help="Max training epochs (early stopping may end sooner). Default: 50")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Training batch size. Default: 16")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Adam learning rate. Default: 1e-4")
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience. Default: 10")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of data held out for validation. Default: 0.2")
parser.add_argument("--save_dir", type=str, default=None,
                    help="Output directory. Defaults to $SCRATCH/autoencoder_models")
parser.add_argument("--season", type=str, default=None,
                    choices=["DJF", "MAM", "JJA", "SON"],
                    help="Season to subset before training: DJF, MAM, JJA, or SON. "
                         "If omitted, all time steps are used.")
args = parser.parse_args()

SCRATCH = os.path.expandvars("/glade/derecho/scratch/$USER")
# assumes you have saved the preprocessed SAM MSL data to $SCRATCH/sam_preprocessed_data.nc
INPUT_FILE = args.input_file or os.path.join(SCRATCH, "sam_preprocessed_data.nc")
SAVE_DIR = args.save_dir or os.path.join(SCRATCH, "autoencoder_models")
os.makedirs(SAVE_DIR, exist_ok=True)

#set all variables from the arguments
TAG = args.tag
SEASON = args.season
COARSEN = args.coarsen
LAT_BOUNDS = tuple(args.lat_bounds) if args.lat_bounds is not None else None
ROUNDS = args.rounds
POOL_SIZE = (args.pool_size, args.pool_size)
CONV_SIZE = (args.conv_size, args.conv_size)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
PATIENCE = args.patience
TEST_SIZE = args.test_size

# start the timer
t0 = time.time()

#same preprocessing functions as run_preprocess_msl.py
def remove_climatology(da: xr.DataArray):
    clim = da.groupby("time.month").mean("time")
    anom = (da.groupby("time.month") - clim).reset_coords("month", drop=True)
    return anom, clim

def remove_linear_trend(da: xr.DataArray):
    fit = da.polyfit(dim="time", deg=1)
    detrended = (da - xr.polyval(da.time, fit.polyfit_coefficients)).astype("float32")
    return detrended, fit.polyfit_coefficients


def apply_cosine_weights(da: xr.DataArray):
    weights = np.sqrt(np.cos(np.deg2rad(da.latitude.values))).astype(np.float32)
    weights = weights / weights.mean()
    weighted = da * xr.DataArray(weights, coords=[da.latitude], dims=["latitude"])
    return weighted.astype("float32"), weights

#load raw MSL data from the preprocessed file which compiled them
msl_data = xr.open_dataset(INPUT_FILE, chunks={"time": 12})
da = msl_data["MSL"].astype("float32")

# crop to the latitude bounds if they are provided
if LAT_BOUNDS is not None:
    lat0, lat1 = float(LAT_BOUNDS[0]), float(LAT_BOUNDS[1])
    lat_slice = slice(lat0, lat1) if float(da.latitude[0]) <= float(da.latitude[-1]) else slice(lat1, lat0)
    da = da.sel(latitude=lat_slice)
    print(
        f"  Cropped lat {min(lat0, lat1):.1f} to {max(lat0, lat1):.1f} "
        f"→ lat range {float(da.latitude.min()):.1f} to {float(da.latitude.max()):.1f} "
        f"({da.sizes.get('latitude', len(da.latitude))} lats)"
    )

# coarsen the data if the coarsen factor is greater than 1
if COARSEN > 1:
    da = da.isel(latitude=slice(None, None, COARSEN),
                 longitude=slice(None, None, COARSEN))
    print(f"  Coarsened {COARSEN}x → {da.shape}")

# filter to the selected season
_SEASON_MONTHS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
                  "JJA": [6, 7, 8],  "SON": [9, 10, 11]}
if SEASON is not None:
    months = _SEASON_MONTHS[SEASON]
    time_index = da.indexes["time"]
    season_mask = time_index.month.isin(months)
    da = da.isel(time=season_mask)

# remove climatology and linear trend computed on the seasonal subset only
da, clim = remove_climatology(da)
da, polyfit_coefs = remove_linear_trend(da)
da, weights_lat = apply_cosine_weights(da)

# get the number of time, latitude, and longitude steps
n_time = len(da.time)
n_lat = len(da.latitude)
n_lon = len(da.longitude)

# create a numpy array to store the data
data_all = np.empty((n_time, n_lat, n_lon), dtype=np.float32)

# load the data in chunks to optimize memory usage
chunk_size = 50
for start in range(0, n_time, chunk_size):
    end = min(start + chunk_size, n_time)
    data_all[start:end] = np.asarray(da.isel(time=slice(start, end)).load().data)
    print(f"    loaded time {start}–{end-1}")

times = da.time.values
lats_coarse = da.latitude.values
lons_coarse = da.longitude.values
n_samples, lat_length, lon_length = data_all.shape

# standardize the data
scaler = StandardScaler()
data_flat = scaler.fit_transform(data_all.reshape(-1, 1))
data_standardized = data_flat.reshape(n_samples, lat_length, lon_length, 1).astype(np.float32)
# replace nan with 0
data_standardized = np.nan_to_num(data_standardized, nan=0.0)
# free up memory
del data_flat, data_all
gc.collect()
# train test split
event_indices = np.arange(n_samples)
x_train, x_test, train_indices, test_indices = train_test_split(
    data_standardized, event_indices, test_size=TEST_SIZE, random_state=5
)

# build the autoencoder
def build_autoencoder(lat_length, lon_length, pool_size, conv_size, rounds):
    input_img = keras.Input(shape=(lat_length, lon_length, 1))
    x = input_img
    for n in rounds:
        x = layers.Conv2D(n, conv_size, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(pool_size, padding="same")(x)
    encoded = x
    for n in reversed(rounds):
        x = layers.Conv2D(n, conv_size, activation="relu", padding="same")(x)
        x = layers.UpSampling2D(pool_size)(x)
    x = layers.Conv2D(1, conv_size, activation="tanh", padding="same")(x)
    # Crop decoder overshoot for non-power-of-2 spatial dims
    n_pools = len(rounds)
    h, w = lat_length, lon_length
    for _ in range(n_pools):
        h, w = (h + 1) // 2, (w + 1) // 2
    out_h, out_w = h * (2 ** n_pools), w * (2 ** n_pools)
    crop_h = max(0, out_h - lat_length)
    crop_w = max(0, out_w - lon_length)
    if crop_h or crop_w:
        x = layers.Cropping2D(cropping=((0, crop_h), (0, crop_w)))(x)
    return keras.Model(input_img, x), keras.Model(input_img, encoded)

# train the autoencoder
autoencoder, encoder = build_autoencoder(lat_length, lon_length,
                                         POOL_SIZE, CONV_SIZE, ROUNDS)
autoencoder.summary()
optimizer = Adam(learning_rate=LR, clipvalue=1.0)
early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                           restore_best_weights=True)
autoencoder.compile(optimizer=optimizer, loss="mean_absolute_error")

init_loss = autoencoder.evaluate(x_train, x_train, batch_size=BATCH_SIZE, verbose=0)
history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[early_stop],
    verbose=1,
)

# save the predictions for future use
encoded_all = encoder.predict(data_standardized)

# save the outputs
def sp(fname):
    return os.path.join(SAVE_DIR, f"{fname}_{TAG}")

autoencoder.save(sp("autoencoder") + ".keras")
encoder.save(sp("encoder") + ".keras")
np.save(sp("encoded_all") + ".npy", encoded_all)
np.save(sp("data_standardized") + ".npy", data_standardized)
np.save(sp("lats") + ".npy", lats_coarse)
np.save(sp("lons") + ".npy", lons_coarse)
np.save(sp("times") + ".npy", times)
np.save(sp("lat_weights") + ".npy", weights_lat)
with open(sp("scaler") + ".pkl", "wb") as f:
    pickle.dump(scaler, f)

clim.to_netcdf(os.path.join(SAVE_DIR, f"climatology_{TAG}.nc"))
polyfit_coefs.to_netcdf(os.path.join(SAVE_DIR, f"polyfit_coefs_{TAG}.nc"))

with open(os.path.join(SAVE_DIR, f"summary_{TAG}.json"), "w") as f:
    json.dump({
    "tag": TAG,
    "season": SEASON or "all",
    "preprocessing": "climatology + linear trend removed on seasonal subset; cosine area weighting applied",
    "input_shape": list(data_standardized.shape[1:]),
    "latent_shape": list(encoded_all.shape[1:]),
    "compression": round(int(np.prod(data_standardized.shape[1:])) / int(np.prod(encoded_all.shape[1:])), 3),
    "epochs_run": len(history.history["loss"]),
    "initial_loss": round(init_loss, 4),
    "final_loss": round(history.history["loss"][-1], 4),
    "best_val_loss": round(min(history.history["val_loss"]), 4),
    "coarsen": COARSEN,
    "rounds": ROUNDS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "wall_seconds": round(time.time() - t0, 1),
}, f, indent=2)

print(f"\n Saving to {SAVE_DIR}/  (tag={TAG})")
print(f"\n Done in {(time.time()-t0)/60:.1f} min")
print(f" Best val loss: {min(history.history['val_loss']):.4f}")
