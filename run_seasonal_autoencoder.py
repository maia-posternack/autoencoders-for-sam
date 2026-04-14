#!/usr/bin/env python3
"""
Train season-specific SAM MSL convolutional autoencoders and save all outputs.

Reads the raw ERA5 MSL field from the preprocessed dataset, subsets to a
single meteorological season (DJF, MAM, JJA, or SON), re-computes the
climatology and linear trend on that seasonal subset, then trains a
convolutional autoencoder on the resulting anomalies.

Running this script once per season (four runs total) produces the seasonal
models used by analyze_seasons.ipynb.

Input
sam_preprocessed_data.nc  (from run_preprocess_msl.py; path set via --input_file
                            or defaults to $SCRATCH/sam_preprocessed_data.nc)

Outputs  (all written to --save_dir, default $SCRATCH/autoencoder_models/)
autoencoder_<tag>.keras       full autoencoder model
encoder_<tag>.keras           encoder sub-model
encoded_all_<tag>.npy         latent representations for every seasonal time step
data_standardized_<tag>.npy   standardised input used for training
lats_<tag>.npy                latitude coordinate array
lons_<tag>.npy                longitude coordinate array
times_<tag>.npy               time coordinate array
climatology_<tag>.nc          seasonal climatology removed during preprocessing
polyfit_coefs_<tag>.nc        linear trend coefficients removed during preprocessing
summary_<tag>.json            training metadata and performance summary

Usage examples
# DJF (boreal winter / austral summer)
python run_seasonal_autoencoder.py --tag sam_ae_DJF --season DJF --rounds 64 32 16 8 4 --coarsen 1 --epochs 50

# All four seasons
python run_seasonal_autoencoder.py --tag sam_ae_MAM --season MAM --rounds 64 32 16 8 4 --coarsen 1 --epochs 50
python run_seasonal_autoencoder.py --tag sam_ae_JJA --season JJA --rounds 64 32 16 8 4 --coarsen 1 --epochs 50
python run_seasonal_autoencoder.py --tag sam_ae_SON --season SON --rounds 64 32 16 8 4 --coarsen 1 --epochs 50
"""
# imports
import argparse
import gc
import os
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
import json


# this takes all our arguements and saves them locally
parser = argparse.ArgumentParser(description="Train seasonal SAM MSL autoencoder")

parser.add_argument("--tag",        type=str,   default="sam_ae",
                    help="Output label used for all saved filenames (default: sam_ae)")
parser.add_argument("--input_file", type=str,   default=None,
                    help="Path to input .nc file. Defaults to $SCRATCH/sam_preprocessed_data.nc")
parser.add_argument("--coarsen",    type=int,   default=4,
                    help="Spatial subsampling factor (1=full res, 4=quarter res). Default: 4")
parser.add_argument(
    "--lat_bounds",
    type=float,
    nargs=2,
    default=[-90.0, -20.0],
    metavar=("LAT_MIN", "LAT_MAX"),
    help="Latitude bounds to subset before training (default: -90 -20). Use --lat_bounds -90 90 for global.",
)
parser.add_argument("--rounds",     type=int,   nargs="+", default=[32, 16, 8],
                    help="Conv filter sizes per encoder stage, e.g. --rounds 32 16 8. Default: 32 16 8")
parser.add_argument("--pool_size",  type=int,   default=2,
                    help="Pooling factor (applied to both H and W). Default: 2")
parser.add_argument("--conv_size",  type=int,   default=3,
                    help="Conv kernel size (applied to both H and W). Default: 3")
parser.add_argument("--epochs",     type=int,   default=100,
                    help="Max training epochs (early stopping may end sooner). Default: 100")
parser.add_argument("--batch_size", type=int,   default=16,
                    help="Training batch size. Default: 16")
parser.add_argument("--lr",         type=float, default=1e-4,
                    help="Adam learning rate. Default: 1e-4")
parser.add_argument("--patience",   type=int,   default=10,
                    help="Early stopping patience. Default: 10")
parser.add_argument("--test_size",  type=float, default=0.2,
                    help="Fraction of data held out for validation. Default: 0.2")
parser.add_argument("--save_dir",   type=str,   default=None,
                    help="Output directory. Defaults to $SCRATCH/autoencoder_models")
parser.add_argument("--season",     type=str,   default=None,
                    choices=["DJF", "MAM", "JJA", "SON"],
                    help="Season to subset before training: DJF, MAM, JJA, or SON. "
                         "If omitted, all time steps are used.")

args = parser.parse_args()

SCRATCH    = os.path.expandvars("/glade/derecho/scratch/$USER")
# assumes you have saved the preprocessed SAM MSL anomalies to $SCRATCH/sam_preprocessed_data.nc
INPUT_FILE = args.input_file or os.path.join(SCRATCH, "sam_preprocessed_data.nc")
SAVE_DIR   = args.save_dir   or os.path.join(SCRATCH, "autoencoder_models")
os.makedirs(SAVE_DIR, exist_ok=True)

#set all variables from the arguments
TAG        = args.tag
SEASON     = args.season
COARSEN    = args.coarsen
LAT_BOUNDS = tuple(args.lat_bounds) if args.lat_bounds is not None else None
ROUNDS     = args.rounds
POOL_SIZE  = (args.pool_size, args.pool_size)
CONV_SIZE  = (args.conv_size, args.conv_size)
EPOCHS     = args.epochs
BATCH_SIZE = args.batch_size
LR         = args.lr
PATIENCE   = args.patience
TEST_SIZE  = args.test_size

# start the timer
t0 = time.time()


# ── Preprocessing helpers (applied to the filtered subset) ─────────────────
def remove_climatology(da: xr.DataArray):
    """Subtract the mean seasonal cycle computed from da itself."""
    clim = da.groupby("time.month").mean("time")
    anom = (da.groupby("time.month") - clim).reset_coords("month", drop=True)
    return anom, clim


def remove_linear_trend(da: xr.DataArray):
    """Fit and subtract a per-pixel linear trend along time."""
    fit       = da.polyfit(dim="time", deg=1)
    detrended = (da - xr.polyval(da.time, fit.polyfit_coefficients)).astype("float32")
    return detrended, fit.polyfit_coefficients


print("=" * 55)
print("  SAM MSL Seasonal Autoencoder Training")
print("=" * 55)
print(f"  tag        : {TAG}")
print(f"  season     : {SEASON or 'all'}")
print(f"  input      : {INPUT_FILE}")
print(f"  save_dir   : {SAVE_DIR}")
print(f"  coarsen    : {COARSEN}x")
print(f"  lat_bounds : {LAT_BOUNDS}")
print(f"  rounds     : {ROUNDS}")
print(f"  pool_size  : {POOL_SIZE}")
print(f"  conv_size  : {CONV_SIZE}")
print(f"  epochs     : {EPOCHS}  (patience={PATIENCE})")
print(f"  batch_size : {BATCH_SIZE}")
print(f"  lr         : {LR}")
print("=" * 55)

#load data
print("\n[1/6] Loading raw MSL data ...")
msl_data = xr.open_dataset(INPUT_FILE, chunks={"time": 12})
da = msl_data["MSL"].astype("float32")
print(f"  Original shape: {da.shape}")

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

# ── Season filter ──────────────────────────────────────────────────────────
_SEASON_MONTHS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
                  "JJA": [6, 7, 8],  "SON": [9, 10, 11]}
if SEASON is not None:
    months      = _SEASON_MONTHS[SEASON]
    time_index  = da.indexes["time"]
    season_mask = time_index.month.isin(months)
    da          = da.isel(time=season_mask)
    print(f"  Season filter ({SEASON}) → {da.sizes['time']} time steps ({months})")

# ── 2. Preprocess on the filtered subset ───────────────────────────────────
print("\n[2/6] Removing climatology from subset ...")
da, clim = remove_climatology(da)
print(f"  climatology shape: {clim.shape}")

print("  Removing linear trend from subset ...")
da, polyfit_coefs = remove_linear_trend(da)
print(f"  polyfit_coefs shape: {polyfit_coefs.shape}")

# get the number of time, latitude, and longitude steps
n_time = len(da.time)
n_lat  = len(da.latitude)
n_lon  = len(da.longitude)

# create a numpy array to store the data
data_all   = np.empty((n_time, n_lat, n_lon), dtype=np.float32)

# load the data in chunks to optimize memory usage
chunk_size = 50
for start in range(0, n_time, chunk_size):
    end = min(start + chunk_size, n_time)
    data_all[start:end] = np.asarray(da.isel(time=slice(start, end)).load().data)
    print(f"    loaded time {start}–{end-1}")

times       = da.time.values
lats_coarse = da.latitude.values
lons_coarse = da.longitude.values
print(f"  data_all shape : {data_all.shape}  ({data_all.nbytes/1e6:.1f} MB)")

# standardize the data
print("\n[3/6] Standardising and splitting ...")
n_samples, lat_length, lon_length = data_all.shape

scaler            = StandardScaler()
data_flat         = scaler.fit_transform(data_all.reshape(-1, 1))
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
print(f"  train: {x_train.shape}   test: {x_test.shape}")

# build the autoencoder
print("\n[4/6] Building autoencoder ...")

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


autoencoder, encoder = build_autoencoder(lat_length, lon_length,
                                         POOL_SIZE, CONV_SIZE, ROUNDS)
autoencoder.summary()

# train the autoencoder
print("\n[5/6] Training ...")
optimizer  = Adam(learning_rate=LR, clipvalue=1.0)
early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                           restore_best_weights=True)
autoencoder.compile(optimizer=optimizer, loss="mean_absolute_error")

init_loss = autoencoder.evaluate(x_train, x_train, batch_size=BATCH_SIZE, verbose=0)
print(f"  Initial loss: {init_loss:.4f}")

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
print("\n[6/6] Encoding full dataset and saving ...")
encoded_all = encoder.predict(data_standardized)
print(f"  encoded_all shape: {encoded_all.shape}")

# save the outputs
def sp(fname):
    return os.path.join(SAVE_DIR, f"{fname}_{TAG}")


autoencoder.save(sp("autoencoder") + ".keras")
encoder.save(sp("encoder") + ".keras")
np.save(sp("encoded_all")       + ".npy", encoded_all)
np.save(sp("data_standardized") + ".npy", data_standardized)
np.save(sp("lats")              + ".npy", lats_coarse)
np.save(sp("lons")              + ".npy", lons_coarse)
np.save(sp("times")             + ".npy", times)

clim.to_netcdf(os.path.join(SAVE_DIR, f"climatology_{TAG}.nc"))
polyfit_coefs.to_netcdf(os.path.join(SAVE_DIR, f"polyfit_coefs_{TAG}.nc"))

n_epochs_run = len(history.history["loss"])
input_size   = int(np.prod(data_standardized.shape[1:]))
latent_size  = int(np.prod(encoded_all.shape[1:]))
summary = {
    "tag":           TAG,
    "season":        SEASON or "all",
    "preprocessing": "climatology + linear trend removed on seasonal subset",
    "input_shape":   list(data_standardized.shape[1:]),
    "latent_shape":  list(encoded_all.shape[1:]),
    "compression":   round(input_size / latent_size, 3),
    "epochs_run":    n_epochs_run,
    "initial_loss":  round(init_loss, 4),
    "final_loss":    round(history.history["loss"][-1], 4),
    "best_val_loss": round(min(history.history["val_loss"]), 4),
    "coarsen":       COARSEN,
    "rounds":        ROUNDS,
    "batch_size":    BATCH_SIZE,
    "lr":            LR,
    "wall_seconds":  round(time.time() - t0, 1),
}
with open(os.path.join(SAVE_DIR, f"summary_{TAG}.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Saved to {SAVE_DIR}/  (tag={TAG})")
print(f"\n{'='*55}")
print(f"  DONE   wall time: {(time.time()-t0)/60:.1f} min")
print(f"  Compression      : {input_size/latent_size:.2f}x")
print(f"  Epochs run       : {n_epochs_run}")
print(f"  Best val loss    : {min(history.history['val_loss']):.4f}")
print(f"{'='*55}")
