#!/usr/bin/env python3
"""
Authors: Maia Posternack (maiaposternack@gmail.com), Kirstin Koepnick (kirstinkoepnick@g.harvard.edu)

Train the SAM MSL convolutional autoencoder on the ERA5 anomalies with a fully
controlled random seed, so that a seed ensemble can be built.

This is the ERA5 counterpart of Kirstin's
/glade/u/home/kkoepnick/sam-ae/run_seed_ensemble.py, which produced the CESM2
(ERA5-grid) ensemble tagged sam_cesm2_era5grid_autoencoder_seed{0..9}.

Differences from that script, and from run_autoencoder.py
---------------------------------------------------------
run_autoencoder.py     fixes only the train/test split (random_state=5) and
                       leaves TensorFlow unseeded, so its weights are drawn
                       from an unrecorded RNG state and the run cannot be
                       reproduced or placed in a seed ensemble.

Kirstin's CESM2 script seeds TensorFlow with --seed but pins the split at
                       SPLIT_SEED = 5 for every member, so the ensemble spread
                       is purely initialisation + shuffle variance.

This script          defaults --split_seed to the value of --seed, so both the
                       weight initialisation and the 80/20 train/test split
                       vary across the ensemble.  The resulting spread is
                       therefore initialisation + shuffle + split variance
                       combined and is NOT directly comparable to the CESM2
                       ensemble.  Pass --split_seed 5 to recover the CESM2
                       design exactly.

Input
sam_preprocessed_data.nc  (from run_preprocess_msl.py; path set via --input_file
                            or defaults to $SCRATCH/sam_preprocessed_data.nc)

Outputs  (all written to --save_dir, default $SCRATCH/autoencoder_models/)
autoencoder_<tag>.keras        full autoencoder model
encoder_<tag>.keras            encoder sub-model
encoded_all_<tag>.npy          latent representations for every time step
data_standardized_<tag>.npy    standardised input used for training
lats_<tag>.npy                 latitude coordinate array
lons_<tag>.npy                 longitude coordinate array
times_<tag>.npy                time coordinate array
train_indices_<tag>.npy        indices of the training months
test_indices_<tag>.npy         indices of the held-out months
history_<tag>.json             per-epoch loss and val_loss
summary_<tag>.json             training metadata and performance summary

Typical use is through submit_era5_seeds.pbs, which runs seeds 0-9 as a job
array.  A single member can be run directly with

  python run_seed_ensemble_era5.py --tag sam_era5_autoencoder_seed0 --seed 0 \
      --rounds 64 32 16 8 4 --coarsen 1 --epochs 50 --batch_size 16 \
      --lr 1e-4 --patience 10
"""
# imports
import argparse
import gc
import os
import time
import json
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# this takes all our arguements and saves them locally
parser = argparse.ArgumentParser(
    description="Train SAM MSL autoencoder on ERA5 with a controlled seed"
)

parser.add_argument("--tag", type=str, default="sam_era5_autoencoder_seed0",
                    help="Output label used for all saved filenames")
parser.add_argument("--input_file", type=str, default=None,
                    help="Path to input .nc file. Defaults to $SCRATCH/sam_preprocessed_data.nc")
parser.add_argument("--coarsen", type=int, default=1,
                    help="Spatial subsampling factor (1=full res, 4=quarter res). Default: 1")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for model initialization and training shuffle.")
parser.add_argument("--split_seed", type=int, default=None,
                    help="Random seed for the train/test split. Defaults to --seed, so the "
                         "split varies across the ensemble. Pass 5 to pin the split and match "
                         "the CESM2 ensemble design.")
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
args = parser.parse_args()

SCRATCH = os.path.expandvars("/glade/derecho/scratch/$USER")
INPUT_FILE = args.input_file or os.path.join(SCRATCH, "sam_preprocessed_data.nc")
SAVE_DIR = args.save_dir or os.path.join(SCRATCH, "autoencoder_models")
os.makedirs(SAVE_DIR, exist_ok=True)

TAG = args.tag
COARSEN = args.coarsen
SEED = args.seed
# Unlike the CESM2 ensemble, the split seed follows the training seed by default.
SPLIT_SEED = args.split_seed if args.split_seed is not None else SEED
LAT_BOUNDS = tuple(args.lat_bounds) if args.lat_bounds is not None else None
ROUNDS = args.rounds
POOL_SIZE = (args.pool_size, args.pool_size)
CONV_SIZE = (args.conv_size, args.conv_size)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
PATIENCE = args.patience
TEST_SIZE = args.test_size

t0 = time.time()

print("=" * 55, flush=True)
print("  SAM MSL Autoencoder - ERA5 seed ensemble member", flush=True)
print("=" * 55, flush=True)
print(f"  tag        : {TAG}", flush=True)
print(f"  input      : {INPUT_FILE}", flush=True)
print(f"  save_dir   : {SAVE_DIR}", flush=True)
print(f"  seed       : {SEED}", flush=True)
print(f"  split_seed : {SPLIT_SEED}", flush=True)
print(f"  coarsen    : {COARSEN}x", flush=True)
print(f"  lat_bounds : {LAT_BOUNDS}", flush=True)
print(f"  rounds     : {ROUNDS}", flush=True)
print(f"  epochs     : {EPOCHS}  (patience={PATIENCE})", flush=True)
print("=" * 55, flush=True)

print("loading data...", flush=True)
msl_data = xr.open_dataset(INPUT_FILE)
print("dataset opened", flush=True)

da = msl_data["removed_trend_and_climatology"]
da = da.transpose("time", "latitude", "longitude")
print("variable selected", da.shape, da.dims, flush=True)

print("cropping...", flush=True)
if LAT_BOUNDS is not None:
    lat0, lat1 = float(LAT_BOUNDS[0]), float(LAT_BOUNDS[1])
    lat_slice = slice(lat0, lat1) if float(da.latitude[0]) <= float(da.latitude[-1]) else slice(lat1, lat0)
    da = da.sel(latitude=lat_slice)

print("coarsening...", flush=True)
if COARSEN > 1:
    da = da.isel(latitude=slice(None, None, COARSEN),
                 longitude=slice(None, None, COARSEN))

n_time = len(da.time)
n_lat = len(da.latitude)
n_lon = len(da.longitude)

data_all = np.empty((n_time, n_lat, n_lon), dtype=np.float32)

print("chunking...", flush=True)
chunk_size = 50
for start in range(0, n_time, chunk_size):
    end = min(start + chunk_size, n_time)
    data_all[start:end] = np.asarray(da.isel(time=slice(start, end)).load().data)

times = da.time.values
lats_coarse = da.latitude.values
lons_coarse = da.longitude.values
n_samples, lat_length, lon_length = data_all.shape
print(f"  data shape : {data_all.shape}", flush=True)

print("standardize...", flush=True)
scaler = StandardScaler()
data_flat = scaler.fit_transform(data_all.reshape(-1, 1))
data_standardized = data_flat.reshape(n_samples, lat_length, lon_length, 1).astype(np.float32)
data_standardized = np.nan_to_num(data_standardized, nan=0.0)

del data_flat, data_all
gc.collect()

print("training test split...", flush=True)
event_indices = np.arange(n_samples)
x_train, x_test, train_indices, test_indices = train_test_split(
    data_standardized, event_indices, test_size=TEST_SIZE, random_state=SPLIT_SEED
)
print(f"  train: {x_train.shape}   test: {x_test.shape}", flush=True)

print("importing tensorflow...", flush=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("tensorflow imported", flush=True)


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


print("build the autoencoder...", flush=True)
autoencoder, encoder = build_autoencoder(lat_length, lon_length, POOL_SIZE, CONV_SIZE, ROUNDS)
print("model built", flush=True)

autoencoder.summary()
optimizer = Adam(learning_rate=LR, clipvalue=1.0)
early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
autoencoder.compile(optimizer=optimizer, loss="mean_absolute_error")
print("compiled", flush=True)

print("skipping initial evaluate", flush=True)
history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[early_stop],
    verbose=1,
)
print("fit done", flush=True)

# save the predictions for future use
print("saving...", flush=True)
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
np.save(sp("train_indices") + ".npy", train_indices)
np.save(sp("test_indices") + ".npy", test_indices)

with open(os.path.join(SAVE_DIR, f"history_{TAG}.json"), "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

init_loss = np.nan
with open(os.path.join(SAVE_DIR, f"summary_{TAG}.json"), "w") as f:
    json.dump({
        "tag": TAG,
        "dataset": "ERA5",
        "input_file": INPUT_FILE,
        "input_shape": list(data_standardized.shape[1:]),
        "n_samples": int(n_samples),
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
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "latent_size": int(np.prod(encoded_all.shape[1:])),
        "latent_channels": int(encoded_all.shape[-1]),
        "pool_size": list(POOL_SIZE),
        "conv_size": list(CONV_SIZE),
        "test_size": TEST_SIZE,
        "patience": PATIENCE,
    }, f, indent=2)

print(f"\n Saving to {SAVE_DIR}/  (tag={TAG})", flush=True)
print(f"\n Done in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"Best val loss: {min(history.history['val_loss']):.4f}", flush=True)
