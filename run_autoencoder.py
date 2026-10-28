#!/usr/bin/env python3
"""
Authors: Maia Posternack (maiaposternack@gmail.com), Kirstin Koepnick (kirstinkoepnick@g.harvard.edu)

In this file, we train a convolutional autoencoder on the preprocessed SAM MSL anomalies and save all
outputs / weights to be analyzed later.

--season DJF|MAM|JJA|SON  train on one season. The climatology and trend are then removed on the
                          seasonal subset, so we start from the raw MSL in the preprocessed file.
--seed N                  seed TensorFlow (weights and minibatch order) and make the ops deterministic,
                          for the initialisation ensemble and the latent sweep. The published all-season
                          run was not seeded.
--split_seed              the 80/20 train/test split; 5 for every run in the paper.

Input
$SCRATCH/sam_preprocessed_data.nc  (from run_preprocess_msl.py)

Outputs (in --save_dir, default $SCRATCH/autoencoder_models/)
autoencoder_<tag>.keras, encoder_<tag>.keras
encoded_all_<tag>.npy          latent representations for every month
data_standardized_<tag>.npy    standardised input used for training
lats_, lons_, times_<tag>.npy
train_indices_, test_indices_<tag>.npy
lat_weights_<tag>.npy          seasonal runs only: the training weight, to undo it
history_<tag>.json, summary_<tag>.json

Example (the published run):
python run_autoencoder.py --tag sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped --rounds 64 32 16 8 4
"""
import argparse
import json
import os
import time
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from run_preprocess_msl import remove_climatology, remove_linear_trend

SCRATCH = os.path.expandvars("/glade/derecho/scratch/$USER")
SEASON_MONTHS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SAM MSL autoencoder")
    p.add_argument("--tag", required=True, help="label used for all saved filenames")
    p.add_argument("--input_file", default=f"{SCRATCH}/sam_preprocessed_data.nc")
    p.add_argument("--save_dir", default=f"{SCRATCH}/autoencoder_models")
    p.add_argument("--season", choices=list(SEASON_MONTHS))
    p.add_argument("--seed", type=int)
    p.add_argument("--split_seed", type=int, default=5)
    p.add_argument("--lat_bounds", type=float, nargs=2, default=[-90.0, -20.0],
                   help="use -90 90 for global")
    p.add_argument("--coarsen", type=int, default=1, help="spatial subsampling factor")
    p.add_argument("--rounds", type=int, nargs="+", default=[64, 32, 16, 8, 4],
                   help="conv filters per encoder stage")
    p.add_argument("--pool_size", type=int, default=2)
    p.add_argument("--conv_size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    return p.parse_args(argv)


def load_data(args):
    """Standardised (n, lat, lon, 1) training input, plus coordinates and (seasonal) weights."""
    ds = xr.open_dataset(args.input_file, chunks={"time": 12})
    da = ds["MSL" if args.season else "removed_trend_and_climatology"].astype("float32")
    da = da.transpose("time", "latitude", "longitude").sel(latitude=slice(*sorted(args.lat_bounds)))
    if args.coarsen > 1:
        da = da.isel(latitude=slice(None, None, args.coarsen), longitude=slice(None, None, args.coarsen))

    weights = None
    if args.season:
        da = da.isel(time=da.indexes["time"].month.isin(SEASON_MONTHS[args.season]))
        da, _ = remove_climatology(da)
        da, _ = remove_linear_trend(da)
        weights = np.sqrt(np.cos(np.deg2rad(da.latitude.values))).astype(np.float32)
        weights /= weights.mean()
        da = (da * xr.DataArray(weights, coords=[da.latitude], dims=["latitude"])).astype("float32")

    data = da.values
    x = StandardScaler().fit_transform(data.reshape(-1, 1)).reshape(*data.shape, 1).astype(np.float32)
    return np.nan_to_num(x), da.time.values, da.latitude.values, da.longitude.values, weights


def build_autoencoder(lat_length, lon_length, pool_size, conv_size, rounds):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(lat_length, lon_length, 1))
    x = inp
    for n in rounds:
        x = layers.Conv2D(n, conv_size, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(pool_size, padding="same")(x)
    encoded = x
    for n in reversed(rounds):
        x = layers.Conv2D(n, conv_size, activation="relu", padding="same")(x)
        x = layers.UpSampling2D(pool_size)(x)
    x = layers.Conv2D(1, conv_size, activation="tanh", padding="same")(x)
    # crop the decoder overshoot for non-power-of-2 dims
    h, w = lat_length, lon_length
    for _ in rounds:
        h, w = (h + 1) // 2, (w + 1) // 2
    crop = (h * 2 ** len(rounds) - lat_length, w * 2 ** len(rounds) - lon_length)
    if any(crop):
        x = layers.Cropping2D(cropping=((0, crop[0]), (0, crop[1])))(x)
    return keras.Model(inp, x), keras.Model(inp, encoded)


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    x, times, lats, lons, weights = load_data(args)
    x_train, x_test, train_idx, test_idx = train_test_split(
        x, np.arange(len(x)), test_size=args.test_size, random_state=args.split_seed)

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()

    autoencoder, encoder = build_autoencoder(x.shape[1], x.shape[2], (args.pool_size,) * 2,
                                             (args.conv_size,) * 2, args.rounds)
    autoencoder.summary()
    autoencoder.compile(optimizer=Adam(learning_rate=args.lr, clipvalue=1.0), loss="mean_absolute_error")
    history = autoencoder.fit(
        x_train, x_train, epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
        validation_data=(x_test, x_test), verbose=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
    ).history
    encoded = encoder.predict(x)

    # save the outputs
    def sp(name):
        return os.path.join(args.save_dir, f"{name}_{args.tag}")

    autoencoder.save(sp("autoencoder") + ".keras")
    encoder.save(sp("encoder") + ".keras")
    arrays = dict(encoded_all=encoded, data_standardized=x, lats=lats, lons=lons, times=times,
                  train_indices=train_idx, test_indices=test_idx)
    if weights is not None:
        arrays["lat_weights"] = weights
    for name, arr in arrays.items():
        np.save(sp(name) + ".npy", arr)

    with open(sp("history") + ".json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    with open(sp("summary") + ".json", "w") as f:
        json.dump({
            **vars(args),
            "input_shape": list(x.shape[1:]),
            "latent_shape": list(encoded.shape[1:]),
            "compression": round(int(np.prod(x.shape[1:])) / int(np.prod(encoded.shape[1:])), 3),
            "epochs_run": len(history["loss"]),
            "final_loss": round(history["loss"][-1], 4),
            "best_val_loss": round(min(history["val_loss"]), 4),
            "wall_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"Saved to {args.save_dir} (tag={args.tag}) in {(time.time() - t0) / 60:.1f} min; "
          f"best val loss {min(history['val_loss']):.4f}")
