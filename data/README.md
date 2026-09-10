# `data/`

Here you can find small, hard-to-regenerate artifacts kept alongside the code!

## Contents

| path | what it is |
|---|---|
| `latent_spaces.nc` | Pre-extracted CAE latent spaces. The one tracked data file. |
| `pca/sam_pca_data.nc` | ERA5 EOF/PC solution, 516 months (1980-2022), on the 281 x 1440 CAE grid. `eofs`, `pcs`, `variance_explained`. Byte-equal to the copy at `/glade/campaign/univ/uhar0025/kkoepnick/sam-autoencoder/sam_pca_data.nc`. Supplies EOF1/PC1 to Figs 4, 5, 6, S9, S10, S16. |
| `model_summaries/summary_*.json` | Training summaries written by `run_autoencoder.py` for the four tested architectures: `final_loss`, `best_val_loss`, `latent_shape`. **These four files are the entire input to Fig S2** — the plotted losses are mean absolute errors, not MSE (every training script compiles with `loss="mean_absolute_error"`). |
| `figure_metrics/reconstruction_loss_metrics.json` | Directly measured held-out MAE (0.2464) and MSE (0.2408) of the published model, which is what proved the old "MSE" axis label wrong. |
| `figure_metrics/latent_sweep_mae.json` | Cached held-out MAE per latent-sweep member, from forward passes through all 22 saved models. Regenerating it costs a TensorFlow pass over every member. |
| `figure_metrics/latent_sweep_metrics.json` | Per-member occupancy, non-annularity, SAM index, amplitude, matched correlation and ARI for the latent sweep. |

`paper_figures.ipynb` does **not** read this directory — it reads the scratch paths below,
unchanged, which is how it was verified. The copies here exist because scratch is purged
and is not backed up.

## Bulk inputs

Too large to track in git. The figure notebook reads them from
`/glade/derecho/scratch/mposternack/`, which is **purged periodically and not backed up**,
so as of 2026-09-10 everything is also archived on campaign storage (see below).

| path, relative to `autoencoder_models/` unless noted | size | used by |
|---|---|---|
| `{autoencoder,encoder}_sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped.keras` + its `{encoded_all,data_standardized,lats,lons,times}_*.npy` | 0.8 G | Parts 2, 4, 5, 6 (Figs S1, S4, S15, S14, S7, S8, S6, S10-S12) |
| `*_lintrend_{DJF,MAM,JJA,SON}.*` | 0.8 G | Part 2 (Figs S13, 8) |
| `era5_seed_ensemble/` | 7.9 G | Part 3 (Figs 2, 3, 4, 5, 6, 7, S5, S9, S16) |
| `era5_latent_sweep/` | 9.5 G | Part 7 (Fig S3) |
| `sam_pca_data_all.nc` (scratch root) | 1.6 G | Part 2 (Figs S15, S14) — all PC modes, not just the leading three |
| `sam_preprocessed_data.nc` (scratch root) | 3.0 G | training input only; not read by the figure notebook |

The four rows above total the 20.6 GB the figures actually need. The full
`autoencoder_models/` directory is 29 G, because it also holds earlier architectures
(`sam_ae`, `sam_ae_8x`, `sam_ae_50x`, `1x_32_16_8`, `1x_64_32_16_8`, `1x_64_32_16_8_4_2_1`)
that no published figure reads. Their `data_standardized_*.npy` files are 2.0 G apiece and
are regenerable from `sam_preprocessed_data.nc` if space is ever needed.

## Durable archive

Copied 2026-09-10, verified by file count and by md5 on the paper-critical models:

```
/glade/campaign/univ/uhar0025/mposternack/sam_archive/
├── autoencoder_models/               29 G, 384 files   full copy of the scratch directory
├── netcdf/                          6.0 G, 8 files     every .nc from the scratch root
└── edits_pre_prune_20260909.tar.gz  304 M              code + figures snapshot, 2026-09-09
```

`netcdf/` holds `sam_pca_data_all.nc` and `sam_preprocessed_data.nc` from the table above,
plus `sam_pca_data.nc`, `sam_preprocessed_wind.nc`, `global_pca_data.nc`,
`sam_eof_analysis.nc`, `ae_cluster_data.nc` and `ae_cluster_ridge_data.nc`.

Note that campaign paths are readable only by the `uhar0025` group on NCAR systems; they
are a backup, not a distribution channel. For outside requests see the contact note in the
top-level `README.md`.
