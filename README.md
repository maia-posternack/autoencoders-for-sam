# Code for: Identifying Regime Structure of the Southern Annular Mode using Clustering Autoencoder Techniques

**Authors**: Maia Posternack, Kirstin Koepnick  
**Journal**: JGR: Machine Learning and Computation  
**Code archive**: https://doi.org/10.5281/zenodo.22690306 

This repository contains all scripts and notebooks required to reproduce the results presented in our paper. The analysis uses a convolutional autoencoder and hierarchical clustering pipeline trained on monthly ERA5 mean sea-level pressure (MSLP) data over the Southern Hemisphere (1980–2022) to reconstruct the Southern Annular Mode (SAM).

---

## Workflow

The analysis proceeds in four steps. Each step produces output files consumed by the next so they must be done in order.

```
Step 1: run_preprocess_msl.py
        Takes raw ERA5 MSL NetCDF files and saves to sam_preprocessed_data.nc

Step 2a: run_pca.py
         Takes sam_preprocessed_data.nc and saves sam_pca_data.nc (leading 3 modes)

Step 2b: run_pca_all.py
         Same input, saves sam_pca_data_all.nc (every mode; the PC-clustering baseline)

Step 3a: run_autoencoder.py
         Reads sam_preprocessed_data.nc → autoencoder_models/<TAG>.*

Step 3b: run_preprocess_and_autoencoder_per_season.py  (run once per season)
         Reads sam_preprocessed_data.nc → autoencoder_models/<SEASON_TAG>.*

Step 3c: run_seed_ensemble_era5.py  (10 initialisations + a 12-member latent sweep)
         Reads sam_preprocessed_data.nc → autoencoder_models/era5_seed_ensemble/
                                        → autoencoder_models/era5_latent_sweep/

Step 4:  edits/paper_figures.ipynb
         Takes outputs from Steps 2–3 and saves every published figure to
         edits/paper_figures_out/{figs,final-figs}/
```

Steps 1–3 are training and are expensive (Step 3c alone is ~9–10 h of wall time on
Derecho). Step 4 is the whole figure set and takes about 25 minutes.

---

## Input Data

The preprocessing script reads ERA5 monthly mean sea-level pressure at 0.25° resolution from the NCAR Research Data Archive (RDA). Also available at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

---

## Step-by-step Instructions

### Step 1 — Preprocess MSL data

```bash
python run_preprocess_msl.py
```

Output: `$SCRATCH/sam_preprocessed_data.nc`

Removes the monthly climatology, per-pixel linear trend, and applies cosine weighting to the global ERA5 MSL field. Saves the anomaly alongside the raw data and preprocessing artefacts.

**Hardware**: Requires ~96 GB RAM and ~30 minutes on a single node. On NCAR Derecho, submit via PBS Batch. 

---

### Step 2 — Compute EOFs (PCA baseline)

```bash
python run_pca.py       # leading three modes  -> $SCRATCH/sam_pca_data.nc
python run_pca_all.py   # every mode           -> $SCRATCH/sam_pca_data_all.nc
```

Computes EOFs of Southern Hemisphere (90°S–20°S) MSL anomalies using full SVD. `run_pca.py` saves the leading three modes, which supply EOF1/PC1 to most
figures; `run_pca_all.py` saves the full score matrix, which is what the principal-component clustering baseline (Figs S14, S15) is built from.

A copy of `sam_pca_data.nc` is kept in `data/pca/` because it is small and slow to rebuild.

**Hardware**: Requires ~100 GB RAM (~10 min). On NCAR Derecho, submit via PBS.

---

### Step 3a — Train the full-year autoencoder

The settings below reproduce the model used in the paper (~250x compression, full 0.25° resolution, 5-stage encoder):

```bash
python run_autoencoder.py \
    --tag        sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped \
    --rounds     64 32 16 8 4 \
    --coarsen    1 \
    --epochs     50 \
    --batch_size 16 \
    --lr         1e-4 \
    --patience   10
```

Outputs (in `$SCRATCH/autoencoder_models/`):
- `autoencoder_<TAG>.keras` / `encoder_<TAG>.keras`
- `encoded_all_<TAG>.npy`, `data_standardized_<TAG>.npy`
- `lats_<TAG>.npy`, `lons_<TAG>.npy`, `times_<TAG>.npy`
- `summary_<TAG>.json`

**Hardware**: This run was performed on a single node with 100 GB RAM (~4 hours) and 4 CPU cores on the NCAR Derecho supercomputer. Full-resolution training is memory intensive; use `--coarsen 4` for a lightweight test on a workstation.

---

### Step 3b — Train seasonal autoencoders

Run once for each season, otherwise the same as the full-year autoencoder. 

```bash
for SEASON in DJF MAM JJA SON; do
    python run_preprocess_and_autoencoder_per_season.py \
        --tag        sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped_lintrend_${SEASON} \
        --season     ${SEASON} \
        --rounds     64 32 16 8 4 \
        --coarsen    1 \
        --epochs     50 \
        --batch_size 16 \
        --lr         1e-4 \
        --patience   10
done
```

Outputs: same as Step 3a plus `climatology_<TAG>.nc` and `polyfit_coefs_<TAG>.nc`.

---

### Step 3c — Train the initialisation ensemble and the latent sweep

The published classes are reported as an ensemble mean over ten random initialisations, so this step is required for most figures. `--split_seed 5` is pinned across all members: it fixes the train/test split so that the spread across members reflects initialisation
and training order only.

```bash
# Ten initialisations at the published latent size (Figs 2-7, S5, S9, S16)
for SEED in 0 1 2 3 4 5 6 7 8 9; do
    python run_seed_ensemble_era5.py \
        --tag        sam_era5_autoencoder_seed${SEED} \
        --seed       ${SEED} \
        --split_seed 5 \
        --rounds     64 32 16 8 4 \
        --coarsen    1 --epochs 50 --batch_size 16 --lr 1e-4 --patience 10 \
        --save_dir   $SCRATCH/autoencoder_models/era5_seed_ensemble
done

# Latent-dimension sweep: channels 1, 2, 8, 16 at seeds 0-2 (Fig S3).  4 channels is the
# published configuration and is not retrained -- the sweep reuses the ensemble above.
for CH in 1 2 8 16; do for SEED in 0 1 2; do
    python run_seed_ensemble_era5.py \
        --tag        sam_era5_latent${CH}ch_seed${SEED} \
        --seed       ${SEED} \
        --split_seed 5 \
        --rounds     64 32 16 8 ${CH} \
        --coarsen    1 --epochs 50 --batch_size 16 --lr 1e-4 --patience 10 \
        --save_dir   $SCRATCH/autoencoder_models/era5_latent_sweep
done; done
```

**Hardware**: ~2.6 h per member. On NCAR Derecho run them as a PBS job array — see
`VARIATION A` and `VARIATION B` in `edits/run_paper_figures.pbs`, which carry the working
queue configuration.

---

### Step 4 — Generate the figures

`edits/paper_figures.ipynb` runs top-to-bottom and writes **every figure in the paper andthe supplement**:

```bash
cd edits
qsub run_paper_figures.pbs      # ~25 min; too heavy for a login node
```

Figures are written to `edits/paper_figures_out/figs/` and `edits/paper_figures_out/final-figs/`, mirroring the two paths the LaTeX sources use. We suggest running this on the develop queue (it takes 20-30 minutes). 

`figures_and_analysis.ipynb` focuses only on figures from the first draft of this paper (a one-off run as opposed to an ensemble) 

---

## Provided Data

The `data/` directory contains pre-extracted latent-space representations from all five trained autoencoders so that the analysis notebook (Step 4) can be explored without re-running the computationally expensive training steps.

It also holds the small artifacts that are slow or impossible to rebuild — see `data/README.md` for the full inventory:

| path | contents |
|---|---|
| `pca/sam_pca_data.nc` | ERA5 EOF/PC solution, 516 months, from Step 2 |
| `model_summaries/summary_*.json` | Training losses for the four tested architectures. These four files are the entire input to Fig S2. |
| `figure_metrics/*.json` | Measured reconstruction errors and per-member latent-sweep diagnostics |

### `data/latent_spaces.nc`

A single NetCDF file (~6.5 MB) holding the encoder output arrays from every run. Each variable has shape `(n_time, 9, 45, 4)` — 9 latent latitude positions × 45 latent longitude positions × 4 feature maps — and carries its own time coordinate:

| Variable | Time dimension | n_time | Description |
|---|---|---|---|
| `latent_annual` | `time_annual` | 516 | Full-year autoencoder, all monthly fields 1980–2022 |
| `latent_DJF` | `time_DJF` | 129 | DJF seasonal autoencoder |
| `latent_MAM` | `time_MAM` | 129 | MAM seasonal autoencoder |
| `latent_JJA` | `time_JJA` | 129 | JJA seasonal autoencoder |
| `latent_SON` | `time_SON` | 129 | SON seasonal autoencoder |

> **Complete dataset**: The full intermediate outputs (trained `.keras` models, standardised input arrays, reconstructed fields, and cluster labels) are too large to include here. If you would like access to the complete dataset, please contact the authors at maiaposternack@gmail.com or kirstinkoepnick@g.harvard.edu.

---

## Output Paths

All intermediate files are written to `$SCRATCH` (`/glade/derecho/scratch/$USER` on NCAR Derecho). To write elsewhere, pass `--save_dir <path>` to the training
scripts and update `SAVE_DIR` / `OUT_DIR` at the top of each file.

The published figures are in `edits/paper_figures_out/`. The `figures/` directory holds an
earlier set from `figures_and_analysis.ipynb` and is superseded by `edits/paper_figures_out/`.

**Note on `$SCRATCH`**: on Derecho it is purged periodically and is not backed up. The
Step 3 outputs the figures depend on are ~20 GB (29 GB for the whole `autoencoder_models/` directory), so copy them somewhere durable if they are needed long
term. Ours are archived at `/glade/campaign/univ/uhar0025/mposternack/sam_archive/` — see `data/README.md` for the layout and for which files each figure needs.

---

## Computing Environment

**Python version**: 3.12  
**Key packages**: TensorFlow 2.19, Keras, NumPy 1.26, Xarray, Dask, scikit-learn 1.6,
Matplotlib 3.10, Cartopy, SciPy, Pandas, Seaborn, Distributed, h5netcdf, netCDF4

Training (Steps 1–3) and figure generation (Step 4) run in the same environment; see
`environment.yml`. TensorFlow is required for Step 4 as well, because Fig S1 does a
forward pass through the saved autoencoder.

We hope you enjoy exploring!