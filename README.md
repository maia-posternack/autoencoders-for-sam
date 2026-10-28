# Code for: Identifying Southern Hemisphere Circulation Classes with a Convolutional Autoencoder

**Authors**: Maia Posternack, Kirstin Koepnick  
**Journal**: JGR: Machine Learning and Computation  
**Code archive**: https://doi.org/10.5281/zenodo.23720471

Hello!! This repository contains everything needed to reproduce the results in our paper. A convolutional autoencoder (CAE) and hierarchical clustering are applied to monthly ERA5 mean sea-level pressure (MSLP) over the Southern Hemisphere (1980–2022) to study the regime structure of the Southern Annular Mode (SAM).

| | |
|---|---|
| `run_preprocess_msl.py` | ERA5 MSL → climatology removed, detrended, cosine weighted |
| `run_pca.py` | EOF/PC baseline |
| `run_autoencoder.py` | trains every CAE in the paper (all-season, seasonal, initialisation ensemble, latent sweep) |
| `figures.ipynb` | every figure in the paper and supplement, written to `figures/` |
| `functions.py` | plotting and analysis helpers used by the notebook |
| `jobs/` | PBS scripts for NCAR Derecho |
| `figures/` | the published figures (Fig. 1, `overview.pdf`, was made in Google Slides) |
| `data/` | small artifacts that ship with the code (below) |

---

## Workflow

Each step produces files consumed by the next, so they must be done in order. I write everything to`$SCRATCH` (`/glade/derecho/scratch/$USER`), but have saved latent spaces available upon request.

### Step 1 — Preprocess MSL data

```bash
python run_preprocess_msl.py      # -> sam_preprocessed_data.nc
```

 Takes ERA5 monthly MSL at 0.25° from the NCAR Research Data Archive as input (also on the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)). Then removes the monthly climatology and a per-pixel linear trend, and applies sqrt(cos(lat)) area weighting.
### Step 2 — EOFs

```bash
python run_pca.py                 # -> sam_pca_data_all.nc, sam_pca_data.nc
```

SVD of the 90°S–20°S anomalies is saved in. `sam_pca_data_all.nc`  (the PC-clustering baseline, Figs S13–S14); I store the  the leading 3 modes in `sam_pca_data.nc`.

### Step 3 — Train the autoencoders

All runs use a 64-32-16-8-4 encoder (1620-scalar latent space, ~250x compression), 50 epochs with early stopping, batch 16, learning rate 1e-4 and the same 80/20 train/test split.

```bash
# the all-season run (Figs S1, S4, S6, S8-S10, S13-S16, S18)
python run_autoencoder.py --tag sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped

# one run per season (Figs 8, S17)
for SEASON in DJF MAM JJA SON; do
    python run_autoencoder.py --tag sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped_lintrend_${SEASON} --season ${SEASON}
done

# ten initialisations (Figs 2-7, S5, S7, S11, S12) and the latent-dimension sweep (Fig S3), ~2.6 h per member
qsub jobs/seed_ensemble.pbs
qsub jobs/latent_sweep.pbs
```

The ensemble runs use a fixed the TensorFlow seed and split, but vary in initialisation and training order. The other three architectures in Fig S2 used `--rounds 64 32 16 8 4 2 1`, `64 32 16 8` and `32 16 8` with `--lat_bounds -90 90`. The all-season run used ~100 GB RAM and ~4 hours on 4 cores; `--coarsen 4` gives a lightweight test on a workstation.

### Step 4 — Figures

```bash
qsub jobs/figures.pbs             # ~25 min, too heavy for a login node
```

`figures.ipynb` creates and writes all figures in `figures/` for the manuscript. Note it needs a  TensorFlow environment because Fig S1 runs the saved autoencoder.

---

## Data

`data/` holds the small artifacts:

| path | contents |
|---|---|
| `pca/sam_pca_data.nc` | the Step 2 EOF/PC solution (leading three modes), 516 months on the 281 x 1440 CAE grid |
| `model_summaries/summary_*.json` | training losses (mean absolute error) of the four architectures, the input to Fig S2 |
| `latent_spaces.nc` | encoder output of the all-season (`latent_annual`) and seasonal (`latent_DJF`, ...) runs, each `(n_time, 9, 45, 4)` |

Note that the trained models and training arrays are too large to include here. They are archived at `/glade/campaign/univ/uhar0025/mposternack/sam_archive/`; for access please contact maiaposternack@gmail.com or kirstinkoepnick@g.harvard.edu.

---

## Computing environment

You will need: 

Python 3.12, TensorFlow 2.18, Keras 3.9, NumPy 1.26, Xarray, Dask, scikit-learn 1.6, Matplotlib 3.10, Cartopy, SciPy, Pandas, h5netcdf, netCDF4; see `environment.yml`. Training and figures run in the same environment.

We hope you enjoy exploring!
