# Code for: Identifying Regime Structure of the Southern Annular Mode using Clustering Autoencoder Techniques (change later_)

**Authors**: Maia Posternack, Kirstin Koepnick  
**Journal**: JGR: Machine Learning and Computation  
**Code archive**: 

This repository contains all scripts and notebooks required to reproduce the results presented in our paper. The analysis uses a convolutional autoencoder and hierarchal clustering pipeline trained on  monthly ERA5 mean sea-level pressure (MSLP) data over the Southern Hemisphere (1980–2022) to reconstruct the Southern Annular Mode (SAM)

---

## Workflow

The analysis proceeds in four steps. Each step produces output files consumed by the next so they must be done in order.

```
Step 1: run_preprocess_msl.py
        Takes raw ERA5 MSL NetCDF files and saves to sam_preprocessed_data.nc

Step 2: run_pca.py
        Takes sam_preprocessed_data.nc and saves sam_pca_data.nc

Step 3a: run_autoencoder.py
         Reads sam_preprocessed_data.nc → autoencoder_models/<TAG>.*

Step 3b: run_seasonal_autoencoder.py  (run once per season)
         Reads sam_preprocessed_data.nc → autoencoder_models/<SEASON_TAG>.*

Step 4:  autoeoncder_analysis.ipynb
         Takes outputs from Steps 2–3 and saves to figures/
```

---

## Input Data

The preprocessing script reads ERA5 monthly mean sea-level pressure at 0.25°
resolution from the NCAR Research Data Archive (RDA), dataset
[d633001](https://rda.ucar.edu/datasets/d633001/):

```
/glade/campaign/collections/rda/data/d633001/e5.moda.an.sfc/
  <year>/e5.moda.an.sfc.128_151_msl.ll025sc.<year>010100_<year>120100.nc
```

If you are working outside NCAR, you can download the equivalent data directly from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) (ERA5 monthly averaged data on single levels, variable `mean_sea_level_pressure`, 1980–2022) and update the `ERA5_MODA_DIR` variable in `run_preprocess_msl.py`.

---

## Step-by-step Instructions

### Step 1 — Preprocess MSL data

```bash
python run_preprocess_msl.py
```

Output: `$SCRATCH/sam_preprocessed_data.nc`

Removes the monthly climatology, per-pixel linear trend, and applies cosine weighting to the global ERA5 MSL field. Saves the anomaly alongside the raw data and preprocessing artefacts.

**Hardware**: Requires ~96 GB RAM and ~30 minutes on a single node. On NCAR Derecho, submit via PBS Batch (see [Running on HPC with PBS](#running-on-hpc-with-pbs) below).

---

### Step 2 — Compute EOFs (PCA baseline)

```bash
python run_pca.py
```

Output: `$SCRATCH/sam_pca_data.nc`

Computes EOFs of Southern Hemisphere (90°S–20°S) MSL anomalies
using full SVD and saves the leading three modes.

**Hardware**: Requires ~100 GB RAM (~10 min). On NCAR Derecho, submit via PBS.

---

### Step 3a — Train the full-year autoencoder

The settings below reproduce the model used in the paper (~250x compression,
full 0.25° resolution, 5-stage encoder):

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
    python run_seasonal_autoencoder.py \
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

### Step 4 — Run the analysis notebook

Open `autoencoder_analysis.ipynb` and run it top-to-bottom. The notebook is organised in
three sequential parts:

| Part | Description | Key outputs |
|---|---|---|
| **Part 1** | Full-year autoencoder: reconstruction quality, elbow analysis, cluster visualisation | `figures/autoencoder_reconstruction.pdf`, `figures/elbow.pdf`, `figures/reconstruction_correlation.pdf` |
| **Part 2** | Seasonal autoencoders: per-season hierarchical clustering, composite maps | `figures/season_linkages.pdf`, `figures/DJF_clusters.pdf`, etc. |
| **Part 3** | Time series and comparison: SAM index construction, EOF comparison, Marshall AAO validation, bootstrap test | `figures/compare.pdf`, `figures/time.pdf`, `figures/corr.pdf`, `figures/bootstrap.pdf` |

Parts 1–3 must be run top-to-bottom in a single session; Part 3 uses `ds_ae` built in Part 1.

CHANGE LATER

---

## Running on HPC with PBS

The scripts were originally submitted via PBS on NCAR Derecho. Template batch scripts are shown below. Replace `<PROJECT_CODE>` with your allocation and
update the conda environment path as needed.

### Step 1 — Preprocess (`run_preprocess_msl.py`)

```bash
#!/bin/bash
#PBS -N run_preprocess_msl
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -l select=1:ncpus=8:mem=96GB
#PBS -l walltime=03:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate sam-autoencoder
export HDF5_USE_FILE_LOCKING=FALSE

python run_preprocess_msl.py
```

### Step 2 — PCA (`run_pca.py`)

```bash
#!/bin/bash
#PBS -N run_pca
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -l select=1:ncpus=8:mem=100GB
#PBS -l walltime=01:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate sam-autoencoder
export HDF5_USE_FILE_LOCKING=FALSE

python run_pca.py
```

### Step 3a — Full-year autoencoder (`run_autoencoder.py`)

```bash
#!/bin/bash
#PBS -N run_autoencoder
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -l walltime=6:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate sam-autoencoder
export HDF5_USE_FILE_LOCKING=FALSE

python run_autoencoder.py \
    --tag        sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped \
    --rounds     64 32 16 8 4 \
    --coarsen    1 \
    --epochs     50 \
    --batch_size 16 \
    --lr         1e-4 \
    --patience   10
```

### Step 3b — Seasonal autoencoders (`run_seasonal_autoencoder.py`)

Submit one job per season. Example for DJF (repeat for MAM, JJA, SON):

```bash
#!/bin/bash
#PBS -N run_seasonal_DJF
#PBS -A <PROJECT_CODE>
#PBS -q develop
#PBS -l select=1:ncpus=4:mem=100GB
#PBS -l walltime=6:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
source /glade/u/apps/opt/conda/etc/profile.d/conda.sh
conda activate sam-autoencoder
export HDF5_USE_FILE_LOCKING=FALSE

SEASON=DJF
python run_seasonal_autoencoder.py \
    --tag        sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped_lintrend_${SEASON} \
    --season     ${SEASON} \
    --rounds     64 32 16 8 4 \
    --coarsen    1 \
    --epochs     50 \
    --batch_size 16 \
    --lr         1e-4 \
    --patience   10
```

Save each script as a `.pbs` file and submit with `qsub <script>.pbs`.
On SLURM-based systems, replace `#PBS` directives with their `#SBATCH` equivalents
(`-N` → `--job-name`, `select=1:ncpus=N:mem=XGB` → `--ntasks=1 --cpus-per-task=N --mem=XGB`,
`-l walltime` → `--time`).

---

## Output Paths

All intermediate files are written to `$SCRATCH` (`/glade/derecho/scratch/$USER` on NCAR Derecho). To write elsewhere, pass `--save_dir <path>` to the training
scripts and update `SAVE_DIR` / `OUT_DIR` at the top of each file.

Pre-generated paper figures are included in the `figures/` directory for reference.

---

## Computing Environment

The scripts were developed and run on the NCAR Derecho supercomputer. The authors
used a pre-existing conda environment named `my-npl-tensor`, located at
`/glade/work/mposternack/conda-envs/my-npl-tensor`. An equivalent reproducible
environment is provided via `environment.yml` (named `sam-autoencoder`).

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate sam-autoencoder
```

Or with pip:

```bash
pip install -r requirements.txt
```

**Python version**: 3.12  
**Key packages**: TensorFlow 2.18, Keras, NumPy 1.26, Xarray, Dask, scikit-learn,
Matplotlib, Cartopy, SciPy
