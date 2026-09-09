#!/usr/bin/env python3
"""
Provenance record for `edits/paper_figures.ipynb` -- the notebook that runs top to
bottom and regenerates every figure cited by the resubmission text in `submission-2/`.

!!  THIS SCRIPT NO LONGER RUNS.  It is kept as documentation of where each cell of
!!  paper_figures.ipynb came from.
!!
!!  It built the notebook by lifting cells out of ten sources, and nine of those were
!!  deleted in the 2026-09-09 prune once the notebook had absorbed them:
!!
!!      era5_ensemble.ipynb          corr_ensemble.ipynb      occupancy_matched.ipynb
!!      common_colorbar.ipynb        seasonal_transitions.ipynb   latent_sweep.ipynb
!!      all_figures.ipynb            make_temporal_e12.py
!!      make_edit08_reconstruction_loss.py
!!
!!  Only `figures_and_analysis.ipynb` (repo root, tracked in git) survives.  All nine
!!  are recoverable from
!!      /glade/campaign/univ/uhar0025/mposternack/sam_archive/edits_pre_prune_20260909.tar.gz
!!  Untar that beside this script and it runs again unchanged.
!!
!!  `paper_figures.ipynb` is self-contained -- every lifted cell is inlined in it -- so
!!  nothing needs this script in order to regenerate the figures.  Edit the notebook
!!  directly and keep the section headers honest.

Why a builder rather than a hand-written notebook
-------------------------------------------------
The 23 code-generated figures were produced by eight different notebooks and two
standalone scripts.  Retyping them would risk silent drift from the versions that
were actually submitted, so every analysis cell here is **lifted verbatim** from
its source at build time and only two things are patched:

  * the output name, so each figure is written under the filename the manuscript
    ``\\includegraphics`` actually asks for; and
  * the duplicated per-notebook preamble (style layer, ``save_figure``,
    ``FIGURE_DIR``, assertion cells), which is hoisted into one shared setup.

Executing the notebook needs only `qsub run_paper_figures.pbs`; rebuilding it needs the
archived sources restored first (see the warning above).

Figure -> source map (checksum-verified against submission-2 on 2026-09-08)
--------------------------------------------------------------------------
    Fig 1   final-figs/overview.pdf              NOT code: a Google Slides export
    Fig 2   final-figs/ensemble_composite.pdf    era5_ensemble.ipynb  cell030
    Fig 3   final-figs/occupancy_box.pdf         era5_ensemble.ipynb  cell010
    Fig 4   final-figs/temporal_proj.pdf         make_temporal_e12.py sections 4, 7
    Fig 5   final-figs/scatter.pdf               era5_ensemble.ipynb  cell016
    Fig 6   final-figs/annularity.pdf            era5_ensemble.ipynb  etabars
    Fig 7   final-figs/transition.pdf            era5_ensemble.ipynb  cell022
    Fig 8   figs/all_seasons_clusters.pdf        figures_and_analysis.ipynb cell 53
    Fig S1  figs/autoencoder_reconstruction.pdf  figures_and_analysis.ipynb cell 15
    Fig S2  final-figs/reconstruction_loss.pdf   make_edit08_reconstruction_loss.py
    Fig S3  final-figs/latent_sweep_composites   latent_sweep.ipynb   ls023
    Fig S4  figs/elbow.pdf                       figures_and_analysis.ipynb cell 19
    Fig S5  final-figs/ensemble.pdf              era5_ensemble.ipynb  cell024
    Fig S6  final-figs/composites_common_colorbar common_colorbar.ipynb cc010
    Fig S7  final-figs/shuffled_null_test.pdf    all_figures.ipynb    e049
    Fig S8  final-figs/seasonal_transition_heatmaps seasonal_transitions.ipynb st015
    Fig S9  final-figs/corr_direct_clusters_ensemble corr_ensemble.ipynb cc019
    Fig S10 final-figs/pc1_distributions_by_cluster occupancy_matched.ipynb om011
    Fig S11 final-figs/occupancy_matched_composites occupancy_matched.ipynb om016
    Fig S12 final-figs/occupancy_matched_difference_maps occupancy_matched.ipynb om019
    Fig S13 figs/season_linkages.pdf             figures_and_analysis.ipynb cell 47
    Fig S14 figs/pca_composites.pdf              figures_and_analysis.ipynb cell 28
    Fig S15 figs/pca_elbow.pdf                   figures_and_analysis.ipynb cell 27
    Fig S16 final-figs/baseline_composites.pdf   era5_ensemble.ipynb  cell014
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "paper_figures.ipynb")

# ── Sources ───────────────────────────────────────────────────────────────────
FA = os.path.join(REPO, "figures_and_analysis.ipynb")   # published single run
ALL_FIGURES = os.path.join(HERE, "all_figures.ipynb")   # style layer, e04x
ERA5 = os.path.join(HERE, "era5_ensemble.ipynb")
CORR = os.path.join(HERE, "corr_ensemble.ipynb")
OCC = os.path.join(HERE, "occupancy_matched.ipynb")
CCB = os.path.join(HERE, "common_colorbar.ipynb")
SEAS = os.path.join(HERE, "seasonal_transitions.ipynb")
LSW = os.path.join(HERE, "latent_sweep.ipynb")
TEMPORAL = os.path.join(HERE, "make_temporal_e12.py")
RECON_LOSS = os.path.join(HERE, "make_edit08_reconstruction_loss.py")


def nb_cells(path):
    with open(path) as fh:
        return json.load(fh)["cells"]


def by_id(path):
    return {c.get("id"): "".join(c["source"]) for c in nb_cells(path)}


def by_index(path):
    return ["".join(c["source"]) for c in nb_cells(path)]


def script(path):
    with open(path) as fh:
        return fh.read()


def lines(text, lo, hi):
    """1-based inclusive line slice, as reported by an editor."""
    return "\n".join(text.split("\n")[lo - 1:hi])


def sub(text, old, new, count=1):
    assert text.count(old) >= 1, f"patch target not found:\n{old[:200]}"
    return text.replace(old, new, count)


def cut_from(text, marker):
    """Drop `marker` and everything after it."""
    i = text.index(marker)
    return text[:i].rstrip() + "\n"


def drop_lines(text, *patterns):
    """Remove whole lines matching any regex, and collapse blank runs."""
    out = []
    for line in text.split("\n"):
        if any(re.search(p, line) for p in patterns):
            continue
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"


# Preamble lines that every built notebook repeats and Part 0 now owns.  Leaving
# `RUN_STARTED` in place would reset the clock the verification cell uses to prove
# each PDF was written by this run.
SHARED_PREAMBLE = (
    r"^FIGURE_DIR = ",
    r"^FIGURE_DIR\.mkdir",
    r"^RUN_STARTED = time\.time\(\)",
    r'^print\(f?"Figures will be written to',
)


def preamble(text):
    """One notebook's configuration cell, minus what Part 0 already defines."""
    text = cut_from(text, "def free(*names):")
    text = re.sub(r"^EXPECTED_FIGURES = \[[^\]]*\]\n", "", text, flags=re.M)
    return drop_lines(text, *SHARED_PREAMBLE)


E5 = by_id(ERA5)
AF = by_id(ALL_FIGURES)
CE = by_id(CORR)
OM = by_id(OCC)
CB = by_id(CCB)
ST = by_id(SEAS)
LS = by_id(LSW)
FAC = by_index(FA)
TE = script(TEMPORAL)

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")})


def code(source):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": source.rstrip("\n")})


# ═════════════════════════════════════════════════════════════════════════════
# Part 0 -- setup
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
# Figures for *Beyond a Single Southern Annular Mode Pattern*

Every figure cited by `submission-2/main-B.tex` and `submission-2/supplemental-B.tex`,
regenerated by one notebook that runs top to bottom.

Outputs are written to `paper_figures_out/figs/` and `paper_figures_out/final-figs/`,
mirroring the two directories the LaTeX sources reference, so the tree can be copied
straight into `submission-2/`.

| Figure | File | Section |
|---|---|---|
| 1 | `final-figs/overview.pdf` | **not generated here** -- hand-drawn schematic, see Part 1 |
| 2 | `final-figs/ensemble_composite.pdf` | Part 3 |
| 3 | `final-figs/occupancy_box.pdf` | Part 3 |
| 4 | `final-figs/temporal_proj.pdf` | Part 3 |
| 5 | `final-figs/scatter.pdf` | Part 3 |
| 6 | `final-figs/annularity.pdf` | Part 3 |
| 7 | `final-figs/transition.pdf` | Part 3 |
| 8 | `figs/all_seasons_clusters.pdf` | Part 2 |
| S1 | `figs/autoencoder_reconstruction.pdf` | Part 2 |
| S2 | `final-figs/reconstruction_loss.pdf` | Part 7 |
| S3 | `final-figs/latent_sweep_composites.pdf` | Part 7 |
| S4 | `figs/elbow.pdf` | Part 2 |
| S5 | `final-figs/ensemble.pdf` | Part 3 |
| S6 | `final-figs/composites_common_colorbar.pdf` | Part 5 |
| S7 | `final-figs/shuffled_null_test.pdf` | Part 6 |
| S8 | `final-figs/seasonal_transition_heatmaps.pdf` | Part 6 |
| S9 | `final-figs/corr_direct_clusters_ensemble.pdf` | Part 3 |
| S10 | `final-figs/pc1_distributions_by_cluster.pdf` | Part 4 |
| S11 | `final-figs/occupancy_matched_composites.pdf` | Part 4 |
| S12 | `final-figs/occupancy_matched_difference_maps.pdf` | Part 4 |
| S13 | `figs/season_linkages.pdf` | Part 2 |
| S14 | `figs/pca_composites.pdf` | Part 2 |
| S15 | `figs/pca_elbow.pdf` | Part 2 |
| S16 | `final-figs/baseline_composites.pdf` | Part 3 |

## How to run it

Not on a login node: Part 3 holds a 0.83 GB de-weighted field for ten members and
bootstraps significance maps on a 281 x 1440 grid.

```bash
cd /glade/u/home/mposternack/autoencoders_for_sam/edits
qsub run_paper_figures.pbs
```

The kernel must be the TensorFlow environment
`/glade/work/mposternack/conda-envs/my-npl-tensor` (matplotlib 3.10.0, TF 2.19).
`npl-2024a` has no TensorFlow, and Figure S1 needs a forward pass through the saved
autoencoder.  Nothing else in the notebook touches Keras.

## Structure

Each part loads one dataset once and then draws every figure that needs it, so the
ten-member ensemble is clustered once rather than seven times.  The parts are
independent: `free(...)` drops the large arrays at the end of each one, so a part can
be re-run on its own after Part 0.

1. Figure 1 -- the schematic (provenance only)
2. The published single CAE run -- Figs S1, S4, S15, S14, S13, 8
3. The ten-member ERA5 initialisation ensemble -- Figs 3, 6, S16, 5, 7, S5, 2, S9, 4
4. PC1-conditioned controls -- Figs S10, S11, S12
5. Composites on one common colour bar -- Fig S6
6. Transition statistics of the published run -- Figs S7, S8
7. Architecture and latent-size sensitivity -- Figs S2, S3
8. Verification

## Provenance and reproducibility notes

* Every analysis cell is lifted verbatim from the notebook or script that produced the
  submitted PDF; `make_paper_figures.py` records the source of each one.
* **Figure S1 is now seeded.** Its source cell drew three test months from an unseeded
  `np.random.default_rng()`, so the submitted panel cannot be reproduced. `RECON_SEED`
  below fixes the draw; the three months shown therefore differ from the submitted PDF,
  which the caption ("three randomly selected monthly MSLP anomaly fields") permits.
* **Figure 4's panel titles** now come from the shared `cluster_names`
  (`Low-amplitude SAM$-$`, `SAM$+$`, `SAM$-$`) rather than the lower-case, ASCII-plus
  variant hard-coded in `make_temporal_e12.py`. Same data, uniform labels.
* Nine of the submitted PDFs were rendered under matplotlib 3.8.2 (`npl-2024a`) and the
  rest under 3.10.0. This notebook uses one kernel, so those nine will differ from the
  submitted files in minor text metrics. No plotted value changes.
* Cluster order is physical, never by size: Cluster 1 = low-amplitude SAM$-$,
  Cluster 2 = SAM$+$, Cluster 3 = SAM$-$, imposed from the SAM index of the reference
  solution and asserted in each part.
""")

md("## Part 0 -- Output paths, figure registry and the shared style layer")

code(r'''
import gc
import os
import time
from pathlib import Path

# One output tree per LaTeX include path.  Overridable so a smoke run can write
# somewhere disposable without editing cells.
OUT_ROOT = Path(os.environ.get(
    "PF_OUT_DIR",
    "/glade/u/home/mposternack/autoencoders_for_sam/edits/paper_figures_out"))

# name -> (subdirectory, manuscript figure number).  The subdirectory is the one
# the .tex files use, so the tree drops straight into submission-2/.
FIGURES = {
    "ensemble_composite":                ("final-figs", "Figure 2"),
    "occupancy_box":                     ("final-figs", "Figure 3"),
    "temporal_proj":                     ("final-figs", "Figure 4"),
    "scatter":                           ("final-figs", "Figure 5"),
    "annularity":                        ("final-figs", "Figure 6"),
    "transition":                        ("final-figs", "Figure 7"),
    "all_seasons_clusters":              ("figs",       "Figure 8"),
    "autoencoder_reconstruction":        ("figs",       "Figure S1"),
    "reconstruction_loss":               ("final-figs", "Figure S2"),
    "latent_sweep_composites":           ("final-figs", "Figure S3"),
    "elbow":                             ("figs",       "Figure S4"),
    "ensemble":                          ("final-figs", "Figure S5"),
    "composites_common_colorbar":        ("final-figs", "Figure S6"),
    "shuffled_null_test":                ("final-figs", "Figure S7"),
    "seasonal_transition_heatmaps":      ("final-figs", "Figure S8"),
    "corr_direct_clusters_ensemble":     ("final-figs", "Figure S9"),
    "pc1_distributions_by_cluster":      ("final-figs", "Figure S10"),
    "occupancy_matched_composites":      ("final-figs", "Figure S11"),
    "occupancy_matched_difference_maps": ("final-figs", "Figure S12"),
    "season_linkages":                   ("figs",       "Figure S13"),
    "pca_composites":                    ("figs",       "Figure S14"),
    "pca_elbow":                         ("figs",       "Figure S15"),
    "baseline_composites":               ("final-figs", "Figure S16"),
}

for _sub in ("figs", "final-figs"):
    (OUT_ROOT / _sub).mkdir(parents=True, exist_ok=True)

WRITTEN = {}
RUN_STARTED = time.time()

# Figure S1 draws three example months at random.  Its source cell used an
# unseeded generator, so the submitted panel is unreproducible; this pins it.
RECON_SEED = 0


def free(*names):
    """Drop large objects from the global namespace and collect."""
    for name in names:
        globals().pop(name, None)
    gc.collect()


print(f"figures -> {OUT_ROOT}")
print(f"{len(FIGURES)} figures registered")
''')

md("""
The style layer is `all_figures.ipynb` cell `c1b`, lifted verbatim: `paper_polar_map`,
`paper_stipple`, `coarsened_cos_weight`, `undo_cos_weight`, `fs_sizes` and the
`FS_TITLE / FS_LABEL / FS_TICK / FS_ANNOT = 26 / 22 / 18 / 15` constants.  Its trailing
`save_figure` is dropped in favour of the registry-aware one below.
""")

code(cut_from(AF["c1b"], "def save_figure(fig, name):"))

code(r'''
def save_figure(fig, name):
    """Write a vector PDF under the filename the manuscript asks for.

    Replaces the `c1b` helper: the name is looked up in FIGURES so the lifted
    analysis cells stay untouched while the output lands in `figs/` or
    `final-figs/` under its manuscript filename.
    """
    subdir, label = FIGURES[name]
    path = OUT_ROOT / subdir / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    WRITTEN[name] = path
    print(f"{label:<12} -> {subdir}/{path.name}  ({path.stat().st_size / 1e3:,.0f} kB)")
    return path


print("save_figure ready")
''')

# ═════════════════════════════════════════════════════════════════════════════
# Part 1 -- the schematic
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 1 -- Figure 1, the pipeline schematic

`final-figs/overview.pdf` is **not** produced by code.  Its PDF metadata reads
`/Creator (Google)` and `/Title (Untitled presentation)`: it is a Google Slides export
of the hand-drawn pipeline diagram, with no matplotlib producer string and no source in
this repository.  It is the only figure in either document that this notebook cannot
regenerate, and it must be carried over from `submission-2/final-figs/` by hand.

Everything else follows.
""")

# ═════════════════════════════════════════════════════════════════════════════
# Part 2 -- the published single run
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 2 -- The published single CAE run

Source: `figures_and_analysis.ipynb`, the notebook behind the submitted `figs/*.pdf`.
This is the single unseeded training run at
`TAG = sam_autoencoder_1x_64_32_16_8_4_50epochs_cropped`; it is not a member of any
ensemble and its weights cannot be reproduced, so the saved `.keras` model and `.npy`
arrays on scratch are the authoritative inputs.

Draws **Figs S1, S4, S15, S14, S13 and 8**.  These six keep their own `_polar_map`
helper rather than the Part 0 style layer, because that is what rendered the submitted
files.
""")

md(r"""
### Imports, paths and helpers (`figures_and_analysis.ipynb` cells 2, 4, 6)

One substantive change: the colour-bar label.  Three labels were in circulation --
`figures_and_analysis.ipynb` renders `MSL (std)`, the submitted `figs/*.pdf` render
`MSLP (std)`, and every `final-figs/*.pdf` renders `MSLP anomaly (std)`.  The plotted
fields have both the climatology and a linear trend removed, so the first two are wrong
twice over; `PAPER_CBAR_LABEL` below sets the third everywhere and makes all 23 figures
agree.  **This changes the rendered label of Figs 8 and S14 relative to the submitted
PDFs** and nothing else about them.
""")
code(FAC[2])
code(FAC[4] + '\n'
     '# The colour-bar label for every panel drawn in this part.  See the note above:\n'
     '# these are anomalies with a linear trend removed, and the rest of the figure set\n'
     '# already says so.\n'
     'PAPER_CBAR_LABEL = "MSLP anomaly (std)"\n')
# Drop plot_cluster_summary: nothing here calls it, and it writes the unreferenced
# figs/composites.pdf.
code(sub(cut_from(FAC[6], "def plot_cluster_summary("),
         "def _polar_map(ax, lons, lats, img, title, cmap='bwr', label='MSL (std)',",
         "def _polar_map(ax, lons, lats, img, title, cmap='bwr', label=PAPER_CBAR_LABEL,"))

md("### Load the trained model and its arrays (cells 9, 11)")
code(FAC[9])
code(FAC[11])

md(r"""
### Figure S1 -- original fields, reconstructions and latent representations

Cell 15, with the one change flagged in the header: the unseeded
`np.random.default_rng()` becomes `np.random.default_rng(RECON_SEED)` so the three
example months are reproducible.
""")
_s1 = sub(FAC[15], "rng = np.random.default_rng()",
          "rng = np.random.default_rng(RECON_SEED)")
_s1 = sub(_s1,
          "save_dir = 'figures'\nplt.savefig(os.path.join(save_dir, 'autoencoder_reconstruction.pdf'), dpi=300)\nplt.show()",
          'save_figure(fig, "autoencoder_reconstruction")\nplt.show()')
code(_s1)

md("### Figure S4 -- Ward linkage-distance elbow for the CAE latent space (cell 19)")
code(sub(FAC[19],
         "    plt.savefig(os.path.join('figures', 'elbow.pdf'), dpi=300, bbox_inches='tight')",
         '    save_figure(fig, "elbow")'))

md(r"""
### Figures S15 and S14 -- the principal-component baseline

Cells 27 and 28 cluster the full standardized PC score matrix (`sam_pca_data_all.nc`,
all modes) with the same Ward linkage, as the reduced-PCA control for the CAE.
""")
code(sub(FAC[27],
         'plt.savefig(os.path.join("figures", "pca_elbow.pdf"), dpi=300, bbox_inches="tight")',
         'save_figure(fig, "pca_elbow")'))
_s14 = sub(FAC[28],
           '    plt.savefig(os.path.join("figures", "pca_composites.pdf"), dpi=300, bbox_inches="tight")',
           '    save_figure(fig, "pca_composites")')
code(sub(_s14, 'def plot_pca_cluster_summary(cluster_data, label="MSL (std)"):',
         'def plot_pca_cluster_summary(cluster_data, label=PAPER_CBAR_LABEL):'))

md(r"""
### The four per-season CAE runs

Cell 43 loads the seasonal models (`..._lintrend_{DJF,MAM,JJA,SON}`) and **overwrites**
`lats_coarse` / `lons_coarse` with the seasonal grid, which is why it comes after the
annual figures above.
""")
code(FAC[43])

md("### Figure S13 -- per-season elbow diagnostics (cell 47)")
code(sub(FAC[47],
         "save_dir = 'figures'\nplt.savefig(os.path.join(save_dir, 'season_linkages.pdf'), dpi=300)",
         'save_figure(fig, "season_linkages")'))

md(r"""
### Figure 8 -- seasonal EOF1 and CAE composites

Cells 49, 51 and 53.  Cluster counts per season come from the elbow diagnostics above
(k = 4 for DJF and MAM, k = 3 for JJA and SON) and the per-panel colour limits are the
hand-tuned `SEASON_PLOT_KW` of the submitted figure.
""")
code(FAC[49])
code(FAC[51])
_f8 = sub(FAC[53],
          '    plt.savefig(\n        os.path.join("figures", "all_seasons_clusters.pdf"),\n        dpi=300,\n        bbox_inches="tight",\n    )',
          '    save_figure(fig, "all_seasons_clusters")')
code(sub(_f8, '    label="MSL (std)",\n', '    label=PAPER_CBAR_LABEL,\n'))

code('free("season_data", "season_cluster_data", "data_standardized", "encoded_all",\n'
     '     "x_train", "x_test", "autoencoder", "encoder", "x_pca_cmp",\n'
     '     "pca_cluster_data", "pca_features", "pca_vectors", "ds_pca_all")\n'
     'print("Part 2 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 3 -- the ERA5 seed ensemble
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 3 -- The ten-member ERA5 initialisation ensemble

Source: `era5_ensemble.ipynb` (cells `cell001`-`cell030`), extended with
`corr_ensemble.ipynb` for Fig S9 and `make_temporal_e12.py` for Fig 4.  Ten CAEs
trained on identical ERA5 input (516 months, 1980-2022) from
`sam_era5_autoencoder_seed{0..9}`, differing only in the random seed that sets both the
weight initialisation and the per-epoch minibatch order.

The pipeline is: cluster each member's latent space with Ward linkage at k = 3, pick the
reference member as the **ensemble medoid** (never hard-coded -- in the CESM2 ensemble
hard-coding picked the outlier), order the reference clusters by SAM index, then match
every member to it by Hungarian assignment on area-weighted spatial correlation.

Draws **Figs 3, 6, S16, 5, 7, S5, 2, S9 and 4** off that one clustering.
""")

md("### Imports, paths and helpers (`era5_ensemble.ipynb` cells `cell001`, `cell003`, `cell005`)")
# cell001 re-defines FIG_DIR/save_figure; Part 0 owns those now.
code(cut_from(E5["cell001"], "FIG_DIR = Path(os.environ.get(")
     + '\nprint("All imports OK")\n')
code(E5["cell003"])
code(E5["cell005"])

md(r"""
### Load the ten members, cluster, and align them

`cell007` holds one copy of the de-weighted physical field and checksums every member's
`data_standardized` against it, so the shared-array optimisation cannot silently
compare members trained on different input.  `cell008` chooses the medoid reference,
imposes the physical cluster order and asserts it.
""")
code(E5["cell007"])
code(E5["cell008"])

md("### Figure 3 -- occupancy across the ten initialisations (`cell010`)")
code(sub(E5["cell010"], 'save_figure(fig, "e12_era5seed_01_occupancy_box")',
         'save_figure(fig, "occupancy_box")'))

md(r"""
### EOFs, PCs and the non-annularity bootstrap

The compute half of `cell012`: loads `sam_pca_data.nc`, checks it lines up with the CAE
month for month, and bootstraps the non-annularity index eta within each cluster of each
member.  Its own diagnostic figure is not cited by the text and is not drawn.
""")
code(cut_from(E5["cell012"], "fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=130,"))

md(r"""
### Figure 6 -- non-annularity of EOF1 and the ensemble-mean composites (`etabars`)

EOF1 enters as a bootstrappable PC1-weighted composite rather than a single fixed
number, so it carries a confidence interval comparable with the clusters'.
""")
code(sub(E5["etabars"], 'save_figure(fig, "e12_era5seed_16_non_annularity_bars")',
         'save_figure(fig, "annularity")'))

md("### Figure S16 -- reduced-PC baselines against the CAE ensemble mean (`cell014`)")
code(sub(E5["cell014"], 'save_figure(fig, "e12_era5seed_03_pca_baseline_composites")',
         'save_figure(fig, "baseline_composites")'))

md("### Figure 5 -- the 516 months in PC1-PC2 space (`cell016`)")
code(sub(E5["cell016"], 'save_figure(fig, "e12_era5seed_04_pc1_pc2_scatter")',
         'save_figure(fig, "scatter")'))

md(r"""
### Transition matrices and their shuffled-label null

The compute half of `cell020`, which also defines the `heatmap` helper and tick labels
used by Figure 7.  Its own three-panel null-test figure duplicates Fig S7 (drawn in
Part 6 from the published run) and is not drawn.
""")
code(cut_from(E5["cell020"], "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)"))

md("### Figure 7 -- transition probability against marginal occupancy (`cell022`)")
code(sub(E5["cell022"], 'save_figure(fig, "e12_era5seed_07_transition_vs_marginal")',
         'save_figure(fig, "transition")'))

md(r"""
### Figure S5 -- composites for all ten initialisations (`cell024`)

Colour scales are fixed within a column, so a member whose Cluster 1 has collapsed or
reversed cannot be rescaled into looking like the rest.  `vmax_col` computed here is
reused by Figure 2.
""")
code(sub(E5["cell024"], 'save_figure(fig, "e12_era5seed_08_all_seed_composites")',
         'save_figure(fig, "ensemble")'))

md(r"""
### Figure 2 -- ensemble-mean composites with consensus stippling (`cell030`)

Per member and per cluster, a within-cluster bootstrap of the monthly fields gives a
90% interval at every grid point; stippling marks points significant in at least
`MIN_SEEDS` of the ten members.
""")
code(sub(E5["cell030"], 'save_figure(fig, "e12_era5seed_11_matched_composites")',
         'save_figure(fig, "ensemble_composite")'))

md(r"""
### Figure S9 -- ensemble-mean composites against the leading EOFs and PCs

Source: `corr_ensemble.ipynb` cells `cc011`, `cc013`, `cc015`, `cc019`, which extend the
same ensemble state.  `cc011` adds the published single run as a control -- it shares
`data_phys` with the ensemble, and the checksum assertion is what proves that is legal.
Correlations are computed on the native 281 x 1440 grid; the large number in each tile
is the ensemble-mean composite's correlation and the small one is the spread across the
ten members computed individually, which is deliberately not an error bar on it.
""")
code('# Modes shown on both axes, exactly as figures_and_analysis.ipynb cell 40: the\n'
     '# leading three.  "Intermediate" was retired in the revision; Cluster 1 is the\n'
     '# low-amplitude, weakly SAM-negative state.\n'
     'N_MODES = 3\n'
     'PLOT_NAMES = [r"Low-amplitude SAM$-$", r"SAM$+$", r"SAM$-$"]\n')
code(CE["cc011"])
code(CE["cc013"])
code(CE["cc015"])
code(CE["cc019"])   # already writes under the manuscript's own filename

md(r"""
### Figure 4 -- temporal evolution of the three regimes

Source: `make_temporal_e12.py` sections 4 and 7, lifted onto the ensemble state already
in memory (the script re-derives an identical pipeline, verified line by line against
`cell007` / `cell008`).

Each monthly anomaly field is projected onto each matched composite pattern, rows
poleward of 89 S excluded, then standardized and sign-aligned to EOF1 so the three
panels can be read against each other and against PC1.  The window is the 120-month
running mean of `figures_and_analysis.ipynb` cell 38.

The script's diagnostic sections 5, 6 and 8 -- turning-point location, hinge fits and
across-seed agreement -- are not needed for the figure and are omitted; run
`make_temporal_e12.py` for those numbers.
""")
_te4 = lines(TE, 359, 438)
_te7 = lines(TE, 744, 809)
# The script's own lower-case, ASCII-plus cluster names are replaced by the shared ones.
_te7 = sub(_te7, "save_figure(fig, FIG_NAME)", 'save_figure(fig, "temporal_proj")')
code('# Cluster colours for the four stacked panels; the shared `cluster_labels` from\n'
     '# `cell008` supply the names.  The ASCII copy keeps the printed sign table\n'
     '# aligned -- mathtext dollar signs are noise in a terminal.\n'
     'cluster_colors = ["steelblue", "tomato", "mediumseagreen"]\n'
     'cluster_labels_txt = ["Cluster 1 (Low-amplitude SAM-)", "Cluster 2 (SAM+)",\n'
     '                      "Cluster 3 (SAM-)"]\n\n'
     + _te4)
code(_te7)

code('free("data_phys", "seed_data", "B", "X_FLAT", "B_COARSE", "sig_maps",\n'
     '     "eta_boot", "null_all", "PROJ", "RM", "proj", "rm", "enc_p")\n'
     'print("Part 3 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 4 -- occupancy-matched PC1 controls
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 4 -- PC1-conditioned controls (Figs S10, S11, S12)

Source: `occupancy_matched.ipynb`.  The question these answer is whether the CAE classes
are anything more than PC1 thresholds: rank the 516 months by PC1 and cut them into three
contiguous groups with **exactly** the CAE occupancies, then compare composite for
composite.  The published single run is used, not the ensemble, so the comparison is
against a single set of labels.

The training latitude weight is undone per row with no clamp on the native 0.25 degree
grid (`AVG_COARSEN = 1`); the clamped `undo_cos_weight` in the style layer is
deliberately not used here, and the printed row-by-row RMS is the check.
""")
code(preamble(OM["om001"])
     + '\nprint(f"AVG_COARSEN = {AVG_COARSEN}   SHARED_PCT = {SHARED_PCT}")\n')
code(OM["om005"])
code(OM["om006"])
code(OM["om008"])
code(OM["om010"])
md("### Figure S10 -- PC1 distributions within the three CAE classes (`om011`)")
code(OM["om011"])
md("### Figure S11 -- occupancy-matched PC1 composites against the CAE composites")
code(OM["om013"])
code(OM["om014"])
code(OM["om016"])
md("### Figure S12 -- what the CAE adds over the PC1 ranking (`om018`, `om019`)")
code(OM["om018"])
code(OM["om019"])
code('free("data_phys", "cae_composites", "pc1_composites", "diff_composites")\n'
     'print("Part 4 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 5 -- common colour bar
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 5 -- Figure S6, the composites on one common colour bar

Source: `common_colorbar.ipynb`.  The manuscript's main composite figures scale each
panel independently to show spatial structure, which makes the amplitude difference
invisible; this is the same three composites on a single shared scale, whose only job is
to show how weak Cluster 1 is.

The one common bar carries 7 ticks at 1.6x the usual tick size -- a deliberate exception
to the collection's three-tick convention, made at the co-author's request, because the
bar spans 93% of the page width and is the figure's only bar.
""")
code(preamble(CB["cc001"])
     + '\nprint(f"AVG_COARSEN = {AVG_COARSEN}   SHARED_PCT = {SHARED_PCT}   '
       'CBAR_NTICKS = {CBAR_NTICKS}")\n')
code(CB["cc005"])
code(CB["cc006"])
code(CB["cc008"])
code(CB["cc010"])
code('free("fields", "cluster_data")\nprint("Part 5 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 6 -- transitions of the published run
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 6 -- Transition statistics of the published run (Figs S7, S8)

Figure S7 is `all_figures.ipynb` cells `e041`-`e049`: the full-record one-month
transition matrix of the published single run against a 1,000-draw shuffled-label null.
Figure S8 is `seasonal_transitions.ipynb`, the same statistics computed within each
season.  Both cluster the published run's latent space directly and need only
`encoded_all`, so they are cheap.
""")
md("### Figure S7 -- shuffled-label null test for the transition matrix")
code(AF["e041"])
code(drop_lines(AF["e043"], r"^FIG_DIR  = ", r"^os\.makedirs\(FIG_DIR"))
code(AF["e045"])
# e046 draws the unreferenced edit04_transition_vs_marginal figure (Fig 7 is the
# ensemble version); keep only the tick labels, font sizes and heatmap helper it
# defines, which e049 needs.
code(cut_from(AF["e046"], "heatmap(axes[0], trans_prob,"))
code(AF["e048"])
code(sub(AF["e049"],
         'plt.savefig(os.path.join(FIG_DIR, "edit04_null_test.pdf"), bbox_inches="tight")\nplt.show()\nprint("Figure saved.")',
         'save_figure(fig, "shuffled_null_test")\nplt.show()'))

md("### Figure S8 -- the same statistics within each season")
code(preamble(ST["st001"])
     + '\nprint(f"N_SHUFFLE = {N_SHUFFLE}  seed = {SHUFFLE_SEED}")\n')
code(ST["st005"])
code(ST["st007"])
code(ST["st009"])
code(ST["st011"])
code(ST["st013"])
code(ST["st015"])
code('free("labels", "pc1_std", "season_labels")\nprint("Part 6 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 7 -- architecture and latent size
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 7 -- Architecture and latent-size sensitivity (Figs S2, S3)

Figure S2 is `make_edit08_reconstruction_loss.py`: the training and validation loss of
the four tested architectures, read from the `summary_<TAG>.json` files that
`run_autoencoder.py` wrote.  The loss is **mean absolute error** in every training
script in this project (`loss="mean_absolute_error"`); the axis said "MSE" for a while
and that was only ever a mislabel, so the axis label here names MAE and its units.

Figure S3 is `latent_sweep.ipynb` cells `ls007`-`ls023`: 22 members spanning five latent
sizes from 405 to 6,480 scalars, all trained on identical input with the same 80/20
split, clustered and matched to the medoid of the published configuration exactly as in
Part 3.  Only the composite grid is cited by the text, so the notebook's reconstruction
error and cluster-validity panels -- and the forward passes through all 22 saved models
they need -- are omitted.
""")
md("### Figure S2 -- reconstruction loss by architecture and latent size")
_rl = script(RECON_LOSS)
_rl = _rl[_rl.index("SAVE_DIR = "):]
_rl = sub(_rl,
          'FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")\n',
          "")
_rl = sub(_rl,
          'out_path = os.path.join(FIG_DIR, "edit08_reconstruction_loss.pdf")\n'
          'fig.savefig(out_path, dpi=300, bbox_inches="tight")\n'
          'print(f"Saved \u2192 {out_path}")',
          'save_figure(fig, "reconstruction_loss")')
code(_rl)

md("### Figure S3 -- composites as a function of latent dimension")
code(LS["ls007"])
code(LS["ls009"])
code(LS["ls011"])
code(LS["ls013"])
code(LS["ls023"])   # already writes under the manuscript's own filename
code('free("models", "data_phys")\nprint("Part 7 arrays released")')

# ═════════════════════════════════════════════════════════════════════════════
# Part 8 -- verification
# ═════════════════════════════════════════════════════════════════════════════
md(r"""
---
## Part 8 -- Verification

Judge the run from this cell, not from the `.log`: `run_paper_figures.log` accumulates
across runs, so a stale traceback from an earlier attempt can sit above a clean one.

Every registered figure must exist, be a real PDF, and have been written by *this*
execution.
""")
code(r'''
missing, stale, bad = [], [], []
print(f"{'figure':<12} {'file':<52} {'kB':>8}  written")
print("-" * 92)
for name, (subdir, label) in FIGURES.items():
    path = OUT_ROOT / subdir / f"{name}.pdf"
    if not path.exists():
        missing.append(f"{label} ({subdir}/{name}.pdf)")
        continue
    fresh = path.stat().st_mtime >= RUN_STARTED
    if not fresh:
        stale.append(f"{label} ({subdir}/{name}.pdf)")
    with open(path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            bad.append(f"{label} ({subdir}/{name}.pdf)")
    print(f"{label:<12} {subdir + '/' + name + '.pdf':<52} "
          f"{path.stat().st_size / 1e3:>8,.0f}  {'yes' if fresh else 'STALE'}")

print()
assert not missing, f"{len(missing)} figure(s) never written: {missing}"
assert not bad, f"{len(bad)} output(s) are not PDFs: {bad}"
assert not stale, f"{len(stale)} figure(s) predate this run: {stale}"
print(f"All {len(FIGURES)} code-generated figures written to {OUT_ROOT} "
      f"in {(time.time() - RUN_STARTED) / 60:.1f} min.")
print("Figure 1 (final-figs/overview.pdf) is the hand-drawn schematic; copy it over.")
''')

# ── Write the notebook ────────────────────────────────────────────────────────
for i, cell in enumerate(CELLS):
    cell["id"] = f"pf{i:03d}"
    cell["source"] = cell["source"].split("\n")
    cell["source"] = [l + "\n" for l in cell["source"][:-1]] + [cell["source"][-1]]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")

n_code = sum(c["cell_type"] == "code" for c in CELLS)
print(f"wrote {OUT}: {len(CELLS)} cells ({n_code} code)")
