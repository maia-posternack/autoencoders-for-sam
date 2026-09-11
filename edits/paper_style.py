"""Shared paper figure style.

Lifted verbatim out of `paper_figures.ipynb` cell pf004 so that stand-alone
scripts (e.g. `eof1_residual.py`) draw maps with exactly the same conventions
as the notebook figures.  The notebook imports these names from here, so there
is a single definition of the style.
"""

# ============================================================
# Paper figure style
#
# Mirrors figures_and_analysis.ipynb so the revision figures are visually
# interchangeable with the manuscript figures:
#
#   fields      bwr, symmetric levels from the 85th percentile of |field|,
#               17 levels, extend="both"
#   contours    black, solid where positive, dashed where negative
#   gridlines   gray dashed, longitude every 30 deg, latitude every 10 deg
#   colorbar    horizontal, 3 ticks (cbar_nticks) at %.2f
#
# The manuscript composite figure is three panels at 9 x 11 inches each, so its
# font sizes (title 40, colourbar label 40) only make sense at that scale.
# Every helper below therefore takes `fs`, a multiplier on the paper sizes:
# fs = 1.0 reproduces the manuscript exactly, and dense grids pass fs < 1.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# Font sizes used by the manuscript's non-map figures.
FS_TITLE = 26
FS_LABEL = 22
FS_TICK = 18
FS_ANNOT = 15

# Field convention.
PAPER_CMAP = "bwr"
PAPER_PCT = 85
PAPER_NLEVELS = 17

# Correlation-matrix convention (manuscript `_heatmap`).
PAPER_CORR_CMAP = "RdBu_r"

PAPER_LON_LABELS = {
    -150: "150°W", -120: "120°W", -90: "90°W",
    -60: "60°W", -30: "30°W",
    30: "30°E", 60: "60°E", 90: "90°E",
    120: "120°E", 150: "150°E",
}


def fix_polar_cap(img, lats, cap_lat=-89.0):
    """Fill rows poleward of `cap_lat` with the zonal mean of the first row equatorward.

    Undoing the sqrt(cos(lat)) training weight leaves the rows next to the pole
    amplified by up to 1e8, because cos(-90) is zero; even the `max(cos, 1e-10)`
    clamp used in the loading cells leaves the -90 row amplified 7.4e4-fold.
    That paints a bullseye over the pole and, worse, hijacks the percentile used
    to set the colour scale.  The cap is 0.02% of the hemisphere's area, so
    filling it costs nothing.

    Only applied on fine grids: if the latitude spacing is 1 degree or coarser,
    the first row is a wide band whose data is real, and it is left alone.
    """
    lats = np.asarray(lats, dtype=float)
    img = np.array(img, dtype=float, copy=True)

    if lats.size < 2:
        return img
    if abs(np.median(np.diff(lats))) >= 1.0:
        return img

    capped = lats <= cap_lat
    if not capped.any() or capped.all():
        return img

    first_good = int(capped.sum())
    if not np.all(capped[:first_good]):
        return img          # cap is not a contiguous run at the start

    img[:first_good, :] = np.nanmean(img[first_good, :])
    return img


def close_polar_cap(img, lats, pole=-90.0):
    """Extend a field to the pole so contourf leaves no hole at the centre.

    `functions.build_cluster_data` labels each coarsened row with the *mean* of
    its source latitudes, which is the right label for the data but means the
    innermost row of a 20-row coarsening sits at -87.625, not -90.  contourf
    cannot fill inside its innermost ring, so every panel gets a white disc over
    the pole.  A field at the pole is single valued, so the cap is filled with
    the zonal mean of the innermost ring.

    No-op when the grid already reaches the pole, and refuses to invent a cap
    wider than 1.5 grid cells so that a genuinely mid-latitude field is left
    alone.  `lats` must be ascending.
    """
    lats = np.asarray(lats, dtype=float)
    img = np.asarray(img, dtype=float)

    gap = lats[0] - pole
    if lats.size < 2 or gap <= 0:
        return img, lats
    if gap > 1.5 * abs(np.median(np.diff(lats))):
        return img, lats

    cap = np.full((1, img.shape[1]), np.nanmean(img[0, :]))
    return np.vstack([cap, img]), np.concatenate([[pole], lats])


def paper_polar_map(
    ax,
    lons,
    lats,
    img,
    title="",
    cmap=PAPER_CMAP,
    label="MSLP anomaly (std)",
    vmax=None,
    vmin=None,
    levels=None,
    add_colorbar=True,
    edge_lat=None,
    percentile=PAPER_PCT,
    nlevels=PAPER_NLEVELS,
    contour=True,
    lon_labels=True,
    gridlines=True,
    pole_fix=True,
    fs=1.0,
    cbar_fmt="%.2f",
    cbar_nticks=3,
    title_pad=60,
    contour_lw=1.5,
):
    """South polar stereographic composite map in the manuscript's style.

    Returns (cf, vmax) so callers can build shared colourbars.
    """
    pc = ccrs.PlateCarree()

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    img = np.asarray(img)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        img = img[::-1, :]

    if edge_lat is None:
        edge_lat = float(np.max(lats))

    # Circular boundary at the outer latitude
    theta = np.linspace(0, 2 * np.pi, 200)
    lons_b = np.rad2deg(theta) % 360
    lons_b[lons_b > 180] -= 360
    lats_b = np.full(200, edge_lat)

    pts_b = ax.projection.transform_points(pc, lons_b, lats_b)
    pts_b = pts_b[np.isfinite(pts_b[:, 0]) & np.isfinite(pts_b[:, 1])]
    if len(pts_b) == 0:
        raise ValueError("No finite boundary points; check the latitude range.")

    R = np.sqrt(pts_b[:, 0] ** 2 + pts_b[:, 1] ** 2).max()
    ax.set_extent([-R, R, -R, R], crs=ax.projection)
    ax.set_boundary(mpath.Path(pts_b[:, :2]), transform=ax.projection)

    img = np.nan_to_num(img, nan=0.0)
    if pole_fix:
        img = fix_polar_cap(img, lats)
        img, lats = close_polar_cap(img, lats)
    if vmax is None:
        vmax = np.percentile(np.abs(img), percentile)
    vmax = max(float(vmax), 1e-4)
    _vmin = -vmax if vmin is None else vmin
    if levels is None:
        levels = np.linspace(_vmin, vmax, nlevels)

    img_c, lons_c = add_cyclic_point(img, coord=lons)

    cf = ax.contourf(
        lons_c, lats, img_c,
        levels=levels, transform=pc, cmap=cmap, extend="both",
    )

    if contour:
        pos_levels = levels[levels > 0]
        neg_levels = levels[levels < 0]
        contour_kw = dict(transform=pc, colors="black", linewidths=contour_lw, alpha=0.7)
        if len(pos_levels):
            ax.contour(lons_c, lats, img_c, levels=pos_levels, linestyles="solid", **contour_kw)
        if len(neg_levels):
            ax.contour(lons_c, lats, img_c, levels=neg_levels, linestyles="dashed", **contour_kw)

    ax.coastlines(linewidth=1.0, color="black")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)

    if gridlines:
        gl = ax.gridlines(
            crs=pc, draw_labels=False,
            linewidth=0.7, color="gray", alpha=0.5, linestyle="--",
        )
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])

    if lon_labels:
        center = np.array([0.5, 0.5])
        for lon_val, lbl in PAPER_LON_LABELS.items():
            xy_data = ax.projection.transform_point(lon_val, edge_lat, pc)
            if not np.all(np.isfinite(xy_data)):
                continue
            xy_ax = ax.transAxes.inverted().transform(ax.transData.transform(xy_data))
            direction = xy_ax - center
            norm_d = np.linalg.norm(direction)
            if norm_d < 1e-6:
                continue
            label_pos = center + (direction / norm_d) * 0.585
            ax.text(
                *label_pos, lbl, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=25 * fs, color="dimgray",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, aspect=8)
        cbar.set_ticks(np.linspace(_vmin, vmax, cbar_nticks))
        cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter(cbar_fmt))
        cbar.ax.tick_params(labelsize=30 * fs)
        cbar.set_label(label, fontsize=40 * fs, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=40 * fs, fontweight="bold", pad=title_pad * fs)

    return cf, vmax


def paper_stipple(ax, lons, lats, mask, stride=(4, 20), size=3, color="black",
                  ref_lat=-60.0, stagger=True):
    """Overlay significance stippling on a polar map drawn by paper_polar_map.

    `stride` is (lat_stride, lon_stride) or a single int for both.  The latitude
    stride is used as given.  The longitude stride is the value *at `ref_lat`*
    and is widened poleward as cos(ref_lat) / cos(lat).

    Meridians converge, so a fixed longitude stride does not sample a polar
    stereographic map evenly.  On the 0.25 deg ERA5 grid, 20 columns is 5 deg of
    longitude everywhere, which is 278 km at 60 S but 96 km at 80 S, 9.7 km at
    89 S and zero at the pole.  The dots then merge into solid concentric rings
    and a black bullseye centred on the pole -- an artefact of the sampling that
    reads as an artefact of the data.  Widening the stride as 1 / cos(lat) holds
    the along-parallel spacing at its `ref_lat` value (278 km for the default 20
    columns) and collapses the polar row to a single dot.

    Equatorward of `ref_lat` the stride is left at its nominal value rather than
    narrowed, so the part of the map where the fixed stride was already sensible
    is unchanged.  `stagger` offsets alternate sampled rows by half a step,
    which breaks up the radial spokes that aligned rows produce.
    """
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    mask = np.asarray(mask, dtype=bool)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        mask = mask[::-1, :]

    s_lat, s_lon = (stride, stride) if np.isscalar(stride) else stride
    n_lon = lons.size
    cos_ref = np.cos(np.deg2rad(ref_lat))

    lon_pts, lat_pts = [], []
    for row, i in enumerate(range(0, lats.size, s_lat)):
        # Floor on cos(lat) so the pole asks for one dot rather than 1 / 0.
        cos_lat = max(np.cos(np.deg2rad(lats[i])), s_lon * cos_ref / n_lon)
        step = min(n_lon, max(s_lon, int(round(s_lon * cos_ref / cos_lat))))
        offset = (row % 2) * (step // 2) if stagger else 0
        cols = np.arange(offset, n_lon, step)
        sel = mask[i, cols]
        if sel.any():
            lon_pts.append(lons[cols][sel])
            lat_pts.append(np.full(int(sel.sum()), lats[i]))

    if not lon_pts:
        return

    ax.scatter(
        np.concatenate(lon_pts), np.concatenate(lat_pts),
        s=size, color=color, marker=".", linewidths=0,
        transform=ccrs.PlateCarree(), zorder=6,
    )


def paper_cluster_summary(cluster_data, label="MSLP anomaly (std)", vmax=None, vmin=None,
                          pole_fix=True):
    """The manuscript's three-panel composite figure (figures_and_analysis.ipynb)."""
    n_clusters = len(cluster_data)
    fig = plt.figure(figsize=(9 * n_clusters, 11))

    for col, ci in enumerate(cluster_data):
        ax = fig.add_subplot(1, n_clusters, col + 1, projection=ccrs.SouthPolarStereo())
        paper_polar_map(
            ax, ci["avg_lons"], ci["avg_lats"], ci["avg_original"],
            title=(
                f'({chr(ord("a") + col)}) Cluster {ci["cluster_id"] + 1}\n'
                f'{ci["size"]} events ({ci["percent"]:.1f}%)'
            ),
            label=label, vmax=vmax, vmin=vmin, pole_fix=pole_fix,
        )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.55)
    return fig


def coarsened_cos_weight(lats_fine, coarsen):
    """The sqrt(cos(lat)/mean(cos(lat))) training weight, coarsened like the data.

    `build_cluster_data` averages `coarsen` adjacent latitude rows, so the weight
    carried by a coarse row is the *mean* of the fine-row weights, not the weight
    evaluated at the coarse row's nominal latitude.  This distinction is what
    keeps the pole finite: the nominal latitude of the first coarse row is -90,
    where sqrt(cos) is 1.1e-8 and dividing by it amplifies the innermost ring by
    9e7, but the mean of the weight across that row's 20 source latitudes is
    O(0.3).
    """
    lats_fine = np.asarray(lats_fine, dtype=float)
    w_fine = np.cos(np.deg2rad(lats_fine))
    w_fine = np.sqrt(w_fine / w_fine.mean())

    if coarsen <= 1:
        return w_fine

    n_coarse = len(lats_fine) // coarsen
    return w_fine[: n_coarse * coarsen].reshape(n_coarse, coarsen).mean(axis=1)


def undo_cos_weight(field, lats):
    """Undo the sqrt(cos(lat)/mean(cos(lat))) training weight on a 2-D field.

    Only for display, so that every map in this notebook is in the same units
    as the manuscript composite figure.  On a 0.25 deg grid the weight at -90 is
    1.1e-8, so the row is clamped and then flattened by `fix_polar_cap` inside
    `paper_polar_map`; sections that need de-weighted *data* rather than a
    de-weighted picture do their own conversion.
    """
    lats = np.asarray(lats, dtype=float)
    cos_lat = np.cos(np.deg2rad(lats))
    weight = np.sqrt(np.maximum(cos_lat, 1e-10) / cos_lat.mean())

    field = np.asarray(field, dtype=float)
    if field.shape[0] != weight.size:
        raise ValueError(
            f"field has {field.shape[0]} latitudes, weight has {weight.size}"
        )
    return field / weight[:, np.newaxis]


def fs_sizes(scale=1.0):
    """Manuscript font sizes, scaled for a given panel density.

    scale = 1.0 suits a one- to three-panel figure, which is what the
    manuscript uses.  Denser grids pass a smaller scale.
    """
    return {
        "title": FS_TITLE * scale,
        "label": FS_LABEL * scale,
        "tick": FS_TICK * scale,
        "annot": FS_ANNOT * scale,
    }