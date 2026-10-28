"""
Authors: Maia Posternack (maiaposternack@gmail.com), Kirstin Koepnick (kirstinkoepnick@g.harvard.edu)

Plotting and analysis helpers shared by figures.ipynb.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 26, 22, 18, 15
PCT = 85
CBAR_LABEL = "MSLP anomaly (std)"
LON_LABELS = {
    -150: "150°W", -120: "120°W", -90: "90°W", -60: "60°W", -30: "30°W",
    30: "30°E", 60: "60°E", 90: "90°E", 120: "120°E", 150: "150°E",
}
SAM_LAT_N = (-42.5, -37.5)
SAM_LAT_S = (-67.5, -62.5)


def fs_sizes(scale=1.0):
    return {"title": FS_TITLE * scale, "label": FS_LABEL * scale,
            "tick": FS_TICK * scale, "annot": FS_ANNOT * scale}


# ── maps ─────────────────────────────────────────────────────────────────────

def fix_polar_cap(img, lats, cap_lat=-89.0):
    """Fill the rows poleward of cap_lat with the zonal mean of the first row outside it.

    Only for the native 0.25 deg grid, where undoing the sqrt(cos) weight leaves a bullseye at the pole.
    """
    n = int((lats <= cap_lat).sum())
    img = img.copy()
    img[:n] = img[n].mean()
    return img


def polar_map(ax, lons, lats, img, title="", label=CBAR_LABEL, vmax=None, vmin=None,
              add_colorbar=True, edge_lat=None, lon_labels=True, pole_fix=False,
              fs=1.0, cbar_nticks=3, title_pad=60, contour_lw=1.5):
    """South polar stereographic composite map. Returns (cf, vmax) for shared colorbars."""
    pc = ccrs.PlateCarree()
    if edge_lat is None:
        edge_lat = float(np.max(lats))

    # circular boundary at the outer latitude
    theta = np.linspace(0, 2 * np.pi, 200)
    lons_b = np.rad2deg(theta) % 360
    lons_b[lons_b > 180] -= 360
    pts = ax.projection.transform_points(pc, lons_b, np.full(200, edge_lat))
    R = np.hypot(pts[:, 0], pts[:, 1]).max()
    ax.set_extent([-R, R, -R, R], crs=ax.projection)
    ax.set_boundary(mpath.Path(pts[:, :2]), transform=ax.projection)

    img = np.nan_to_num(np.asarray(img, dtype=float))
    if pole_fix:
        img = fix_polar_cap(img, lats)
    if vmax is None:
        vmax = np.percentile(np.abs(img), PCT)
    vmax = max(float(vmax), 1e-4)
    vmin = -vmax if vmin is None else vmin
    levels = np.linspace(vmin, vmax, 17)

    # cyclic wrapping because of the dateline
    img_c, lons_c = add_cyclic_point(img, coord=lons)
    cf = ax.contourf(lons_c, lats, img_c, levels=levels, transform=pc, cmap="bwr", extend="both")
    kw = dict(transform=pc, colors="black", linewidths=contour_lw, alpha=0.7)
    if (levels > 0).any():
        ax.contour(lons_c, lats, img_c, levels=levels[levels > 0], linestyles="solid", **kw)
    if (levels < 0).any():
        ax.contour(lons_c, lats, img_c, levels=levels[levels < 0], linestyles="dashed", **kw)

    ax.coastlines(linewidth=1.0, color="black")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=0.7, color="gray", alpha=0.5, linestyle="--")
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])

    # longitude labels around the perimeter
    if lon_labels:
        for lon, lbl in LON_LABELS.items():
            xy = ax.projection.transform_point(lon, edge_lat, pc)
            d = ax.transAxes.inverted().transform(ax.transData.transform(xy)) - 0.5
            ax.text(*(0.5 + 0.585 * d / np.linalg.norm(d)), lbl, transform=ax.transAxes,
                    ha="center", va="center", fontsize=25 * fs, color="dimgray",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, aspect=8)
        cbar.set_ticks(np.linspace(vmin, vmax, cbar_nticks))
        cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=30 * fs)
        cbar.set_label(label, fontsize=40 * fs, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=40 * fs, fontweight="bold", pad=title_pad * fs)
    return cf, vmax


def stipple(ax, lons, lats, mask, stride=(4, 20), size=3, ref_lat=-60.0):
    """Significance dots on a polar map.

    The longitude stride is the one at ref_lat and widens poleward as 1/cos(lat),
    otherwise the dots merge into rings and a black bullseye at the pole.
    Alternate rows are offset by half a step.
    """
    s_lat, s_lon = stride
    n_lon = lons.size
    cos_ref = np.cos(np.deg2rad(ref_lat))
    lon_pts, lat_pts = [], []
    for row, i in enumerate(range(0, lats.size, s_lat)):
        cos_lat = max(np.cos(np.deg2rad(lats[i])), s_lon * cos_ref / n_lon)
        step = min(n_lon, max(s_lon, int(round(s_lon * cos_ref / cos_lat))))
        cols = np.arange((row % 2) * (step // 2), n_lon, step)
        sel = mask[i, cols]
        lon_pts.append(lons[cols][sel])
        lat_pts.append(np.full(int(sel.sum()), lats[i]))
    ax.scatter(np.concatenate(lon_pts), np.concatenate(lat_pts), s=size, color="black",
               marker=".", linewidths=0, transform=ccrs.PlateCarree(), zorder=6)


# ── small panels ─────────────────────────────────────────────────────────────

def elbow(ax, Z, color, ms=8):
    """Ward linkage distance of the last 20 merges."""
    k = range(1, 21)
    ax.plot(k, Z[-20:, 2][::-1], marker="o", lw=1.5, markersize=ms, color=color)
    ax.set_xticks(list(k))
    ax.grid(True, alpha=0.3, linestyle="--")


def threshold_color(vmin, vmax):
    return lambda v: "black" if abs(v - vmin) < (vmax - vmin) * 0.6 else "white"


def luminance_color(cmap, vmin, vmax):
    """White text on dark cells. Right for diverging maps too, where the middle is palest."""
    def color(v):
        r, g, b, _ = plt.get_cmap(cmap)((v - vmin) / max(vmax - vmin, 1e-9))
        return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
    return color


def heatmap(ax, mat, title, cmap, vmin, vmax, ticks, fs, fmt=".3f", stars=None,
            text_color=None, xlabel="Destination cluster", ylabel="Origin cluster"):
    """Annotated transition matrix."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=fs["tick"] * 0.8)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=fs["tick"])
    ax.set_yticks(range(len(ticks)))
    ax.set_yticklabels(ticks, fontsize=fs["tick"])
    ax.set_xlabel(xlabel, fontsize=fs["label"])
    ax.set_ylabel(ylabel, fontsize=fs["label"])
    ax.set_title(title, fontsize=fs["title"], fontweight="bold")
    color = text_color or threshold_color(vmin, vmax)
    for (i, j), v in np.ndenumerate(mat):
        star = " *" if stars is not None and stars[i, j] else ""
        ax.text(j, i, f"{v:{fmt}}{star}", ha="center", va="center",
                fontsize=fs["annot"], fontweight="bold", color=color(v))


# ── clustering ───────────────────────────────────────────────────────────────

def ward(x):
    """Flatten, standardize, Ward linkage."""
    return linkage(StandardScaler().fit_transform(x.reshape(len(x), -1)), method="ward")


def ward_labels(x, k):
    """Ward clusters cut at k, relabelled so 0 is the largest."""
    raw = fcluster(ward(x), k, criterion="maxclust") - 1
    order = np.argsort(-np.bincount(raw, minlength=k), kind="stable")
    return np.argsort(order)[raw]


def match_clusters(composites, ref_composites, w):
    """Hungarian assignment on spatial correlation. perm[i] is the cluster matching reference i."""
    k = len(ref_composites)
    corr = np.array([[spatial_corr(ref_composites[i], composites[j], w) for j in range(k)]
                     for i in range(k)])
    rows, cols = linear_sum_assignment(-corr)
    return cols[np.argsort(rows)], corr


def physical_order(composites, lats):
    """Cluster 1 = smallest |SAM index| (low-amplitude SAM-), 2 = SAM+, 3 = SAM-."""
    sam = [sam_index(c, lats) for c in composites]
    weak = int(np.argmin(np.abs(sam)))
    rest = [j for j in range(len(sam)) if j != weak]
    return [weak, max(rest, key=sam.__getitem__), min(rest, key=sam.__getitem__)]


# ── fields ───────────────────────────────────────────────────────────────────

def coarsen(img, f):
    # coarsens the image so faster to plot
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def undo_lat_weight(data_std, lats):
    """(n, lat, lon, 1) training input -> (n, lat, lon), dividing out sqrt(cos/mean cos).

    No clamp at the pole: the stored value there is itself ~1e-8, so the inversion is exact.
    """
    cos = np.cos(np.deg2rad(lats))
    return (data_std[..., 0] / np.sqrt(cos / cos.mean())[:, None]).astype(np.float32)


def spatial_corr(a, b, w):
    """Area-weighted Pearson correlation of two fields; w broadcasts against them."""
    w = np.broadcast_to(w, a.shape)
    a = a - np.sum(a * w) / np.sum(w)
    b = b - np.sum(b * w) / np.sum(w)
    return float(np.sum(a * b * w) / np.sqrt(np.sum(a ** 2 * w) * np.sum(b ** 2 * w)))


def sam_index(field, lats):
    """Area-weighted zonal-mean anomaly at 40S minus 65S. Positive => SAM+."""
    w = np.cos(np.deg2rad(lats))
    out = []
    for lo, hi in (SAM_LAT_N, SAM_LAT_S):
        m = (lats >= lo) & (lats <= hi)
        out.append(np.sum(field[m].mean(axis=1) * w[m]) / np.sum(w[m]))
    return float(out[0] - out[1])


def area_weighted_rms(field, lats):
    w = np.broadcast_to(np.cos(np.deg2rad(lats))[:, None], field.shape)
    return float(np.sqrt(np.sum(w * field ** 2) / np.sum(w)))


def non_annularity(field, cos_lat):
    """eta = weighted variance of the zonally asymmetric part / weighted total variance."""
    asym = field - field.mean(axis=-1, keepdims=True)
    return float(np.sum(asym ** 2 * cos_lat[:, None]) / np.sum(field ** 2 * cos_lat[:, None]))


# ── transitions ──────────────────────────────────────────────────────────────

def transition_matrix(labels, k=3, pairs=None):
    """P(j at t+1 | i at t), optionally over a subset of the (t, t+1) pairs."""
    idx = labels[:-1] * k + labels[1:]
    if pairs is not None:
        idx = idx[pairs]
    counts = np.bincount(idx, minlength=k * k).reshape(k, k)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def shuffle_null(labels, rng, n=1000, k=3):
    """Transition matrices of n shuffled label sequences: keeps occupancy, removes memory."""
    return np.array([transition_matrix(rng.permutation(labels), k) for _ in range(n)])


def p_values(obs, null):
    """One-sided: fraction of the null at least as far from its mean as obs."""
    return np.where(obs >= null.mean(axis=0), (null >= obs).mean(axis=0), (null <= obs).mean(axis=0))
