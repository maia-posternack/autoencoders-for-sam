import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from cartopy.util import add_cyclic_point

def _coarsen_img(img, factor):
    if factor <= 1:
        return img
    h, w = img.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def build_cluster_data(encoded_imgs, x_original_data, event_lats_padded, event_lons_padded,
                       k=4, top_n=None, max_instances_per_cluster=10, avg_coarsen=4,
                       linkage_method='ward'):

    latent_vectors = encoded_imgs.reshape((encoded_imgs.shape[0], -1))
    latent_vectors = StandardScaler().fit_transform(latent_vectors)

    Z = linkage(latent_vectors, method=linkage_method)
    cluster_labels = fcluster(Z, k, criterion='maxclust') - 1
    total_events = len(cluster_labels)

    cluster_data = []
    for cluster_id in range(k):
        indices = np.where(cluster_labels == cluster_id)[0]
        percent = 100 * len(indices) / total_events
        original_imgs = [x_original_data[i, :, :, 0] for i in indices]
        avg_original = np.mean([_coarsen_img(img, avg_coarsen) for img in original_imgs], axis=0)

        first_idx = indices[0]
        # avg_lons = event_lons_padded[first_idx][::avg_coarsen][:avg_original.shape[1]]
        # avg_lats = event_lats_padded[first_idx][::avg_coarsen][:avg_original.shape[0]]
        lats = np.asarray(event_lats_padded[first_idx])
        lons = np.asarray(event_lons_padded[first_idx])
        
        nlat = avg_original.shape[0] * avg_coarsen
        nlon = avg_original.shape[1] * avg_coarsen
        
        avg_lats = (lats[:nlat].reshape(avg_original.shape[0], avg_coarsen).mean(axis=1))
        avg_lons = (lons[:nlon].reshape(avg_original.shape[1], avg_coarsen).mean(axis=1))

        entry = {
            'cluster_id': cluster_id,
            'indices': indices,
            'indices_to_show': indices[:max_instances_per_cluster],
            'percent': percent,
            'size': len(indices),
            'avg_original': avg_original,
            'avg_lons': avg_lons,
            'avg_lats': avg_lats,
            'original_imgs': original_imgs[:max_instances_per_cluster],
        }
        cluster_data.append(entry)

    cluster_data.sort(key=lambda x: x['percent'], reverse=True)
    if top_n is not None:
        cluster_data = cluster_data[:top_n]

    for rank, ci in enumerate(cluster_data):
        ci['cluster_id'] = rank

    return cluster_data, cluster_labels, Z

def _polar_map(ax, lons, lats, img, title, cmap='bwr', label='MSL (std)',
               vmax=None, vmin=None, levels=None, add_colorbar=True):

    pc = ccrs.PlateCarree()
    edge_lat = float(np.max(lats))
    theta = np.linspace(0, 2 * np.pi, 200)
    lons_b = np.rad2deg(theta) % 360
    lons_b[lons_b > 180] -= 360
    lats_b = np.full(200, edge_lat)
    pts_b = ax.projection.transform_points(pc, lons_b, lats_b)
    R = np.sqrt(pts_b[:, 0] ** 2 + pts_b[:, 1] ** 2).max()
    ax.set_extent([-R, R, -R, R], crs=ax.projection)
    circle = mpath.Path(pts_b[:, :2])
    ax.set_boundary(circle, transform=ax.projection)

    img = np.nan_to_num(img, nan=0.0)
    if vmax is None:
        vmax = np.percentile(np.abs(img), 85)
    if vmax < 0.0001:
        vmax = 0.0001
    _vmin = -vmax if vmin is None else vmin
    if levels is None:
        levels = np.linspace(_vmin, vmax, 17)

    img_c, lons_c = add_cyclic_point(img, coord=lons)

    cf = ax.contourf(lons_c, lats, img_c, levels=levels, transform=pc,
                     cmap=cmap, extend='both')

    pos_levels = levels[levels > 0]
    neg_levels = levels[levels < 0]
    contour_kw = dict(transform=pc, colors='black', linewidths=1.5, alpha=0.7)
    if len(pos_levels):
        ax.contour(lons_c, lats, img_c, levels=pos_levels, linestyles='solid', **contour_kw)
    if len(neg_levels):
        ax.contour(lons_c, lats, img_c, levels=neg_levels, linestyles='dashed', **contour_kw)

    ax.coastlines(linewidth=1.0, color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)

    gl = ax.gridlines(crs=pc, draw_labels=False,
                      linewidth=0.7, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])

    lon_labels = {
        -150: '150°W', -120: '120°W', -90: '90°W',
        -60: '60°W', -30: '30°W',
        30: '30°E', 60: '60°E', 90: '90°E',
        120: '120°E', 150: '150°E',
    }
    center = np.array([0.5, 0.5])
    for lon_val, lbl in lon_labels.items():
        xy_data = ax.projection.transform_point(lon_val, -20., pc)
        if not np.all(np.isfinite(xy_data)):
            continue
        xy_disp = ax.transData.transform(xy_data)
        xy_ax = ax.transAxes.inverted().transform(xy_disp)
        direction = xy_ax - center
        norm_d = np.linalg.norm(direction)
        if norm_d < 1e-6:
            continue
        label_pos = center + (direction / norm_d) * 0.585
        ax.text(*label_pos, lbl, transform=ax.transAxes,
                ha='center', va='center', fontsize=25, color='dimgray',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, aspect=8)
        cbar.set_ticks(np.linspace(_vmin, vmax, 3))
        cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label, fontsize=22, fontweight='bold')

    ax.set_title(title, fontsize=28, fontweight='bold', pad=30)
    return cf, vmax


def plot_cluster_summary(cluster_data, label='MSLP (std)'):
    n_clusters = len(cluster_data)
    fig = plt.figure(figsize=(9 * n_clusters, 11))

    for col, ci in enumerate(cluster_data):
        ax = fig.add_subplot(1, n_clusters, col + 1, projection=ccrs.SouthPolarStereo())
        _polar_map(
            ax, ci['avg_lons'], ci['avg_lats'], ci['avg_original'],
            title=f'Cluster {ci["cluster_id"]+1}\n{ci["size"]} events ({ci["percent"]:.1f}%)',
            label=label
        )
        ax.set_title(f'({chr(ord("a")+col)}) ' + ax.get_title(),
                     fontsize=30, fontweight='bold', pad=40)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.11, wspace=0.55)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from cartopy.util import add_cyclic_point


def polar_map_general(
    ax,
    lons,
    lats,
    img,
    title="",
    cmap="RdBu_r",
    label="",
    vmax=None,
    vmin=None,
    add_colorbar=True,
    edge_lat=None,
    percentile=95,
    nlevels=17,
    contour=True,
):
    pc = ccrs.PlateCarree()

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    img = np.asarray(img)

    # Ensure latitude increasing for consistent plotting logic
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        img = img[::-1, :]

    if edge_lat is None:
        edge_lat = float(np.max(lats))

    # Build circular boundary from chosen outer latitude
    theta = np.linspace(0, 2 * np.pi, 200)
    lons_b = np.rad2deg(theta) % 360
    lons_b[lons_b > 180] -= 360
    lats_b = np.full(200, edge_lat)

    pts_b = ax.projection.transform_points(pc, lons_b, lats_b)
    good = np.isfinite(pts_b[:, 0]) & np.isfinite(pts_b[:, 1])
    pts_b = pts_b[good]

    if len(pts_b) == 0:
        raise ValueError("No finite boundary points found. Check latitude range passed to plot.")

    R = np.sqrt(pts_b[:, 0] ** 2 + pts_b[:, 1] ** 2).max()
    ax.set_extent([-R, R, -R, R], crs=ax.projection)
    circle = mpath.Path(pts_b[:, :2])
    ax.set_boundary(circle, transform=ax.projection)

    # Color limits
    img = np.nan_to_num(img, nan=0.0)

    if vmax is None:
        vmax = np.nanpercentile(np.abs(img), percentile)
    vmax = max(float(vmax), 1e-6)

    if vmin is None:
        vmin = -vmax

    levels = np.linspace(vmin, vmax, nlevels)

    # Add cyclic point for dateline
    img_c, lons_c = add_cyclic_point(img, coord=lons)

    cf = ax.contourf(
        lons_c, lats, img_c,
        levels=levels,
        transform=pc,
        cmap=cmap,
        extend="both"
    )

    if contour:
        pos_levels = levels[levels > 0]
        neg_levels = levels[levels < 0]
        contour_kw = dict(transform=pc, colors="black", linewidths=1.2, alpha=0.7)
        if len(pos_levels):
            ax.contour(lons_c, lats, img_c, levels=pos_levels, linestyles="solid", **contour_kw)
        if len(neg_levels):
            ax.contour(lons_c, lats, img_c, levels=neg_levels, linestyles="dashed", **contour_kw)

    ax.coastlines(linewidth=1.0, color="black")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)

    gl = ax.gridlines(
        crs=pc,
        draw_labels=False,
        linewidth=0.7,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])

    lon_labels = {
        -150: "150°W", -120: "120°W", -90: "90°W",
        -60: "60°W", -30: "30°W",
         30: "30°E",   60: "60°E",   90: "90°E",
        120: "120°E", 150: "150°E",
    }
    center = np.array([0.5, 0.5])
    for lon_val, lbl in lon_labels.items():
        xy_data = ax.projection.transform_point(lon_val, edge_lat, pc)
        if not np.all(np.isfinite(xy_data)):
            continue
        xy_disp = ax.transData.transform(xy_data)
        xy_ax = ax.transAxes.inverted().transform(xy_disp)
        direction = xy_ax - center
        norm_d = np.linalg.norm(direction)
        if norm_d < 1e-6:
            continue
        label_pos = center + (direction / norm_d) * 0.585
        ax.text(
            *label_pos, lbl,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=18, color="dimgray",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, aspect=8)
        cbar.set_ticks(np.linspace(vmin, vmax, 3))
        cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label, fontsize=16, fontweight="bold")

    ax.set_title(title, fontsize=22, fontweight="bold", pad=25)
    return cf, vmax


def plot_cluster_summary_general(
    cluster_data,
    label="",
    cmap="RdBu_r",
    figsize_per_panel=(8.5, 9),
    edge_lat=-20,
    percentile=95,
    vmax=None,
    vmin=None,
    contour=True,
):
    n_clusters = len(cluster_data)
    fig = plt.figure(figsize=(figsize_per_panel[0] * n_clusters, figsize_per_panel[1]))

    # Optional common color scale across all panels
    if vmax is None and vmin is None:
        all_vals = np.concatenate([
            np.ravel(np.asarray(ci["avg_original"])) for ci in cluster_data
        ])
        vmax_use = np.nanpercentile(np.abs(all_vals), percentile)
        vmax_use = max(float(vmax_use), 1e-6)
        vmin_use = -vmax_use
    else:
        vmax_use = vmax
        vmin_use = -vmax if vmin is None and vmax is not None else vmin

    for col, ci in enumerate(cluster_data):
        ax = fig.add_subplot(1, n_clusters, col + 1, projection=ccrs.SouthPolarStereo())

        polar_map_general(
            ax,
            ci["avg_lons"],
            ci["avg_lats"],
            ci["avg_original"],
            title=f'Cluster {ci["cluster_id"]+1}\n{ci["size"]} events ({ci["percent"]:.1f}%)',
            cmap=cmap,
            label=label,
            vmax=vmax_use,
            vmin=vmin_use,
            edge_lat=edge_lat,
            percentile=percentile,
            contour=contour,
        )

        ax.set_title(
            f'({chr(ord("a")+col)}) ' + ax.get_title(),
            fontsize=24,
            fontweight="bold",
            pad=30
        )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, wspace=0.45)
    plt.show()