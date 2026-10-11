"""Fig. 2 (respiration-EEG PAC): violin plots of whole-head coupling (row A) and cluster-test topoplots (row B).

Ported from analysis notebook cell 96 (``band_label``, ``plot_rowA_violin_wholehead`` and ``make_2row_figure`` are the
notebook's functions, extracted from its source) plus cells 84 (font sizes / rc settings, which cell 96 silently used) and
93-95 (``load_plot_info``: rescaling and rotation of the electrode montage for plotting).
"""
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels import make_dig_montage

try:                                     # private MNE helper used by the notebook; its location differs between versions
    from mne.channels.layout import _find_topomap_coords
except ImportError:                      # pragma: no cover
    from mne.viz.topomap import _find_topomap_coords

from .pac_stats import (BAND_COLORS, BANDS_ORDER, FREQ_BANDS, GROUP_P_SOURCE, MIN_CHANNELS_WHOLEHEAD,
                        N_PERMUTATIONS_CLUSTER, NULL_PER_SUBJECT_FOR_PLOT, ROWB_VLIM, compute_group_stats,
                        compute_subject_wholehead_stats_with_ids, p_to_stars, run_spatial_cluster_permutation_pac)

# Font sizes: defined in the habituation-figure cell (84) of the notebook and used by cell 96 without being defined there.
SMALLEST_SIZE, SMALL_SIZE, MED_SIZE, BIG_SIZE = 10, 12, 16, 20
RC = {"font.size": MED_SIZE, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
      "axes.titlesize": MED_SIZE, "axes.labelsize": MED_SIZE, "legend.fontsize": SMALLEST_SIZE,
      "figure.labelsize": BIG_SIZE, "figure.titlesize": BIG_SIZE, "xtick.labelsize": SMALLEST_SIZE,
      "ytick.labelsize": SMALLEST_SIZE}
RNG = np.random.default_rng(42)          # subsamples the null distributions drawn in the violins


def band_label(band):
    return {
        "theta": "Theta",
        "alpha": "Alpha",
        "beta": "Beta",
        "low_gamma": "Low gamma",
    }.get(band, band)


def plot_rowA_violin_wholehead(fig, gs_row, df_obs, null_by_subj_band, group_stats,
                                bands_order=BANDS_ORDER, colors=BAND_COLORS,
                                null_per_subject=1000):
    axes = []
    for i, band in enumerate(bands_order):
        ax = fig.add_subplot(gs_row[0, i])
        axes.append(ax)

        d = df_obs[df_obs['band'] == band].copy()
        if len(d) == 0:
            ax.set_title(f"{band_label(band)}\n(no data)", fontsize=MED_SIZE, fontweight='bold', color='black')
            ax.axis('off')
            continue

        z_obs = d['z'].to_numpy()

        pooled_null = []
        for _, row in d.iterrows():
            subj = row['subject_id']
            null = null_by_subj_band.get((subj, band), None)
            if null is None:
                continue
            mu = row['MI_null_mean']
            sd = row['MI_null_std']
            z_null = (null - mu) / sd

            if len(z_null) > null_per_subject:
                z_null = RNG.choice(z_null, size=null_per_subject, replace=False)
            pooled_null.append(z_null)

        pooled_null = np.concatenate(pooled_null) if len(pooled_null) else np.array([])

        # smaller violins
        parts = ax.violinplot([pooled_null, z_obs], positions=[1, 2],
                              widths=0.65, showmeans=False, showmedians=False, showextrema=False)

        # Null
        parts['bodies'][0].set_facecolor('#95A5A6')
        parts['bodies'][0].set_alpha(0.65)
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][0].set_linewidth(1.0)

        # Observed
        parts['bodies'][1].set_facecolor(colors.get(band, '#d62728'))
        parts['bodies'][1].set_alpha(0.85)
        parts['bodies'][1].set_edgecolor('black')
        parts['bodies'][1].set_linewidth(1.2)

        ax.boxplot([pooled_null, z_obs], positions=[1, 2], widths=0.22,
                   showfliers=False, patch_artist=False,
                   medianprops=dict(color='black', linewidth=2.0),
                   boxprops=dict(color='black', linewidth=1.2),
                   whiskerprops=dict(color='black', linewidth=1.2),
                   capprops=dict(color='black', linewidth=1.2))

        ax.axhline(0, color='black', linewidth=1, alpha=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Null', 'Observed'], fontsize=SMALLEST_SIZE, fontweight='bold')

        lo, hi = FREQ_BANDS[band]
        ax.set_title(f"{band_label(band)}\n({lo}–{hi} Hz)", fontsize=MED_SIZE, fontweight='bold', color='black', pad=0)

        if i == 0:
            ax.set_ylabel('MI z-score', fontsize=MED_SIZE, fontweight='bold')
            # Use figure coordinates for consistent placement
            fig.text(0.02, 0.95, 'A', fontsize=BIG_SIZE, fontweight='bold',
                    va='top', ha='left')
            
        gs_band = group_stats[group_stats['band'] == band]
        if len(gs_band):
            gs_band = gs_band.iloc[0]
            stars = p_to_stars(gs_band['p_fdr'])
            yl0, yl1 = ax.get_ylim()
            yr = yl1 - yl0

            y = yl1 - 0.18 * yr     # bracket baseline (lower = more space to title)
            h = 0.05 * yr           # bracket height

            ax.plot([1, 1, 2, 2], [y, y+h, y+h, y], color='k', lw=1.6)
            ax.text(1.5, y+h+0.01*yr, stars, ha='center', va='bottom',
                    fontsize=MED_SIZE, fontweight='bold')

    return axes


def make_2row_figure(pac_results_humidity, full_respiration_EEG_humidity, subject_info):
    """
    UPDATED layout + labeling:
    1) A/B/C letters consistent size/location; remove duplicates in row titles
    2) More space between suptitle and Row A; Row A titles show band + freq on 2 lines
    3) Row B: show cluster listing as "Cluster 1: p=..., n=..." etc (or "No significant clusters")
    4) Smaller Row A + tighter figure; reduce spacing between B topoplots
    Analysis logic unchanged.
    """

    # --- Run clusters using YOUR exact function
    all_cluster_results = {}
    for band in BANDS_ORDER:
        all_cluster_results[band] = run_spatial_cluster_permutation_pac(
            pac_results_humidity,
            full_respiration_EEG_humidity,
            band,
            n_permutations=N_PERMUTATIONS_CLUSTER,
            threshold=None
        )

    # --- Whole-head stats
    df_obs, null_by_subj_band = compute_subject_wholehead_stats_with_ids(
        pac_results_humidity, bands_order=BANDS_ORDER, min_channels=MIN_CHANNELS_WHOLEHEAD
    )
    group_stats = compute_group_stats(df_obs, bands_order=BANDS_ORDER, p_source=GROUP_P_SOURCE)

    # --- Create figure (smaller width; tighter; more room above Row A)
    fig = plt.figure(figsize=(14.5, 7.0))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[0.75, 1.00],  # Row A smaller
        hspace=0.45, wspace=0.02,    # tighter spacing between topoplots
        left=0.05, right=0.86, top=0.85, bottom=0.06
    )
    gsA = gs[0, :].subgridspec(1, 4, wspace=0.18)  # increase/decrease wspace as needed

    # Row A: violins
    rowA_axes = plot_rowA_violin_wholehead(fig, gsA, df_obs, null_by_subj_band, group_stats,
        bands_order=BANDS_ORDER, colors=BAND_COLORS,
        null_per_subject=NULL_PER_SUBJECT_FOR_PLOT
    )

    # -------------------------
    # Row B: topomaps
    # -------------------------

    rowB_axes = []
    for band_idx, band in enumerate(BANDS_ORDER):
        ax = fig.add_subplot(gs[1, band_idx])
        rowB_axes.append(ax)

        cluster_result = all_cluster_results[band]
        valid_channels = cluster_result['valid_channels']
        T_obs = cluster_result['T_obs']

        matching_channels = [ch for ch in valid_channels if ch in subject_info.ch_names]
        matching_ch_indices = [subject_info.ch_names.index(ch) for ch in matching_channels]
        info_subset = mne.pick_info(subject_info, matching_ch_indices)

        channel_to_idx = {ch: i for i, ch in enumerate(valid_channels)}
        t_obs_indices = [channel_to_idx[ch] for ch in matching_channels]
        T_obs_matched = T_obs[t_obs_indices]

        sig_mask = np.zeros(len(matching_channels), dtype=bool)
        for sig_cluster in cluster_result['significant_clusters']:
            for ch_name in sig_cluster['channels']:
                if ch_name in matching_channels:
                    sig_mask[matching_channels.index(ch_name)] = True

        mne.viz.plot_topomap(
            T_obs_matched,
            info_subset,
            mask=sig_mask,
            mask_params=dict(marker='o', markerfacecolor='none',
                             linewidth=2.5, markersize=11, linestyle='None'),
            axes=ax,
            show=False,
            cmap='bwr',
            vlim=ROWB_VLIM,
            contours=0,
            sensors=False,#True,
            names=None,
            sphere=0.13#0.095
        )

        # # electrode indices (unchanged)
        pos = _find_topomap_coords(info_subset, picks=None)
        for i, (x, y) in enumerate(pos):
            ax.text(x, y, str(i + 1), ha='center', va='center', fontsize=7, color='k')
        # pos = _find_topomap_coords(info_subset, picks=None)

        # label_spread = 1.1  # >1 pushes labels outward from center; tune this
        # for i, (x, y) in enumerate(pos):
        #     lx, ly = x * label_spread, y * label_spread
        #     ax.text(lx, ly, str(i + 1), ha='center', va='center',
        #             fontsize=7, color='k')

        # Title (UPDATED): cluster listing as Cluster 1/2/...; no band duplication
        n_sig = len(cluster_result['significant_clusters'])
        if n_sig == 0:
            xlab = "No significant clusters"
        else:
            lines = []
            for j, c in enumerate(cluster_result['significant_clusters'], start=1):
                lines.append(f"Cluster {j}: p={c['p_value']:.3f}, n={c['n_channels']}")
            xlab = "\n".join(lines)

        ax.set_xlabel(xlab, fontsize=SMALLEST_SIZE, fontweight='bold', color='black')

        if band_idx == 0:
            # Use figure coordinates for consistent placement
            fig.text(0.02, 0.48, 'B', fontsize=BIG_SIZE, fontweight='bold',
                    va='top', ha='left')

    # Column headers for Row B (same as Row A columns)
    col_names = [band_label(b) for b in BANDS_ORDER]
    for ax, name in zip(rowB_axes, col_names):
        ax.text(
            0.5, 1.0, name, transform=ax.transAxes,  # try 1.06–1.12
            ha='center', va='bottom',
            fontsize=MED_SIZE, fontweight='bold', color='black'
        )

    # Row B colorbar spanning row B only
    y0 = min(ax.get_position().y0 for ax in rowB_axes)
    y1 = max(ax.get_position().y1 for ax in rowB_axes)
    cbar_ax = fig.add_axes([0.87, y0, 0.02, (y1 - y0)])
    norm = mcolors.Normalize(vmin=ROWB_VLIM[0], vmax=ROWB_VLIM[1])
    sm = cm.ScalarMappable(norm=norm, cmap='bwr')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('T-statistic\n(PAC strength)', fontsize=MED_SIZE, fontweight='bold',
                   rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=SMALLEST_SIZE)

    # Row title for B (single, no duplicate "B" in title text)
    x0b = min(ax.get_position().x0 for ax in rowB_axes)
    x1b = max(ax.get_position().x1 for ax in rowB_axes)
    fig.text((x0b + x1b) / 2, y1 + 0.03, "2D Topoplots",
             ha='center', va='bottom', fontsize=BIG_SIZE, fontweight='bold', color='black')

    # Suptitle (more space above Row A thanks to top=0.89)
    fig.suptitle(
        'Respiration–EEG Phase–Amplitude Coupling',
        fontsize=BIG_SIZE, fontweight='bold', y=0.97
    )

    plt.tight_layout(rect=[0.0, 0.03, 0.88, 0.93])

    return fig, all_cluster_results, df_obs, group_stats


def build_fig2(pac_results, recordings, subject_info):
    """Run the statistics and draw Fig. 2.  Returns (fig, cluster_results, df_obs, group_stats).

    recordings: anything exposing ['EEG'][subject].ch_names/.info and ['metadata'][subject]['bad_channels'] (a full
    recordings dict or the light metadata extract made by recordings.make_recordings_meta).
    """
    global RNG
    RNG = np.random.default_rng(42)       # the notebook created it fresh when the cell was run
    with plt.rc_context(RC):
        out = make_2row_figure(pac_results, recordings, subject_info)
        out[0].canvas.draw()
    return out


def load_plot_info(info_fif, reference_radius=0.00991496, target_radius=0.095, rotate_deg=90):
    """MNE Info with the electrode montage rescaled and rotated for topoplots (notebook cells 93-95, final state).

    The stored montage has a head radius of ~0.0099 m (units issue), so all positions (and fiducials) are scaled by
    target_radius / reference_radius (= 0.095 / 0.00991496) and rotated about z by 90 degrees.
    """
    info = mne.io.read_info(info_fif, verbose=False)
    pos = info.get_montage().get_positions()
    scale = target_radius / reference_radius
    theta = np.deg2rad(rotate_deg)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    ch_pos = {ch: Rz @ (p * scale) for ch, p in pos["ch_pos"].items()}
    fid = {k: (Rz @ (pos[k] * scale) if pos[k] is not None else None) for k in ("nasion", "lpa", "rpa")}
    info.set_montage(make_dig_montage(ch_pos=ch_pos, nasion=fid["nasion"], lpa=fid["lpa"], rpa=fid["rpa"],
                                      coord_frame="head"))
    return info
