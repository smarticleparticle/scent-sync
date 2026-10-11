"""Figure builders ported from the analysis notebook (plotting code unchanged; data come from the results objects).

  build_fig4_AB   notebook cell 29   Fig. 4A (selection-frequency heatmaps) + Fig. 4B (ROC curves)
  build_figS3     notebook cell 31   Fig. S3 (per-odor selection frequency) -- the notebook typed the numbers in by hand
  build_fig5      notebook cell 84   Fig. 5 (habituation: AUC early vs late, selection frequency early vs late)

Each builder sets its matplotlib rc parameters locally (``plt.rc_context``) instead of changing them globally,
and returns the Figure.  Fig. 4C/D (odor recognition) come from the word-embedding analysis (not ported yet).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.metrics import roc_curve

from . import classify as clf

SMALLEST_SIZE, SMALL_SIZE = 10, 12
ODORS = ["peppermint", "oregano", "citrus"]
DISPLAY = {"peppermint": "Peppermint", "oregano": "Oregano", "citrus": "Grapefruit"}
BANDS_ROWS = ["Beta", "Alpha", "Theta"]            # row order of the heatmaps (top to bottom)
BANDS_KEYS = ("beta", "alpha", "theta")


# ---------------------------------------------------------------------------------------------------------
# Fig. 4A/B   (notebook cell 29)
# ---------------------------------------------------------------------------------------------------------
def build_fig4_AB(per_odor, phase_bins, time_bins):
    """per_odor: {odor: {"phase", "time_2s", "time_378s": LOSO results}} (the classification stage's output).

    phase_bins / time_bins: ordered bin names of the 4-bin RPL and 3.78-s strategies (for the heatmap columns).
    """
    SMALLEST, SMALL, MED = SMALLEST_SIZE, SMALL_SIZE, 14
    rc = {"font.size": SMALL, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
          "axes.titlesize": SMALL, "axes.labelsize": SMALL, "legend.fontsize": SMALLEST,
          "xtick.labelsize": SMALLEST, "ytick.labelsize": SMALLEST}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(16, 3.5))
        gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 0.05, 0.9, 0.9, 0.9], height_ratios=[1, 0.05],
                      hspace=0.25, wspace=0.35, top=0.88, bottom=0.12)

        # ---- Panel A: heatmaps (mean over odors of the per-odor selection frequency, %)
        data_phase = np.mean([clf.selection_matrix(per_odor[o]["phase"], phase_bins, BANDS_KEYS) for o in ODORS], axis=0)
        data_time = np.mean([clf.selection_matrix(per_odor[o]["time_378s"], time_bins, BANDS_KEYS) for o in ODORS], axis=0)
        phase_labels = ["0-90°", "90-180°", "180-270°", "270-360°"]
        time_labels = ["0-0.945s", "0.945-1.89s", "1.89-2.835s", "2.835-3.78s"]
        datasets = [data_phase, data_time]
        titles = ["Respiratory Phase \nLocked ERSP", "Time-Locked ERSP"]
        xticks = [phase_labels, time_labels]

        cmap, norm = "OrRd", plt.Normalize(vmin=0, vmax=100)
        axes_heat = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        for idx, (ax, data, title, xtick) in enumerate(zip(axes_heat, datasets, titles, xticks)):
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect=0.7)
            ax.set_xticks(np.arange(len(xtick)))
            ax.set_xticklabels(xtick, rotation=45, ha="right", fontsize=SMALLEST - 1)
            ax.set_yticks(np.arange(len(BANDS_ROWS)))
            if idx == 0:
                ax.set_yticklabels(BANDS_ROWS, fontsize=SMALLEST)
            else:
                ax.set_yticklabels([])
            for i in range(len(BANDS_ROWS)):
                for j in range(4):
                    value = data[i, j]
                    if value > 0:
                        ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                                color="white" if value > 50 else "black", fontsize=SMALLEST - 2, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Frequency Band", fontsize=SMALLEST, fontweight="bold")
            ax.set_title(title, fontsize=SMALL, fontweight="bold", pad=10, loc="center")
            ax.set_xticks(np.arange(4) - .5, minor=True)
            ax.set_yticks(np.arange(len(BANDS_ROWS)) - .5, minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

        cbar_ax = fig.add_subplot(gs[1, 0:2])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Selection Frequency (%)", fontsize=SMALLEST, fontweight="bold", labelpad=3)
        cbar.ax.tick_params(labelsize=SMALLEST - 2, pad=1)

        # ---- Panel B: ROC curves
        odor_colors = {"Peppermint": "#2ECC71", "Oregano": "#9B59B6", "Citrus": "#E67E22"}
        color_time_2s, color_time_avg = "#7F8C8D", "#95A5A6"
        roc_axes = [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 5])]
        for i, (ax, odor) in enumerate(zip(roc_axes, ["Peppermint", "Oregano", "Citrus"])):
            r = per_odor[ODORS[i]]
            res_phase, res_time, res_time_avg = r["phase"], r["time_2s"], r["time_378s"]
            odor_color = odor_colors[odor]
            fpr_phase, tpr_phase, _ = roc_curve(res_phase["y_true"], res_phase["y_proba"])
            fpr_time, tpr_time, _ = roc_curve(res_time["y_true"], res_time["y_proba"])
            fpr_avg, tpr_avg, _ = roc_curve(res_time_avg["y_true"], res_time_avg["y_proba"])

            ax.plot(fpr_phase, tpr_phase, linewidth=2, color=odor_color,
                    label=f'RPL-ERSP ({res_phase["auc"]:.3f})', alpha=1, zorder=3)
            ax.fill_between(fpr_phase, tpr_phase, alpha=0.15, color=odor_color, zorder=1)
            ax.plot(fpr_time, tpr_time, linewidth=2, color=color_time_2s,
                    label=f'Traditional ERSP ({res_time["auc"]:.3f})', alpha=0.6, linestyle="--", zorder=2)
            ax.plot(fpr_avg, tpr_avg, linewidth=1.5, color=color_time_avg,
                    label=f'Longer ERSP ({res_time_avg["auc"]:.3f})', alpha=0.9, linestyle=":", zorder=1)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=SMALLEST, fontweight="bold")
            if i == 0:
                ax.set_ylabel("True Positive Rate", fontsize=SMALLEST, fontweight="bold")
            title = "Grapefruit vs Water" if odor == "Citrus" else f"{odor} vs Water"
            ax.set_title(title, fontsize=SMALL, fontweight="bold", color=odor_color, loc="center", pad=10)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=SMALLEST - 2, framealpha=0.95, ncol=1)
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=SMALLEST - 1)

        fig.suptitle("Feature Selection & Classification Performance", fontsize=MED, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.canvas.draw()
    return fig


# ---------------------------------------------------------------------------------------------------------
# Fig. S3   (notebook cell 31)
# ---------------------------------------------------------------------------------------------------------
def build_figS3(data_by_odor):
    """data_by_odor: {"Peppermint": 3x4 array (%), ...}; rows Beta/Alpha/Theta, columns the four phase bins."""
    SMALLEST, SMALL = SMALLEST_SIZE, SMALL_SIZE
    rc = {"font.size": SMALL, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
          "axes.titlesize": SMALL, "axes.labelsize": SMALL, "legend.fontsize": SMALLEST,
          "xtick.labelsize": SMALLEST, "ytick.labelsize": SMALLEST}
    phase_labels = ["0-90°", "90-180°", "180-270°", "270-360°"]
    with plt.rc_context(rc):
        cmap, norm = "OrRd", plt.Normalize(vmin=0, vmax=100)
        fig = plt.figure(figsize=(9, 3.2))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.06], hspace=0.3, wspace=0.25, top=0.85, bottom=0.15)
        axes_heat = [fig.add_subplot(gs[0, i]) for i in range(3)]
        for idx, (ax, (odor, data)) in enumerate(zip(axes_heat, data_by_odor.items())):
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect=0.7)
            ax.set_xticks(np.arange(len(phase_labels)))
            ax.set_xticklabels(phase_labels, rotation=45, ha="right", fontsize=SMALLEST - 1)
            ax.set_yticks(np.arange(len(BANDS_ROWS)))
            if idx == 0:
                ax.set_yticklabels(BANDS_ROWS, fontsize=SMALLEST)
                ax.set_ylabel("Frequency Band", fontsize=SMALLEST, fontweight="bold")
            else:
                ax.set_yticklabels([])
            for i in range(len(BANDS_ROWS)):
                for j in range(len(phase_labels)):
                    value = data[i, j]
                    ax.text(j, i, f"{value:.0f}%", ha="center", va="center",
                            color="white" if value > 50 else "black", fontsize=SMALLEST - 2, fontweight="bold")
            ax.set_title(odor, fontsize=SMALL, fontweight="bold", pad=10, loc="center")
            ax.set_xticks(np.arange(len(phase_labels)) - .5, minor=True)
            ax.set_yticks(np.arange(len(BANDS_ROWS)) - .5, minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        cbar_ax = fig.add_subplot(gs[1, :])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Selection Frequency (%)", fontsize=SMALLEST, fontweight="bold", labelpad=3)
        cbar.ax.tick_params(labelsize=SMALLEST - 2, pad=1)
        fig.canvas.draw()
    return fig


def figS3_data(per_odor, phase_bins):
    """Per-odor selection-frequency matrices rounded to whole percent (the notebook typed the rounded values)."""
    return {DISPLAY[o]: np.round(clf.selection_matrix(per_odor[o]["phase"], phase_bins, BANDS_KEYS)) for o in ODORS}


# ---------------------------------------------------------------------------------------------------------
# Fig. 5   (notebook cell 84)
# ---------------------------------------------------------------------------------------------------------
def build_fig5(per_odor, count_unselected_as_zero=False, robust_threshold=0.75):
    """per_odor: {odor: {"res_early", "res_late", "ci_early", "ci_late"}} from the habituation stage.

    Panel A plots the AUC of each block on ALL subjects with data (not only those with both blocks) with bootstrap
    95% CI whiskers.  Panel B shows the features with early mean selection >= robust_threshold; the averaging
    convention across odors is set by ``count_unselected_as_zero`` (False = published figure; see classify.mean_selection_frequency).
    """
    SMALLEST, SMALL, MED, BIG = 10, 12, 16, 20
    rc = {"font.size": MED, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
          "axes.titlesize": MED, "axes.labelsize": MED, "legend.fontsize": SMALLEST,
          "figure.labelsize": BIG, "figure.titlesize": BIG, "xtick.labelsize": SMALLEST, "ytick.labelsize": SMALLEST}
    colors_odor = {"peppermint": "#2ECC71", "oregano": "#9B59B6", "citrus": "#E67E22"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(14, 3.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.5, hspace=0.5)

        # ---- Panel A: classification AUC with bootstrap CI
        ax_a = fig.add_subplot(gs[0, 0])
        for odor in ODORS:
            r = per_odor[odor]
            first_auc, last_auc = r["res_early"]["auc"], r["res_late"]["auc"]
            x, y = [0, 1], [first_auc, last_auc]
            (lo_early, hi_early), (lo_late, hi_late) = r["ci_early"], r["ci_late"]
            yerr_lower = [first_auc - lo_early, last_auc - lo_late]
            yerr_upper = [hi_early - first_auc, hi_late - last_auc]
            ax_a.errorbar(x, y, yerr=[yerr_lower, yerr_upper], color=colors_odor[odor], capsize=6, capthick=2,
                          linewidth=2, zorder=2, fmt="none", alpha=0.6)
            ax_a.plot(x, y, "o-", color=colors_odor[odor], linewidth=4, markersize=12, alpha=0.85,
                      label=DISPLAY[odor], markeredgewidth=2, markeredgecolor="white", zorder=3)

        ax_a.axhline(0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5)
        ax_a.text(1.06, 0.5, "Chance", fontsize=SMALL, va="center", style="italic", color="gray")
        ax_a.set_xlim([-0.1, 1.15])
        ax_a.set_ylim([0.2, 0.9])
        ax_a.set_xticks([0, 1])
        ax_a.set_xticklabels(["Early Trials\n(1-3)", "Late Trials\n(6-8)"], fontsize=MED, fontweight="bold")
        ax_a.set_ylabel("Classification AUC\n(Odor vs Water)", fontsize=MED, fontweight="bold")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.grid(axis="y", alpha=0.3)
        ax_a.legend(loc="upper right", fontsize=SMALL, framealpha=0.95)
        ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_a.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_a.tick_params(which="major", direction="out", length=6, width=1.5, color="black", bottom=True, left=True)
        ax_a.tick_params(which="minor", direction="out", length=4, width=1, color="black", bottom=False, left=True)
        fig.text(0.02, 0.95, "A", fontsize=BIG, fontweight="bold", va="top")

        # ---- Panel B: feature selection, early vs late
        first_mean_sel = clf.mean_selection_frequency({o: per_odor[o]["res_early"] for o in ODORS}, count_unselected_as_zero)
        last_mean_sel = clf.mean_selection_frequency({o: per_odor[o]["res_late"] for o in ODORS}, count_unselected_as_zero)

        thr = robust_threshold
        robust_features = {f: v for f, v in first_mean_sel.items() if v >= thr}
        if len(robust_features) == 0:
            thr = 0.5
            robust_features = {f: v for f, v in first_mean_sel.items() if v >= thr}
        robust_sorted = sorted(robust_features.items(), key=lambda x: x[1], reverse=True)

        ax_b = fig.add_subplot(gs[0, 1])
        feature_names = [f for f, _ in robust_sorted]
        first_freqs = np.array([first_mean_sel[f] for f in feature_names])
        last_freqs = np.array([last_mean_sel.get(f, 0) for f in feature_names])
        declines = first_freqs - last_freqs

        def get_sort_key(idx):
            parts = feature_names[idx].split("_")
            if len(parts) >= 3:
                band_priority = {"theta": 0, "alpha": 1, "beta": 2, "gamma": 3}.get(parts[0], 99)
                return (band_priority, -declines[idx])
            return (99, -declines[idx])

        sort_idx = sorted(range(len(feature_names)), key=get_sort_key)
        feature_names = [feature_names[i] for i in sort_idx]
        first_freqs, last_freqs, declines = first_freqs[sort_idx], last_freqs[sort_idx], declines[sort_idx]

        def make_readable(feat):
            parts = feat.split("_")
            if len(parts) >= 3:
                phase_map = {"p0_90": "Early Inhale\n(0-90°)", "p90_180": "Late Inhale\n(90-180°)",
                             "p180_270": "Early Exhale\n(180-270°)", "p270_360": "Late Exhale\n(270-360°)"}
                phase_str = parts[1] + "_" + parts[2]
                return f"{parts[0].capitalize()} - {phase_map.get(phase_str, phase_str)}"
            return feat

        readable_names = [make_readable(f) for f in feature_names]
        y_pos = np.arange(len(feature_names))
        width = 0.35
        ax_b.barh(y_pos + width / 2, first_freqs, width, label="Early Trials", color="#3498DB",
                  alpha=0.85, edgecolor="#2980B9", linewidth=1.5)
        ax_b.barh(y_pos - width / 2, last_freqs, width, label="Late Trials", color="#AED6F1",
                  alpha=0.75, edgecolor="#85C1E9", linewidth=1.5)
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(readable_names, fontsize=SMALLEST)
        ax_b.set_xlabel("Selection Frequency", fontsize=MED, fontweight="bold")
        ax_b.set_xlim([0, 1.05])
        ax_b.grid(axis="x", alpha=0.3, linestyle=":")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.legend(fontsize=SMALL, framealpha=0.95, loc="upper right", bbox_to_anchor=(0.98, 1.2))
        fig.text(0.52, 0.95, "B", fontsize=BIG, fontweight="bold", va="top")

        fig.suptitle("Olfactory Habituation: Neural Signatures Diminish with Repeated Exposure",
                     fontsize=BIG, fontweight="bold", y=1.1)
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.canvas.draw()
    return fig
