"""SI Fig. S4: cluster-permutation t-maps, time-locked (3.78 s) vs. phase-locked ERSP (notebook cell 128).

``plot_winning_cluster_shared`` is the notebook's, unchanged.  The only difference in the figure code: the colorbar label size (16 pt) is set
explicitly; in the notebook it came from a matplotlib rc setting left over from an earlier cell (calibrated on the saved figure).
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_winning_cluster_shared(result, band_name, region_key, x_label, ax, vmax, show_ylabel=False):
    sub_result = result[region_key]
    freqs_axis = result['freqs_axis']
    bins_axis = result['bins_axis']

    if region_key == 'theta_result':
        freq_mask = result['freq_mask_theta']
        bin_mask = result['bin_mask_first']
    else:
        freq_mask = result['freq_mask_alpha']
        bin_mask = result['bin_mask_second']

    t_map = sub_result['t_map']
    labeled = sub_result.get('labeled', None)

    region_freqs = freqs_axis[freq_mask]
    region_bins = bins_axis[bin_mask]

    im = ax.pcolormesh(region_bins, region_freqs, t_map, cmap='bwr',
                         vmin=-vmax, vmax=vmax, shading='auto')

    if len(sub_result['clusters']) > 0:
        best_cluster = min(sub_result['clusters'], key=lambda c: c['p_value'])
        best_id = best_cluster['cluster_id']
        winning_mask = (labeled == best_id)
        ax.contour(region_bins, region_freqs, winning_mask.astype(int),
                    levels=[0.5], colors='black', linewidths=2)
        p_str = f"\np={best_cluster['p_value']:.4f}, n_pixels={best_cluster['n_pixels']}"
    else:
        p_str = "\np=1.000"

    ax.set_title(f"{band_name}{p_str}", fontsize=11)
    ax.set_xlabel(x_label, fontsize=10)
    if show_ylabel:
        ax.set_ylabel('Frequency (Hz)', fontsize=11)

    return im


def make_figS4(all_results, out_prefix=None, dpi=600):
    """2 x 2 figure (rows: time-locked 3.78 s, phase-locked; columns: theta, alpha) with one shared colour scale (notebook cell 128)."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    def get_tmap(result_dict, region_key):
        return result_dict[region_key]['t_map']

    all_tmaps = [
        get_tmap(all_results['2_longer_5050'], 'theta_result'),
        get_tmap(all_results['2_longer_5050'], 'alpha_result'),
        get_tmap(all_results['3_phase_locked'], 'theta_result'),
        get_tmap(all_results['3_phase_locked'], 'alpha_result'),
    ]
    global_vmax = max(np.abs(t).max() for t in all_tmaps)

    plot_winning_cluster_shared(all_results['2_longer_5050'], 'Theta', 'theta_result',
                                x_label='Time (s)', ax=axes[0, 0], vmax=global_vmax, show_ylabel=True)
    im_last = plot_winning_cluster_shared(all_results['2_longer_5050'], 'Alpha', 'alpha_result',
                                          x_label='Time (s)', ax=axes[0, 1], vmax=global_vmax)
    plot_winning_cluster_shared(all_results['3_phase_locked'], 'Theta', 'theta_result',
                                x_label='Respiratory phase (\u00b0)', ax=axes[1, 0], vmax=global_vmax, show_ylabel=True)
    plot_winning_cluster_shared(all_results['3_phase_locked'], 'Alpha', 'alpha_result',
                                x_label='Respiratory phase (\u00b0)', ax=axes[1, 1], vmax=global_vmax)

    plt.subplots_adjust(hspace=0.7, top=0.85, right=0.85)
    fig.text(0.5, 0.96, 'Time-Locked ERSP', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.46, 'Respiratory Phase-Locked ERSP', ha='center', fontsize=13, fontweight='bold')
    cbar_ax = fig.add_axes([0.88, 0.12, 0.03, 0.73])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label('t-statistic', fontsize=16)
    if out_prefix:
        fig.savefig(out_prefix + ".png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    return fig, global_vmax
