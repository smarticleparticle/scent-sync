"""SI Fig. S2: example FOOOF fits of channels with and without an oscillatory peak in each band (theta, alpha, beta, low gamma).

Ported from analysis-notebook cell 106.  The notebook recomputed each example's spectrum from the continuous EEG
(``compute_psd(raw_eeg.get_data()[ch], sfreq)``); the spectra cache of Section 1.2 holds exactly that output for every channel
(``fooof_gating.compute_psd_cache``), so the figure is drawn from the deposited cache instead and needs no continuous EEG.  The selection logic
(six hand-picked examples, plus the first channel with an alpha peak and the first channel without a clean beta peak, which are searched in the
order of ``peak_gate``) and the plotting code are the notebook's.  Cells 107-109 (a bar chart of peak prevalence and random example strips) are
exploratory and not in the paper.
"""
import matplotlib.pyplot as plt
import numpy as np

from .fooof_gating import fit_fooof

band_colors = {
    'theta': '#B5EAD7',
    'alpha': '#C7E6FB',
    'beta': '#FFDAC1',
    'low_gamma': '#FFB7B2',
}

freq_bands_list = ['theta', 'alpha', 'beta', 'low_gamma']

# ---- Hardcoded examples you've hand-picked as clean/illustrative ---- (notebook cell 106, verbatim)
manual_examples = {
    ('beta', True): ('33', 'E58'),
    ('low_gamma', True): ('34', 'E5'),
    ('alpha', False): ('29', 'E14'),
    ('theta', True): ('23', 'E3'),
    ('theta', False): ('32', 'E111'),
    ('low_gamma', False): ('32', 'E99')
}


def find_example_channel(peak_gate, band_name, want_peak=True, max_search=500):
    """notebook cell 106 (``peak_gate`` is now an argument)"""
    count = 0
    for subject_id, chans in peak_gate.items():
        for ch_name, bands in chans.items():
            count += 1
            if count > max_search:
                return None
            if bands.get(band_name, False) == want_peak:
                return subject_id, ch_name
    return None


def find_clean_band_example(peak_gate, psd_cache, freq_bands, band_name, want_peak=True, center_margin=3.0, max_search=1000,
                            settings=None):
    """notebook cell 106; the spectrum of each channel is read from the spectra cache instead of being recomputed from the raw EEG."""
    band_range = freq_bands[band_name]
    safe_min = band_range[0] + center_margin
    safe_max = band_range[1] - center_margin

    count = 0
    for subject_id, chans in peak_gate.items():
        for ch_name in chans.keys():
            count += 1
            if count > max_search:
                return None
            if ch_name not in psd_cache['psd'][subject_id]:
                continue
            freqs, psd = psd_cache['psd'][subject_id][ch_name]
            try:
                fm = fit_fooof(freqs, psd, settings=settings)
            except Exception:
                continue

            peaks_in_band = [p for p in fm.peak_params_ if band_range[0] <= p[0] <= band_range[1]]

            if want_peak:
                clean_peak = any(safe_min <= p[0] <= safe_max for p in peaks_in_band)
                if clean_peak:
                    return subject_id, ch_name
            else:
                if len(peaks_in_band) == 0:
                    return subject_id, ch_name
    return None


def make_figS2(peak_gate, psd_cache, freq_bands, settings=None, out_prefix=None, dpi=150):
    """The 4 x 2 figure of notebook cell 106 (rows: bands; columns: peak present / no peak).  Returns (fig, {(band, want_peak): (subject, channel)})."""
    fig, axes = plt.subplots(len(freq_bands_list), 2, figsize=(11, 14), facecolor='white')
    chosen = {}

    for row, band_name in enumerate(freq_bands_list):
        band_range = freq_bands[band_name]

        for col, want_peak in enumerate([True, False]):
            ax = axes[row, col]

            # ---- check for a manual override first ----
            if (band_name, want_peak) in manual_examples:
                example = manual_examples[(band_name, want_peak)]
            elif band_name == 'beta':
                example = find_clean_band_example(peak_gate, psd_cache, freq_bands, band_name, want_peak=want_peak,
                                                  center_margin=2.0, settings=settings)
            else:
                example = find_example_channel(peak_gate, band_name, want_peak=want_peak)

            if example is None:
                ax.text(0.5, 0.5, "No example found", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{band_name.capitalize()} — {'HAS' if want_peak else 'NO'} peak")
                continue

            subject_id, ch_name = example
            chosen[(band_name, want_peak)] = example
            freqs, psd = psd_cache['psd'][subject_id][ch_name]
            fm = fit_fooof(freqs, psd, settings=settings)

            fm.plot(ax=ax, plt_log=False, add_legend=(row == 0 and col == 0))
            ax.axvspan(band_range[0], band_range[1], color=band_colors[band_name], alpha=0.3, zorder=0)

            status = "peak present" if want_peak else "no peak detected"
            ax.set_title(f"{band_name.capitalize()} ({band_range[0]}-{band_range[1]} Hz): {status}\n"
                         f"Subject {subject_id}, channel {ch_name}", fontsize=10)
            ax.set_xlim(1, 45)
            ax.tick_params(labelsize=9)

    plt.tight_layout()
    if out_prefix:
        fig.savefig(out_prefix + ".pdf", bbox_inches='tight')
    return fig, chosen
