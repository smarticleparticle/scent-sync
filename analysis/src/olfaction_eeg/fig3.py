"""Fig. 3: respiratory phase-locked ERSPs (conceptual schematic A, two single subjects B-C, grand averages D-E).

Ported from analysis-notebook cell 60 (``create_main_figure_rearranged`` and its helpers, unchanged).  Edits to the notebook code (both marked):
  * the three data sets the function used as notebook globals (``phase_aligned_2``, ``timealigned_ersp_results_avgbreath``, ``breath_data``) are
    now arguments;
  * before returning, the numbers behind the figure (colour limits, subject counts) are attached as ``fig.fig3_info`` so they can be checked.
The supplement figure defined in the same notebook cell (odor-specific ERSPs, ``create_supplement_odor_figure``) is not in the paper and not ported.

Inputs: the paper's phase-aligned ERSP, the 3.78-s time-locked ERSP, and ``breath_data`` (first inhale / exhale of every trial, Table S3 detection;
``breath.extract_first_breath_data``).  Subjects 33 and 40 are shown as "Subject 1" and "Subject 2" in panels B and C.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import interp1d


def compute_vmin_vmax_from_data(data_list, percentile=99, symmetric=True):
    """Compute vmin/vmax from actual data."""
    all_data = np.concatenate([d.flatten() for d in data_list if d is not None])
    all_data = all_data[~np.isnan(all_data)]
    
    if symmetric:
        vmax = np.percentile(np.abs(all_data), percentile)
        vmin = -vmax
    else:
        vmin = np.percentile(all_data, 100 - percentile)
        vmax = np.percentile(all_data, percentile)
    
    return vmin, vmax

def get_first_n_trials_average(data_dict, subject, condition, n_trials=3):
    """Average first n trials for a subject/condition."""
    if condition not in data_dict or subject not in data_dict[condition]:
        return None
    
    trial_keys = sorted([k for k in data_dict[condition][subject].keys() 
                        if isinstance(k, (int, np.integer)) or 
                        (isinstance(k, str) and k.isdigit())])
    
    if len(trial_keys) == 0:
        return None
    
    selected_trials = trial_keys[:min(n_trials, len(trial_keys))]
    trial_data = [data_dict[condition][subject][t] for t in selected_trials]
    avg_data = np.nanmean(np.stack(trial_data, axis=0), axis=0)
    
    return avg_data


def get_averaged_respiration_all_odors(breath_data, subject, odor_list=['peppermint', 'citrus', 'oregano'], n_trials=3):
    """Average respiratory data across ALL odor conditions for a subject."""
    eeg_sfreq = 250
    all_inhale_signals = []
    all_exhale_signals = []
    all_inhale_durations = []
    all_exhale_durations = []
    
    for odor in odor_list:
        if odor not in breath_data or subject not in breath_data[odor]:
            continue
        
        trial_keys = sorted([k for k in breath_data[odor][subject].keys() 
                            if isinstance(k, (int, np.integer))])
        
        if len(trial_keys) == 0:
            continue
        
        selected_trials = trial_keys[:min(n_trials, len(trial_keys))]
        
        for trial_idx in selected_trials:
            trial_data = breath_data[odor][subject][trial_idx]
            all_inhale_signals.append(trial_data['inhale']['resp_data'])
            all_exhale_signals.append(trial_data['exhale']['resp_data'])
            all_inhale_durations.append(trial_data['inhale']['duration_seconds'])
            all_exhale_durations.append(trial_data['exhale']['duration_seconds'])
    
    if len(all_inhale_signals) == 0:
        return None, None, None, None, None
    
    avg_inhale_dur = np.mean(all_inhale_durations)
    avg_exhale_dur = np.mean(all_exhale_durations)
    
    target_inhale_len = int(avg_inhale_dur * eeg_sfreq)
    target_exhale_len = int(avg_exhale_dur * eeg_sfreq)
    
    inhale_interp = []
    for inh_signal in all_inhale_signals:
        x_orig = np.linspace(0, 1, len(inh_signal))
        x_new = np.linspace(0, 1, target_inhale_len)
        f_interp = interp1d(x_orig, inh_signal, kind='cubic', fill_value='extrapolate')
        inhale_interp.append(f_interp(x_new))
    
    exhale_interp = []
    for exh_signal in all_exhale_signals:
        x_orig = np.linspace(0, 1, len(exh_signal))
        x_new = np.linspace(0, 1, target_exhale_len)
        f_interp = interp1d(x_orig, exh_signal, kind='cubic', fill_value='extrapolate')
        exhale_interp.append(f_interp(x_new))
    
    resp_inhale_avg = np.mean(np.stack(inhale_interp, axis=0), axis=0)
    resp_exhale_avg = np.mean(np.stack(exhale_interp, axis=0), axis=0)
    
    mid_inhale = len(resp_inhale_avg) // 2
    if np.mean(resp_inhale_avg[:mid_inhale]) < np.mean(resp_inhale_avg[mid_inhale:]):
        resp_inhale_avg = -resp_inhale_avg
        resp_exhale_avg = -resp_exhale_avg
    
    resp_combined = np.concatenate([resp_inhale_avg, resp_exhale_avg])
    resp_combined = (resp_combined - np.mean(resp_combined)) / (np.std(resp_combined) + 1e-6)
    
    time_combined = np.arange(len(resp_combined)) / eeg_sfreq
    
    n_inhale = len(resp_inhale_avg)
    n_exhale = len(resp_exhale_avg)
    
    phase_inhale = np.linspace(0, 180, n_inhale, endpoint=False)
    phase_exhale = np.linspace(180, 360, n_exhale, endpoint=False)
    phase_combined = np.concatenate([phase_inhale, phase_exhale])
    
    print(f"  ✓ Averaged respiration from {len(all_inhale_signals)} trials across {len(odor_list)} odors")
    
    return resp_combined, time_combined, phase_combined, avg_inhale_dur, avg_exhale_dur

def draw_panel_A_concept(ax_container):
    """
    Panel A (fixed):
    - Push curves down by giving more headroom in ylim (legend stays above, no overlap)
    - Remove "strange side stuff" by turning off right/top spines + clipping + clean ticks
    - Time-locked dots:
        * ONLY fast respiration gets 0°,180°,360° dots (in the blue box)
        * Slow respiration gets dots at its OWN inhale/trough/next inhale (not aligned in time)
          -> demonstrates misalignment under time-locking
    - Phase-locked dots:
        * Both traces get dots at 0°,180°,360° (aligned by definition)
    - Legend includes inhale/exhale dot meaning and is positioned above curves.
    """
    ax_container.axis('off')
    ax_container.set_title('A', fontsize=10, fontweight='bold', loc='left', pad=2)

    ax_right = ax_container.inset_axes([0.1, 0.10, 0.8, 0.90])
    ax_right.axis('off')

    # Make panels a bit taller and leave a touch more gap between them
    # ax_time  = ax_right.inset_axes([0.00, 0.8, 0.90, 0.6])
    # ax_phase = ax_right.inset_axes([0.00, 0.02, 0.90, 0.6])

    ax_time  = ax_right.inset_axes([0.00, 0.7, 0.90, 0.4])
    ax_phase = ax_right.inset_axes([0.00, 0.05, 0.90, 0.4])

    # ---------- helpers ----------
    def clean_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=8, width=1.0, length=4)
        # helps prevent odd drawing outside axes
        for line in ax.lines:
            line.set_clip_on(True)

    # Visual style for dots
    dot_blue_edge = 'royalblue'
    dot_blue_face = 'white'
    dot_black_edge = 'black'
    dot_black_face = 'black'
    ms = 5
    mew = 1.0

    # ---------- TIME-LOCKED ----------
    time = np.linspace(0, 5, 1000)
    f_fast = 0.4
    f_slow = 0.2
    offset_fast = 2.0
    offset_slow = -0.5

    person_fast = np.cos(2 * np.pi * f_fast * time)
    person_slow = np.cos(2 * np.pi * f_slow * time)

    ax_time.plot(time, person_fast + offset_fast, color='b', lw=1.8, label='Fast respiration')
    ax_time.plot(time, person_slow + offset_slow, color='k', lw=1.8, label='Slow respiration')

    ax_time.axvline(x=0, color='red', linestyle='--', linewidth=1.8, alpha=0.8, zorder=10)
    ax_time.text(0.12, 3.60, 'Odor onset', fontsize=8, color='red',
                 fontweight='bold', ha='left')

    # grid lines
    shared_times = [0, 1.25, 2.5, 3.75, 5.0]
    for t in shared_times:
        ax_time.axvline(x=t, color='gray', linestyle=':', linewidth=1.0, alpha=0.35)

    # blue box = first FAST cycle (0 -> 2.5s)
    first_cycle_end = 1.0 / f_fast  # 2.5 s
    rect = FancyBboxPatch(
        (0, offset_fast - 1.2), first_cycle_end, 2.4,
        boxstyle="round,pad=0.05",
        edgecolor='blue', facecolor='lightblue',
        linewidth=1.4, alpha=0.25, zorder=5
    )
    ax_time.add_patch(rect)

    ax_time.set_xlim(0, 5)
    # Give more headroom so legend doesn't cover curves
    ax_time.set_ylim(-2.2, 4.8)

    ax_time.set_xticks([0, 1.25, 2.5, 3.75, 5.0])
    ax_time.set_xticklabels(['0', '1', '2', '3', '4'], fontsize=8, fontweight='bold')
    ax_time.set_xlabel('Time (s)',fontsize = 8, fontweight = 'bold')
    # ax_time.set_title('Time-locked',
    #                   fontsize=9, fontweight='bold', pad=6)

    clean_axis(ax_time)

    # --- Time dots ---
    # FAST: 0°, 180°, 360° within first fast cycle (in the blue box)
    T_fast = 1.0 / f_fast  # 2.5
    t0_fast, t180_fast, t360_fast = 0.0, T_fast/2.0, T_fast

    def y_fast_time(tt):  # fast trace y at time
        return np.cos(2 * np.pi * f_fast * tt) + offset_fast

    # only FAST gets these three dots
    ax_time.plot([t0_fast],   [y_fast_time(t0_fast)],   marker='o', ms=ms,
                 mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)
    ax_time.plot([t180_fast], [y_fast_time(t180_fast)], marker='o', ms=ms,
                 mfc=dot_black_face, mec=dot_black_edge, mew=mew, zorder=20)
    ax_time.plot([t360_fast], [y_fast_time(t360_fast)], marker='o', ms=ms,
                 mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)

    # SLOW: dots at its OWN 0°,180°,360° times (NOT aligned with the fast box)
    # This is the point: time-locked aligns *time*, not phase.
    T_slow = 1.0 / f_slow  # 5.0
    t0_slow, t180_slow, t360_slow = 0.0, T_slow/2.0, T_slow  # 0, 2.5, 5

    def y_slow_time(tt):
        return np.cos(2 * np.pi * f_slow * tt) + offset_slow

    # Put slow dots at its inhale/trough/next inhale across its cycle
    ax_time.plot([t0_slow],   [y_slow_time(t0_slow)],   marker='o', ms=ms,
                 mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)
    ax_time.plot([t180_slow], [y_slow_time(t180_slow)], marker='o', ms=ms,
                 mfc=dot_black_face, mec=dot_black_edge, mew=mew, zorder=20)
    ax_time.plot([t360_slow], [y_slow_time(t360_slow)], marker='o', ms=ms,
                 mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)

    # ---------- PHASE-LOCKED ----------
    phase = np.linspace(0, 360, 1000)
    normalized_signal = np.cos(2 * np.pi * phase / 360)

    ax_phase.plot(phase, normalized_signal + offset_fast, color='b', lw=1.8, label='Fast respiration')
    ax_phase.plot(phase, normalized_signal + offset_slow, color='k', lw=1.8, label='Slow respiration')

    ax_phase.axvline(x=0, color='red', linestyle='--', linewidth=1.8, alpha=0.8, zorder=10)
    ax_phase.text(8, 3.60, 'Odor onset', fontsize=8, color='red',
                  fontweight='bold', ha='left')

    phase_points = [0, 90, 180, 270, 360]
    for p in phase_points:
        ax_phase.axvline(x=p, color='gray', linestyle=':', linewidth=1.0, alpha=0.35)

    rect2 = FancyBboxPatch(
        (0, offset_fast - 1.2), 360, 2.4,
        boxstyle="round,pad=0.05",
        edgecolor='blue', facecolor='lightblue',
        linewidth=1.4, alpha=0.25, zorder=5
    )
    ax_phase.add_patch(rect2)

    ax_phase.set_xlim(0, 360)
    ax_phase.set_ylim(-2.2, 4.8)
    ax_phase.set_xticks(phase_points)
    ax_phase.set_xticklabels([f'{p}°' for p in phase_points], fontsize=8, fontweight='bold')
    ax_phase.set_xlabel('Respiratory Phase',fontsize = 8, fontweight = 'bold')
    #ax_phase.set_title('Phase-locked',
     #                  fontsize=9, fontweight='bold', pad=6)

    clean_axis(ax_phase)

    # Phase dots: BOTH traces at same phase points (this is the key)
    def y_fast_phase(p): return np.cos(2 * np.pi * p / 360) + offset_fast
    def y_slow_phase(p): return np.cos(2 * np.pi * p / 360) + offset_slow

    for yfun in (y_fast_phase, y_slow_phase):
        ax_phase.plot([0],   [yfun(0)],   marker='o', ms=ms, mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)
        ax_phase.plot([180], [yfun(180)], marker='o', ms=ms, mfc=dot_black_face, mec=dot_black_edge, mew=mew, zorder=20)
        ax_phase.plot([360], [yfun(360)], marker='o', ms=ms, mfc=dot_blue_face, mec=dot_blue_edge, mew=mew, zorder=20)

    # ---------- Legend: keep it up, away from curves ----------
    legend_elements = [
        Line2D([0], [0], color='b', lw=1.8, label='Person A \nFast respiration'),
        Line2D([0], [0], color='k', lw=1.8, label='Person B \nSlow respiration'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='white', markeredgecolor='royalblue',
               markeredgewidth=1.2, markersize=ms, label='Inhale (0°/360°)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor='black', markeredgecolor='black',
               markeredgewidth=1.2, markersize=ms, label='Exhale (180°)'),
    ]

    ax_time.legend(
        handles=legend_elements,
        loc='upper left',              # anchor the legend's left edge
        bbox_to_anchor=(1.02, 1.1),   # x>1 pushes legend outside to the right
        frameon=False,
        fontsize=7,
        ncol=1,
        borderaxespad=0.0
    )

def create_main_figure_rearranged(
    phase_aligned_2,
    timealigned_ersp_results_avgbreath,
    breath_data,
    subjects_individual,
    subjects_grand,
    odor_list=['peppermint', 'oregano', 'citrus'],
    n_trials=3,
    freq_min=4,
    freq_max=40,
    time_max_long=3.78,
    percentile=99,
    save_path=None
):
    freqs = np.geomspace(2, 40, 100)
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    desired_yticks = [4, 8, 12, 20, 30, 40]
    eeg_sfreq = 250

    print("\n=== COLLECTING DATA (MAIN FIG) ===")

    # Individual subjects (all odors vs water) - phase-locked
    individual_data = {}
    for subject in subjects_individual:
        phase_water = get_first_n_trials_average(phase_aligned_2, subject, 'water', n_trials)

        phase_odors = []
        for odor in odor_list:
            phase_o = get_first_n_trials_average(phase_aligned_2, subject, odor, n_trials)
            if phase_o is not None:
                phase_odors.append(phase_o)

        if len(phase_odors) == 0 or phase_water is None:
            continue

        phase_odor_avg = np.nanmean(np.stack(phase_odors, axis=0), axis=0)
        phase_diff = phase_odor_avg - phase_water

        individual_data[subject] = {'phase_diff': phase_diff[freq_mask, :]}
        print(f"✓ Subject {subject}: {len(phase_odors)} odors")

    # Grand average - PHASE-locked
    grand_phase_diffs_all = []
    for subject in subjects_grand:
        phase_water = get_first_n_trials_average(phase_aligned_2, subject, 'water', n_trials)

        phase_odors = []
        for odor in odor_list:
            phase_o = get_first_n_trials_average(phase_aligned_2, subject, odor, n_trials)
            if phase_o is not None:
                phase_odors.append(phase_o)

        if len(phase_odors) > 0 and phase_water is not None:
            phase_odor_avg = np.nanmean(np.stack(phase_odors, axis=0), axis=0)
            phase_diff = phase_odor_avg - phase_water
            grand_phase_diffs_all.append(phase_diff[freq_mask, :])

    grand_avg_phase_all = np.mean(np.stack(grand_phase_diffs_all, axis=0), axis=0) if len(grand_phase_diffs_all) > 0 else None
    print(f"✓ Grand phase-locked: n={len(grand_phase_diffs_all)}")

    # Grand average - TIME-locked
    grand_time_diffs_all = []
    grand_resp_traces_all = []

    for subject in subjects_grand:
        time_water = get_first_n_trials_average(timealigned_ersp_results_avgbreath, subject, 'water', n_trials)

        time_odors = []
        for odor in odor_list:
            time_o = get_first_n_trials_average(timealigned_ersp_results_avgbreath, subject, odor, n_trials)
            if time_o is not None:
                time_odors.append(time_o)

        if len(time_odors) > 0 and time_water is not None:
            time_odor_avg = np.nanmean(np.stack(time_odors, axis=0), axis=0)
            time_diff = time_odor_avg - time_water
            grand_time_diffs_all.append(time_diff[freq_mask, :])

        resp_combined, time_resp, phase_resp, inhale_dur, exhale_dur = \
            get_averaged_respiration_all_odors(breath_data, subject, odor_list, n_trials)

        if resp_combined is not None:
            grand_resp_traces_all.append({
                'signal': resp_combined,
                'time': time_resp,
                'phase': phase_resp,
                'subject': subject,
                'inhale_dur': inhale_dur,
                'exhale_dur': exhale_dur
            })

    grand_avg_time_all = np.mean(np.stack(grand_time_diffs_all, axis=0), axis=0) if len(grand_time_diffs_all) > 0 else None
    print(f"✓ Grand time-locked: n={len(grand_time_diffs_all)}")

    # Compute vmin/vmax
    indiv_subj_diffs = [d['phase_diff'] for d in individual_data.values()]
    vmin_subj, vmax_subj = compute_vmin_vmax_from_data(indiv_subj_diffs, percentile, symmetric=True)

    grand_data = [grand_avg_phase_all, grand_avg_time_all]
    vmin_grand, vmax_grand = compute_vmin_vmax_from_data(grand_data, percentile, symmetric=True)

    print(f"Individual subjects: vmin={vmin_subj:.3f}, vmax={vmax_subj:.3f}")
    print(f"Grand averages: vmin={vmin_grand:.3f}, vmax={vmax_grand:.3f}")

    # CREATE FIGURE (slightly wider for more rectangular ERSP panels)
    fig = plt.figure(figsize=(7.4, 7.2))

    # More height for A; more horizontal space between left block and right block
    gs = GridSpec(
        3, 5, figure=fig,
        width_ratios=[1.0, 0.03, 1.0, 0.03, 0.01],
        height_ratios=[1, 0.92, 0.92],
        hspace=0.4, wspace=0.65,
        left=0.10, right=0.96, top=0.87, bottom=0.06
    )

    # A: Conceptual framework (full width top row)
    axA = fig.add_subplot(gs[0, 0:4])
    draw_panel_A_concept(axA)

    # B, C: Individual subjects (phase-locked)
    subject_items = list(individual_data.items())[:2]
    im_subj = None
    axes_BC = []

    for idx, (subject, data) in enumerate(subject_items):
        row = idx + 1
        ax = fig.add_subplot(gs[row, 0])
        axes_BC.append(ax)

        im_subj = ax.imshow(
            data['phase_diff'],
            aspect='auto',
            origin='lower',
            extent=[0, 360, freq_min, freq_max],
            cmap='bwr',#'bwr',
            vmin=vmin_subj, vmax=vmax_subj,
            interpolation='bilinear'
        )

        ax.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.2)
        ax.axvline(360, color='white', linestyle='--', alpha=0.5, linewidth=1.2)

        ax.set_yticks(desired_yticks)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=8)
        ax.set_ylim(freq_min, freq_max)
        ax.set_xlim(0, 360)

        # Only show y tick labels on B (remove from C for cleanliness)
       #if idx == 0:
        ax.set_yticklabels([f'{int(y)}' for y in desired_yticks], fontsize=8)
        # else:
        #     ax.set_yticklabels([])

        ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.2,
                       color='black', labelsize=8, pad=2)
        ax.tick_params(axis='both', which='minor', length=0, width=0)
        ax.minorticks_off()

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        label = 'B' if idx == 0 else 'C'
        subj_name = 'Subject 1 (All Odors)' if idx == 0 else 'Subject 2 (All Odors)'
        ax.set_title(f'{label}. {subj_name}', fontsize=10, fontweight='bold', loc='left', pad=5)

        if idx == 1:
            ax.set_xlabel('Respiratory Phase', fontsize=9, fontweight='bold')

    # Shared Y-label centered between B and C (computed from axes positions)
    if len(axes_BC) == 2:
        pos0 = axes_BC[0].get_position()
        pos1 = axes_BC[1].get_position()
        y_center = 0.5 * ((pos0.y0 + pos0.y1) / 2 + (pos1.y0 + pos1.y1) / 2)
        x_left = min(pos0.x0, pos1.x0)
        fig.text(x_left - 0.06, y_center, 'Frequency (Hz)',
                 fontsize=9, fontweight='bold',
                 rotation=90, va='center', ha='center')

    # D: Grand avg phase-locked
    axD = fig.add_subplot(gs[1, 2])
    im_phase = None

    if grand_avg_phase_all is not None:
        im_phase = axD.imshow(
            grand_avg_phase_all,
            aspect='auto',
            origin='lower',
            extent=[0, 360, freq_min, freq_max],
            cmap='bwr',#'bwr',
            vmin=vmin_grand, vmax=vmax_grand,
            interpolation='bilinear'
        )

        # Individual resp traces (thinner)
        for resp_data in grand_resp_traces_all:
            phase_resp = resp_data['phase']
            resp_signal = resp_data['signal']
            resp_min = np.min(resp_signal)
            resp_max = np.max(resp_signal)
            resp_range = resp_max - resp_min + 1e-6
            resp_normalized = (resp_signal - resp_min) / resp_range
            resp_y = resp_normalized * (freq_max - freq_min) + freq_min
            axD.plot(phase_resp, resp_y, color='black', lw=0.5, alpha=0.75, zorder=5)

        # Mean resp trace
        common_phase = np.linspace(0, 360, 360)
        interp_resp_signals = []
        for resp_data in grand_resp_traces_all:
            f = interp1d(resp_data['phase'], resp_data['signal'], kind='cubic', fill_value='extrapolate')
            interp_resp_signals.append(f(common_phase))

        if len(interp_resp_signals) > 0:
            mean_resp = np.mean(np.stack(interp_resp_signals, axis=0), axis=0)
            mean_resp_min = np.min(mean_resp)
            mean_resp_max = np.max(mean_resp)
            mean_resp_range = mean_resp_max - mean_resp_min + 1e-6
            resp_normalized = (mean_resp - mean_resp_min) / mean_resp_range
            resp_y = resp_normalized * (freq_max - freq_min) + freq_min
            axD.plot(common_phase, resp_y, color='black', lw=1.5, alpha=0.9, zorder=10)

        axD.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.2)
        axD.axvline(360, color='white', linestyle='--', alpha=0.5, linewidth=1.2)

        axD.set_yticks(desired_yticks)
        axD.set_yticklabels([f'{int(y)}' for y in desired_yticks], fontsize=8)
        axD.set_xticks([0, 90, 180, 270, 360])
        axD.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=8)
        axD.set_ylim(freq_min, freq_max)
        axD.set_xlim(0, 360)

        axD.tick_params(axis='both', which='major', direction='out', length=4, width=1.2,
                        labelsize=8, pad=2)
        axD.tick_params(axis='both', which='minor', length=0, width=0)
        axD.minorticks_off()

        for spine in axD.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        axD.set_title('D. Grand Avg Phase-Locked', fontsize=10, fontweight='bold', loc='left', pad=5)
        axD.set_xlabel('Respiratory Phase', fontsize=9, fontweight='bold')

    # E: Grand avg time-locked
    axE = fig.add_subplot(gs[2, 2])
    im_time = None

    if grand_avg_time_all is not None:
        ersp_start = -0.5
        prestim_samples = int(abs(ersp_start) * eeg_sfreq)
        grand_avg_time_plot = grand_avg_time_all[:, prestim_samples:]

        n_times = grand_avg_time_plot.shape[1]
        time_vector = np.linspace(0, n_times / eeg_sfreq, n_times)

        actual_max_time = min(time_vector[-1], time_max_long)
        time_mask = time_vector <= actual_max_time
        time_data_plot = grand_avg_time_plot[:, time_mask]

        im_time = axE.imshow(
            time_data_plot,
            aspect='auto',
            origin='lower',
            extent=[0, actual_max_time, freq_min, freq_max],
            cmap='bwr',#'bwr',
            vmin=vmin_grand, vmax=vmax_grand,
            interpolation='bilinear'
        )

        # Individual resp traces (thinner)
        for resp_data in grand_resp_traces_all:
            time_resp = resp_data['time']
            resp_signal = resp_data['signal']
            resp_min = np.min(resp_signal)
            resp_max = np.max(resp_signal)
            resp_range = resp_max - resp_min + 1e-6

            resp_time_mask = time_resp <= actual_max_time
            if np.sum(resp_time_mask) > 10:
                time_resp_plot = time_resp[resp_time_mask]
                resp_plot = resp_signal[resp_time_mask]
                resp_normalized = (resp_plot - resp_min) / resp_range
                resp_y = resp_normalized * (freq_max - freq_min) + freq_min
                axE.plot(time_resp_plot, resp_y, color='black', lw=0.5, alpha=0.75, zorder=5)

        # Mean resp trace
        max_time = min([resp_data['time'][-1] for resp_data in grand_resp_traces_all]) if len(grand_resp_traces_all) else actual_max_time
        max_time = min(max_time, actual_max_time)
        common_time = np.linspace(0, max_time, int(max_time * eeg_sfreq)) if max_time > 0 else np.array([0])

        interp_resp_signals = []
        for resp_data in grand_resp_traces_all:
            if resp_data['time'][-1] >= max_time:
                f = interp1d(resp_data['time'], resp_data['signal'], kind='cubic', fill_value='extrapolate')
                interp_resp_signals.append(f(common_time))

        if len(interp_resp_signals) > 0:
            mean_resp = np.mean(np.stack(interp_resp_signals, axis=0), axis=0)
            mean_resp_min = np.min(mean_resp)
            mean_resp_max = np.max(mean_resp)
            mean_resp_range = mean_resp_max - mean_resp_min + 1e-6
            resp_normalized = (mean_resp - mean_resp_min) / mean_resp_range
            resp_y = resp_normalized * (freq_max - freq_min) + freq_min
            axE.plot(common_time, resp_y, color='black', lw=1.5, alpha=0.9, zorder=10)

        axE.set_xlabel('Time (s)', fontsize=9, fontweight='bold')
        axE.set_yticks(desired_yticks)
        axE.set_yticklabels([f'{int(y)}' for y in desired_yticks], fontsize=8)
        axE.set_ylim(freq_min, freq_max)
        axE.set_xlim(0, actual_max_time)

        axE.tick_params(axis='both', which='major', direction='out', length=4, width=1.2,
                        labelsize=8, pad=2)
        axE.tick_params(axis='both', which='minor', length=0, width=0)
        axE.minorticks_off()

        for spine in axE.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        axE.set_title('E. Grand Avg Time-Locked', fontsize=10, fontweight='bold', loc='left', pad=5)

    # THINNER COLORBARS + slightly smaller labelpad to avoid overlap
    if im_subj is not None:
        cbar_ax_subj = fig.add_subplot(gs[1:3, 1])
        cbar_subj = plt.colorbar(im_subj, cax=cbar_ax_subj)
        # shift the B/C colorbar left (negative = left, positive = right)
        pos = cbar_ax_subj.get_position()
        cbar_ax_subj.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])
        cbar_subj.set_label('log ratio\n(Odor/Water)', fontsize=8, rotation=270, labelpad=12)
        cbar_subj.ax.tick_params(labelsize=7)

    if im_time is not None:
        cbar_ax_grand = fig.add_subplot(gs[1:3, 3])
        cbar_grand = plt.colorbar(im_time, cax=cbar_ax_grand)
        pos = cbar_ax_grand.get_position()
        cbar_ax_grand.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])
        cbar_grand.set_label('log ratio\n(Odor/Water)', fontsize=8, rotation=270, labelpad=12)
        cbar_grand.ax.tick_params(labelsize=7)

    # Legend for respiration overlays
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label='Mean respiration'),
        Line2D([0], [0], color='black', lw=0.8, alpha=0.8, label='Individual subjects'),
    ]

    fig.legend(handles=legend_elements,
           loc='center',
           ncol=2, fontsize=7, framealpha=0.95,
           bbox_to_anchor=(0.5, 0.6))  # <-- between A and B/D

    # fig.suptitle('Respiratory Phase-Locked ERSPs Reveal Odor-Specific Neural Signatures',
    #              fontsize=11, fontweight='bold', y=0.985)
    # fig.suptitle('Respiratory Phase-Locked ERSPs Reveal a Respiration-Locked Odor Detection Signature',
    #              fontsize=11, fontweight='bold', y=0.985)
    fig.suptitle('Respiratory Phase-Locked ERSPs Reveal a \nRespiration-Locked Neural Signature of Odor Presence',
                fontsize=11, fontweight='bold', y=0.985)

    if save_path:
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path)#, bbox_inches='tight')
        print(f"\n✓ Main figure saved to {save_path}")
    else:
        plt.show()

    # (added) numbers behind the figure, for the checks against the published figure
    fig.fig3_info = {
        "vmin_subj": vmin_subj, "vmax_subj": vmax_subj, "vmin_grand": vmin_grand, "vmax_grand": vmax_grand,
        "n_grand_phase": len(grand_phase_diffs_all), "n_grand_time": len(grand_time_diffs_all),
        "n_resp_traces": len(grand_resp_traces_all), "subjects_individual": list(individual_data.keys()),
    }
    return fig
