"""Fig. 4C/D: how a free response is scored (C) and recognition accuracy by odor with significance stars (D).

``create_bottom_row_figure`` is notebook cell 27 (preprocessing notebook), unchanged.  The notebook's module-level matplotlib font settings of that
cell are applied with ``rc_context`` (``make_fig4cd``).  The stars of panel D come from PAIRED t-tests (as in the notebook), whereas the text and
caption quote p-values from the unpaired permutation test of cell 22 -- see README.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

SMALLEST_SIZE = 10
SMALL_SIZE = 12
MED_SIZE = 14
BIG_SIZE = 16
RC = {"font.size": SMALL_SIZE, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.titlesize": SMALL_SIZE,
      "axes.labelsize": SMALL_SIZE, "legend.fontsize": SMALLEST_SIZE, "xtick.labelsize": SMALLEST_SIZE, "ytick.labelsize": SMALLEST_SIZE}

COLORS = {
    'peppermint': '#2ECC71',
    'oregano': '#9B59B6',
    'grapefruit': '#E67E22'
}


def create_bottom_row_figure(df_combined, combined_profiles_v2, empirical_profiles_expanded, 
                             semantic_model_v2, save_path=None):
    """
    Bottom row figure:
    Left: Infographic (How We Score Participant Responses)
    Right: Performance violin plots (with significance stars)
    """
    
    # Create figure matching top row dimensions
    fig = plt.figure(figsize=(14, 3.5))
    
    # # GridSpec: 1 row, 2 main columns (infographic | violin plots)
    # gs = GridSpec(1, 2, figure=fig, 
    #               width_ratios=[2,2],  # Left panel wider for infographic
    #               wspace=0.05,
    #               top=0.85, bottom=0.15)
    gs = GridSpec(1, 6, figure=fig, 
                  width_ratios=[1, 1, 0.05, 0.75, 0.75, 0.75],  # EXACT same as top
                  wspace=0.2,  # Same as top figure
                  top=0.85, bottom=0.15,
                  left=0.05, right=0.98)
    
    
    # ========================================================================
    # LEFT PANEL: INFOGRAPHIC - MORE SPREAD OUT
    # ========================================================================
    
    # ax_info = fig.add_subplot(gs[0, 0])
    # ax_info.axis('off')
    # ax_info.set_xlim(0, 12)  # Wider to spread boxes out
    # ax_info.set_ylim(0, 4.5)
    ax_info = fig.add_subplot(gs[0, 0:2])  # Span columns 0 and 1
    ax_info.axis('off')
    ax_info.set_xlim(0, 12)
    ax_info.set_ylim(0, 4.5)
    
    # Calculate ACTUAL scores for "mint" → "peppermint"
    emp_freq = empirical_profiles_expanded.get('peppermint', {}).get('mint', 0.0)
    
    try:
        desc_vec = semantic_model_v2.encode("mint", convert_to_numpy=True).reshape(1, -1)
        odor_vec = semantic_model_v2.encode("peppermint", convert_to_numpy=True).reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = cosine_similarity(desc_vec, odor_vec)[0, 0]
    except:
        cos_sim = 0.636  # fallback
    
    final_score = 0.5 * emp_freq + 0.5 * cos_sim
    
    # Step 1: Participant response - MORE SPREAD OUT
    box1 = FancyBboxPatch((0.1, 2.3), 2, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#E8E8E8', linewidth=2)
    ax_info.add_patch(box1)
    ax_info.text(1.1, 2.9, 'Subject says:\n"mint"', ha='center', va='center',
                fontsize=SMALLEST_SIZE, fontweight='bold')
    
    # Arrow 1 - LONGER
    arrow1 = FancyArrowPatch((2.2, 2.9), (3.1, 2.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
    ax_info.add_artist(arrow1)
    
    # Step 2a: GoodScents - MORE SPREAD OUT
    box2a = FancyBboxPatch((3.2, 2.1), 2.0, 1.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#F5F5F5', linewidth=2)
    ax_info.add_patch(box2a)
    ax_info.text(4.2, 3.35, 'GoodScents', ha='center', fontsize=SMALLEST_SIZE, fontweight='bold')
    ax_info.text(4.2, 3.05, 'Database', ha='center', fontsize=SMALLEST_SIZE, fontweight = 'bold')
    ax_info.text(4.2, 2.65, f'"mint" frequency:', ha='center', fontsize=SMALLEST_SIZE-2)
    ax_info.text(4.2, 2.35, f'{emp_freq:.3f}', ha='center', fontsize=SMALLEST_SIZE, fontweight='bold')
    
    # Arrow 2 - LONGER
    arrow2 = FancyArrowPatch((5.3, 2.9), (6.2, 2.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
    ax_info.add_artist(arrow2)
    
    # Step 2b: Word Embeddings - MORE SPREAD OUT
    box2b = FancyBboxPatch((6.3, 2.1), 2.0, 1.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#F5F5F5', linewidth=2)
    ax_info.add_patch(box2b)
    ax_info.text(7.3, 3.35, 'Word', ha='center', fontsize=SMALLEST_SIZE, fontweight='bold')
    ax_info.text(7.3, 3.05, 'Embeddings', ha='center', fontsize=SMALLEST_SIZE,fontweight = 'bold')
    ax_info.text(7.3, 2.65, 'Cosine similarity:', ha='center', fontsize=SMALLEST_SIZE-2)
    ax_info.text(7.3, 2.35, f'{cos_sim:.3f}', ha='center', fontsize=SMALLEST_SIZE, fontweight='bold')
    
    # Arrow 3 - LONGER
    arrow3 = FancyArrowPatch((8.4, 2.9), (9.3, 2.9),
                            arrowstyle='->', mutation_scale=25, linewidth=2.5,
                            color='black')
    ax_info.add_artist(arrow3)
    
    # Step 3: Final score - MORE SPREAD OUT
    box3 = FancyBboxPatch((9.4, 2.5), 1.7, 0.9, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#D3D3D3', linewidth=2)
    ax_info.add_patch(box3)
    ax_info.text(10.25, 3.1, 'Final Score:', ha='center', fontsize=SMALLEST_SIZE-1, fontweight='bold')
    ax_info.text(10.15, 2.75, f'{final_score:.3f}', ha='center', fontsize=SMALL_SIZE, fontweight='bold')
    
    # Formula at bottom
    ax_info.text(6, 0.9, 'Score = 0.5 × (Empirical Frequency) + 0.5 × (Cosine Similarity)',
                ha='center', fontsize=SMALLEST_SIZE, style='italic',
                bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8, pad=0.4, 
                         edgecolor='black', linewidth=1))
    
    # ========================================================================
    # RIGHT PANEL: PERFORMANCE VIOLIN PLOTS - NO INDIVIDUAL CIRCLES
    # ========================================================================
    
    #ax_violin = fig.add_subplot(gs[0, 1])
    ax_violin = fig.add_subplot(gs[0, 3:6])  # Span columns 3, 4, 5
    
    pivot = df_combined.pivot_table(
        values='similarity',
        index='subject',
        columns='odor',
        aggfunc='mean'
    )
    
    melted = pivot.melt(var_name='Odor', value_name='Similarity')
    melted['Odor'] = melted['Odor'].str.capitalize()
    
    # Order: Peppermint, Oregano, Grapefruit
    odor_order = ['Peppermint', 'Oregano', 'Grapefruit']
    positions = [0, 1, 2]
    
    # Store data for statistics
    data_by_odor = {}
    
    # Plot violins WITHOUT individual points
    for i, odor in enumerate(odor_order):
        data = melted[melted['Odor'] == odor]['Similarity'].values
        data_by_odor[odor] = data
        
        # Violin
        parts = ax_violin.violinplot([data], positions=[i], widths=0.7,
                                     showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[odor.lower()])
            pc.set_alpha(0.7)
            pc.set_edgecolor(COLORS[odor.lower()])
            pc.set_linewidth(2.5)
        
        # Mean line
        mean_val = np.mean(data)
        ax_violin.hlines(mean_val, i-0.35, i+0.35, colors='black', 
                        linewidth=3, zorder=5)
        
        # REMOVED: Individual scatter points
    
    # Add significance bars
    def add_significance_bar(ax, x1, x2, y, p_value, h=0.01):
        if p_value >= 0.05:
            return
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=1.5, color='black')
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        ax.text((x1 + x2) / 2, y-0.025, sig_text, ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Pairwise comparisons
    comparisons = [
        ('Peppermint', 'Oregano', 0, 1, 1.02),
        ('Peppermint', 'Grapefruit', 0, 2, 1.08),
        ('Oregano', 'Grapefruit', 1, 2, 0.96)
    ]
    
    for odor1, odor2, pos1, pos2, y_pos in comparisons:
        t_stat, p_value = stats.ttest_rel(data_by_odor[odor1], data_by_odor[odor2])
        add_significance_bar(ax_violin, pos1, pos2, y_pos, p_value)
    
    # Formatting
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(odor_order, fontsize=SMALL_SIZE, fontweight='bold')
    for label, color in zip(ax_violin.get_xticklabels(), [COLORS[o.lower()] for o in odor_order]):
        label.set_color(color)
    
    ax_violin.set_ylabel('Recognition Accuracy', fontsize=SMALLEST_SIZE, fontweight='bold')
    ax_violin.set_ylim(-0.05, 1.15)
    
    # Simplified legend - NO individual subjects
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=3, label='Mean')
    ]
    ax_violin.legend(handles=legend_elements, loc='upper right', fontsize=SMALLEST_SIZE)
    
    ax_violin.grid(axis='y', alpha=0.3, linestyle=':')
    ax_violin.set_xlim(-0.5, 2.5)
    
    # Overall title
    fig.suptitle('Odor Recognition Methodology & Performance', 
                fontsize=MED_SIZE, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def make_fig4cd(df_combined, combined_profiles_v2, empirical_profiles_expanded, semantic_model_v2, out_prefix=None):
    """Run ``create_bottom_row_figure`` under the notebook's font settings and save it as a PDF."""
    with plt.rc_context(RC):
        fig = create_bottom_row_figure(df_combined=df_combined, combined_profiles_v2=combined_profiles_v2,
                                       empirical_profiles_expanded=empirical_profiles_expanded,
                                       semantic_model_v2=semantic_model_v2)
        if out_prefix:
            fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    return fig
