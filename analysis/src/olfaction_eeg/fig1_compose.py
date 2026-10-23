"""Fig. 1 for the journal, as one vector PDF: the hand-made olfactometer / participant illustration (left) next to the respiration trace drawn from data (right).

The illustration is a PDF exported from the Inkscape drawing (``figure_sources/figure1_left_infographic.pdf``; the editable source is the .svg next to it).  It is placed *unchanged*
(scaled, never redrawn) on a 180 x 96 mm page; the trace is drawn with matplotlib (live text, fonts embedded) and placed beside it.  The page is assembled with ``pypdf``, which is pure
Python: no Inkscape or other converter is needed.  Every text of the trace is reported with its printed size, because the journal asks for 6-12 pt (the illustration contains no live text).
"""
import io
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from scipy.signal import find_peaks

MM = 72.0 / 25.4                       # points per millimetre
W_MM, H_MM = 180.0, 96.0              # page (full-page width of a PNAS figure)
LEFT_W_MM = 118.0                      # width of the illustration
PANEL_X, PANEL_W = 117.0, 63.0         # the trace panel
# the plot itself (the original figure had a wide, short plot)
GRAPH_LEFT_MM, GRAPH_W_MM, GRAPH_H_MM, GRAPH_BOTTOM_MM = 17.5, 45.0, 28.0, 34.0
FS_LABEL, FS_TEXT = 9.0, 7.5
FS_TITLE, FS_AXLABEL, FS_TICK, FS_ANNOT = 10.0, 8.5, 8.0, 8.5        # text of the trace panel (all within the journal's 6-12 pt)
YTICKS = (75, 80)                                                      # as in the original figure
MIN_PT, MAX_PT = 6.0, 12.0


# ------------------------------------------------------------------------------------------------------ respiration trace (right panel)
def draw_respiration(ax, trial, trial_idx=0, pre_ttl=4, post_ttl=4, yticks=YTICKS):
    """The notebook's trace (fig1_resp.plot_example_trial) on a given axes: 'Inhale' sits just above the peak and 'Exhale' just below the trough before the TTL pulse,
    offset in points so they hug the curve whatever the height of the plot."""
    t = np.linspace(-pre_ttl, post_ttl, len(trial), endpoint=False)
    ax.plot(t, trial, color="black", linewidth=1.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ttl_idx = int(np.argmin(np.abs(t - 0)))
    pre, pre_t = np.asarray(trial[:ttl_idx]), t[:ttl_idx]
    peaks, _ = find_peaks(pre, distance=3)
    if len(peaks):
        i = peaks[-1]
        ax.annotate("Inhale", xy=(pre_t[i], pre[i]), xytext=(0, 3), textcoords="offset points", fontsize=FS_ANNOT, fontweight="bold", color="red", ha="center", va="bottom")
    troughs, _ = find_peaks(-pre, distance=3)
    if len(troughs):
        j = troughs[-1]
        ax.annotate("Exhale", xy=(pre_t[j], pre[j]), xytext=(2, -3), textcoords="offset points", fontsize=FS_ANNOT, fontweight="bold", color="red", ha="center", va="top")
    ax.set_title(f"Respiration\nSubject 1, Trial {trial_idx + 1}", fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("Time from TTL pulse (s)", fontsize=FS_AXLABEL, fontweight="bold")
    ax.set_ylabel("Relative Humidity (%)", fontsize=FS_AXLABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
    if yticks is not None:
        ax.set_yticks(list(yticks))
    ax.grid(False)                                                          # no grid lines (the original had none); only the dashed TTL line
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_linewidth(0.6)
    ymin, ymax = float(np.min(trial)), float(np.max(trial))
    ax.set_ylim(ymin - 0.15 * (ymax - ymin), ymax + 0.24 * (ymax - ymin))      # little space below the lowest point; just enough for 'Exhale' under the trough


def respiration_panel_pdf(trial, trial_idx=0, placeholder=False, panel_w_mm=PANEL_W, panel_h_mm=H_MM, graph=None):
    """The trace as a one-page PDF (live text, fonts embedded, transparent background) of panel_w x panel_h mm.  Returns (pdf bytes, [(text, printed size in pt), ...])."""
    rc = {"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(panel_w_mm / 25.4, panel_h_mm / 25.4))
        fig.patch.set_visible(False)
        gl, gw, gh, gb = graph or (GRAPH_LEFT_MM, GRAPH_W_MM, GRAPH_H_MM, GRAPH_BOTTOM_MM)
        ax = fig.add_axes([gl / panel_w_mm, gb / panel_h_mm, gw / panel_w_mm, gh / panel_h_mm])
        ax.patch.set_visible(False)
        draw_respiration(ax, np.asarray(trial, float), trial_idx=trial_idx)
        if placeholder:                                                      # layout previews only: never submit these
            ax.text(0.5, 0.5, "PREVIEW\nplaceholder trace", transform=ax.transAxes, fontsize=11, color="#E3A3AD", ha="center", va="center", rotation=25, fontweight="bold")
        fig.canvas.draw()                                                    # tick labels only exist once the figure has been drawn
        tick_labels = list(ax.xaxis.get_ticklabels()) + list(ax.yaxis.get_ticklabels())
        shown = []                                                           # tick labels inside the plotted range (matplotlib keeps, but does not draw, the others)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            shown += [lab for lab, loc in zip(axis.get_ticklabels(), axis.get_ticklocs()) if lo - 1e-9 <= loc <= hi + 1e-9]
        texts = [t for t in fig.findobj(Text) if t not in tick_labels] + shown
        findings = [(t.get_text().replace("\n", " "), t.get_fontsize()) for t in texts if t.get_text().strip() and t.get_visible()]
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf")
        plt.close(fig)
    return buf.getvalue(), findings


def compose_pdf(left_pdf, trial, out_pdf, trial_idx=0, placeholder=False, graph=None):
    """Write the 180 x 96 mm figure PDF.  Returns the text sizes of the trace panel [(text, pt), ...]."""
    left = PdfReader(left_pdf).pages[0]
    lw, lh = float(left.mediabox.width), float(left.mediabox.height)
    k = LEFT_W_MM * MM / lw
    page = PageObject.create_blank_page(width=W_MM * MM, height=H_MM * MM)
    page.merge_transformed_page(left, Transformation().scale(k, k).translate(0, (H_MM * MM - lh * k) / 2.0))
    panel, findings = respiration_panel_pdf(trial, trial_idx, placeholder, graph=graph)
    page.merge_transformed_page(PdfReader(io.BytesIO(panel)).pages[0], Transformation().translate(PANEL_X * MM, 0))
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({"/Title": "Figure 1"})                              # nothing else: no author, no dates, no file paths
    with open(out_pdf, "wb") as f:
        writer.write(f)
    return findings


def summarize_findings(findings):
    """{printed size (pt): [texts]}, number below the journal's minimum, number above its maximum."""
    rows = {}
    for text, pt in findings:
        rows.setdefault(round(pt, 1), []).append(text)
    small = sum(len(v) for k, v in rows.items() if k < MIN_PT)
    big = sum(len(v) for k, v in rows.items() if k > MAX_PT)
    return rows, small, big


def illustration_matches_svg(left_pdf, left_svg):
    """True if the illustration PDF has the same aspect ratio as the SVG source (a guard against editing one and forgetting to re-export the other)."""
    page = PdfReader(left_pdf).pages[0]
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', open(left_svg, encoding="utf-8").read(5000)).group(1).split()]
    return abs(float(page.mediabox.width) / float(page.mediabox.height) - vb[2] / vb[3]) < 0.005
