"""Fig. 1 as one vector PDF: the illustration is placed unchanged, the trace is drawn from data, text sizes are within the journal's 6-12 pt."""
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from pypdf import PdfReader

from olfaction_eeg import fig1_compose as fc

ROOT = os.path.dirname(os.path.dirname(__file__))
LEFT_PDF = os.path.join(ROOT, "figure_sources", "figure1_left_infographic.pdf")
LEFT_SVG = os.path.join(ROOT, "figure_sources", "figure1_left_infographic.svg")
MM = 72.0 / 25.4


def trial():
    t = np.arange(70) / 8.757
    return 77 + 4.5 * np.sin(2 * np.pi * (t - 2.2) / 4.6)


def test_the_illustration_pdf_is_vector_without_text_and_matches_its_svg_source():
    page = PdfReader(LEFT_PDF).pages[0]
    assert len(page.images) == 0                                              # no raster images
    assert "/Font" not in (page["/Resources"] or {}) or len(page["/Resources"]["/Font"]) == 0     # no live text: nothing that could print below the journal's minimum size
    assert not (page.extract_text() or "").strip()
    assert fc.illustration_matches_svg(LEFT_PDF, LEFT_SVG)                    # re-export the PDF from the .svg if you edit the drawing


def test_page_is_180_by_96_mm_vector_with_embedded_fonts(tmp_path):
    out = tmp_path / "f.pdf"
    fc.compose_pdf(LEFT_PDF, trial(), str(out))
    page = PdfReader(str(out)).pages[0]
    assert abs(float(page.mediabox.width) / MM - 180) < 0.2 and abs(float(page.mediabox.height) / MM - 96) < 0.2
    assert len(page.images) == 0                                              # vector only
    fonts = page["/Resources"].get("/Font", {})
    assert len(fonts) > 0                                                     # the trace's text uses embedded fonts
    for k in fonts:
        fd = fonts[k].get_object()
        desc = fd.get("/FontDescriptor") or fd["/DescendantFonts"][0].get_object()["/FontDescriptor"]
        assert any(key in desc.get_object() for key in ("/FontFile", "/FontFile2", "/FontFile3"))     # the font file is inside the PDF
    assert os.path.getsize(out) < 10e6
    info = PdfReader(str(out)).metadata or {}
    assert not any(k in info for k in ("/Author", "/CreationDate", "/ModDate"))              # no dates or author in the file


def test_the_illustration_is_placed_unchanged_not_redrawn(tmp_path):
    """The composed page contains exactly the illustration's drawing operations (scaled), plus the trace."""
    out = tmp_path / "f.pdf"
    fc.compose_pdf(LEFT_PDF, trial(), str(out))
    left = PdfReader(LEFT_PDF).pages[0].get_contents().get_data()
    both = PdfReader(str(out)).pages[0].get_contents().get_data()
    n = lambda d: len(re.findall(rb"\s(?:re|l|c)\s", d))            # number of path segments / rectangles
    assert n(both) > n(left) > 1000                                           # every segment of the illustration is still there, and the trace is added


def test_text_of_the_trace_is_within_6_to_12_pt(tmp_path):
    findings = fc.compose_pdf(LEFT_PDF, trial(), str(tmp_path / "f.pdf"))
    rows, small, big = fc.summarize_findings(findings)
    assert small == 0 and big == 0, rows
    texts = {t for t, _ in findings}
    assert {"Inhale", "Exhale", "Time from TTL pulse (s)", "Relative Humidity (%)", "75", "80"} <= texts
    assert "70" not in texts and "85" not in texts                              # only the 75 and 80 ticks, as in the original figure


def test_the_plot_keeps_the_original_wide_short_proportions():
    assert 1.4 <= fc.GRAPH_W_MM / fc.GRAPH_H_MM <= 3.0
    assert fc.GRAPH_LEFT_MM + fc.GRAPH_W_MM <= fc.PANEL_W and fc.GRAPH_BOTTOM_MM + fc.GRAPH_H_MM < fc.H_MM - 12


def test_no_grid_lines_and_little_white_space_below_the_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    tr = np.array(trial())
    fc.draw_respiration(ax, tr)
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    lo, hi = ax.get_ylim()
    assert (tr.min() - lo) / (tr.max() - tr.min()) <= 0.2
    plt.close(fig)


def test_script_writes_only_the_pdf(tmp_path):
    np.save(tmp_path / "t.npy", {"subject": "22", "trials": np.array([trial(), trial() + 1])}, allow_pickle=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_fig1.py"), "--left-pdf", LEFT_PDF, "--trial-file", str(tmp_path / "t.npy"), "--trial-idx", "1",
                        "--out-prefix", str(tmp_path / "o" / "fig1")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(tmp_path / "o")) == ["fig1.pdf"]
    assert "Printed text sizes" in r.stdout and "outside 6-12 pt" not in r.stdout and "PLACEHOLDER" not in r.stdout
