"""Fig. 1 composition: the illustration is placed unchanged, the trace is drawn from data, text sizes are audited, exports work where a converter exists."""
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from olfaction_eeg import fig1_compose as fc

ROOT = os.path.dirname(os.path.dirname(__file__))
LEFT = os.path.join(ROOT, "figure_sources", "figure1_left_infographic.svg")
LIVE = os.path.join(ROOT, "figure_sources", "figure1_left_infographic_live_text.svg")        # the same drawing with live text (30 tiny items)
SVGNS = "{http://www.w3.org/2000/svg}"


def trial():
    t = np.arange(70) / 8.757
    return 77 + 4.5 * np.sin(2 * np.pi * (t - 2.2) / 4.6)


def count(root, tag):
    return sum(1 for _ in root.iter(SVGNS + tag))


def test_illustration_is_placed_unchanged_and_page_is_180_by_96_mm(tmp_path):
    out = tmp_path / "f.svg"
    fc.compose_svg(LEFT, trial(), str(out))
    root = ET.parse(out).getroot()
    assert root.get("width") == "180mm" and root.get("height") == "96mm"
    left = [g for g in root.iter(SVGNS + "g") if g.get("id") == "panel_left_illustration"][0]
    orig = ET.parse(LEFT).getroot()
    for tag in ("path", "rect", "ellipse", "circle", "text", "g"):
        assert count(left, tag) == count(orig, tag) + (1 if tag == "g" else 0), tag     # same drawing elements: none lost, none added (+1 = the wrapper group)
    assert count(left, "path") > 500
    assert not any("image" in e.tag for e in root.iter())                                # no raster images
    assert [g for g in root.iter(SVGNS + "g") if g.get("id") == "panel_right_respiration"]


def test_the_committed_illustration_has_no_text_the_journal_could_flag(tmp_path):
    """The illustration used for the figure must not contain text below 6 pt (the live-text archive does: that is why the outlined file exists)."""
    f = fc.compose_svg(LEFT, trial(), str(tmp_path / "f.svg"))
    rows, small, big = fc.summarize_findings(f)
    assert small == 0 and big == 0, rows
    for pt in rows:
        assert fc.MIN_PT <= pt <= fc.MAX_PT, (pt, rows[pt])                              # what text remains (the trace) is within 6-12 pt
    assert count(ET.parse(LEFT).getroot(), "text") == 0                                  # the labels A-I etc. are outlines


def test_text_audit_flags_tiny_text_in_the_live_text_version(tmp_path):
    f = fc.compose_svg(LIVE, trial(), str(tmp_path / "f.svg"))
    rows, small, big = fc.summarize_findings(f)
    assert small == 30 and big == 0                                      # 24 digits, 4 x "L/min", "UNO", "Arduino"
    assert {"A", "B", "C", "D", "E", "F", "G", "H", "I"} <= {f_["text"] for f_ in f}


def test_drop_tiny_text_removes_exactly_those_items(tmp_path):
    f = fc.compose_svg(LIVE, trial(), str(tmp_path / "f.svg"), drop_tiny_text=True)
    assert fc.summarize_findings(f)[1] == 0
    root = ET.parse(tmp_path / "f.svg").getroot()
    left = [g for g in root.iter(SVGNS + "g") if g.get("id") == "panel_left_illustration"][0]
    assert count(left, "text") == 43 - 30                                      # the 13 readable labels remain
    assert count(left, "path") == count(ET.parse(LIVE).getroot(), "path")      # the artwork itself is untouched


def test_export_pdf_has_the_page_size_embedded_fonts_and_no_images(tmp_path):
    pytest.importorskip("cairosvg")
    pypdf = pytest.importorskip("pypdf")
    fc.compose_svg(LEFT, trial(), str(tmp_path / "f.svg"))
    made = fc.export(str(tmp_path / "f.svg"), str(tmp_path / "f"), formats=("pdf", "png"), png_dpi=100)
    reader = pypdf.PdfReader(made["pdf"])
    page = reader.pages[0]
    assert abs(float(page.mediabox.width) / 72 * 25.4 - 180) < 0.5 and abs(float(page.mediabox.height) / 72 * 25.4 - 96) < 0.5
    fonts = page["/Resources"].get("/Font", {})
    assert len(fonts) > 0 and all("/FontDescriptor" in fonts[k].get_object() or "/DescendantFonts" in fonts[k].get_object() for k in fonts)
    assert len(page.images) == 0                                              # vector only
    assert os.path.getsize(made["pdf"]) < 10e6


def test_script_writes_the_svg_and_reports_sizes(tmp_path):
    np.save(tmp_path / "t.npy", {"subject": "22", "trials": np.array([trial(), trial() + 1])}, allow_pickle=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_fig1.py"), "--left-svg", LEFT, "--trial-file", str(tmp_path / "t.npy"), "--trial-idx", "1",
                        "--out-prefix", str(tmp_path / "o" / "fig1"), "--formats", "svg"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "fig1.svg").exists() and "Printed text sizes" in r.stdout and "would print below 6 pt" not in r.stdout and "PLACEHOLDER" not in r.stdout


def test_the_plot_keeps_the_original_wide_short_proportions():
    assert 1.4 <= fc.GRAPH_W_MM / fc.GRAPH_H_MM <= 3.0                       # a wide plot, not a tall one
    assert fc.GRAPH_LEFT_MM + fc.GRAPH_W_MM <= fc.PANEL_W and fc.GRAPH_BOTTOM_MM + fc.GRAPH_H_MM < fc.H_MM - 12      # fits the panel, leaves room for the title


def test_only_the_75_and_80_ticks_are_labelled_on_the_humidity_axis():
    root = fc.respiration_panel_svg(trial())
    labels = ["".join(e.itertext()).strip() for e in root.iter(SVGNS + "text")]
    assert "75" in labels and "80" in labels and "70" not in labels and "85" not in labels
    assert {"Inhale", "Exhale"} <= set(labels)


def test_no_grid_lines_and_little_white_space_below_the_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    tr = np.array(trial())
    fc.draw_respiration(ax, tr)
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    lo, hi = ax.get_ylim()
    span = tr.max() - tr.min()
    assert (tr.min() - lo) / span <= 0.2                                     # the empty band between the lowest point and the axis stays small
    plt.close(fig)
