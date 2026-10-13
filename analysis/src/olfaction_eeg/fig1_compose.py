"""Fig. 1 for the journal: the hand-made olfactometer / participant illustration (SVG, left) next to the respiration trace drawn from data (right), as one vector figure.

The illustration is placed *unchanged* (as a scaled group) into a 180 x 96 mm page whose user unit is the millimetre; the trace is drawn with matplotlib as SVG (live text) and placed
beside it.  The composed SVG is exported to PDF / EPS / PNG with Inkscape (if installed) or CairoSVG.  ``audit_text`` reports the printed size of every text element, because the journal
asks for 6-12 pt.
"""
import io
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

SVG = "http://www.w3.org/2000/svg"
SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
W_MM, H_MM = 180.0, 96.0              # page (full-page width of a PNAS figure)
LEFT_W_MM = 118.0                      # width of the illustration
PANEL_X, PANEL_W = 117.0, 63.0         # the trace panel
# the plot itself (the original figure had a wide, short plot: about 2.6 : 1)
GRAPH_LEFT_MM, GRAPH_W_MM, GRAPH_H_MM, GRAPH_BOTTOM_MM = 17.5, 45.0, 28.0, 34.0
FS_LABEL, FS_TEXT = 9.0, 7.5
FS_TITLE, FS_AXLABEL, FS_TICK, FS_ANNOT = 10.0, 8.5, 8.0, 8.5        # text of the trace panel (all within the journal's 6-12 pt)
YTICKS = (75, 80)                                                      # as in the original figure
MIN_PT, MAX_PT = 6.0, 12.0
MAC_INKSCAPE = "/Applications/Inkscape.app/Contents/MacOS/inkscape"


def _t(name):
    return f"{{{SVG}}}{name}"


def _register_namespaces(*paths):
    ET.register_namespace("", SVG)
    for p in ("xlink", "http://www.w3.org/1999/xlink"), ("inkscape", "http://www.inkscape.org/namespaces/inkscape"), ("sodipodi", SODIPODI), \
             ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"), ("cc", "http://creativecommons.org/ns#"), ("dc", "http://purl.org/dc/elements/1.1/"):
        ET.register_namespace(*p)
    for path in paths:
        for pre, uri in re.findall(r'xmlns:(\w+)="([^"]+)"', open(path, encoding="utf-8", errors="ignore").read(4000)):
            ET.register_namespace(pre, uri)


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


def respiration_panel_svg(trial, trial_idx=0, placeholder=False, panel_w_mm=PANEL_W, panel_h_mm=H_MM, graph=None):
    """The trace as an SVG root element (live text, transparent background), sized panel_w x panel_h mm (1 pt = 1 user unit)."""
    rc = {"svg.fonttype": "none", "svg.hashsalt": "fig1", "font.family": "sans-serif",
          "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(panel_w_mm / 25.4, panel_h_mm / 25.4))
        fig.patch.set_visible(False)
        gl, gw, gh, gb = graph or (GRAPH_LEFT_MM, GRAPH_W_MM, GRAPH_H_MM, GRAPH_BOTTOM_MM)
        ax = fig.add_axes([gl / panel_w_mm, gb / panel_h_mm, gw / panel_w_mm, gh / panel_h_mm])
        ax.patch.set_visible(False)
        draw_respiration(ax, np.asarray(trial, float), trial_idx=trial_idx)
        if placeholder:                                                      # layout previews only: never submit these
            ax.text(0.5, 0.5, "PREVIEW\nplaceholder trace", transform=ax.transAxes, fontsize=11, color="#E3A3AD", ha="center", va="center", rotation=25, fontweight="bold")
        buf = io.BytesIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)
    return ET.fromstring(buf.getvalue())


# ------------------------------------------------------------------------------------------------------ composition
def compose_svg(left_svg, trial, out_svg, trial_idx=0, placeholder=False, drop_tiny_text=False, graph=None):
    """Write the composed 180 x 96 mm SVG.  Returns the list of text-size findings (see ``audit_text``)."""
    _register_namespaces(left_svg)
    left = ET.parse(left_svg).getroot()
    vb = [float(v) for v in left.get("viewBox").split()]
    k = LEFT_W_MM / vb[2]
    y0 = (H_MM - vb[3] * k) / 2.0
    outer = ET.Element(_t("svg"), {"width": f"{W_MM:g}mm", "height": f"{H_MM:g}mm", "viewBox": f"0 0 {W_MM:g} {H_MM:g}", "version": "1.1"})
    ET.SubElement(outer, _t("rect"), {"x": "0", "y": "0", "width": f"{W_MM:g}", "height": f"{H_MM:g}", "fill": "#ffffff"})
    gl = ET.SubElement(outer, _t("g"), {"id": "panel_left_illustration", "transform": f"translate(0,{y0:.4f}) scale({k:.7f})"})
    for child in left:
        if child.tag in (_t("metadata"), f"{{{SODIPODI}}}namedview"):
            continue
        gl.append(child)
    right = respiration_panel_svg(trial, trial_idx, placeholder, graph=graph)
    kr = 25.4 / 72.0                                                          # matplotlib's SVG is in points
    gr = ET.SubElement(outer, _t("g"), {"id": "panel_right_respiration", "transform": f"translate({PANEL_X:g},0) scale({kr:.7f})", "stroke-linejoin": "round"})
    for child in right:
        if child.tag in (_t("metadata"), _t("style")):
            continue
        for sty in list(child.iter(_t("style"))):                             # matplotlib's global CSS would also restyle the illustration
            for parent in child.iter():
                if sty in list(parent):
                    parent.remove(sty)
        gr.append(child)
    findings = audit_text(outer)
    if drop_tiny_text:
        tiny = {id(f["element"]) for f in findings if f["pt"] < MIN_PT}
        for parent in outer.iter():
            for el in list(parent):
                if id(el) in tiny:
                    parent.remove(el)
        findings = audit_text(outer)
    ET.ElementTree(outer).write(out_svg, encoding="utf-8", xml_declaration=True)
    return findings


# ------------------------------------------------------------------------------------------------------ text size audit
def _font_size(el):
    st = el.get("style") or ""
    m = re.search(r"(?<![\w-])font-size:\s*([\d.]+)", st) or re.search(r"(?<![\w-])font:\s*(?:[a-z0-9 ]*?\s)?([\d.]+)px", st)
    if m:
        return float(m.group(1))
    v = el.get("font-size")
    m = re.match(r"[\d.]+", v) if v else None
    return float(m.group(0)) if m else None


def _scale(el):
    s = 1.0
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", el.get("transform") or ""):
        a = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        if name == "scale":
            s *= (a[0] * a[-1]) ** 0.5 if len(a) > 1 else a[0]
        elif name == "matrix":
            s *= abs(a[0] * a[3] - a[1] * a[2]) ** 0.5
    return s


def audit_text(root):
    """Printed size (pt) of every <text>, taking nested scale transforms into account (user unit of the page = 1 mm)."""
    out = []

    def walk(el, scale, font):
        s = scale * _scale(el)
        f = _font_size(el) or font
        if el.tag == _t("text"):
            sizes = [x for x in (_font_size(e) for e in el.iter()) if x] or ([f] if f else [])
            if sizes:
                out.append({"text": "".join(el.itertext()).strip(), "pt": min(sizes) * s * 72.0 / 25.4, "element": el})
        for c in el:
            walk(c, s, f)

    walk(root, 1.0, None)
    return out


def summarize_findings(findings):
    rows = {}
    for f in findings:
        rows.setdefault(round(f["pt"], 1), []).append(f["text"])
    small = sum(len(v) for k, v in rows.items() if k < MIN_PT)
    big = sum(len(v) for k, v in rows.items() if k > MAX_PT)
    return rows, small, big


# ------------------------------------------------------------------------------------------------------ export
def find_inkscape(explicit=None):
    for c in (explicit, shutil.which("inkscape"), MAC_INKSCAPE):
        if c and os.path.exists(c):
            return c
    return None


def export(svg_path, out_prefix, formats=("pdf", "eps", "png"), png_dpi=600, inkscape=None, text_to_path=False):
    """PDF / EPS / PNG from the composed SVG with Inkscape if available, else CairoSVG.  Returns {format: path or None}."""
    ink = find_inkscape(inkscape)
    made = {}
    for fmt in formats:
        if fmt == "svg":
            continue
        out = f"{out_prefix}.{fmt}"
        made[fmt] = None
        if ink:
            cmd = [ink, svg_path, f"--export-type={fmt}", f"--export-filename={out}"]
            if fmt == "png":
                cmd += [f"--export-dpi={png_dpi}", "--export-background=white"]
            elif text_to_path:                                                 # all text becomes outlines: no font files needed, nothing left to flag as small text
                cmd += ["--export-text-to-path"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(out):
                made[fmt] = out
                continue
        try:
            import cairosvg
            if fmt == "pdf":
                cairosvg.svg2pdf(url=svg_path, write_to=out)
            elif fmt == "eps":
                cairosvg.svg2eps(url=svg_path, write_to=out)
            elif fmt == "png":
                cairosvg.svg2png(url=svg_path, write_to=out, dpi=png_dpi, background_color="white")
            made[fmt] = out
        except Exception:
            made[fmt] = None
    return made
