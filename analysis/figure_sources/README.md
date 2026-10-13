# Figure sources

* `figure1_left_infographic.svg` -- the hand-made vector illustration of the olfactometer and the participant (left side of Fig. 1; labels A-I as in the figure legend), edited in Inkscape.
  **All text and shapes are converted to paths** (no live text, so no fonts are needed and nothing can print below the journal's minimum text size). This is the file `scripts/make_fig1.py` uses.
* `figure1_left_infographic_live_text.svg` -- the previous version of the same drawing with live (editable) text; keep it if you want to change a label, then outline the text again before using it.
  Its 30 tiny text items (flow-meter digits, "L/min", "UNO", "Arduino") print at 1-2 pt, which is why the outlined version exists.

`scripts/make_fig1.py` places the illustration, unchanged, next to the respiration trace that is drawn from data, and exports the finished figure (`figures/fig1.svg`, `.pdf`, `.eps`).
Edit the illustration in Inkscape / Illustrator and rerun the script; do not edit the composed `figures/fig1.*` files.
