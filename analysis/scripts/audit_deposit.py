#!/usr/bin/env python
"""Look through the data deposit for anything that could identify a participant or a session: acquisition dates / clock times, names, free text, experimenter fields.

    python scripts/audit_deposit.py --root deposit                 # skips files > 1 GB (the epochs file); add --include-large to open it too

What it looks at:
  * every info/*.fif (MNE measurement info): meas_date, meas_id (contains a date), subject_info, experimenter, description, proj_name, device_info
  * every .npy dictionary: all keys (recursively), any text, and any numeric array that looks like a calendar time (unix seconds ~ 2001-2040) or is called *time*/*date*
The script only reports; it changes nothing.  What counts as identifying is a decision for you and your IRB: absolute dates of a session are the usual concern
(HIPAA's safe-harbor rule, for example, lists all date elements except the year); relative times from the start of a recording are not.
"""
import argparse
import datetime as dt
import glob
import os
import sys

import numpy as np

UNIX_LO, UNIX_HI = 1.0e9, 2.2e9          # 2001-09 .. 2039


def looks_like_calendar_time(a):
    a = np.asarray(a)
    if a.dtype.kind not in "fiu" or a.size == 0:
        return False
    v = a.astype(float).ravel()
    v = v[np.isfinite(v)]
    return v.size > 0 and bool(np.all((v > UNIX_LO) & (v < UNIX_HI)))


def walk(obj, path, findings, texts, depth=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}/{k}", findings, texts, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:50]):
            walk(v, f"{path}[{i}]", findings, texts, depth + 1)
    elif isinstance(obj, (str, bytes)):
        texts.append((path, str(obj)[:80]))
    elif isinstance(obj, (dt.datetime, dt.date, np.datetime64)):
        findings.append((path, f"date/time object: {obj}"))
    elif isinstance(obj, np.ndarray):
        name = path.lower()
        if obj.dtype.kind in "OUS":
            vals = {str(x)[:60] for x in obj.ravel()[:20]}
            texts.append((path, "; ".join(sorted(vals))[:120]))
        elif obj.dtype.kind == "M":
            findings.append((path, f"datetime64 array, e.g. {obj.ravel()[0]}"))
        elif looks_like_calendar_time(obj):
            first = float(obj.astype(float).ravel()[0])
            findings.append((path, f"values look like UNIX CLOCK TIMES, e.g. {first:.0f} = {dt.datetime.utcfromtimestamp(first):%Y-%m-%d %H:%M:%S} UTC"))
        elif ("time" in name or "date" in name) and obj.dtype.kind in "fiu" and obj.size:
            v = obj.astype(float).ravel()
            findings.append((path, f"named like a time; range {np.nanmin(v):.3f} .. {np.nanmax(v):.3f} (relative to recording start? please check)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--include-large", action="store_true")
    a = ap.parse_args()
    problems = 0

    infos = sorted(glob.glob(os.path.join(a.root, "info", "*.fif")))
    if infos:
        import mne
        mne.set_log_level("ERROR")
        print(f"== MNE info files ({len(infos)})")
        for f in infos:
            i = mne.io.read_info(f)
            mid = (i.get("meas_id") or {}).get("secs") if i.get("meas_id") else None
            mid = f"{dt.datetime.utcfromtimestamp(mid):%Y-%m-%d %H:%M} UTC" if mid and UNIX_LO < mid < UNIX_HI else None   # 0 = MNE's empty placeholder
            fields = {"meas_date": i.get("meas_date"), "meas_id": mid,
                      "subject_info": i.get("subject_info"), "experimenter": i.get("experimenter"), "description": i.get("description"),
                      "proj_name": i.get("proj_name"), "device_info": i.get("device_info")}
            hit = {k: v for k, v in fields.items() if v not in (None, "", {})}
            if hit:
                problems += 1
                print(f"  [CHECK] {os.path.basename(f)}: " + "; ".join(f"{k}={str(v)[:60]}" for k, v in hit.items()))
        if not problems:
            print("  none of meas_date / meas_id / subject_info / experimenter / description / proj_name / device_info is set")

    for f in sorted(glob.glob(os.path.join(a.root, "**", "*.npy"), recursive=True)):
        size = os.path.getsize(f)
        rel = os.path.relpath(f, a.root)
        if size > 1e9 and not a.include_large:
            print(f"== {rel}: skipped ({size / 1e9:.1f} GB; use --include-large)")
            continue
        try:
            obj = np.load(f, allow_pickle=True)
            obj = obj.item() if obj.dtype == object and obj.shape == () else obj
        except Exception as e:
            print(f"== {rel}: could not be read ({e})")
            continue
        findings, texts = [], []
        walk(obj, "", findings, texts)
        # collapse repeated paths of one kind (per subject / condition) to keep the report short
        seen = {}
        for p, msg in findings:
            key = (p.split("/")[1] if p.count("/") > 0 else p, msg.split(",")[0].split(";")[0])
            seen.setdefault(key, (p, msg, 0))
            seen[key] = (seen[key][0], seen[key][1], seen[key][2] + 1)
        print(f"== {rel}: {len(findings)} date/time-like findings, {len(texts)} text values")
        for (top, _), (p, msg, n) in seen.items():
            problems += 1
            print(f"  [CHECK] {p}  ({n}x)  {msg}")
        for p, t in texts[:8]:
            print(f"  [text] {p}: {t}")
        if len(texts) > 8:
            print(f"  [text] ... {len(texts) - 8} more")

    print("\n" + ("Nothing date-like or personal found." if not problems else
                  f"{problems} item(s) to look at above: absolute dates / clock times of sessions, or identifying fields, should be removed or shifted before release."))
    sys.exit(0)


if __name__ == "__main__":
    main()
