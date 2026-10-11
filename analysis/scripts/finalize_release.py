#!/usr/bin/env python
"""Fill in the release placeholders in one go: LICENSE, repository URL, DOIs, pyrfume-data commit, and the remaining package pins.

Dry run by default (shows what would change); add --apply to write.  Example:

    python scripts/finalize_release.py --license MIT --holder "Alisa Y. Hathaway and Todd P. Coleman" \
        --repo-url https://github.com/<user>/<repo> --paper-doi 10.1073/pnas.XXXXXXXXXX --data-doi 10.5281/zenodo.XXXXXXX \
        --pyrfume-commit <sha> --versions results/versions.txt --apply

Every option is optional; whatever is not given is left as `TODO` (and listed at the end).  Paper / data DOIs: pass the bare DOI (10.xxxx/...).
"""
import argparse
import json
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIT = """MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
BSD3 = """BSD 3-Clause License

Copyright (c) {year}, {holder}
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
CC_HEADER = """Copyright (c) {year} {holder}

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
(SPDX: CC-BY-NC-4.0). To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.  The full legal text follows.

------------------------------------------------------------------------------------------------------------------

"""
LICENSES = {"MIT": MIT, "BSD-3-Clause": BSD3, "CC-BY-NC-4.0": None}   # CC-BY-NC-4.0: header + the official text in licenses/
PIP_NAME = {"pywt": "pywavelets", "yaml": "pyyaml", "sklearn": "scikit-learn"}


def doi_url(doi):
    return f"https://doi.org/{doi}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--license", choices=sorted(LICENSES))
    ap.add_argument("--holder", help="copyright holder(s) for LICENSE, e.g. 'Alisa Y. Hathaway and Todd P. Coleman' or 'Stanford University'")
    ap.add_argument("--repo-url")
    ap.add_argument("--paper-doi", help="bare DOI, or 'forthcoming' if it is not assigned yet")
    ap.add_argument("--software-doi", help="DOI of the archived code release (e.g. from Zenodo's GitHub integration); optional, can be added after the first release")
    ap.add_argument("--data-doi")
    ap.add_argument("--data-license", default="CC-BY-4.0", help="license of the data deposit (stated in DEPOSIT.md and the README)")
    ap.add_argument("--pyrfume-commit")
    ap.add_argument("--versions", help="results/versions.txt of a real run: pins h5py, pywavelets, pyyaml, snakemake, fire, statsmodels in environment.yml")
    ap.add_argument("--analysis-subdir", default=None, help="folder of this code inside a larger repository (e.g. analysis): the README's clone instruction then ends with `cd <repo>/<subdir>`")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    R = a.root
    changed, notes = [], []

    def read(p):
        return open(os.path.join(R, p), encoding="utf-8").read()

    def write(p, text):
        path = os.path.join(R, p)
        if os.path.exists(path) and open(path, encoding="utf-8").read() == text:
            return                                                       # already up to date
        if a.apply:
            open(path, "w", encoding="utf-8").write(text)
        changed.append(p)

    # ---- LICENSE
    if a.license:
        if not a.holder:
            sys.exit("--license needs --holder (the name that goes into the copyright line)")
        if a.license == "CC-BY-NC-4.0":
            body = CC_HEADER.format(year=date.today().year, holder=a.holder) + open(os.path.join(ROOT, "licenses", "CC-BY-NC-4.0.txt"), encoding="utf-8").read()
        else:
            body = LICENSES[a.license].format(year=date.today().year, holder=a.holder)
        write("LICENSE", body)

    # ---- README
    readme = read("README.md")
    r = readme
    forthcoming = bool(a.paper_doi) and a.paper_doi.lower() in ("forthcoming", "none")
    if a.paper_doi and forthcoming:
        r = r.replace("PNAS (2026). DOI: `TODO`", "PNAS (2026). DOI forthcoming")
        r = r.replace("Please cite the paper (DOI `TODO`)", "Please cite the paper (DOI forthcoming)")
    elif a.paper_doi:
        r = r.replace("PNAS (2026). DOI: `TODO`", f"PNAS (2026). [doi:{a.paper_doi}]({doi_url(a.paper_doi)})")
        r = r.replace("Please cite the paper (DOI `TODO`)", f"Please cite the paper ([doi:{a.paper_doi}]({doi_url(a.paper_doi)}))")
    if a.data_doi:
        r = r.replace("(DOI: `TODO`; contents", f"([doi:{a.data_doi}]({doi_url(a.data_doi)}); contents")
        r = r.replace("and the data deposit (DOI `TODO`)", f"and the data deposit ([doi:{a.data_doi}]({doi_url(a.data_doi)}))")
    if a.repo_url:
        name = a.repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
        sub = f"/{a.analysis_subdir.strip('/')}" if a.analysis_subdir else ""
        r = r.replace("git clone https://github.com/TODO/TODO.git && cd TODO", f"git clone {a.repo_url.rstrip('/').removesuffix('.git')}.git && cd {name}{sub}")
    if a.license:
        r = r.replace("License: `TODO`.", f"License: {a.license} (code), {a.data_license} (data deposit).")
    if r != readme:
        write("README.md", r)

    # ---- CITATION.cff
    cff = read("CITATION.cff")
    c = cff
    if a.repo_url:
        c = re.sub(r'repository-code: ".*?".*', f'repository-code: "{a.repo_url.rstrip("/")}"', c)
    if a.license:
        c = re.sub(r"# license: TODO.*", f"license: {a.license}", c)
    if a.paper_doi:
        c = re.sub(r'  # doi: "TODO".*paper DOI', "" if forthcoming else f'  doi: "{a.paper_doi}"', c)
        c = c.replace("\n\n", "\n") if forthcoming else c
    if a.software_doi:
        c = re.sub(r'# doi: "TODO".*archived release.*', f'doi: "{a.software_doi}"', c)
    elif a.apply and a.repo_url:
        c = re.sub(r'# doi: "TODO".*archived release.*\n', "", c)               # added after the first release if wanted
    if c != cff:
        write("CITATION.cff", c)

    # ---- DEPOSIT.md: license + DOI line
    dep = read("DEPOSIT.md")
    d = dep
    if (a.data_doi or a.license) and "**License of the deposit:**" not in d:
        line = f"**License of the deposit:** {a.data_license}." + (f" DOI: [{a.data_doi}]({doi_url(a.data_doi)})." if a.data_doi else "") + "\n\n"
        i = d.index("\n", d.index("# "))  # after the title line
        d = d[: i + 1] + "\n" + line + d[i + 1:].lstrip("\n")
    if d != dep:
        write("DEPOSIT.md", d)

    # ---- pyrfume commit
    if a.pyrfume_commit:
        p = "config/goodscents_selected_source.json"
        j = json.loads(read(p))
        j["commit"] = a.pyrfume_commit
        j["ref"] = a.pyrfume_commit
        write(p, json.dumps(j, indent=2) + "\n")

    # ---- pins from versions.txt
    if a.versions:
        vers = {}
        for line in open(a.versions, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2 and re.match(r"^\d", parts[1]):
                vers[PIP_NAME.get(parts[0], parts[0])] = parts[1]
        env = read("environment.yml")
        e = env
        for pkg in ("h5py", "pywavelets", "pyyaml", "fire", "statsmodels"):
            if pkg in vers:
                e = re.sub(rf"(\n      - {pkg})(\s*(#.*)?\n)", rf"\1=={vers[pkg]}\2", e, count=1)
        if "snakemake" in vers:
            e = re.sub(r"(\n      - snakemake)>=8", rf"\1=={vers['snakemake']}", e, count=1)
        if e != env:
            write("environment.yml", e)
        else:
            notes.append("environment.yml: nothing to pin (already pinned, or the packages are missing from versions.txt)")

    print(("APPLIED" if a.apply else "DRY RUN (add --apply to write)") + ": " + (", ".join(changed) if changed else "nothing to change"))
    for n in notes:
        print("  note:", n)
    left = []
    for dp, _, fs in os.walk(R):
        if any(x in dp for x in (".git", "__pycache__", "tests/data")):
            continue
        for f in fs:
            if f.endswith((".md", ".cff", ".yml", ".yaml", ".json", ".py")) and f not in ("NOTES_FOR_AUTHORS.md", "finalize_release.py", "test_finalize.py"):
                path = os.path.join(dp, f)
                for k, line in enumerate(open(path, encoding="utf-8", errors="ignore"), 1):
                    if "TODO" in line:
                        left.append(f"{os.path.relpath(path, R)}:{k}: {line.strip()[:110]}")
    if not os.path.exists(os.path.join(R, "LICENSE")) and not (a.license and a.apply):
        left.append("LICENSE: missing (pass --license and --holder)")
    print("\nStill open:" if left else "\nNo TODO placeholders left.")
    for l in left:
        print("  " + l)


if __name__ == "__main__":
    main()
