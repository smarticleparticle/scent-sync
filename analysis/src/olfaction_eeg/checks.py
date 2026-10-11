"""Compare pipeline results with numbers published in the paper (config/paper_values.yaml)."""
import numpy as np
import pandas as pd


class Checks:
    """Collects (name, got, expected, ok) rows; numbers use an absolute tolerance, strings must match exactly."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.rows = []

    def number(self, name, got, expected, tol):
        if not self.enabled or expected is None:
            return
        got = float(got)
        self.rows.append({"check": name, "got": got, "expected": float(expected),
                          "ok": bool(abs(got - float(expected)) <= tol)})

    def less_than(self, name, got, bound):
        if not self.enabled or bound is None:
            return
        self.rows.append({"check": name, "got": float(got), "expected": f"< {bound}", "ok": bool(got < bound)})

    def text(self, name, got, expected):
        if not self.enabled or expected is None:
            return
        self.rows.append({"check": name, "got": str(got), "expected": str(expected), "ok": str(got) == str(expected)})

    def matrix(self, name, got, expected, tol, row_labels, col_labels):
        """Element-wise comparison of two equally shaped arrays (expected may be None)."""
        if not self.enabled or expected is None:
            return
        got, expected = np.asarray(got, float), np.asarray(expected, float)
        for i, r in enumerate(row_labels):
            for j, c in enumerate(col_labels):
                self.number(f"{name} [{r}, {c}]", got[i, j], expected[i, j], tol)

    def to_frame(self):
        return pd.DataFrame(self.rows, columns=["check", "got", "expected", "ok"])

    def report(self, path=None, strict=False):
        df = self.to_frame()
        if path:
            df.to_csv(path, index=False)
        print("\n--- checks against numbers published in the paper ---")
        if df.empty:
            print("(none configured / disabled)")
            return True
        n_ok = int(df["ok"].sum())
        for _, r in df[~df["ok"]].iterrows():
            print(f"[MISMATCH] {r['check']}: got {r['got']}, expected {r['expected']}")
        print(f"{n_ok}/{len(df)} checks passed" + ("" if n_ok == len(df) else f"  ({len(df) - n_ok} mismatches listed above)"))
        if strict and n_ok != len(df):
            raise SystemExit("Strict mode: at least one published value was not reproduced.")
        return n_ok == len(df)
