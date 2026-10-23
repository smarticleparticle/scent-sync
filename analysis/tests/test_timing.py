"""Odor-arrival timing: tubing transit time (SI 1.2), margins, flow-rate sensitivity."""
import numpy as np
import pandas as pd
import pytest

from olfaction_eeg import timing

SEGMENTS = [{"name": "connector", "inner_diameter_mm": 2.0, "length_m": 0.0053},
            {"name": "cannula", "inner_diameter_mm": 2.794, "length_m": 1.2192}]


def test_transit_time_reproduces_the_si_numbers():
    r = timing.transit_time_seconds(0.41, SEGMENTS)
    assert r["flow_mL_per_sec"] == pytest.approx(6.8333, abs=1e-3)
    assert r["segments"][0]["transit_time_sec"] == pytest.approx(0.002, abs=5e-4)
    assert r["segments"][1]["transit_time_sec"] == pytest.approx(1.094, abs=1e-3)
    assert r["total_transit_time_sec"] == pytest.approx(1.096, abs=1e-3)


def test_transit_time_scales_inversely_with_flow():
    t1 = timing.transit_time_seconds(0.41, SEGMENTS)["total_transit_time_sec"]
    t2 = timing.transit_time_seconds(0.82, SEGMENTS)["total_transit_time_sec"]
    assert t1 / t2 == pytest.approx(2.0)


def test_margin_summary_and_sensitivity_on_a_crafted_table():
    lags = [2.0] * 8 + [1.0, 0.9]                      # transit 1.096: the last two trials are late (by 0.096 s and 0.196 s)
    df = pd.DataFrame({"lag_sec": lags})
    df["arrival_relative"] = 1.096 - df["lag_sec"]
    m = timing.margin_summary(df)
    assert (m["n_trials"], m["n_arrived_in_time"], m["n_late"]) == (10, 8, 2)
    assert m["pct_in_time"] == pytest.approx(80.0)
    assert m["median_margin_s"] == pytest.approx(0.904)
    assert m["late_delays_s"] == pytest.approx([0.096, 0.196])
    s = timing.flow_sensitivity_table(df, 0.41, SEGMENTS, [-25, 0, 25])
    assert list(s["n_contaminated"]) == sorted(s["n_contaminated"], reverse=True)       # faster flow -> fewer late trials
    assert s.loc[s.pct_error_in_flow_rate == 0, "n_contaminated"].iloc[0] == 2


def test_lag_table_columns():
    df = pd.DataFrame({"subject": ["1", "1", "2"], "lag_sec": [2.0, 3.0, 1.5]})
    t = timing.lag_table(df)
    assert list(t.columns) == ["mean_lag_sec", "sd_lag_sec", "n_trials"] and t.loc["1", "mean_lag_sec"] == 2.5 and t.loc["1", "n_trials"] == 2
