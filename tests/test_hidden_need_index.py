"""Tests for Phase 5: bias correction and the Hidden Need Index."""

import numpy as np
import pandas as pd
import pytest

from hidden_need_index import build_index, correct_live_signal, summarize


def make_inputs(n: int = 20):
    codes = list(range(101, 101 + n))
    table = pd.DataFrame({
        "cd_code": codes,
        "district_name": [f"D{c}" for c in codes],
        "total_2025_per_1k_renters": np.linspace(100, 300, n),
        "psych_hosp_per_100k": np.linspace(200, 800, n),
    })
    bias = pd.DataFrame({"cd_code": codes, "reporting_ratio": 1.0, "silent": False})
    # D105 is silent: complains at 40% of its expected level
    bias.loc[bias["cd_code"] == 105, ["reporting_ratio", "silent"]] = [0.4, True]
    # D120 has no ratio (like Tottenville)
    bias = bias[bias["cd_code"] != 120]
    return table, bias


def test_correction_divides_by_ratio():
    table, bias = make_inputs()
    df = correct_live_signal(table, bias, 2025).set_index("cd_code")
    assert df.loc[105, "corrected_rate"] == pytest.approx(df.loc[105, "raw_rate"] / 0.4, abs=0.01)
    assert df.loc[106, "corrected_rate"] == pytest.approx(df.loc[106, "raw_rate"], abs=0.01)


def test_missing_ratio_means_no_correction():
    table, bias = make_inputs()
    df = correct_live_signal(table, bias, 2025).set_index("cd_code")
    assert not df.loc[120, "ratio_available"]
    assert df.loc[120, "corrected_rate"] == pytest.approx(df.loc[120, "raw_rate"], abs=0.01)


def test_silent_district_climbs_the_ranking():
    table, bias = make_inputs()
    df = build_index(correct_live_signal(table, bias, 2025)).set_index("cd_code")
    assert df.loc[105, "rank_change"] > 0          # moved up
    assert df.loc[105, "rank_corrected"] < df.loc[105, "rank_raw"]


def test_index_is_average_of_two_zscores():
    table, bias = make_inputs()
    df = build_index(correct_live_signal(table, bias, 2025))
    expected = ((df["housing_z_corrected"] + df["mental_health_z"]) / 2).round(3)
    assert np.allclose(df["index_corrected"], expected, atol=0.002)


def test_summary_lists_new_top_entries():
    table, bias = make_inputs()
    df = build_index(correct_live_signal(table, bias, 2025))
    s = summarize(df, top_n=5)
    assert s["districts_without_ratio"] == [120]
    assert len(s["top_n_corrected"]) == 5   
