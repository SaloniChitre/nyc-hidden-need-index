"""Tests for Phase 4: reporting-bias model.

Fake districts follow a known rule (complaints rise with defects), with one
district planted to complain half as much as it should. The model must find it.
"""

import numpy as np
import pandas as pd

from reporting_bias import (NEED_COL, RATE_COL, explain_reporting,
                            fit_expected, flag_silent)


def make_districts(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    defects = np.linspace(40, 85, n)
    lep = rng.uniform(5, 40, n)
    # complaints follow defects, and drop where limited English is higher
    z = (lep - lep.mean()) / lep.std()
    rate = np.exp(3 + 0.04 * defects - 0.2 * z + rng.normal(0, 0.05, n))
    df = pd.DataFrame({
        "cd_code": range(101, 101 + n),
        "district_name": [f"D{i}" for i in range(n)],
        NEED_COL: defects,
        RATE_COL: rate,
        "pct_limited_english": lep,
        "pct_foreign_born": rng.uniform(20, 60, n),
        "pct_poverty": rng.uniform(5, 35, n),
    })
    # plant a silent neighborhood: high defects, half the complaints
    df.loc[n - 3, RATE_COL] *= 0.5
    return df


def test_defects_predict_complaints():
    _, stats = fit_expected(make_districts())
    assert stats["pct_more_complaints_per_defect_point"] > 0
    assert stats["r_squared"] > 0.5


def test_missing_need_is_excluded():
    df = make_districts()
    df.loc[0, NEED_COL] = np.nan               # like Tottenville's "^"
    data, stats = fit_expected(df)
    assert stats["n_districts"] == len(df) - 1


def test_planted_silent_district_is_found():
    df = make_districts()
    data = flag_silent(fit_expected(df)[0]).set_index("cd_code")
    planted = int(df.loc[len(df) - 3, "cd_code"])
    assert data.loc[planted, "silent"]
    assert data.loc[planted, "reporting_ratio"] < 0.8


def test_low_need_district_is_never_silent():
    df = make_districts()
    df.loc[0, RATE_COL] *= 0.3                 # low defects, few complaints
    data = flag_silent(fit_expected(df)[0]).set_index("cd_code")
    assert not data.loc[101, "silent"]         # quiet, but not silent


def test_limited_english_effect_is_negative():
    data = flag_silent(fit_expected(make_districts())[0])
    effects = explain_reporting(data)["effects"]
    assert effects["pct_limited_english"]["pct_change_per_sd"] < 0