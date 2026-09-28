"""Tests for Phase 3a: yearly district summary."""

from datetime import date

import pandas as pd

from summarize_311 import summarize


def make_monthly() -> pd.DataFrame:
    rows = []
    for m in range(1, 13):                       # full year 2025
        rows.append([101, "MANHATTAN", f"2025-{m:02d}", "HEAT/HOT WATER", "HEAT", 10])
        rows.append([101, "MANHATTAN", f"2025-{m:02d}", "PLUMBING", "WATER_PLUMBING", 2])
    rows.append([101, "MANHATTAN", "2026-01", "HEAT/HOT WATER", "HEAT", 7])  # partial year
    return pd.DataFrame(rows, columns=["cd_code", "borough", "month",
                                       "complaint_type", "category", "complaints"])


def test_totals_by_year_and_category():
    out = summarize(make_monthly(), today=date(2026, 9, 24))
    y2025 = out[out["year"] == 2025].iloc[0]
    assert y2025["HEAT"] == 120
    assert y2025["WATER_PLUMBING"] == 24
    assert y2025["total"] == 144


def test_partial_year_is_flagged():
    out = summarize(make_monthly(), today=date(2026, 9, 24)).set_index("year")
    assert out.loc[2025, "full_year"]
    assert not out.loc[2026, "full_year"]


def test_nothing_lost():
    monthly = make_monthly()
    out = summarize(monthly, today=date(2026, 9, 24))
    assert out["total"].sum() == monthly["complaints"].sum()