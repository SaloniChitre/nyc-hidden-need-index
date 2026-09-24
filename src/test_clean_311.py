"""
Tests for Phase 2 cleaning.

Unit tests use small made-up data, so they run anywhere (including
GitHub Actions) without downloading anything.

Run from the project root:   pytest -v
"""

from pathlib import Path

import pandas as pd
import pytest

from clean_311 import (VALID_CD_CODES, categorize, clean, parse_board,
                       quality_checks)


# ---------- parse_board ----------------------------------------------------
@pytest.mark.parametrize("label, expected", [
    ("12 QUEENS",         ("valid", 412, "QUEENS")),
    ("01 MANHATTAN",      ("valid", 101, "MANHATTAN")),
    ("18 BROOKLYN",       ("valid", 318, "BROOKLYN")),
    ("03 STATEN ISLAND",  ("valid", 503, "STATEN ISLAND")),
    ("64 MANHATTAN",      ("non_residential", None, "MANHATTAN")),
    ("0 Unspecified",     ("unspecified", None, None)),
    ("Unspecified BRONX", ("unspecified", None, "BRONX")),
    (None,                ("unspecified", None, None)),
    ("",                  ("unspecified", None, None)),
    ("XX NOWHERE",        ("invalid", None, None)),
])
def test_parse_board(label, expected):
    assert parse_board(label) == expected


def test_exactly_59_districts():
    assert len(VALID_CD_CODES) == 59


# ---------- categorize -----------------------------------------------------
def test_categorize_known_and_unknown():
    assert categorize("HEAT/HOT WATER") == "HEAT"
    assert categorize("unsanitary condition") == "PESTS_MOLD"
    assert categorize("SOMETHING NEW") == "OTHER"
    assert categorize(None) == "OTHER"


# ---------- clean + quality checks on a tiny fake dataset ------------------
def make_fake_raw() -> pd.DataFrame:
    """Every valid district for 2 months, plus some messy rows."""
    rows = []
    boroughs = {1: "MANHATTAN", 2: "BRONX", 3: "BROOKLYN",
                4: "QUEENS", 5: "STATEN ISLAND"}
    for code in VALID_CD_CODES:
        label = f"{code % 100:02d} {boroughs[code // 100]}"
        for month in ["2025-01-01T00:00:00.000", "2025-02-01T00:00:00.000"]:
            rows.append([label, month, "HEAT/HOT WATER", "10"])
    rows.append(["0 Unspecified", "2025-01-01T00:00:00.000",
                 "HEAT/HOT WATER", "5"])
    rows.append(["64 MANHATTAN", "2025-01-01T00:00:00.000", "PLUMBING", "2"])
    return pd.DataFrame(rows, columns=["community_board", "month",
                                       "complaint_type", "complaints"])


def test_clean_fake_data_passes_all_checks():
    report = quality_checks(clean(make_fake_raw()))
    assert report["passed"], report["checks"]
    assert report["complaints_by_location_status"]["unspecified"] == 5
    assert report["complaints_by_location_status"]["non_residential"] == 2


def test_negative_count_is_caught():
    raw = make_fake_raw()
    raw.loc[0, "complaints"] = "-3"
    report = quality_checks(clean(raw))
    assert not report["checks"]["no_negative_counts"]


def test_missing_district_is_caught():
    raw = make_fake_raw()
    raw = raw[raw["community_board"] != "12 QUEENS"]
    report = quality_checks(clean(raw))
    assert not report["checks"]["all_59_districts_present"]


def test_exact_duplicate_is_caught():
    """The same row loaded twice is an error and must FAIL the check."""
    raw = make_fake_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    report = quality_checks(clean(raw))
    assert not report["checks"]["no_duplicate_rows"]
    assert report["exact_duplicate_rows"] == 1


def test_case_variants_are_merged():
    """ "HEAT/HOT WATER" and "Heat/Hot Water" should become one row."""
    raw = make_fake_raw()
    variant = raw.iloc[[0]].copy()
    variant["complaint_type"] = "Heat/Hot Water"
    variant["complaints"] = "4"
    raw = pd.concat([raw, variant], ignore_index=True)

    cleaned = clean(raw)
    report = quality_checks(cleaned)
    assert report["checks"]["no_duplicate_rows"]
    assert report["rows_merged_case_variants"] == 1
    assert report["passed"]

    merged = cleaned[(cleaned["community_board"] == raw.loc[0, "community_board"])
                     & (cleaned["month"].astype(str) == "2025-01")]
    assert merged["complaints"].sum() == 14   # 10 + 4, nothing lost


# ---------- checks on the REAL clean output (skipped if not built yet) -----
CLEAN_FILE = (Path(__file__).resolve().parent.parent
              / "data" / "clean" / "311_hpd_monthly.parquet")


@pytest.mark.skipif(not CLEAN_FILE.exists(),
                    reason="run src/clean_311.py first")
def test_real_clean_output():
    df = pd.read_parquet(CLEAN_FILE)
    assert set(df["cd_code"]) == VALID_CD_CODES
    assert (df["complaints"] >= 0).all()
    assert not df.duplicated(["cd_code", "month", "complaint_type"]).any()