"""Tests for Phase 3b: building the district table."""

from pathlib import Path

import pandas as pd
import pytest

from build_district_table import build_table, clean_chp, validate

CODES = [b * 100 + n for b, c in {1: 12, 2: 12, 3: 18, 4: 14, 5: 3}.items()
         for n in range(1, c + 1)]


def make_chp_raw() -> pd.DataFrame:
    """Looks like the CHP sheet: city/borough rows, 59 districts, footnotes."""
    rows = [{"ID": 0, "Name": "NYC"}, {"ID": 1, "Name": "Manhattan"}]
    for c in CODES:
        rows.append({"ID": c, "Name": f"District {c}"})
    rows.append({"ID": "^ Data are suppressed", "Name": None})   # footnote row
    df = pd.DataFrame(rows)
    df["Overall_Pop"] = 100_000
    for col in ["Ltd_Eng_Prof", "Born_Outside_US", "Poverty", "Rent_Burden",
                "Homes_Any_Defects", "Homes_Roach"]:
        df[col] = 50
    df["Homes_Any_Defects"] = df["Homes_Any_Defects"].astype(object)  # holds "^"
    df["Psych_Hosp"] = 400.0
    df["Homes_Any_Defects_reliability_note"] = pd.Series([None] * len(df), dtype=object)
    df["Homes_Roach_reliability_note"] = pd.Series([None] * len(df), dtype=object)
    # quirks: one suppressed value, one caution flag
    df.loc[df["ID"] == 503, "Homes_Any_Defects"] = "^"
    df.loc[df["ID"] == 409, "Homes_Any_Defects_reliability_note"] = "*"
    return df


def make_yearly() -> pd.DataFrame:
    rows = []
    for c in CODES:
        for y, full in [(2023, True), (2025, True), (2026, False)]:
            rows.append(dict(cd_code=c, borough="X", year=y, full_year=full,
                             months_in_data=12, total=500, HEAT=300,
                             WATER_PLUMBING=200))
    return pd.DataFrame(rows)


def test_keeps_only_59_districts():
    chp = clean_chp(make_chp_raw())
    assert len(chp) == 59
    assert set(chp["cd_code"]) == set(CODES)


def test_suppressed_becomes_missing_and_caution_is_flagged():
    chp = clean_chp(make_chp_raw()).set_index("cd_code")
    assert pd.isna(chp.loc[503, "pct_homes_with_defects"])
    assert chp.loc[409, "pct_homes_with_defects_caution"]
    assert not chp.loc[101, "pct_homes_with_defects_caution"]


def test_rates_per_1k_and_validation():
    table = build_table(clean_chp(make_chp_raw()), make_yearly())
    # 500 complaints / 100,000 people * 1,000 = 5.0
    assert (table["total_2023_per_1k"] == 5.0).all()
    assert "total_2025" in table.columns           # latest full year, not 2026
    report = validate(table)
    assert report["passed"], report["checks"]
    assert report["suppressed_values_by_column"]["pct_homes_with_defects"] == [503]


def test_partial_year_is_rejected():
    yearly = make_yearly()
    yearly.loc[yearly["year"] == 2023, "full_year"] = False
    with pytest.raises(ValueError):
        build_table(clean_chp(make_chp_raw()), yearly)


CHP_FILE = (Path(__file__).resolve().parent.parent
            / "data" / "raw" / "2026-chp-pud.xlsx")


@pytest.mark.skipif(not CHP_FILE.exists(), reason="CHP file not downloaded")
def test_real_chp_file():
    from build_district_table import load_chp
    chp = load_chp(CHP_FILE)
    assert len(chp) == 59
    assert chp["population"].sum() > 8_000_000       # NYC is ~8.5M people