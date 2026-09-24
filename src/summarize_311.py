"""
Phase 3a - Summarize 311 by district and year
Project: Hidden Need Index

What this does
--------------
Turns the clean monthly table (district x month x complaint type) into one
row per district per year, with a total and a column for each category:

    cd_code | borough | year | total | HEAT | PESTS_MOLD | WATER_PLUMBING | ...

Why yearly: the survey data we compare against is published by year, so 311
must be summarized the same way before the two can be joined fairly.

The current year is only partly finished, so it is flagged (full_year=False)
and should NOT be compared with full years.

Output: data/clean/311_district_yearly.parquet

Run from the project root:   python src/summarize_311.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
IN_FILE = CLEAN_DIR / "311_hpd_monthly.parquet"
OUT_FILE = CLEAN_DIR / "311_district_yearly.parquet"


def summarize(monthly: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """District x year table with a total and one column per category."""
    today = today or date.today()
    df = monthly.copy()
    df["year"] = df["month"].str[:4].astype(int)   # "2025-01" -> 2025

    wide = df.pivot_table(index=["cd_code", "borough", "year"],
                          columns="category", values="complaints",
                          aggfunc="sum", fill_value=0)
    wide.columns.name = None
    wide = wide.reset_index()

    categories = sorted(df["category"].unique())
    wide["total"] = wide[categories].sum(axis=1)

    months_per_year = df.groupby("year")["month"].nunique()
    wide["months_in_data"] = wide["year"].map(months_per_year)
    wide["full_year"] = (wide["months_in_data"] == 12) & (wide["year"] < today.year)

    cols = ["cd_code", "borough", "year", "full_year", "months_in_data",
            "total"] + categories
    return wide[cols].sort_values(["year", "cd_code"]).reset_index(drop=True)


if __name__ == "__main__":
    if not IN_FILE.exists():
        raise SystemExit("Clean file missing - run src/clean_311.py first.")
    monthly = pd.read_parquet(IN_FILE)
    yearly = summarize(monthly)

    # Quick checks: every district in every year, nothing lost
    assert yearly.groupby("year")["cd_code"].nunique().eq(59).all(), \
        "Some year is missing districts"
    assert yearly["total"].sum() == monthly["complaints"].sum(), \
        "Totals don't match the monthly data"

    print("=== 311 BY DISTRICT & YEAR ===")
    print(yearly.groupby(["year", "full_year"])["total"].sum().to_string())
    latest = yearly.loc[yearly["full_year"], "year"].max()
    top = (yearly[yearly["year"] == latest]
           .nlargest(5, "total")[["cd_code", "borough", "total"]])
    print(f"\nTop 5 districts by raw complaints ({latest}):")
    print(top.to_string(index=False))
    print("\n(Raw counts - bigger districts naturally have more. "
          "Per-1,000 rates come after we add population.)")

    yearly.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")