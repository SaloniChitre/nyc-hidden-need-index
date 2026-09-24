"""
Phase 3b - Build the district table (one row per community district)
Project: Hidden Need Index

What this does
--------------
Joins three things into ONE 59-row table:
  1. DOHMH Community Health Profiles 2026 (population, poverty, language,
     survey-reported housing problems, psychiatric hospitalizations)
  2. 311 HPD complaints for 2023 - the SAME year as the housing survey
     (NYC Housing and Vacancy Survey 2023), so the comparison is fair
  3. 311 complaints for the latest full year - the "live" signal we will
     correct later

Then converts complaint counts into rates per 1,000 residents.

Data quirks handled (documented in DOHMH's Metadata sheet):
  * "^" = suppressed (estimate too unreliable to publish) -> stored as missing
  * "*" = interpret with caution -> kept, but flagged in a *_caution column
  * Some districts share one estimate because the source data comes from a
    larger area (e.g. 101 & 102 share poverty and language values)

Output: data/clean/district_table.parquet  (+ .csv copy for easy viewing)
        data/clean/district_table_report.json

Run from the project root:   python src/build_district_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CHP_FILE = RAW_DIR / "2026-chp-pud.xlsx"
YEARLY_311 = CLEAN_DIR / "311_district_yearly.parquet"
OUT_FILE = CLEAN_DIR / "district_table.parquet"

SURVEY_YEAR = 2023  # year of the Housing and Vacancy Survey in the CHP file

# CHP column -> our column name
CHP_COLUMNS = {
    "ID": "cd_code",
    "Name": "district_name",
    "Overall_Pop": "population",
    "Ltd_Eng_Prof": "pct_limited_english",
    "Born_Outside_US": "pct_foreign_born",
    "Poverty": "pct_poverty",
    "Rent_Burden": "pct_rent_burden",
    "Homes_Any_Defects": "pct_homes_with_defects",   # survey: REAL problems
    "Homes_Roach": "pct_homes_roaches",
    "Psych_Hosp": "psych_hosp_per_100k",             # mental health outcome
}
CAUTION_COLUMNS = {
    "Homes_Any_Defects_reliability_note": "pct_homes_with_defects_caution",
    "Homes_Roach_reliability_note": "pct_homes_roaches_caution",
}
PERCENT_COLUMNS = ["pct_limited_english", "pct_foreign_born", "pct_poverty",
                   "pct_rent_burden", "pct_homes_with_defects",
                   "pct_homes_roaches"]


# ---------------------------------------------------------------------------
# 1. Load and clean the Community Health Profiles
# ---------------------------------------------------------------------------
def load_chp(path: Path) -> pd.DataFrame:
    """Read the CHP data sheet and keep the 59 community districts."""
    # Row 0 holds section names ("Who We Are"...), row 1 holds column names
    raw = pd.read_excel(path, sheet_name="CHP_all_data", header=1)
    return clean_chp(raw)


def clean_chp(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]  # some names end in a space

    # Keep only community districts: 3-digit IDs 101-599.
    # (Rows 0-5 are NYC and borough totals; bottom rows are footnotes.)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df = df[df["ID"].between(101, 599)].copy()

    out = df[list(CHP_COLUMNS)].rename(columns=CHP_COLUMNS)
    out["cd_code"] = out["cd_code"].astype(int)

    # "^" means suppressed -> missing.  Everything else becomes a number.
    for col in out.columns.drop(["cd_code", "district_name"]):
        out[col] = pd.to_numeric(out[col].replace("^", pd.NA), errors="coerce")

    # "*" means interpret with caution -> True/False flag
    for src, dst in CAUTION_COLUMNS.items():
        out[dst] = df[src].astype(str).str.contains(r"\*", na=False).values

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Prepare 311 for the join
# ---------------------------------------------------------------------------
def prepare_311(yearly: pd.DataFrame, year: int, suffix: str) -> pd.DataFrame:
    """One year of 311, with columns renamed like total_2023, HEAT_2023."""
    one = yearly[yearly["year"] == year].copy()
    if one.empty:
        raise ValueError(f"No 311 data for {year}")
    if not one["full_year"].all():
        raise ValueError(f"{year} is not a full year - pick another")
    keep = ["cd_code", "borough", "total"] + [
        c for c in one.columns
        if c.isupper() and c not in ("cd_code",)]          # category columns
    one = one[keep]
    return one.rename(columns={c: f"{c}_{suffix}"
                               for c in keep if c not in ("cd_code", "borough")})


# ---------------------------------------------------------------------------
# 3. Join and compute rates
# ---------------------------------------------------------------------------
def build_table(chp: pd.DataFrame, yearly: pd.DataFrame,
                survey_year: int = SURVEY_YEAR) -> pd.DataFrame:
    latest = int(yearly.loc[yearly["full_year"], "year"].max())

    survey = prepare_311(yearly, survey_year, str(survey_year))
    live = prepare_311(yearly, latest, str(latest)).drop(columns="borough")

    table = (chp.merge(survey, on="cd_code", how="left", validate="1:1")
                .merge(live, on="cd_code", how="left", validate="1:1"))

    # Complaints per 1,000 residents, so big and small districts compare fairly
    for col in [c for c in table.columns if c.startswith(("total_", "HEAT_",
                "PESTS_MOLD_", "WATER_PLUMBING_", "STRUCTURAL_",
                "SAFETY_UTILITIES_", "OTHER_"))]:
        table[f"{col}_per_1k"] = (table[col] / table["population"] * 1000).round(2)

    table.attrs["latest_full_year"] = latest
    front = ["cd_code", "borough", "district_name", "population"]
    return table[front + [c for c in table.columns if c not in front]]


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------
def validate(table: pd.DataFrame, survey_year: int = SURVEY_YEAR) -> dict:
    latest = table.attrs.get("latest_full_year")
    pct_ok = all(table[c].dropna().between(0, 100).all() for c in PERCENT_COLUMNS)
    suppressed = {c: table.loc[table[c].isna(), "cd_code"].tolist()
                  for c in PERCENT_COLUMNS + ["psych_hosp_per_100k"]
                  if table[c].isna().any()}
    caution = table.loc[table["pct_homes_with_defects_caution"], "cd_code"].tolist()

    checks = {
        "exactly_59_districts": len(table) == 59,
        "district_codes_unique": table["cd_code"].is_unique,
        "population_positive": bool((table["population"] > 0).all()),
        "percentages_between_0_and_100": bool(pct_ok),
        "every_district_has_311_data":
            bool(table[[f"total_{survey_year}", f"total_{latest}"]].notna().all().all()),
    }
    return {
        "rows": len(table),
        "survey_year_311": survey_year,
        "latest_full_year_311": latest,
        "suppressed_values_by_column": suppressed,
        "housing_survey_interpret_with_caution": caution,
        "checks": checks,
        "passed": all(checks.values()),
    }


def print_report(report: dict, table: pd.DataFrame) -> None:
    print("=== DISTRICT TABLE ===")
    print(f"Rows: {report['rows']}   311 years: "
          f"{report['survey_year_311']} (matches survey) and "
          f"{report['latest_full_year_311']} (latest full year)")
    if report["suppressed_values_by_column"]:
        print("Suppressed by DOHMH (^):")
        for col, codes in report["suppressed_values_by_column"].items():
            print(f"  {col:<26} {codes}")
    print(f"Housing survey 'interpret with caution' (*): "
          f"{report['housing_survey_interpret_with_caution']}")
    print("\nChecks:")
    for name, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nOverall: {'PASSED' if report['passed'] else 'NEEDS REVIEW'}")

    y = report["survey_year_311"]
    preview = table.sort_values(f"total_{y}_per_1k", ascending=False)
    print(f"\nHighest 311 housing complaints per 1,000 residents ({y}):")
    print(preview[["cd_code", "district_name", f"total_{y}_per_1k",
                   "pct_homes_with_defects"]].head(5).to_string(index=False))
    print(f"\nLowest:")
    print(preview[["cd_code", "district_name", f"total_{y}_per_1k",
                   "pct_homes_with_defects"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    if not CHP_FILE.exists():
        raise SystemExit(f"Missing {CHP_FILE.name} in data/raw/")
    if not YEARLY_311.exists():
        raise SystemExit("Run src/summarize_311.py first.")

    chp = load_chp(CHP_FILE)
    yearly = pd.read_parquet(YEARLY_311)
    table = build_table(chp, yearly)
    report = validate(table)
    print_report(report, table)

    table.to_parquet(OUT_FILE, index=False)
    table.to_csv(OUT_FILE.with_suffix(".csv"), index=False)
    (CLEAN_DIR / "district_table_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUT_FILE} (+ .csv)")