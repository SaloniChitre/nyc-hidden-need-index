"""
Phase 2 - Cleaning & Data Quality: NYC 311 housing complaints
Project: Hidden Need Index

What this does
--------------
1. Loads the newest raw file from data/raw/
2. Converts 311's "12 QUEENS" labels into standard community district
   codes (borough digit x 100 + district number -> 412)
3. Sorts every row into one of four buckets:
     valid           - one of NYC's 59 community districts
     unspecified     - no usable location
     non_residential - parks/airports ("Joint Interest Areas", e.g. 64 MANHATTAN)
     invalid         - anything we can't parse
4. Merges complaint types that differ only in capitalization
   (e.g. "APPLIANCE" vs "Appliance") by adding their counts
5. Groups complaint types into 5 analysis categories
6. Runs data quality checks and writes a quality report
7. Saves the clean table as Parquet

Output: data/clean/311_hpd_monthly.parquet
        data/clean/quality_report.json

Run from the project root:   python src/clean_311.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Paths and reference data
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Borough digit used in standard NYC community district codes
BOROUGH_CODES = {
    "MANHATTAN": 1,
    "BRONX": 2,
    "BROOKLYN": 3,
    "QUEENS": 4,
    "STATEN ISLAND": 5,
}

# How many residential community districts each borough has (total = 59)
DISTRICTS_PER_BOROUGH = {
    "MANHATTAN": 12,
    "BRONX": 12,
    "BROOKLYN": 18,
    "QUEENS": 14,
    "STATEN ISLAND": 3,
}

# All 59 valid codes, e.g. 101..112, 201..212, 301..318, 401..414, 501..503
VALID_CD_CODES = {
    BOROUGH_CODES[b] * 100 + n
    for b, count in DISTRICTS_PER_BOROUGH.items()
    for n in range(1, count + 1)
}

# Group HPD complaint types into categories we can analyze.
# Anything not listed falls into "OTHER" (and gets reported).
COMPLAINT_CATEGORIES = {
    "HEAT/HOT WATER": "HEAT",
    "UNSANITARY CONDITION": "PESTS_MOLD",     # includes pests and mold
    "WATER LEAK": "WATER_PLUMBING",
    "PLUMBING": "WATER_PLUMBING",
    "PAINT/PLASTER": "STRUCTURAL",
    "DOOR/WINDOW": "STRUCTURAL",
    "FLOORING/STAIRS": "STRUCTURAL",
    "OUTSIDE BUILDING": "STRUCTURAL",
    "ELEVATOR": "STRUCTURAL",
    "ELECTRIC": "SAFETY_UTILITIES",
    "APPLIANCE": "SAFETY_UTILITIES",
    "SAFETY": "SAFETY_UTILITIES",
    "GENERAL": "OTHER",
}


# ---------------------------------------------------------------------------
# 2. Parsing helpers (small, pure functions = easy to test)
# ---------------------------------------------------------------------------
def parse_board(label: str | float) -> tuple[str, int | None, str | None]:
    """
    Turn a 311 community_board label into (status, cd_code, borough).

    "12 QUEENS"      -> ("valid", 412, "QUEENS")
    "64 MANHATTAN"   -> ("non_residential", None, "MANHATTAN")
    "0 Unspecified"  -> ("unspecified", None, None)
    "Unspecified BRONX" -> ("unspecified", None, "BRONX")
    """
    if not isinstance(label, str) or not label.strip():
        return ("unspecified", None, None)

    text = label.strip().upper()
    if "UNSPECIFIED" in text:
        borough = next((b for b in BOROUGH_CODES if b in text), None)
        return ("unspecified", None, borough)

    number, _, borough = text.partition(" ")
    borough = borough.strip()
    if not number.isdigit() or borough not in BOROUGH_CODES:
        return ("invalid", None, None)

    code = BOROUGH_CODES[borough] * 100 + int(number)
    if code in VALID_CD_CODES:
        return ("valid", code, borough)
    return ("non_residential", None, borough)


def categorize(complaint_type: str) -> str:
    """Map an HPD complaint type to an analysis category."""
    if not isinstance(complaint_type, str):
        return "OTHER"
    return COMPLAINT_CATEGORIES.get(complaint_type.strip().upper(), "OTHER")


# ---------------------------------------------------------------------------
# 3. Load
# ---------------------------------------------------------------------------
def load_latest_raw() -> tuple[pd.DataFrame, Path]:
    """Load the newest combined raw file (names sort by date)."""
    files = sorted(RAW_DIR.glob("311_hpd_monthly_*.csv"))
    if not files:
        raise SystemExit("No raw file found - run src/fetch_311.py first.")
    latest = files[-1]
    return pd.read_csv(latest, dtype=str), latest


# ---------------------------------------------------------------------------
# 4. Clean
# ---------------------------------------------------------------------------
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and tag every row with a location status."""
    df = raw.copy()

    # Exact duplicates (same raw label, month AND identical type text) would
    # mean the same data was loaded twice - that's an error, not a variant.
    # Count them BEFORE merging so they can't be silently added together.
    exact_dups = int(df.duplicated(
        ["community_board", "month", "complaint_type"]).sum())

    df["complaints"] = pd.to_numeric(df["complaints"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")

    # Standardize text BEFORE merging. 311 stores some complaint types in two
    # formats (e.g. "APPLIANCE" and "Appliance"), which the API returns as
    # separate rows. After upper-casing they become the same type, so we add
    # their counts together.
    df["community_board"] = df["community_board"].str.strip().str.upper()
    df["complaint_type"] = df["complaint_type"].str.strip().str.upper()

    keys = ["community_board", "month", "complaint_type"]
    sizes = df.groupby(keys, dropna=False).size()
    variant_groups = sizes[sizes > 1]
    variant_months = sorted(
        str(m) for m in variant_groups.index.get_level_values("month").unique()
        if pd.notna(m))

    rows_before = len(df)
    # min_count=1 keeps a group as NaN if ALL its counts are missing,
    # so the no_missing_counts check can still catch bad values.
    df = (df.groupby(keys, dropna=False, as_index=False)["complaints"]
            .sum(min_count=1))

    parsed = df["community_board"].apply(parse_board)
    df["location_status"] = parsed.str[0]
    df["cd_code"] = parsed.str[1].astype("Int64")
    df["borough"] = parsed.str[2]

    df["category"] = df["complaint_type"].apply(categorize)

    # Remember what the merge did, for the quality report
    df.attrs["exact_duplicates"] = exact_dups
    df.attrs["rows_merged"] = rows_before - len(df) - exact_dups
    df.attrs["months_with_case_variants"] = variant_months
    return df


# ---------------------------------------------------------------------------
# 5. Data quality checks
# ---------------------------------------------------------------------------
def quality_checks(df: pd.DataFrame) -> dict:
    """Run checks and return a report. 'passed' is False if any check fails."""
    total = int(df["complaints"].sum())
    by_status = df.groupby("location_status")["complaints"].sum()
    valid = df[df["location_status"] == "valid"]

    months = valid["month"].dropna().unique()
    expected_rows = len(VALID_CD_CODES) * len(months)
    present = valid.groupby(["cd_code", "month"]).size()

    unmapped_types = sorted(
        df.loc[(df["category"] == "OTHER") &
               (df["complaint_type"] != "GENERAL"),
               "complaint_type"].dropna().unique())

    checks = {
        "no_missing_counts": bool(df["complaints"].notna().all()),
        "no_negative_counts": bool((df["complaints"].dropna() >= 0).all()),
        "no_bad_months": bool(df["month"].notna().all()),
        "no_duplicate_rows":
            df.attrs.get("exact_duplicates", 0) == 0
            and not bool(df.duplicated(
                ["community_board", "month", "complaint_type"]).any()),
        "all_59_districts_present":
            set(valid["cd_code"].dropna()) == VALID_CD_CODES,
        "every_district_has_every_month": len(present) == expected_rows,
        "totals_reconcile": int(by_status.sum()) == total,
    }

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "total_complaints": total,
        "complaints_by_location_status": {
            k: int(v) for k, v in by_status.items()},
        "share_mapped_to_valid_district":
            round(float(by_status.get("valid", 0)) / total, 4) if total else 0,
        "months_covered": len(months),
        "district_months_expected": expected_rows,
        "district_months_present": int(len(present)),
        "unmapped_complaint_types": unmapped_types,
        "exact_duplicate_rows": int(df.attrs.get("exact_duplicates", 0)),
        "rows_merged_case_variants": int(df.attrs.get("rows_merged", 0)),
        "months_with_case_variants":
            len(df.attrs.get("months_with_case_variants", [])),
        "first_month_with_case_variants":
            (df.attrs.get("months_with_case_variants") or [None])[0],
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report


def print_report(report: dict) -> None:
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Rows:                  {report['rows']:,}")
    print(f"Total complaints:      {report['total_complaints']:,}")
    print(f"Mapped to a district:  "
          f"{report['share_mapped_to_valid_district']:.1%}")
    print("By location status:")
    for status, n in report["complaints_by_location_status"].items():
        print(f"  {status:<16} {n:>12,}")
    print(f"Months covered:        {report['months_covered']}")
    print(f"District-months:       {report['district_months_present']:,} "
          f"of {report['district_months_expected']:,} expected")
    if report["rows_merged_case_variants"]:
        print(f"Case variants merged:  "
              f"{report['rows_merged_case_variants']:,} rows across "
              f"{report['months_with_case_variants']} months "
              f"(first: {report['first_month_with_case_variants']})")
    if report["unmapped_complaint_types"]:
        print(f"New complaint types -> OTHER: "
              f"{report['unmapped_complaint_types']}")
    print("\nChecks:")
    for name, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nOverall: {'PASSED' if report['passed'] else 'NEEDS REVIEW'}")


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame, report: dict, source: Path) -> None:
    valid = df[df["location_status"] == "valid"].copy()
    valid["month"] = valid["month"].astype(str)          # Parquet-friendly
    cols = ["cd_code", "borough", "month", "complaint_type",
            "category", "complaints"]
    out = CLEAN_DIR / "311_hpd_monthly.parquet"
    valid[cols].to_parquet(out, index=False)

    report["source_file"] = source.name
    (CLEAN_DIR / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")
    print(f"Saved: {CLEAN_DIR / 'quality_report.json'}")


if __name__ == "__main__":
    raw, source = load_latest_raw()
    print(f"Loaded {source.name}")
    cleaned = clean(raw)
    report = quality_checks(cleaned)
    print_report(report)
    save(cleaned, report, source)