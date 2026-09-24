"""
Phase 1 - Ingestion: NYC 311 housing complaints
Project: Hidden Need Index (NYC housing distress + mental health)

What this does
--------------
Pulls 311 housing complaints (agency = HPD) from NYC Open Data and asks the
API to COUNT them by community board, month, and complaint type. We never
download millions of raw rows - the server does the aggregation for us.

Design choices
--------------
* One request per MONTH: small queries finish fast and avoid timeouts.
* Checkpointing: each month is saved to its own file. If the run stops,
  re-running skips finished months and only fetches what's missing.
* The current month is always re-fetched, because it is still growing.
* Failed months are logged, not fatal - just re-run to fill the gaps.

Output: data/raw/311_monthly/<YYYY-MM>.csv        (one file per month)
        data/raw/311_hpd_monthly_<YYYYMMDD>.csv  (combined table)
        (optional) combined file uploaded to S3 if S3_BUCKET is set

Run from the project root:   python src/fetch_311.py
"""

from __future__ import annotations  # allows modern type hints on older Python

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
# "311 Service Requests from 2020 to Present" dataset on NYC Open Data
API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

START_YEAR = 2021                        # ~5 years of history
PAGE_SIZE = 50_000                       # max rows per request
TIMEOUT = 180                            # seconds to wait per request
APP_TOKEN = os.getenv("NYC_APP_TOKEN")   # optional, reduces throttling
S3_BUCKET = os.getenv("S3_BUCKET")       # optional, e.g. "saloni-hidden-need"

# Paths are built from this file's location, so the script works no matter
# which folder you run it from.  src/fetch_311.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MONTH_DIR = RAW_DIR / "311_monthly"
MONTH_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. The query (SoQL = SQL for Socrata/NYC Open Data)
# ---------------------------------------------------------------------------
# HPD handles indoor housing conditions: heat/hot water, pests & mold
# (UNSANITARY CONDITION), leaks, plumbing, paint/plaster, etc.
# The $where (date filter) is added per month in fetch_month().
QUERY = {
    "$select": (
        "community_board, "
        "date_trunc_ym(created_date) AS month, "
        "complaint_type, "
        "count(*) AS complaints"
    ),
    "$group": "community_board, month, complaint_type",
}


def month_range(start_year: int) -> list[tuple[int, int]]:
    """All (year, month) pairs from Jan of start_year to the current month."""
    today = date.today()
    months = []
    for y in range(start_year, today.year + 1):
        last = today.month if y == today.year else 12
        months += [(y, m) for m in range(1, last + 1)]
    return months


def next_month(y: int, m: int) -> tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(y: int, m: int, headers: dict) -> pd.DataFrame | None:
    """Pull one month of aggregated HPD complaints. Returns None on failure."""
    ny, nm = next_month(y, m)
    where = (
        f"agency = 'HPD' "
        f"AND created_date >= '{y}-{m:02d}-01T00:00:00' "
        f"AND created_date < '{ny}-{nm:02d}-01T00:00:00'"
    )
    params = {**QUERY, "$where": where, "$limit": PAGE_SIZE}

    for attempt in range(3):  # retry with backoff for network hiccups
        try:
            resp = requests.get(API_URL, params=params,
                                headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            if len(df) == PAGE_SIZE:  # safety check: result may be cut off
                print(f"  WARNING: {y}-{m:02d} hit the row limit")
            return df
        except requests.RequestException as err:
            print(f"    attempt {attempt + 1} failed: {err}")
            time.sleep(10 * (attempt + 1))
    return None


def fetch_all_pages() -> pd.DataFrame:
    """Fetch every month (skipping cached ones) and stack into one table."""
    headers = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}
    today = date.today()
    failed = []

    for y, m in month_range(START_YEAR):
        path = MONTH_DIR / f"{y}-{m:02d}.csv"
        is_current = (y, m) == (today.year, today.month)

        if path.exists() and not is_current:
            continue  # already fetched in an earlier run

        print(f"  pulling {y}-{m:02d}...", end=" ", flush=True)
        df = fetch_month(y, m, headers)
        if df is None:
            print("FAILED")
            failed.append(f"{y}-{m:02d}")
            continue
        df.to_csv(path, index=False)
        print(f"{len(df):,} rows")

    if failed:
        print(f"\n{len(failed)} month(s) failed: {', '.join(failed)}")
        print("Re-run the script to fetch only the missing months.")

    files = sorted(MONTH_DIR.glob("*.csv"))
    frames = [pd.read_csv(f, dtype=str) for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. First-look sanity checks (full quality checks come in Phase 2)
# ---------------------------------------------------------------------------
def sanity_report(df: pd.DataFrame) -> None:
    df["complaints"] = pd.to_numeric(df["complaints"])
    total = df["complaints"].sum()
    unspecified = df.loc[
        df["community_board"].str.contains("Unspecified", na=True),
        "complaints"].sum()

    print("\n=== SANITY REPORT ===")
    print(f"Rows (board x month x type): {len(df):,}")
    print(f"Total complaints:            {total:,}")
    print(f"Months covered:              {df['month'].min()[:7]} "
          f"to {df['month'].max()[:7]} "
          f"({df['month'].str[:7].nunique()} months)")
    print(f"Distinct community boards:   {df['community_board'].nunique()}")
    print(f"'Unspecified' board share:   {unspecified / total:.1%}")
    print("\nTop complaint types:")
    print(df.groupby("complaint_type")["complaints"].sum()
            .sort_values(ascending=False).head(10).to_string())


# ---------------------------------------------------------------------------
# 4. Save combined file locally (and to S3 if configured)
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame) -> Path:
    out = RAW_DIR / f"311_hpd_monthly_{date.today():%Y%m%d}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    if S3_BUCKET:
        import boto3  # only needed if you use S3
        key = f"raw/311/{out.name}"
        boto3.client("s3").upload_file(str(out), S3_BUCKET, key)
        print(f"Uploaded: s3://{S3_BUCKET}/{key}")
    return out


if __name__ == "__main__":
    print("Pulling 311 HPD housing complaints (one month at a time)...")
    data = fetch_all_pages()
    if data.empty:
        raise SystemExit("No data returned - check the query or dataset ID.")
    sanity_report(data)
    save(data)