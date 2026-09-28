"""
Phase 3c - Ingestion: renter-occupied homes from the Census Bureau
Project: Hidden Need Index

Why this exists
---------------
HPD complaints come from RENTERS, but so far we divided complaints by ALL
residents. Districts where most people own their homes would then look
"silent" just because they have fewer renters. To compare fairly we need
the number of renter-occupied homes in each district.

Source
------
American Community Survey (ACS) 5-year estimates, 2019-2023, table B25003
(Tenure), by PUMA (Public Use Microdata Area). The 2023 5-year release
lines up with the 2023 housing survey used in the Community Health Profiles.

NYC PUMAs follow community districts, and their names say which ones, e.g.
    "NYC-Manhattan Community District 3--Lower East Side & Chinatown"
    "NYC-Manhattan Community Districts 1 & 2--Financial District & ..."
A few PUMAs cover two districts - those get split later, in
build_district_table.py.

Output: data/raw/acs_renters_2023.csv   (one row per NYC PUMA, untouched)

Run from the project root:   python src/fetch_acs_renters.py
No API key is usually needed for a single request this small. If the API
refuses, get a free key at https://api.census.gov/data/key_signup.html and
run:  export CENSUS_API_KEY="your_key"
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RAW_DIR / "acs_renters_2023.csv"

# The URL is written out in full (spaces as %20) because the Census API can
# reject a "+" in place of a space, which is how requests encodes spaces.
# B25003_001E = all occupied homes, B25003_003E = renter-occupied homes
# state:36 = New York State
API_URL = (
    "https://api.census.gov/data/2023/acs/acs5"
    "?get=NAME,B25003_001E,B25003_003E"
    "&for=public%20use%20microdata%20area:*"
    "&in=state:36"
)
CENSUS_KEY = os.getenv("CENSUS_API_KEY")   # optional

BOROUGH_CODES = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3,
                 "QUEENS": 4, "STATEN ISLAND": 5}

# "NYC-<Borough> Community District(s) <numbers>--<neighborhood names>"
PUMA_PATTERN = re.compile(
    r"NYC-(Manhattan|Bronx|Brooklyn|Queens|Staten Island)\s+"
    r"Community Districts?\s+([\d\s&,and]+?)\s*--",
    re.IGNORECASE)


def parse_puma_name(name: str) -> list[int] | None:
    """
    Turn a PUMA name into the community district codes it covers.

    "NYC-Manhattan Community District 3--Lower East Side"  -> [103]
    "NYC-Manhattan Community Districts 1 & 2--Financial..." -> [101, 102]
    "Westchester County (Southwest)--Yonkers City"         -> None
    """
    if not isinstance(name, str):
        return None
    match = PUMA_PATTERN.search(name)
    if not match:
        return None
    borough = BOROUGH_CODES[match.group(1).upper()]
    numbers = [int(n) for n in re.findall(r"\d+", match.group(2))]
    return [borough * 100 + n for n in numbers] or None


def fetch() -> pd.DataFrame:
    url = API_URL + (f"&key={CENSUS_KEY}" if CENSUS_KEY else "")
    for attempt in range(3):
        resp = None
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            rows = resp.json()
            break
        except (requests.RequestException, ValueError) as err:
            # Show what the server actually sent back, to make fixing easy
            status = resp.status_code if resp is not None else "no response"
            preview = resp.text[:300] if resp is not None else ""
            print(f"  attempt {attempt + 1} failed ({status}): {err}")
            if preview:
                print(f"  server said: {preview!r}")
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit("Census API failed 3 times - see messages above.")

    # First row is the header, the rest are data
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.rename(columns={
        "NAME": "puma_name",
        "B25003_001E": "occupied_homes",
        "B25003_003E": "renter_homes",
        "public use microdata area": "puma",
    })


def keep_nyc(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NYC PUMAs and record which districts each covers."""
    df = df.copy()
    df["cd_codes"] = df["puma_name"].apply(parse_puma_name)
    nyc = df[df["cd_codes"].notna()].copy()
    for col in ["occupied_homes", "renter_homes"]:
        nyc[col] = pd.to_numeric(nyc[col], errors="coerce")
    # Store the list as text like "101;102" so it survives a CSV round trip
    nyc["cd_codes"] = nyc["cd_codes"].apply(lambda c: ";".join(map(str, c)))
    return nyc[["puma", "puma_name", "cd_codes",
                "occupied_homes", "renter_homes"]].reset_index(drop=True)


if __name__ == "__main__":
    print("Pulling renter-occupied homes (ACS 2019-2023) for NY PUMAs...")
    nyc = keep_nyc(fetch())

    covered = sorted(int(c) for codes in nyc["cd_codes"]
                     for c in codes.split(";"))
    shared = nyc[nyc["cd_codes"].str.contains(";")]

    print(f"NYC PUMAs found:        {len(nyc)}")
    print(f"Districts covered:      {len(set(covered))} of 59")
    print(f"PUMAs covering 2 CDs:   {len(shared)} -> "
          f"{', '.join(shared['cd_codes'])}")
    print(f"Renter-occupied homes:  {nyc['renter_homes'].sum():,} of "
          f"{nyc['occupied_homes'].sum():,} "
          f"({nyc['renter_homes'].sum() / nyc['occupied_homes'].sum():.0%})")

    nyc.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")