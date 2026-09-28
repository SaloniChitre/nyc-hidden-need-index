"""Tests for Phase 3c: reading Census PUMA names."""

import pandas as pd
import pytest

from fetch_acs_renters import keep_nyc, parse_puma_name


@pytest.mark.parametrize("name, expected", [
    ("NYC-Manhattan Community District 3--Lower East Side & Chinatown; New York", [103]),
    ("NYC-Manhattan Community Districts 1 & 2--Financial District & Greenwich Village", [101, 102]),
    ("NYC-Bronx Community Districts 3 & 6--Belmont & East Tremont", [203, 206]),
    ("NYC-Queens Community District 14--Far Rockaway, Breezy Point & Broad Channel", [414]),
    ("NYC-Staten Island Community District 3--Tottenville & Great Kills", [503]),
    ("Westchester County (Southwest)--Yonkers City PUMA; New York", None),
    (None, None),
])
def test_parse_puma_name(name, expected):
    assert parse_puma_name(name) == expected


def test_keep_nyc_drops_other_counties():
    raw = pd.DataFrame({
        "puma_name": ["NYC-Bronx Community District 4--Concourse",
                      "Albany County (East Central)--Albany City"],
        "occupied_homes": ["50000", "60000"],
        "renter_homes": ["40000", "30000"],
        "puma": ["04204", "02001"],
    })
    out = keep_nyc(raw)
    assert len(out) == 1
    assert out.loc[0, "cd_codes"] == "204"
    assert out.loc[0, "renter_homes"] == 40000