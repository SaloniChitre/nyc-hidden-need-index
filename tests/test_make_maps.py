"""Tests for Phase 6: joining results to map boundaries."""

import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")        # skip if geopandas isn't installed
from shapely.geometry import box               # noqa: E402

import make_maps                               # noqa: E402

CODES = [b * 100 + n for b, c in {1: 12, 2: 12, 3: 18, 4: 14, 5: 3}.items()
         for n in range(1, c + 1)]


def write_boundaries(path, column_name: str, extra_codes=(164, 480)):
    codes = CODES + list(extra_codes)           # districts + parks/airports
    geoms = [box(-74 + i * 0.01, 40.6, -73.995 + i * 0.01, 40.605)
             for i in range(len(codes))]
    gpd.GeoDataFrame({column_name: [str(c) for c in codes]}, geometry=geoms,
                     crs=4326).to_file(path, driver="GeoJSON")


@pytest.mark.parametrize("column_name", ["BoroCD", "boro_cd"])
def test_finds_district_column_either_name(tmp_path, monkeypatch, column_name):
    f = tmp_path / "cd.geojson"
    write_boundaries(f, column_name)
    monkeypatch.setattr(make_maps, "BOUNDARY_FILE", f)
    gdf = make_maps.load_boundaries()
    assert set(CODES) <= set(gdf["cd_code"].dropna().astype(int))


def test_join_keeps_parks_as_empty(tmp_path, monkeypatch):
    f = tmp_path / "cd.geojson"
    write_boundaries(f, "BoroCD")
    monkeypatch.setattr(make_maps, "BOUNDARY_FILE", f)
    results = pd.DataFrame({"cd_code": CODES,
                            "district_name": [f"D{c}" for c in CODES]})
    merged = make_maps.join_results(make_maps.load_boundaries(), results)
    assert merged["district_name"].notna().sum() == 59
    assert merged["district_name"].isna().sum() == 2        # the parks


def test_join_fails_loudly_if_districts_missing(tmp_path, monkeypatch):
    f = tmp_path / "cd.geojson"
    write_boundaries(f, "BoroCD")
    monkeypatch.setattr(make_maps, "BOUNDARY_FILE", f)
    results = pd.DataFrame({"cd_code": CODES[:50],
                            "district_name": ["x"] * 50})
    with pytest.raises(SystemExit):
        make_maps.join_results(make_maps.load_boundaries(), results)