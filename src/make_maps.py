"""
Phase 6 - Maps
Project: Hidden Need Index

Draws two maps of NYC's 59 community districts:

  1. fig_map_silent.png       Reporting ratio (complaints vs. expected).
                              Orange = complains LESS than its housing problems
                              predict; Silent Neighborhoods outlined and
                              numbered to match the scatter chart.
  2. fig_map_hidden_need.png  The corrected Hidden Need Index, with the
                              top 10 numbered by rank.

Boundaries: NYC Department of City Planning "Community Districts", from NYC
Open Data. Downloaded once and saved to data/raw/community_districts.geojson.
Parks and airports ("Joint Interest Areas") are drawn in light grey.

Run from the project root:   python src/make_maps.py
Needs:  python -m pip install geopandas
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BOUNDARY_FILE = RAW_DIR / "community_districts.geojson"
INDEX_FILE = CLEAN_DIR / "hidden_need_index.parquet"

# NYC Open Data "Community Districts" (dataset 5crt-au7u). Two URL styles are
# tried because Open Data serves map files through different endpoints.
BOUNDARY_URLS = [
    "https://data.cityofnewyork.us/resource/5crt-au7u.geojson?$limit=100",
    "https://data.cityofnewyork.us/api/geospatial/5crt-au7u?method=export&format=GeoJSON",
]
NY_STATE_PLANE = 2263   # a flat projection for NYC, so shapes aren't stretched

INK, TEAL, CORAL, GREY = "#0B3C49", "#1B998B", "#F46036", "#E6E9EA"
TOP_N = 10


# ---------------------------------------------------------------------------
# 1. Boundaries
# ---------------------------------------------------------------------------
def download_boundaries() -> None:
    """Save the boundary file once; later runs reuse it."""
    if BOUNDARY_FILE.exists() and BOUNDARY_FILE.stat().st_size > 10_000:
        return
    for url in BOUNDARY_URLS:
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                if '"FeatureCollection"' not in resp.text[:500]:
                    raise ValueError("response is not GeoJSON")
                BOUNDARY_FILE.write_text(resp.text)
                print(f"Downloaded boundaries from {url.split('?')[0]}")
                return
            except (requests.RequestException, ValueError) as err:
                print(f"  {url.split('?')[0]} attempt {attempt + 1}: {err}")
                time.sleep(3)
    raise SystemExit(
        "Could not download boundaries. In a browser, open NYC Open Data "
        "'Community Districts', choose Export -> GeoJSON, and save it as "
        f"{BOUNDARY_FILE}")


def load_boundaries():
    import geopandas as gpd
    gdf = gpd.read_file(BOUNDARY_FILE)
    gdf.columns = [c.lower() for c in gdf.columns]
    # The district code column is called "borocd" or "boro_cd" depending on
    # the version of the dataset - find whichever exists
    cd_col = next(c for c in gdf.columns if c.replace("_", "") == "borocd")
    gdf["cd_code"] = pd.to_numeric(gdf[cd_col], errors="coerce").astype("Int64")
    return gdf[["cd_code", "geometry"]].to_crs(NY_STATE_PLANE)


def join_results(gdf, results: pd.DataFrame):
    """Attach results; parks/airports have no match and stay empty."""
    merged = gdf.merge(results, on="cd_code", how="left")
    matched = merged["district_name"].notna().sum()
    if matched != 59:
        raise SystemExit(f"Only {matched} of 59 districts matched the map - "
                         "check the district code column.")
    return merged


# ---------------------------------------------------------------------------
# 2. Drawing helpers
# ---------------------------------------------------------------------------
def base_axes(title: str, subtitle: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9.5))
    ax.set_axis_off()
    fig.text(0.03, 0.975, title, fontsize=16, fontweight="bold", color=INK,
             va="top")
    fig.text(0.03, 0.945, subtitle, fontsize=10, color="#5F7178", va="top")
    return fig, ax


def number_districts(ax, gdf, labels: dict[int, str]) -> None:
    """Write a label (e.g. a rank) at a point inside each chosen district."""
    for _, r in gdf[gdf["cd_code"].isin(labels)].iterrows():
        p = r.geometry.representative_point()   # always inside the shape
        ax.annotate(labels[int(r["cd_code"])], (p.x, p.y), ha="center",
                    va="center", fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="white",
                              lw=0.8))


def draw_parks(ax, gdf) -> None:
    gdf[gdf["district_name"].isna()].plot(ax=ax, color=GREY, edgecolor="white",
                                          linewidth=0.4)


# ---------------------------------------------------------------------------
# 3. Map 1: reporting ratio + Silent Neighborhoods
# ---------------------------------------------------------------------------
def map_silent(gdf, path: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import matplotlib.pyplot as plt

    fig, ax = base_axes("Where complaints under-count real problems",
                        "311 complaints vs. what housing defects predict "
                        "(2023). Outlined = Silent Neighborhood.")
    draw_parks(ax, gdf)
    has = gdf[gdf["reporting_ratio"].notna()]
    cmap = LinearSegmentedColormap.from_list("gap", [CORAL, "#FBE3DA", "#F4F7F7",
                                                     "#CBE8E3", TEAL])
    has.plot(ax=ax, column="reporting_ratio", cmap=cmap,
             norm=TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.6),
             edgecolor="white", linewidth=0.5)
    gdf[gdf["reporting_ratio"].isna() & gdf["district_name"].notna()].plot(
        ax=ax, color="white", edgecolor="#9AA5A9", hatch="///", linewidth=0.5)

    silent = gdf[gdf["silent"].fillna(False).astype(bool)]
    silent.plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=2)
    order = silent.sort_values("reporting_ratio")["cd_code"].astype(int)
    number_districts(ax, gdf, {c: str(i + 1) for i, c in enumerate(order)})

    # colour bar explained in words
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.6))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035,
                      pad=0.01, shrink=0.6)
    cb.set_ticks([0.4, 0.7, 1.0, 1.3, 1.6])
    cb.set_ticklabels(["-60%", "-30%", "as expected", "+30%", "+60%"])
    cb.set_label("Complaints vs. expected", fontsize=9)
    cb.outline.set_visible(False)

    names = [f"{i + 1}  {gdf.loc[gdf.cd_code == c, 'district_name'].iloc[0]}"
             for i, c in enumerate(order)]
    ax.text(0.01, 0.99, "Silent Neighborhoods\n" + "\n".join(names),
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=CORAL))
    ax.text(0.99, 0.01, "Hatched: no survey estimate (Tottenville)\n"
            "Grey: parks & airports", transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", color="#5F7178")
    fig.tight_layout(rect=(0, 0, 1, 0.93))   # leave room for the title
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Map 2: Hidden Need Index
# ---------------------------------------------------------------------------
def map_hidden_need(gdf, path: Path, top_n: int = TOP_N) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    fig, ax = base_axes("Hidden Need Index",
                        "Corrected housing distress + psychiatric "
                        f"hospitalizations. Numbers = top {top_n} by need.")
    draw_parks(ax, gdf)
    cmap = LinearSegmentedColormap.from_list("need", ["#F4F7F7", "#9CD3CB",
                                                      TEAL, INK])
    res = gdf[gdf["district_name"].notna()]
    res.plot(ax=ax, column="index_corrected", cmap=cmap, edgecolor="white",
             linewidth=0.5, legend=True,
             legend_kwds={"orientation": "horizontal", "fraction": 0.035,
                          "pad": 0.01, "shrink": 0.6,
                          "label": "Hidden Need Index (higher = more need)"})

    silent = gdf[gdf["silent"].fillna(False).astype(bool)]
    silent.plot(ax=ax, facecolor="none", edgecolor=CORAL, linewidth=2.2)

    top = res.nsmallest(top_n, "rank_corrected")
    number_districts(ax, gdf, {int(r.cd_code): str(int(r.rank_corrected))
                               for r in top.itertuples()})

    new = top[top["rank_raw"] > top_n]
    lines = [f"{int(r.rank_corrected):>2}  {r.district_name}"
             + ("  (new)" if r.cd_code in set(new["cd_code"]) else "")
             for r in top.sort_values("rank_corrected").itertuples()]
    ax.text(0.01, 0.99, f"Top {top_n} by need\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=INK))
    ax.text(0.99, 0.01, "Orange outline: Silent Neighborhood\n"
            "(new) = enters top 10 only after correction",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="#5F7178")
    fig.tight_layout(rect=(0, 0, 1, 0.93))   # leave room for the title
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    if not INDEX_FILE.exists():
        raise SystemExit("Run src/hidden_need_index.py first.")
    download_boundaries()
    gdf = join_results(load_boundaries(), pd.read_parquet(INDEX_FILE))

    map_silent(gdf, OUT_DIR / "fig_map_silent.png")
    map_hidden_need(gdf, OUT_DIR / "fig_map_hidden_need.png")
    print(f"Districts on map: {gdf['district_name'].notna().sum()} "
          f"(+ {gdf['district_name'].isna().sum()} parks/airports)")
    print(f"Saved: {OUT_DIR / 'fig_map_silent.png'}")
    print(f"Saved: {OUT_DIR / 'fig_map_hidden_need.png'}")