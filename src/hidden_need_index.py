"""
Phase 5 - Bias Correction & the Hidden Need Index
Project: Hidden Need Index

The idea
--------
311 is timely but biased: Silent Neighborhoods complain less than their
real problems. Phase 4 measured each district's reporting ratio using the
2023 survey. Now we use it to CORRECT the latest full year of 311 data:

    corrected rate = actual complaint rate / reporting ratio

Example: a district complains at half its expected level (ratio 0.5).
Its 2025 complaints are doubled - an estimate of what they would be if it
reported like a typical district. Result: a signal that is timely (2025)
AND fair (adjusted for under-reporting).

The Hidden Need Index
---------------------
Combines two kinds of need, each converted to a z-score (distance from the
city average in standard deviations) so they count equally:
    1. Housing distress   - corrected 2025 complaint rate (log scale)
    2. Mental health need - psychiatric hospitalizations per 100,000 adults
    Hidden Need Index = average of the two z-scores

The key comparison
------------------
The same index built from RAW (uncorrected) 311 data. Districts that climb
the ranking after correction are the ones standard data would overlook.

Assumption: each district's reporting behaviour in 2025 is similar to 2023.
Tottenville (503) has no reporting ratio (survey suppressed), so it is left
uncorrected and flagged.

Output: data/clean/hidden_need_index.parquet (+ .csv)
        outputs/hidden_need_summary.json
        outputs/fig_rank_change.png

Run from the project root:   python src/hidden_need_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_FILE = CLEAN_DIR / "district_table.parquet"
BIAS_FILE = CLEAN_DIR / "reporting_bias.parquet"
OUT_FILE = CLEAN_DIR / "hidden_need_index.parquet"

MH_COL = "psych_hosp_per_100k"
TOP_N = 10


def zscore(s: pd.Series) -> pd.Series:
    """Distance from the average, in standard deviations."""
    return (s - s.mean()) / s.std()


# ---------------------------------------------------------------------------
# Step 1: correct the live 311 signal
# ---------------------------------------------------------------------------
def correct_live_signal(table: pd.DataFrame, bias: pd.DataFrame,
                        live_year: int) -> pd.DataFrame:
    live_col = f"total_{live_year}_per_1k_renters"
    df = table.merge(bias[["cd_code", "reporting_ratio", "silent"]],
                     on="cd_code", how="left", validate="1:1")

    df["ratio_available"] = df["reporting_ratio"].notna()
    df["ratio_used"] = df["reporting_ratio"].fillna(1.0)   # no ratio -> no change
    df["silent"] = df["silent"].fillna(False).astype(bool)

    df["raw_rate"] = df[live_col]
    df["corrected_rate"] = (df["raw_rate"] / df["ratio_used"]).round(2)
    return df


# ---------------------------------------------------------------------------
# Step 2: build the index (raw and corrected versions)
# ---------------------------------------------------------------------------
def build_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mh = zscore(df[MH_COL])

    for version, rate_col in [("raw", "raw_rate"), ("corrected", "corrected_rate")]:
        housing = zscore(np.log(df[rate_col]))
        df[f"housing_z_{version}"] = housing.round(3)
        df[f"index_{version}"] = ((housing + mh) / 2).round(3)
        df[f"rank_{version}"] = (df[f"index_{version}"]
                                 .rank(ascending=False, method="min").astype(int))

    df["mental_health_z"] = mh.round(3)
    # positive = moved UP the need ranking after correction
    df["rank_change"] = df["rank_raw"] - df["rank_corrected"]
    return df.sort_values("rank_corrected").reset_index(drop=True)


def summarize(df: pd.DataFrame, top_n: int = TOP_N) -> dict:
    top_raw = set(df.loc[df["rank_raw"] <= top_n, "cd_code"])
    top_corr = set(df.loc[df["rank_corrected"] <= top_n, "cd_code"])
    name = df.set_index("cd_code")["district_name"]

    def describe(codes):
        rows = df[df["cd_code"].isin(codes)].sort_values("rank_corrected")
        return [{"cd_code": int(r.cd_code), "district_name": r.district_name,
                 "rank_raw": int(r.rank_raw), "rank_corrected": int(r.rank_corrected)}
                for r in rows.itertuples()]

    return {
        "top_n": top_n,
        "entered_top_n_after_correction": describe(top_corr - top_raw),
        "left_top_n_after_correction": describe(top_raw - top_corr),
        "biggest_climbers": describe(df.nlargest(5, "rank_change")["cd_code"]),
        "top_n_corrected": [name[c] for c in
                            df.nsmallest(top_n, "rank_corrected")["cd_code"]],
        "districts_without_ratio": df.loc[~df["ratio_available"], "cd_code"].tolist(),
    }


# ---------------------------------------------------------------------------
# Chart: how ranks change after correction (top 15 + all Silent)
# ---------------------------------------------------------------------------
def plot_rank_change(df: pd.DataFrame, path: Path, top_n: int = TOP_N) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = df[(df["rank_corrected"] <= 15) | df["silent"]].sort_values(
        "rank_corrected", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(show) + 1.5))
    for i, r in enumerate(show.itertuples()):
        color = "#F46036" if r.silent else "#1B998B"
        ax.plot([r.rank_raw, r.rank_corrected], [i, i], color="#B8D8D4", lw=2.5,
                zorder=1)
        ax.scatter(r.rank_raw, i, s=45, color="white", edgecolor="#5F7178",
                   zorder=2)
        ax.scatter(r.rank_corrected, i, s=70, color=color, zorder=3)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["district_name"], fontsize=9)
    ax.invert_xaxis()                              # rank 1 on the right
    ax.axvline(top_n + 0.5, color="grey", ls=":", lw=1)
    ax.text(top_n + 0.5, len(show) - 0.3, f"  top {top_n} →", color="grey",
            fontsize=9, ha="right")
    ax.set_xlabel("Hidden Need rank  (○ raw 311  →  ● corrected; 1 = highest need)")
    ax.set_title("Correcting for under-reporting moves Silent Neighborhoods up",
                 loc="left", fontsize=13, fontweight="bold")
    ax.scatter([], [], color="#F46036", label="Silent Neighborhood")
    ax.scatter([], [], color="#1B998B", label="Other district")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for f, step in [(TABLE_FILE, "build_district_table.py"),
                    (BIAS_FILE, "reporting_bias.py")]:
        if not f.exists():
            raise SystemExit(f"Run src/{step} first.")

    table = pd.read_parquet(TABLE_FILE)
    bias = pd.read_parquet(BIAS_FILE)
    live_year = max(int(c.split("_")[1]) for c in table.columns
                    if c.startswith("total_") and c.endswith("_per_1k_renters"))

    df = build_index(correct_live_signal(table, bias, live_year))
    summary = summarize(df)
    summary["live_year"] = live_year

    cols = ["rank_corrected", "rank_raw", "rank_change", "cd_code",
            "district_name", "silent", "index_corrected"]
    print(f"=== HIDDEN NEED INDEX ({live_year} 311, corrected) ===")
    print(df[cols].head(TOP_N).to_string(index=False))

    print(f"\nEntered the top {TOP_N} only after correction:")
    for d in summary["entered_top_n_after_correction"] or [{"district_name": "none"}]:
        extra = (f"  (rank {d['rank_raw']} -> {d['rank_corrected']})"
                 if "rank_raw" in d else "")
        print(f"  {d['district_name']}{extra}")

    print("\nBiggest climbers:")
    for d in summary["biggest_climbers"]:
        print(f"  {d['district_name']:<32} rank {d['rank_raw']:>2} -> "
              f"{d['rank_corrected']:>2}")
    if summary["districts_without_ratio"]:
        print(f"\nNot corrected (no reporting ratio): "
              f"{summary['districts_without_ratio']}")

    keep = ["cd_code", "borough", "district_name", "silent", "reporting_ratio",
            "ratio_available", "raw_rate", "corrected_rate", MH_COL,
            "housing_z_raw", "housing_z_corrected", "mental_health_z",
            "index_raw", "index_corrected", "rank_raw", "rank_corrected",
            "rank_change"]
    df[keep].to_parquet(OUT_FILE, index=False)
    df[keep].to_csv(OUT_FILE.with_suffix(".csv"), index=False)
    (OUT_DIR / "hidden_need_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    plot_rank_change(df, OUT_DIR / "fig_rank_change.png")

    print(f"\nSaved: {OUT_FILE} (+ .csv)")
    print(f"Saved: {OUT_DIR / 'hidden_need_summary.json'}")
    print(f"Saved: {OUT_DIR / 'fig_rank_change.png'}")