"""
Phase 4 - Reporting-Bias Model: finding "Silent Neighborhoods"
Project: Hidden Need Index

The question
------------
Do some neighborhoods complain to 311 LESS than their real housing problems
would predict?

How
---
Step 1  Expected complaints
        Across all districts, fit a line: how many complaints per 1,000
        renter homes do we usually see for a given % of homes with defects?
            log(complaint rate) = a + b * (% homes with defects)
        (log, because complaint rates are skewed and the effect of defects
        is closer to "x% more complaints per point" than "+N complaints")

Step 2  Reporting ratio
        reporting_ratio = actual complaint rate / expected complaint rate
            1.0  -> complains as much as its defects predict
            0.6  -> complains 40% LESS than predicted  (under-reports)
            1.4  -> complains 40% MORE than predicted  (over-reports)

Step 3  Silent Neighborhoods
        High real need (defects at or above the city median) AND
        reporting ratio below 0.8 (at least 20% fewer complaints than predicted)

Step 4  Who under-reports?
        Regress log(reporting ratio) on limited English, foreign-born share
        and poverty. Predictors are standardized (z-scores) so their effects
        can be compared: "1 standard deviation more X -> Y% change".

Data used: 2023 311 complaints vs the 2023 Housing and Vacancy Survey
(same year). Tottenville (503) is excluded: DOHMH suppressed its estimate.

Output: data/clean/reporting_bias.parquet (+ .csv)
        outputs/reporting_bias_summary.json
        outputs/fig_silent_neighborhoods.png

Run from the project root:   python src/reporting_bias.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IN_FILE = CLEAN_DIR / "district_table.parquet"
OUT_FILE = CLEAN_DIR / "reporting_bias.parquet"

SURVEY_YEAR = 2023
RATE_COL = f"total_{SURVEY_YEAR}_per_1k_renters"
NEED_COL = "pct_homes_with_defects"
SILENT_RATIO = 0.8                  # 20%+ fewer complaints than predicted
EXPLANATORY = ["pct_limited_english", "pct_foreign_born", "pct_poverty"]


# ---------------------------------------------------------------------------
# Step 1 + 2: expected complaints and reporting ratio
# ---------------------------------------------------------------------------
def fit_expected(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit log(rate) ~ defects and add expected rate + reporting ratio."""
    data = df.dropna(subset=[RATE_COL, NEED_COL]).copy()
    data = data[data[RATE_COL] > 0]

    X = sm.add_constant(data[[NEED_COL]])
    y = np.log(data[RATE_COL])
    model = sm.OLS(y, X).fit()

    data["expected_rate"] = np.exp(model.predict(X)).round(2)
    data["reporting_ratio"] = (data[RATE_COL] / data["expected_rate"]).round(3)

    b = model.params[NEED_COL]
    stats = {
        "n_districts": int(len(data)),
        "r_squared": round(float(model.rsquared), 3),
        "pct_more_complaints_per_defect_point": round(float(np.expm1(b) * 100), 2),
        "p_value": float(model.pvalues[NEED_COL]),
    }
    return data, stats


# ---------------------------------------------------------------------------
# Step 3: flag Silent Neighborhoods
# ---------------------------------------------------------------------------
def flag_silent(data: pd.DataFrame, ratio_cutoff: float = SILENT_RATIO) -> pd.DataFrame:
    data = data.copy()
    median_need = data[NEED_COL].median()
    data["high_need"] = data[NEED_COL] >= median_need
    data["silent"] = data["high_need"] & (data["reporting_ratio"] < ratio_cutoff)
    data["silence_gap_pct"] = ((1 - data["reporting_ratio"]) * 100).round(1)
    data.attrs["median_need"] = float(median_need)
    return data


# ---------------------------------------------------------------------------
# Step 4: who under-reports?
# ---------------------------------------------------------------------------
def explain_reporting(data: pd.DataFrame) -> dict:
    """Standardized regression of log(reporting ratio) on community traits."""
    d = data.dropna(subset=EXPLANATORY + ["reporting_ratio"])
    X = (d[EXPLANATORY] - d[EXPLANATORY].mean()) / d[EXPLANATORY].std()
    X = sm.add_constant(X)
    y = np.log(d["reporting_ratio"])
    model = sm.OLS(y, X).fit()

    effects = {}
    for var in EXPLANATORY:
        effects[var] = {
            # "+1 standard deviation of var -> X% change in reporting ratio"
            "pct_change_per_sd": round(float(np.expm1(model.params[var]) * 100), 1),
            "p_value": round(float(model.pvalues[var]), 4),
            "correlation": round(float(d[var].corr(np.log(d["reporting_ratio"]),
                                                   method="spearman")), 3),
        }
    return {"n_districts": int(len(d)),
            "r_squared": round(float(model.rsquared), 3),
            "effects": effects}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def plot(data: pd.DataFrame, stats: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")                       # save to file, no window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    normal = data[~data["silent"]]
    silent = data[data["silent"]]
    ax.scatter(normal[NEED_COL], normal[RATE_COL], s=45, color="#1B998B",
               alpha=0.75, label="Other districts")
    ax.scatter(silent[NEED_COL], silent[RATE_COL], s=90, color="#F46036",
               edgecolor="black", label="Silent Neighborhoods")

    line = data.sort_values(NEED_COL)
    ax.plot(line[NEED_COL], line["expected_rate"], color="#0B3C49", lw=2,
            label="Expected complaints")
    ax.axvline(data.attrs["median_need"], color="grey", ls=":", lw=1)

    # alternate labels below/above the dot so neighbors don't overlap
    for i, (_, r) in enumerate(silent.sort_values(NEED_COL).iterrows()):
        dy = -14 if i % 2 == 0 else 10
        ax.annotate(r["district_name"], (r[NEED_COL], r[RATE_COL]),
                    xytext=(7, dy), textcoords="offset points", fontsize=9)

    ax.set_xlabel("% of renter homes with housing defects (survey, 2023)")
    ax.set_ylabel("311 housing complaints per 1,000 renter homes (2023)")
    ax.set_title("Silent Neighborhoods: real problems, fewer complaints",
                 loc="left", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not IN_FILE.exists():
        raise SystemExit("Run src/build_district_table.py first.")
    table = pd.read_parquet(IN_FILE)

    data, fit_stats = fit_expected(table)
    data = flag_silent(data)
    explained = explain_reporting(data)

    print("=== STEP 1: DO DEFECTS PREDICT COMPLAINTS? ===")
    print(f"Districts analyzed: {fit_stats['n_districts']} "
          f"(503 excluded - suppressed by DOHMH)")
    print(f"Each extra percentage point of homes with defects -> "
          f"{fit_stats['pct_more_complaints_per_defect_point']}% more complaints")
    print(f"R-squared: {fit_stats['r_squared']}   p-value: {fit_stats['p_value']:.2g}")

    cols = ["cd_code", "district_name", NEED_COL, RATE_COL,
            "expected_rate", "reporting_ratio"]
    silent = data[data["silent"]].sort_values("reporting_ratio")
    print(f"\n=== STEP 3: SILENT NEIGHBORHOODS ({len(silent)}) ===")
    print(f"High need = defects >= {data.attrs['median_need']:.0f}% (city median), "
          f"and reporting ratio < {SILENT_RATIO}")
    print(silent[cols].to_string(index=False) if len(silent) else "  none found")

    print("\n=== STEP 4: WHO UNDER-REPORTS? ===")
    print(f"(n = {explained['n_districts']}, R-squared = {explained['r_squared']})")
    for var, e in explained["effects"].items():
        print(f"  {var:<22} +1 SD -> {e['pct_change_per_sd']:+.1f}% reporting "
              f"(p = {e['p_value']:.3f}, Spearman r = {e['correlation']:+.2f})")
    print("  Negative % = communities with more of this complain LESS than "
          "their defects predict.")

    keep = ["cd_code", "borough", "district_name", "pct_renter", NEED_COL,
            f"{NEED_COL}_caution", RATE_COL, "expected_rate", "reporting_ratio",
            "silence_gap_pct", "high_need", "silent"] + EXPLANATORY
    data[keep].to_parquet(OUT_FILE, index=False)
    data[keep].to_csv(OUT_FILE.with_suffix(".csv"), index=False)

    summary = {"survey_year": SURVEY_YEAR, "silent_ratio_cutoff": SILENT_RATIO,
               "median_need_pct": data.attrs["median_need"],
               "expected_complaints_model": fit_stats,
               "silent_neighborhoods": silent[["cd_code", "district_name",
                                               "reporting_ratio"]].to_dict("records"),
               "who_under_reports": explained}
    (OUT_DIR / "reporting_bias_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    plot(data, fit_stats, OUT_DIR / "fig_silent_neighborhoods.png")

    print(f"\nSaved: {OUT_FILE} (+ .csv)")
    print(f"Saved: {OUT_DIR / 'reporting_bias_summary.json'}")
    print(f"Saved: {OUT_DIR / 'fig_silent_neighborhoods.png'}")