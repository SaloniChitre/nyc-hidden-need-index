# Data Catalog

Every dataset used in the Hidden Need Index, where it comes from, and what to watch out for.

## Source datasets

### 1. NYC 311 Service Requests (2020 to Present)

| | |
|---|---|
| **Owner** | NYC Office of Technology and Innovation, via NYC Open Data |
| **Dataset ID** | `erm2-nwe9` |
| **Access** | Socrata API (SoQL), no key required; optional `NYC_APP_TOKEN` reduces throttling |
| **Filter used** | `agency = 'HPD'` (Housing Preservation & Development: indoor housing conditions) |
| **Years used** | January 2021 – latest month |
| **Geography** | Community board label, e.g. `"12 QUEENS"` → district code 412 |
| **Grain pulled** | Count of complaints by community board × month × complaint type (aggregated on the server) |
| **Script** | `src/fetch_311.py` |
| **Raw file** | `data/raw/311_monthly/<YYYY-MM>.csv`, `data/raw/311_hpd_monthly_<date>.csv` |

**Known issues**
- Some complaints have no usable location ("Unspecified") or fall in parks/airports; these are counted and excluded (0.1% of complaints).
- Some complaint types appear in two formats (`APPLIANCE` / `Appliance`); merged in `clean_311.py`.
- Large aggregated queries can time out, so data is pulled one month at a time with checkpointing.
- In December 2025 the 311 dataset was split: 2010–2019 moved to a separate dataset; `erm2-nwe9` covers 2020 onward.
- Complaints reflect **who reports**, not only where problems are. Measuring that bias is the point of this project.

### 2. DOHMH Community Health Profiles, 2026 Public Use Dataset

| | |
|---|---|
| **Owner** | NYC Department of Health and Mental Hygiene |
| **Access** | Excel download from the Community Health Profiles page on nyc.gov. **Must be downloaded in a browser**: nyc.gov returns an HTML page to scripted requests |
| **File** | `data/raw/2026-chp-pud.xlsx`, sheet `CHP_all_data` (header on row 2) |
| **Geography** | 59 community districts (same 3-digit codes as 311) |
| **Script** | `src/build_district_table.py` |

**Columns used**

| CHP column | Our name | Meaning | Year(s) | Original source |
|---|---|---|---|---|
| `Overall_Pop` | `population` | Residents | 2024 | DOHMH estimates, modified from Census intercensal estimates |
| `Homes_Any_Defects` | `pct_homes_with_defects` | % of renter homes with any health-related housing problem | 2023 | NYC Housing and Vacancy Survey |
| `Homes_Roach` | `pct_homes_roaches` | % of renter homes with cockroaches | 2023 | NYC Housing and Vacancy Survey |
| `Ltd_Eng_Prof` | `pct_limited_english` | % who speak English less than "very well" | 2019–2023 | American Community Survey |
| `Born_Outside_US` | `pct_foreign_born` | % born outside the US | 2019–2023 | American Community Survey |
| `Poverty` | `pct_poverty` | % below the NYC poverty threshold | 2018–2022 | American Community Survey / NYC Opportunity |
| `Rent_Burden` | `pct_rent_burden` | % of renters paying 30%+ of income on rent | 2019–2023 | American Community Survey |
| `Psych_Hosp` | `psych_hosp_per_100k` | Psychiatric hospitalizations per 100,000 adults | 2023 | NYS DOH, SPARCS hospital discharge data |

Years and sources are taken from the file's own `Metadata` sheet, which has full definitions.

**Known issues**
- `^` = estimate suppressed as unreliable. Stored as **missing, never zero**. Tottenville (503) has no housing-defects estimate.
- `*` = interpret with caution. Kept but flagged (`*_caution` columns): districts 409, 410, 411, 413, 502 for housing defects.
- Some district pairs share one value because the underlying estimate covers a larger area (e.g. 101 & 102). Effective independent sample size is below 59.
- Rows for NYC and borough totals, and footnote rows, are removed by keeping only IDs 101–599.
- Psychiatric hospitalizations reflect **service use** and can be higher near facilities.

### 3. American Community Survey, 5-year estimates (2019–2023): Tenure

| | |
|---|---|
| **Owner** | US Census Bureau |
| **Table** | B25003 (Tenure): `B25003_001E` occupied homes, `B25003_003E` renter-occupied homes |
| **Access** | Census Data API; **requires a free key** in the `CENSUS_API_KEY` environment variable (never commit it) |
| **Geography** | PUMA (Public Use Microdata Area), New York State (`state:36`) |
| **Script** | `src/fetch_acs_renters.py` |
| **Raw file** | `data/raw/acs_renters_2023.csv` |

**Known issues**
- NYC PUMAs are built from community districts, and their names list them (e.g. "NYC-Manhattan Community Districts 1 & 2--..."). The script parses these names.
- Four PUMAs cover two districts (101/102, 105/106, 201/202, 203/206). Their renter homes are split by each district's share of population and flagged `renter_homes_estimated_split`.
- ACS figures are estimates with margins of error.

### 4. Community District boundaries

| | |
|---|---|
| **Owner** | NYC Department of City Planning, via NYC Open Data |
| **Dataset ID** | `5crt-au7u` |
| **Access** | GeoJSON from NYC Open Data, no key |
| **Contents** | 59 community districts + 12 Joint Interest Areas (major parks and airports, drawn grey) |
| **Key field** | `BoroCD` / `boro_cd` (borough digit + district number) |
| **Script** | `src/make_maps.py` |
| **Raw file** | `data/raw/community_districts.geojson` |

## Derived tables

| File | Grain | Built by | Key columns |
|---|---|---|---|
| `data/clean/311_hpd_monthly.parquet` | district × month × complaint type | `clean_311.py` | `cd_code`, `month`, `category`, `complaints` |
| `data/clean/quality_report.json` | one report per run | `clean_311.py` | checks, counts by location status, merged variants |
| `data/clean/311_district_yearly.parquet` | district × year | `summarize_311.py` | `total`, category columns, `full_year` |
| `data/clean/district_table.parquet` | district (59 rows) | `build_district_table.py` | survey measures, `renter_homes`, `total_<year>_per_1k_renters` |
| `data/clean/reporting_bias.parquet` | district (58 rows) | `reporting_bias.py` | `expected_rate`, `reporting_ratio`, `silent` |
| `data/clean/hidden_need_index.parquet` | district (59 rows) | `hidden_need_index.py` | `corrected_rate`, `index_raw`, `index_corrected`, `rank_raw`, `rank_corrected` |

## Lineage

```
311 API ──> fetch_311 ──> clean_311 ──> summarize_311 ─┐
DOHMH CHP (browser download) ──────────────────────────┼──> build_district_table ──> reporting_bias ──> hidden_need_index ──> make_maps
Census ACS API ──> fetch_acs_renters ──────────────────┘                                                    ▲
Community District boundaries (Open Data) ──────────────────────────────────────────────────────────────────┘
```

## Privacy and security

- All sources are public and aggregated to the neighborhood level; no individual records are stored.
- API keys are read from environment variables and are never written to code or committed.
- `data/` is git-ignored; the pipeline rebuilds it from source.