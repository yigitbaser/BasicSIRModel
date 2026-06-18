# Trusted Data Sources for COVID-19 Modelling

To fit or validate any of the models in this repository you need real
case/death/hospitalisation time series. Use **authoritative, well-documented**
sources only — never scrape unverified dashboards or social-media figures. The
sources below are the ones the epidemiological modelling community actually
relied on, with notes on their current status (several stopped updating once
the WHO declared the end of the global health emergency in **May 2023**).

## Primary global repositories

| Source | What it provides | Granularity | Status / notes | URL |
|---|---|---|---|---|
| **WHO COVID-19 Dashboard** | Official cases & deaths reported by member states | Country, weekly | **Actively maintained**; the authoritative global reference | https://data.who.int/dashboards/covid19/ |
| **Our World in Data (OWID)** | Cases, deaths, tests, vaccinations, hospitalisations, policy indices — cleaned & per-capita normalised | Country, daily | Excellent for modelling; well-documented CSV/JSON; ongoing | https://github.com/owid/covid-19-data |
| **JHU CSSE COVID-19 Data** | The canonical daily case/death time series 2020–2023 | Country + US county, daily | **Archived 10 Mar 2023** (read-only) — still the best historical record | https://github.com/CSSEGISandData/COVID-19 |

## Regional / national authorities

| Source | Coverage | URL |
|---|---|---|
| **US CDC COVID Data Tracker** | United States (cases, deaths, hospitalisations, variants, wastewater) | https://covid.cdc.gov/covid-data-tracker/ |
| **ECDC** (European CDC) | EU/EEA surveillance data | https://www.ecdc.europa.eu/en/covid-19/data |
| **UK Health Security Agency** | United Kingdom dashboard & API | https://ukhsa-dashboard.data.gov.uk/ |
| **US HHS / HealthData.gov** | US hospital capacity & utilisation | https://healthdata.gov/ |

## Genomic / variant surveillance

| Source | What it provides | URL |
|---|---|---|
| **GISAID** | Global SARS-CoV-2 genome sequences & variant prevalence | https://gisaid.org/ |
| **Nextstrain** | Real-time phylogenetic tracking of variants | https://nextstrain.org/ncov/gisaid/global |
| **CoVariants** | Variant prevalence over time by country | https://covariants.org/ |

## Why source selection matters for a model

1. **Reporting artefacts.** Raw case counts dip on weekends and jump after
   backlog corrections. Use 7-day rolling averages (OWID provides these) before
   fitting, or the model will "see" noise as dynamics.
2. **Cases ≠ infections.** Confirmed cases undercount true infections by a
   factor that changed over time with test availability. To estimate the IFR
   (not just the CFR) you need **seroprevalence** studies, not case counts.
3. **Denominators.** Per-capita comparison requires reliable population
   denominators — OWID bundles these; raw JHU does not.
4. **Definitions drift.** "COVID death" definitions, PCR vs antigen, and
   hospitalisation criteria changed across jurisdictions and time. Prefer a
   single harmonised source (OWID/WHO) over stitching national feeds together.

## Quick start: loading OWID data

```python
import pandas as pd

URL = ("https://raw.githubusercontent.com/owid/covid-19-data/"
       "master/public/data/owid-covid-data.csv")
df = pd.read_csv(URL, parse_dates=["date"])

# Example: Italy's smoothed daily new cases, ready to fit against a model.
italy = df[df["location"] == "Italy"][["date", "new_cases_smoothed", "population"]]
```

> Network access in this environment is governed by the session's policy; if the
> download is blocked, fetch the CSV where you have connectivity and commit a
> snapshot, or point the loader at a local copy.
