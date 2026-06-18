# Scientifically-Backed COVID-19 Parameters

This file lists the epidemiological constants used to drive the models in
`models.py`, together with the peer-reviewed source for each number. These are
the **ancestral / wild-type SARS-CoV-2** values (the strain circulating in
early 2020). Later variants (Alpha, Delta, Omicron) had materially different
transmissibility and severity — see the "Variant note" at the bottom.

> **Caveat — read this first.** None of these numbers is a single universal
> constant. Every one is an *estimate with a confidence interval*, and the
> estimates vary by setting, population age structure, surveillance quality and
> control measures in place. The point estimates below are the most widely
> cited pooled/meta-analytic values; always carry the interval, not just the
> mean.

## Core transmission & natural-history parameters

| Parameter | Symbol | Point estimate | Range / 95% CI | Source |
|---|---|---|---|---|
| Basic reproduction number (ancestral) | R₀ | **2.79** (median of 14 estimates); pooled meta-analytic **2.66** | 1.9 – 3.9 (pooled CI 2.41–2.94) | Billah et al. 2020 (PLOS ONE); Alimohamadi et al. 2020 |
| Incubation period (infection → symptoms) | — | **5.1 days** | 95% CI 4.5–5.8; 97.5% symptomatic by 11.5 d | Lauer et al. 2020 (Ann Intern Med) |
| Latent period (infection → infectious) | 1/σ | **≈ 3–4 days** | shorter than incubation (pre-symptomatic transmission) | He et al. 2020 (Nat Med); Lauer et al. 2020 |
| Serial interval | — | **5.2 days** (pooled) | 95% CI 4.9–5.5 (range 4.2–7.5) | Rai et al. 2021; Nishiura et al. 2020 |
| Infectious period | 1/γ | **≈ 6–10 days** (modelling: 6.5 d used here) | viral shedding context-dependent | Byrne et al. 2020 (BMJ Open); WHO |
| Generation time | — | **≈ 5 days** | 95% CI ~4–6 | Ganyani et al. 2020 (Eurosurveillance) |

## Severity parameters (pre-vaccine era)

| Parameter | Point estimate | Notes | Source |
|---|---|---|---|
| Population infection-fatality ratio (IFR) | **≈ 0.5 – 1.0%** (we use 0.68%) | Extremely age-dependent; depends on population age structure | COVID-19 Forecasting Team, *Lancet* 2022; Levin et al. 2020 |
| Age-specific IFR | 0.001% @ age 5 → 8.4% @ age 80 | log-linear with age; >8000× difference | Levin et al. 2020 (Eur J Epidemiol) |
| IFR @ 55 / 65 / 75 / 85 yrs | 0.4% / 1.3% / 4.2% / 14% | pre-vaccine | Levin et al. 2020 |

**Why IFR ≠ CFR.** The *case* fatality ratio (deaths / confirmed cases)
overestimates lethality because many infections are never confirmed. The
*infection* fatality ratio (deaths / all infections) is the epidemiologically
meaningful quantity and requires seroprevalence data to estimate.

## Derived rates used in the code

The models convert the interpretable quantities above into ODE rate constants:

```
gamma = 1 / t_infectious      # recovery rate (per day)
sigma = 1 / t_latent          # E -> I progression rate (per day)
beta  = R0 * gamma            # transmission rate (per day)
```

With the defaults (`R0 = 2.79`, `t_infectious = 6.5 d`, `t_latent = 4.0 d`):

```
gamma ≈ 0.154 /day
sigma  = 0.250 /day
beta  ≈ 0.429 /day
```

## Derived epidemic quantities (computed, not assumed)

These fall out of the parameters and are reported by `statistics.py`:

| Quantity | Formula | Ancestral value |
|---|---|---|
| Herd-immunity threshold | 1 − 1/R₀ | ≈ **64%** at R₀ = 2.79 |
| Final attack rate (unmitigated) | implicit final-size equation | ≈ **92%** at R₀ = 2.79 |
| Epidemic doubling time (early) | ln(2) / growth rate | a few days (depends on structure) |

## Variant note (why ancestral values are not universal)

| Variant | Approx. R₀ | Relative to ancestral |
|---|---|---|
| Ancestral (Wuhan, 2020) | ~2.8 | baseline |
| Alpha (B.1.1.7) | ~4–5 | ~1.5× |
| Delta (B.1.617.2) | ~5–8 | ~2× |
| Omicron (B.1.1.529) | ~8–10+ | highest transmissibility, lower intrinsic severity |

Source for Delta vs ancestral: Liu & Rocklöv 2021 (*J Travel Med*).

---

## Full citations

1. **Lauer SA, et al.** "The Incubation Period of Coronavirus Disease 2019
   (COVID-19) From Publicly Reported Confirmed Cases." *Annals of Internal
   Medicine*, 2020. https://www.acpjournals.org/doi/10.7326/m20-0504
2. **Billah MA, Miah MM, Khan MN.** "Reproductive number of coronavirus: A
   systematic review and meta-analysis based on global level evidence."
   *PLOS ONE*, 2020. https://doi.org/10.1371/journal.pone.0242128
3. **Alimohamadi Y, et al.** "Estimate of the Basic Reproduction Number for
   COVID-19: A Systematic Review and Meta-analysis." *J Prev Med Public
   Health*, 2020. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9517346/
4. **He X, et al.** "Temporal dynamics in viral shedding and transmissibility
   of COVID-19." *Nature Medicine*, 2020. https://doi.org/10.1038/s41591-020-0869-5
5. **Rai B, Shukla A, Dwivedi LK.** "Estimates of serial interval for COVID-19:
   A systematic review and meta-analysis." *Clinical Epidemiology and Global
   Health*, 2021. https://www.sciencedirect.com/science/article/pii/S2213398420301895
6. **Nishiura H, Linton NM, Akhmetzhanov AR.** "Serial interval of novel
   coronavirus (COVID-19) infections." *Int J Infect Dis*, 2020.
   https://www.sciencedirect.com/science/article/pii/S1201971220301193
7. **Ganyani T, et al.** "Estimating the generation interval for COVID-19 based
   on symptom onset data." *Eurosurveillance*, 2020.
   https://doi.org/10.2807/1560-7917.ES.2020.25.17.2000257
8. **Byrne AW, et al.** "Inferred duration of infectious period of SARS-CoV-2."
   *BMJ Open*, 2020. https://bmjopen.bmj.com/content/10/8/e039856
9. **Levin AT, et al.** "Assessing the age specificity of infection fatality
   rates for COVID-19." *European Journal of Epidemiology*, 2020.
   https://doi.org/10.1007/s10654-020-00698-1
10. **COVID-19 Forecasting Team.** "Variation in the COVID-19 infection–fatality
    ratio by age, time, and geography during the pre-vaccine era." *The Lancet*,
    2022. https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)02867-1/fulltext
11. **Liu Y, Rocklöv J.** "The reproductive number of the Delta variant of
    SARS-CoV-2 is far higher compared to the ancestral SARS-CoV-2 virus."
    *Journal of Travel Medicine*, 2021. https://academic.oup.com/jtm/article/28/7/taab124/6346388
