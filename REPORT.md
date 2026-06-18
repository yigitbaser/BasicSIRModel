# Modelling COVID-19: An Assessment of the SIR Model and Better Alternatives

**Author's brief:** assess the SIR-family models in general, recommend models
that give better predictions, and build a model that produces statistics. This
report covers the first two; the accompanying `models.py` and `statistics.py`
deliver the third. Parameter sources are in `PARAMETERS.md`; data sources in
`DATA_SOURCES.md`.

---

## 1. What the original model does

The original `model.py` is a textbook **deterministic SIR** model integrated
with forward Euler:

```
S' = -β·S·I/N
I' =  β·S·I/N − γ·I
R' =  γ·I
```

It partitions a fixed population into **S**usceptible, **I**nfectious and
**R**ecovered, with a transmission rate β and a recovery rate γ. It is a
faithful, clean implementation of the 1927 Kermack–McKendrick model and is
excellent for *teaching* the core idea: an epidemic grows while the effective
reproduction number `Rₜ = R₀·S/N` exceeds 1, peaks when `Rₜ = 1`, and burns out
as susceptibles are depleted.

For *predicting* COVID-19, however, plain SIR is structurally too simple. The
sections below explain why, and what to use instead.

---

## 2. Assessment of the SIR model — strengths and limitations

### Strengths
- **Interpretable & analytically tractable.** R₀, the herd-immunity threshold
  (1 − 1/R₀) and the final-size relation all drop out in closed form.
- **Few parameters**, so it is hard to over-fit and easy to reason about.
- **Good qualitative behaviour**: it reproduces the single-wave epidemic curve.

### Limitations (and why each one matters for COVID-19)

| # | Limitation of plain SIR | Consequence for COVID-19 | Fix |
|---|---|---|---|
| 1 | **No latent/exposed state.** Infection is assumed to confer immediate infectiousness. | COVID-19 has a ~3–5 day latent period; SIR makes the epidemic rise and peak too early. | **SEIR** (add E) |
| 2 | **No deaths compartment.** R lumps "recovered" and "dead" together. | Can't read off mortality or compute IFR/CFR. | **SIRD / SEIRD** |
| 3 | **Homogeneous mixing.** Everyone contacts everyone equally. | Ignores age structure, households, super-spreading; COVID-19 risk is hugely age-dependent. | **Age-structured / network / metapopulation** |
| 4 | **Constant β.** No lockdowns, masks, seasonality or behaviour change. | Cannot reproduce multiple waves or NPIs. | **Time-varying β(t)** |
| 5 | **Permanent immunity.** | COVID-19 immunity wanes; reinfection happens. | **SIRS / SEIRS + waning** |
| 6 | **Deterministic & continuous.** Fractional people; no randomness. | Wrong at low counts (introductions, extinction, early-outbreak stochasticity). | **Stochastic / agent-based** |
| 7 | **No asymptomatic / pre-symptomatic class.** | A large share of COVID-19 transmission was pre-/asymptomatic; omitting it biases β and control estimates. | **SEIAR / compartment split** |
| 8 | **No vaccination, no hospital capacity.** | Can't evaluate the interventions that mattered most. | **SEIRDV + hospital compartments** |
| 9 | **Point estimates, no uncertainty.** | A single curve with no confidence band is dangerous for policy. | **Bayesian fitting / ensembles** |
| 10 | **Forward-Euler integration** (original code). | Accumulates numerical error; can go unstable for large β·Δt. | **Adaptive ODE solver** (done here via `solve_ivp`) |

---

## 3. Better models, in increasing order of realism

### 3.1 SEIR — add an Exposed (latent) compartment *(low effort, high value)*
```
S' = -β·S·I/N
E' =  β·S·I/N − σ·E
I' =  σ·E − γ·I
R' =  γ·I
```
The single most important upgrade for COVID-19. The latent period (1/σ) delays
infectiousness, which **slows early growth and pushes the peak later** — exactly
what we see in `run_statistics.py`: the SIR peak lands on day ~36, the SEIR peak
on day ~74 for identical R₀. Getting the *timing* right matters as much as the
height for hospital planning.

### 3.2 SEIRD — add an explicit Dead compartment *(recommended baseline)*
Splits the outflow from I into recovery and death using the **infection fatality
ratio**. This is the minimum structure that lets you predict the quantity policy
cares about most — deaths — and it is the default model in this repo.

### 3.3 SEIRS / waning immunity — endemic dynamics
Add a flow R → S at rate ω (1/duration-of-immunity). Produces recurring waves
and an endemic steady state rather than a single burn-out. Essential once you
model beyond the first year.

### 3.4 Time-varying transmission β(t) — interventions & seasonality
Make β a function of time to encode lockdowns, mask mandates, school closures
and seasonality. `models.py` already accepts a `beta_t(t, beta0)` callback. This
is what lets a model reproduce **multiple waves** instead of one.

### 3.5 Age-structured / compartment-stratified models
Replace each compartment with an age-stratified vector and couple them with a
**contact matrix** (e.g. POLYMOD/Prem matrices). Because COVID-19 IFR rises
>8000× from age 5 to 80 (see `PARAMETERS.md`), age structure is the single
biggest driver of realistic mortality predictions and of targeted-vaccination
analysis.

### 3.6 Metapopulation models — spatial spread
Many coupled sub-populations (cities/regions) with a mobility matrix between
them. Captures importation, travel restrictions and asynchronous regional waves.

### 3.7 Network models — contact heterogeneity & super-spreading
Individuals are nodes; transmission flows along edges. Captures the
**over-dispersion** of COVID-19 (a small fraction of cases caused most
transmission), which mean-field SIR cannot represent.

### 3.8 Agent-based models (ABMs) — maximum realism
Simulate individuals with households, workplaces, schools and explicit
behaviour. Tools: **Covasim** (IDM), **OpenABM-Covid19**, **EpiABM**. Best for
detailed what-if policy analysis (test-trace-isolate, school reopening). Cost:
many parameters, heavy computation, harder to fit.

### 3.9 Stochastic compartmental models
Replace ODEs with a continuous-time Markov chain (Gillespie) or chain-binomial
process. Necessary at low case counts where chance extinction and
introduction-timing dominate.

### 3.10 Statistical / data-driven & hybrid models
- **Rₜ estimation** from incidence (EpiEstim / Cori method, Wallinga–Teunis) —
  no mechanistic model needed; directly tracks whether transmission is growing.
- **Time-series / ML** (ARIMA, Prophet, LSTMs) for short-horizon forecasting.
- **Renewal-equation / semi-mechanistic** models (e.g. the Imperial College
  EpiNow2 / Flaxman *Nature* 2020 approach) — the workhorses of operational
  short-term forecasting.

### 3.11 Bayesian inference & ensembles — *quantify the uncertainty*
Whatever the structure, fit it with **Bayesian methods** (MCMC / particle
filtering via PyMC, Stan, NumPyro) to obtain **credible intervals**, not a
single line. Operationally, **multi-model ensembles** (the US/EU COVID-19
Forecast Hubs) consistently beat any single model — the right answer to "which
model?" is often "a calibrated ensemble of several."

---

## 4. Recommendation

| Goal | Recommended model |
|---|---|
| Teaching / intuition | **SIR** (the original) |
| Realistic single-wave baseline with deaths | **SEIRD** *(this repo's default)* |
| Multi-wave / interventions | **SEIRD with time-varying β(t)** + waning |
| Mortality & targeted vaccination | **Age-structured SEIRD** with contact matrix |
| Detailed policy what-ifs | **Agent-based (Covasim)** |
| Operational short-term forecasting | **Semi-mechanistic renewal model + Bayesian fit, in an ensemble** |
| Any forecast for decision-making | Always report **uncertainty** (Bayesian / ensemble) |

**Bottom line:** keep the original SIR as the teaching artefact, adopt **SEIRD**
as the working baseline (provided here), and layer on time-varying transmission,
age structure and Bayesian uncertainty as the question demands. No model is
"correct"; the goal is a model whose structure matches the decision being made,
fit to trusted data (`DATA_SOURCES.md`) with honest confidence intervals.

---

## 5. What ships in this repository

| File | Purpose |
|---|---|
| `model.py` | The original primitive SIR model (unchanged, for reference) |
| `models.py` | New ODE-based SIR / SIRD / SEIR / SEIRD / SEIRDV models with interpretable parameters and time-varying β support |
| `statistics.py` | Computes R₀, Rₜ(t), peak size & timing, attack rate, deaths, realised IFR, doubling time, herd-immunity threshold, epidemic duration; plotting helpers |
| `run_statistics.py` | Driver: runs every model, prints the statistics table, writes figures + CSV |
| `PARAMETERS.md` | Literature-backed parameter values with citations |
| `DATA_SOURCES.md` | Trusted data repositories for fitting/validation |
| `REPORT.md` | This report |

### Reproduce the statistics
```bash
pip install numpy scipy pandas matplotlib
python run_statistics.py
```
Outputs: `seird_trajectory.png`, `model_comparison.png`, `statistics_summary.csv`,
and a console table of headline statistics for SIR vs SEIR vs SEIRD vs a
vaccinated SEIRDV scenario.

### Illustrative results (ancestral strain, N = 1,000,000, R₀ = 2.79)

| Model | Peak infectious | Peak day | Attack rate | Deaths |
|---|---|---|---|---|
| SIR | ~273,700 (27%) | ~36 | ~92% | n/a |
| SEIR | ~166,400 (17%) | ~74 | ~92% | n/a |
| SEIRD | ~166,400 (17%) | ~74 | ~92% | ~6,300 |
| SEIRDV (0.5%/day) | ~69,300 (7%) | ~84 | ~53% | ~3,600 |

The comparison makes the report's central point concrete: **adding the latent
compartment (SEIR) nearly halves the predicted peak and doubles the time-to-peak
versus SIR**, and vaccination flattens the curve and roughly halves both the
attack rate and deaths — differences that are decisive for hospital planning yet
entirely invisible to the plain SIR model.
