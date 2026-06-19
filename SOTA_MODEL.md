# A State-of-the-Art COVID-19 Model

This is a clean-slate, state-of-the-art model built **independently** of the
SIR/SEIRD code elsewhere in this repository. It is the model family that defines
the current standard for COVID-19 nowcasting and short-term forecasting:
a **Bayesian semi-mechanistic renewal model with a time-varying reproduction
number**, fit by gradient-based Hamiltonian Monte Carlo.

It is the approach behind:

- **EpiNow2** (Abbott et al.) — the reference open-source nowcasting tool;
- the **Imperial College** report estimating the effect of NPIs in Europe
  (Flaxman et al., *Nature*, 2020);
- the **US and European COVID-19 Forecast Hubs**, whose ensembles of such
  models were used by the CDC/ECDC for operational forecasting.

> Files: [`sota_model.py`](sota_model.py) (the model + inference) and
> [`sota_run.py`](sota_run.py) (data loading, validation, plotting).

---

## 1. Why not just a bigger compartmental model?

The natural instinct is "SIR is too simple, so build SEIR, then SEIRDV, then an
agent-based model." But adding compartments does not fix the two things that
actually matter for *forecasting*:

1. **Transmission is not constant.** What drives a real epidemic curve is how
   the reproduction number changes over time (lockdowns, variants, behaviour).
   A mechanistic model bakes this into a fixed `beta`; you then have to *guess*
   `beta(t)`. The state-of-the-art move is to **infer the time-varying R_t
   directly from data** and let it be whatever the data say.

2. **Data are dirty.** Reported cases are delayed, under-ascertained,
   weekly-periodic and over-dispersed. A model that ignores this fits noise.
   The state-of-the-art move is an explicit **observation model** plus **full
   Bayesian uncertainty**.

So the SOTA model is *less* mechanistic in its contact structure but *more*
honest about transmission change and observation. That trade is exactly why it
forecasts better.

---

## 2. The model

### 2.1 Latent epidemic: the renewal equation

Instead of compartments, infections evolve through the **renewal equation**:

```
I_t = R_t · Σ_{s≥1} g_s · I_{t−s}
```

- `I_t` — latent (true) new infections on day *t*;
- `g_s` — the **generation-interval** distribution (probability that a secondary
  case is infected *s* days after the primary). Discretised Gamma, mean 5.2 d,
  sd 3.8 d (Ganyani et al. 2020; see `PARAMETERS.md`);
- `R_t` — the **effective reproduction number** on day *t*.

This is the same epidemic engine that underlies SIR/SEIR (in the right limit
they coincide), but it makes `R_t` — the quantity we care about — explicit.

### 2.2 R_t as a latent stochastic process

`R_t` is **not** a parameter to tune; it is a latent time series we infer:

```
log R_t = log R_0 + σ_rw · RW_t,     RW ~ GaussianRandomWalk
```

The random walk lets `R_t` move smoothly over time, with `σ_rw` (itself
inferred) controlling how fast. This is what lets the model discover a lockdown
or a variant takeover **from the data alone**, with no hand-coded `beta(t)`.

### 2.3 Observation model: from infections to reported cases

Latent infections are not observed; reported cases are. The map is:

```
E[cases_t] = ρ · ( Σ_{d≥0} π_d · I_{t−d} ) · weekday_effect_t
cases_t   ~ NegativeBinomial( mean = E[cases_t], dispersion = φ )
```

- `π_d` — **infection-to-report delay** (discretised log-normal, mean 9 d:
  incubation + test/report lag);
- `ρ` — **ascertainment** (fraction of infections ever reported). In case-only
  data only `ρ·I` is identified, so `ρ` gets an informative prior
  (Beta(6,14): mean 0.30) reflecting external seroprevalence evidence;
- **weekday effect** — a multiplicative day-of-week factor (sum-to-zero on the
  log scale) for the very real weekend reporting dip;
- **Negative-Binomial** likelihood — captures over-dispersion (variance ≫ mean)
  that a Poisson would underestimate, so credible intervals are honest.

### 2.4 Inference

Everything is fit jointly with the **No-U-Turn Sampler (NUTS)**, a
self-tuning, gradient-based Hamiltonian Monte Carlo, via **NumPyro/JAX**. The
renewal recursion is implemented with `jax.lax.scan` so the whole model is
differentiable and fast. Output is a full **posterior** over `R_t`, latent
infections, and all parameters.

### 2.5 Probabilistic forecasting (dampened R_t)

To forecast *H* days ahead, `R_t` is projected forward and the renewal equation
propagates infections, producing a **posterior-predictive** case forecast.

Crucially, the forecast `R_t` does **not** follow a pure random walk (whose
variance grows without bound, giving absurdly wide intervals — e.g. a final-day
`R_t` interval of 0.1–8). Instead the forecast weeks follow a **dampened AR(1)**
process that reverts toward the last in-sample `R_t` level:

```
log R_w = anchor + d · (log R_{w−1} − anchor) + σ · ε_w,   0 < d < 1
```

With damping `d < 1` the innovation variance **saturates** instead of growing,
so forecast credible intervals stay realistic while still widening with the
horizon. This matches the dampened-RW projection used by EpiNow2.

### 2.6 Seeding

The first weeks of any renewal model are sensitive to how the epidemic is
*seeded* before the data window opens. The model seeds the initial
generation-interval window at a single inferred level. (A more flexible
exponential seed — inferring a per-day growth across the window — was tried and
**reverted**: it created a degeneracy in which a decaying seed could explain the
data *instead of* transmission, biasing R_t low and inflating its uncertainty.
This is a good example of why every change is re-validated against a known
synthetic truth before being kept.) The short start-up burn-in the single-level
seed leaves is excluded from plots and interpretation.

---

## 3. Validation (this is important)

A model you cannot validate is not science. There are two layers of validation.

**(a) Recovering a known R_t.** `sota_run.py --synthetic`
**simulates an epidemic from a known, changing R_t** (a rise to 2.4, a lockdown
decline below 1, then a partial rebound to 1.1), adds realistic reporting noise,
then fits the model **blind** and checks recovery.

**(b) Proper scoring & calibration (`evaluate.py`).** Qualitative recovery is
not enough. `evaluate.py` runs **backtests** — fit on a training window, forecast
a held-out horizon, and score the forecast with the **Weighted Interval Score
(WIS)**, the COVID-19 Forecast Hub's headline metric — plus **interval coverage**
(do the 50%/90% intervals contain truth 50%/90% of the time?) and a **PIT
calibration histogram** (flat ⇒ calibrated). Run `python evaluate.py` for a
self-contained rolling backtest; it writes `sota_calibration.png`. This turns
"looks right" into a measured, falsifiable claim.

Result (see `sota_Rt_synthetic.png`): the posterior median R_t tracks the true
trajectory closely, the true curve stays within the credible band throughout,
and forecast uncertainty fans out beyond the data. This is the right way to earn
trust in a forecasting model before pointing it at real data.

### Real data: Italy, second wave (autumn 2020)

Fitting JHU CSSE reported cases for Italy from 2020-09-15
(`sota_Rt_italy.png`, `sota_forecast_italy.png`) recovers the textbook second
wave with **zero divergences** and good convergence (R̂ ≈ 1.0):

- R_t crosses 1 in mid-October 2020 as the second wave takes off;
- it peaks around **1.4** in late October;
- it falls back **below 1 by mid-November**, exactly when Italy's tiered
  regional restrictions (the late-October/November DPCM measures) took effect —
  the model recovers the policy response purely from case data;
- the 14-day forecast then fans out, honestly reflecting that a volatile R_t is
  hard to extrapolate.

> **Burn-in.** The first ~14 days (one generation interval) are a seeding
> region: the renewal window has not yet "forgotten" the initial seed, so R_t
> there is an artefact and is excluded from the plots and from interpretation —
> standard practice for renewal models. The model still uses those days
> internally to initialise.

---

## 4. Running it

```bash
pip install numpyro jax arviz numpy scipy pandas matplotlib

# 1) Self-test: recover a known R_t from synthetic data
python sota_run.py --synthetic

# 2) Real data: fit a country window from the JHU CSSE archive
python sota_run.py --country Italy --start 2020-08-15 --days 120 --horizon 14

# 3) Backtest + proper scoring (WIS / coverage / PIT calibration)
python evaluate.py

# 4) Save the posterior once, then re-plot without re-running MCMC
python sota_run.py --synthetic --save post.npz
python sota_run.py --synthetic --load post.npz

# 5) Run the fast unit tests
pytest -q
```

Outputs (suffixed `_synthetic` or `_<country>`):

- **`sota_Rt_*.png`** — the time-varying R_t with 50%/90% credible intervals and
  the R_t = 1 epidemic threshold;
- **`sota_forecast_*.png`** — the fit and the probabilistic forecast against
  observed cases;
- a console summary: ascertainment, current R_t with credible interval, and
  `P(R_t > 1)` (i.e. the posterior probability the epidemic is growing).

---

## 5. How this compares to the SIR/SEIRD models in this repo

| Capability | SIR (`model.py`) | SEIRD (`models.py`) | **SOTA renewal (`sota_model.py`)** |
|---|---|---|---|
| Transmission over time | fixed β | fixed β (β(t) optional) | **inferred latent R_t(t)** |
| Fit to real data | no | no | **yes (NUTS)** |
| Reporting delay & under-ascertainment | no | no | **yes** |
| Day-of-week / over-dispersion | no | no | **yes (NegBin + weekday)** |
| Uncertainty quantification | none | none | **full Bayesian posterior** |
| Probabilistic forecast | no | no | **yes, with growing CrI** |
| Validated by recovering known truth | n/a | n/a | **yes (synthetic self-test)** |
| Best use | teaching | scenario baseline | **nowcasting & forecasting** |

The SIR/SEIRD models answer *"what would an epidemic with these assumptions look
like?"*. The SOTA model answers the operational question *"given the messy data
we actually have, what is transmission doing right now and what happens next —
and how sure are we?"*

---

## 6. Limitations & honest caveats

- **Scale confounding.** From case data alone, ascertainment `ρ` and the
  absolute number of infections are confounded; we resolve it with an
  informative prior. `R_t` and the case forecast are unaffected.
- **Fixed delay/generation distributions.** These are taken from the literature
  (`PARAMETERS.md`) and held fixed; a fuller treatment would propagate their
  uncertainty too.
- **Right-truncation.** The most recent days are based on incomplete reporting;
  operational tools add an explicit truncation model. The random walk partially
  absorbs this but recent `R_t` should be read with care.
- **Short-horizon only.** Like all such models, it is a 1–3 week forecaster, not
  a long-range predictor — transmission can change for reasons no model sees.
- **One signal.** It fits a single stream (cases). The strongest operational
  setups jointly fit cases, hospitalisations and deaths, and combine multiple
  models into a calibrated **ensemble** — empirically the best performer.

---

## References

- Cori A, et al. "A new framework and software to estimate time-varying
  reproduction numbers during epidemics." *Am J Epidemiol*, 2013.
- Abbott S, et al. "Estimating the time-varying reproduction number of
  SARS-CoV-2 (EpiNow2)." *Wellcome Open Res*, 2020.
- Flaxman S, et al. "Estimating the effects of non-pharmaceutical interventions
  on COVID-19 in Europe." *Nature*, 2020. https://doi.org/10.1038/s41586-020-2405-7
- Ganyani T, et al. "Estimating the generation interval for COVID-19."
  *Eurosurveillance*, 2020.
- Cramer EY, et al. "Evaluation of individual and ensemble probabilistic
  forecasts of COVID-19 mortality in the US." *PNAS*, 2022.
- Bhatt S, et al. "Semi-mechanistic Bayesian modelling of COVID-19." *J R Stat
  Soc A*, 2023.
