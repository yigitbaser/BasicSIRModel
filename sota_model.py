"""
State-of-the-art COVID-19 model: a Bayesian semi-mechanistic renewal model.

This is a from-scratch implementation of the model family that defines the
current state of the art for COVID-19 nowcasting and short-term forecasting —
the approach behind tools such as EpiNow2, the Imperial College report
(Flaxman et al., *Nature* 2020) and the operational US/EU COVID-19 Forecast
Hubs. It is deliberately *not* another compartmental (SIR/SEIR) model.

Why this is state of the art
----------------------------
1. **Semi-mechanistic renewal process.** Instead of fixing a mechanistic
   contact structure, latent infections evolve through the renewal equation

        I_t = R_t * sum_{s>=1} g_s * I_{t-s}

   where g is the generation-interval distribution. This is the same epidemic
   engine as a compartmental model but makes the *time-varying* reproduction
   number R_t the object we infer, rather than a fixed beta.

2. **R_t is a latent stochastic process**, not a constant. We model log R_t as a
   Gaussian random walk, so the model *learns* how transmission changed over
   time (lockdowns, variants, behaviour) directly from data — no hand-tuned
   beta(t).

3. **A realistic observation model.** Latent infections are mapped to *reported*
   cases through (a) an infection-to-report delay distribution, (b) an
   ascertainment fraction, (c) a day-of-week reporting effect, and (d) a
   Negative-Binomial likelihood that accounts for over-dispersion. Real
   surveillance data is delayed, under-ascertained, weekly-periodic and noisy;
   this model says so explicitly.

4. **Full Bayesian uncertainty.** Inference is done with the No-U-Turn Sampler
   (NUTS, gradient-based HMC) in NumPyro/JAX. Every output — R_t, the infection
   nowcast, and the forecast — comes with posterior credible intervals, which is
   the entire point of a decision-grade epidemic model.

5. **Probabilistic forecasting.** The latent R_t random walk is projected
   forward, the renewal equation propagates infections, and the observation
   model generates a posterior-predictive forecast with growing uncertainty.

Run ``python sota_model.py`` to fit synthetic data (self-test) and, if network
is available, real data for a chosen country, producing figures + a summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_host_device_count(2)


# ---------------------------------------------------------------------------
# Epidemiological input distributions (discretised), from PARAMETERS.md.
# ---------------------------------------------------------------------------
def discretise_gamma(mean: float, sd: float, length: int) -> np.ndarray:
    """Discretise a Gamma(mean, sd) onto days 1..length, renormalised to sum 1.

    Used for the generation interval (no mass on day 0 — you cannot infect
    someone the same instant you are infected).
    """
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    # Probability mass in each daily bin via the CDF.
    edges = np.arange(0, length + 1)
    cdf = gamma_dist.cdf(edges, a=shape, scale=scale)
    pmf = np.diff(cdf)
    pmf = pmf / pmf.sum()
    return pmf  # index 0 == lag 1 day


def discretise_lognormal(mean: float, sd: float, length: int) -> np.ndarray:
    """Discretise a log-normal delay (e.g. infection->report) onto days 0..length-1."""
    # Convert mean/sd of the variable to log-space parameters.
    sigma2 = np.log(1 + (sd / mean) ** 2)
    mu = np.log(mean) - sigma2 / 2
    sigma = np.sqrt(sigma2)
    edges = np.arange(0, length + 1)
    cdf = lognorm.cdf(edges, s=sigma, scale=np.exp(mu))
    pmf = np.diff(cdf)
    pmf = pmf / pmf.sum()
    return pmf  # index 0 == same-day (lag 0)


@dataclass
class EpiConfig:
    """Fixed epidemiological inputs (ancestral SARS-CoV-2; see PARAMETERS.md)."""

    gen_mean: float = 5.2     # generation interval mean (days), Ganyani 2020
    gen_sd: float = 3.8       # generation interval sd (days)
    gen_max: int = 21         # truncation length for the generation interval
    rep_mean: float = 9.0     # infection -> case-report delay mean (incubation+reporting)
    rep_sd: float = 4.5       # infection -> case-report delay sd
    rep_max: int = 30         # truncation length for the reporting delay
    seed_days: int = 7        # number of days of seeded initial infections
    forecast_damping: float = 0.85  # AR(1) damping of R_t over the forecast horizon
    dow_prior_sd: float = 0.15      # prior sd of the day-of-week (log) effect


# ---------------------------------------------------------------------------
# The probabilistic model.
# ---------------------------------------------------------------------------
def renewal_model(cases, cfg: EpiConfig, horizon: int = 0, gen_pmf=None, rep_pmf=None):
    """NumPyro model: latent renewal process + reporting observation model.

    Parameters
    ----------
    cases : array (T,) or None
        Observed daily reported cases. None => prior predictive only.
    cfg : EpiConfig
        Fixed epidemiological inputs.
    horizon : int
        Number of future days to forecast beyond the observed window.
    gen_pmf, rep_pmf : arrays
        Pre-computed generation-interval / reporting-delay pmfs.
    """
    T = len(cases) if cases is not None else 60
    n_steps = T + horizon                      # total days of latent infections to model
    L = len(gen_pmf)
    seed_days = cfg.seed_days

    gen = jnp.asarray(gen_pmf)
    gen_flip = gen[::-1]                        # align with a chronological window
    rep = jnp.asarray(rep_pmf)

    # --- Priors ---------------------------------------------------------------
    # Initial daily infections (seed), on a log scale, centred on the first week
    # of observed cases. The whole length-L generation window is seeded at this
    # single level. (An earlier attempt to infer a free per-day seed *growth*
    # over the window was reverted: it created a degeneracy in which a decaying
    # seed could explain the data instead of transmission, biasing R_t and
    # inflating its uncertainty. The single-level seed keeps R_t identified; the
    # short start-up burn-in it leaves is excluded from plots, see sota_run.py.)
    if cases is not None:
        early = float(np.maximum(np.mean(np.asarray(cases)[:7]), 1.0))
    else:
        early = 100.0
    log_I0 = numpyro.sample("log_I0", dist.Normal(jnp.log(early), 1.5))
    init_window = jnp.exp(log_I0) * jnp.ones(L)

    # Time-varying reproduction number on a WEEKLY grid (smoothly interpolated to
    # daily). The weekly grid avoids the funnel geometry that makes daily random
    # walks diverge under NUTS, matching the smoothing used by EpiNow2.
    #
    # In-sample weeks follow a log Gaussian random walk. Forecast weeks instead
    # follow a *dampened* AR(1) process that reverts toward the last in-sample
    # R_t level: log R_w = anchor + d * (log R_{w-1} - anchor) + sigma * eps.
    # With damping d < 1 the innovation variance saturates instead of growing
    # without bound, so forecast credible intervals stay realistic rather than
    # fanning out to absurd values (the main weakness of a pure random walk).
    n_weeks = n_steps // 7 + 2
    n_weeks_obs = T // 7 + 1                       # weeks covering the observed window
    log_R0 = numpyro.sample("log_R0", dist.Normal(jnp.log(1.0), 0.5))
    sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(0.2))   # week-to-week R_t volatility
    eps = numpyro.sample("rw", dist.Normal(jnp.zeros(n_weeks), 1.0))
    d = cfg.forecast_damping

    def week_step(carry, inputs):
        prev, w = carry
        eps_w, anchor = inputs
        in_sample = w < n_weeks_obs
        rw_val = prev + sigma_rw * eps_w                       # random walk (in-sample)
        ar_val = anchor + d * (prev - anchor) + sigma_rw * eps_w  # dampened (forecast)
        val = jnp.where(in_sample, rw_val, ar_val)
        return (val, w + 1), val

    # The AR(1) anchor is the last in-sample weekly level; compute the in-sample
    # path first (cumulative sum), then run the scan for all weeks.
    insample_cumsum = log_R0 + sigma_rw * jnp.cumsum(eps[:n_weeks_obs])
    anchor = insample_cumsum[-1]
    (_, _), log_Rt_weekly = jax.lax.scan(
        week_step,
        (log_R0, jnp.array(0)),
        (eps, jnp.broadcast_to(anchor, (n_weeks,))),
    )
    week_nodes = jnp.arange(n_weeks) * 7.0
    log_Rt = jnp.interp(jnp.arange(n_steps).astype(jnp.float32), week_nodes, log_Rt_weekly)
    # Bound R_t to a plausible epidemic range so the renewal recursion can never
    # overflow to infinity (which would produce NaN gradients / divergences).
    log_Rt = jnp.clip(log_Rt, jnp.log(0.1), jnp.log(8.0))
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))

    # Ascertainment (fraction of infections that become reported cases).
    # In case-only data only (rho * infections) is identified, so we use an
    # informative prior — Beta(6, 14): mean 0.30, sd ~0.10 — reflecting external
    # seroprevalence evidence that ~1 in 3 infections were reported. This
    # regularises the otherwise-confounded infection scale.
    rho = numpyro.sample("rho", dist.Beta(6.0, 14.0))

    # Day-of-week reporting multiplier (sum-to-zero on the log scale).
    dow_raw = numpyro.sample("dow", dist.Normal(jnp.zeros(7), cfg.dow_prior_sd))
    dow = dow_raw - jnp.mean(dow_raw)

    # Negative-Binomial over-dispersion.
    phi = numpyro.sample("phi", dist.Exponential(0.2))

    # --- Latent renewal process (deterministic recursion via scan) -----------
    def step(window, Rt_t):
        # window holds the most recent L infections in chronological order.
        infectiousness = jnp.dot(window, gen_flip)
        # Clip to keep the recursion finite even for extreme proposed R_t paths.
        I_t = jnp.clip(Rt_t * infectiousness, 1e-6, 1e12)
        new_window = jnp.concatenate([window[1:], I_t[None]])
        return new_window, I_t

    # `init_window` (the length-L exponential seed) was built with the priors above.
    _, I_scanned = jax.lax.scan(step, init_window, Rt)
    infections = numpyro.deterministic("infections", I_scanned)  # length n_steps

    # --- Observation model ----------------------------------------------------
    # Expected reported cases = ascertainment * (infections convolved with delay).
    # Full convolution then crop to the modelled days.
    conv = jnp.convolve(infections, rep)[:n_steps]
    weekday = jnp.arange(n_steps) % 7
    expected = rho * conv * jnp.exp(dow[weekday])
    expected = jnp.clip(expected, 1e-3, None)
    numpyro.deterministic("expected_cases", expected)

    # Negative-Binomial likelihood on the observed window only.
    if cases is not None:
        with numpyro.plate("obs_time", T):
            numpyro.sample(
                "obs",
                dist.GammaPoisson(concentration=phi, rate=phi / expected[:T]),
                obs=jnp.asarray(cases),
            )
    # Posterior-predictive cases (incl. the forecast horizon) are generated
    # afterwards in numpy from `expected_cases` + `phi`, to keep the model free
    # of discrete latent sites that gradient-based NUTS cannot sample.


# ---------------------------------------------------------------------------
# Fitting & forecasting orchestration.
# ---------------------------------------------------------------------------
def fit(cases, cfg: EpiConfig, horizon: int = 14, num_warmup=600, num_samples=600, chains=2, seed=0):
    gen_pmf = discretise_gamma(cfg.gen_mean, cfg.gen_sd, cfg.gen_max)
    rep_pmf = discretise_lognormal(cfg.rep_mean, cfg.rep_sd, cfg.rep_max)

    kernel = NUTS(renewal_model, target_accept_prob=0.95, max_tree_depth=12)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=chains,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        cases=jnp.asarray(cases, dtype=jnp.float32),
        cfg=cfg,
        horizon=horizon,
        gen_pmf=gen_pmf,
        rep_pmf=rep_pmf,
    )
    return mcmc, gen_pmf, rep_pmf


def summarise(mcmc):
    """Print the latest reproduction number from a fitted MCMC object."""
    return summarise_samples(mcmc.get_samples(group_by_chain=False))


def summarise_samples(samples):
    """Print the latest reproduction number from a posterior-samples dict."""
    Rt = np.asarray(samples["Rt"])  # (draws, n_steps)
    rho = np.asarray(samples["rho"])
    Rt_now = Rt[:, -1]  # last modelled day (end of forecast horizon)
    print("\n=== Posterior summary ===")
    print(f"  Ascertainment rho....... {np.median(rho):.2%}  "
          f"(90% CrI {np.quantile(rho,0.05):.2%}–{np.quantile(rho,0.95):.2%})")
    print(f"  Final-day R_t........... {np.median(Rt_now):.2f}  "
          f"(90% CrI {np.quantile(Rt_now,0.05):.2f}–{np.quantile(Rt_now,0.95):.2f})")
    prob_above_1 = float(np.mean(Rt_now > 1.0))
    print(f"  P(R_t > 1) at horizon... {prob_above_1:.0%}  "
          f"=> epidemic {'GROWING' if prob_above_1 > 0.5 else 'SHRINKING'}")
    return samples


def credible_band(arr, lo=0.05, hi=0.95):
    return np.quantile(arr, lo, axis=0), np.median(arr, axis=0), np.quantile(arr, hi, axis=0)


def save_samples(samples, path):
    """Persist a posterior-samples dict to a compressed .npz so you can re-plot
    or re-score without re-running the (minutes-long) MCMC."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in samples.items()})


def load_samples(path):
    """Load a posterior-samples dict previously written by `save_samples`."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def posterior_predictive_cases(samples, seed=0):
    """Draw posterior-predictive reported cases (fit + forecast) in numpy.

    For each posterior draw we sample Negative-Binomial observation noise around
    that draw's `expected_cases`, using its over-dispersion `phi`. The spread of
    the result is the full predictive uncertainty shown on the forecast plot.
    """
    rng = np.random.default_rng(seed)
    expected = np.clip(np.asarray(samples["expected_cases"]), 1e-3, 1e9)  # (draws, n_steps)
    phi = np.asarray(samples["phi"])[:, None]                 # (draws, 1)
    # Draw from the Gamma-Poisson (Negative-Binomial) mixture directly: this is
    # numerically stable for the large case counts where the (n, p) form of
    # numpy's negative_binomial underflows.
    rate = rng.gamma(shape=phi, scale=expected / phi)
    rate = np.clip(rate, 0.0, 1e12)
    return rng.poisson(rate)
