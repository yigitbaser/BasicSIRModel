"""
Hierarchical (partially-pooled) multi-region renewal model.

Fitting each region in isolation wastes information: regions share structure
(how volatile R_t tends to be, the weekly reporting rhythm). A hierarchical
model fits all regions jointly, with **shared hyperparameters drawn from common
hyperpriors** — genuine partial pooling. Data-rich regions then inform data-poor
ones, and the shared quantities are estimated more precisely.

What is pooled here:
  * `sigma_rw`  — the week-to-week R_t volatility (one shared value);
  * `dow`       — the day-of-week reporting effect (one shared pattern);
while each region keeps its **own** seed, initial R_t, ascertainment, R_t
trajectory and over-dispersion. This is the standard "partial pooling on the
hyperparameters" design.

Run ``python hierarchical.py`` to fit 3 synthetic regions jointly and recover
their distinct R_t trajectories.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

from sota_model import EpiConfig, discretise_gamma, discretise_lognormal


def hierarchical_model(cases, cfg: EpiConfig, gen_pmf, rep_pmf):
    """Partially-pooled renewal model over R regions.

    cases : array (R, T) of daily reported cases (no forecast horizon here —
            this module focuses on the pooled R_t estimation).
    """
    R, T = cases.shape
    n_steps = T
    L = len(gen_pmf)
    gen_flip = jnp.asarray(gen_pmf)[::-1]
    rep = jnp.asarray(rep_pmf)
    n_weeks = n_steps // 7 + 2
    week_nodes = jnp.arange(n_weeks) * 7.0
    day_idx = jnp.arange(n_steps).astype(jnp.float32)

    # --- Shared hyperparameters (partial pooling) ----------------------------
    sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(0.2))   # shared R_t volatility
    dow_raw = numpyro.sample("dow", dist.Normal(jnp.zeros(7), cfg.dow_prior_sd))
    dow = dow_raw - jnp.mean(dow_raw)

    # --- Per-region parameters ----------------------------------------------
    early = jnp.maximum(jnp.mean(cases[:, :7], axis=1), 1.0)       # (R,)
    with numpyro.plate("region", R, dim=-1):
        log_I0 = numpyro.sample("log_I0", dist.Normal(jnp.log(early), 1.5))
        log_R0 = numpyro.sample("log_R0", dist.Normal(jnp.log(1.0), 0.5))
        logit_rho = numpyro.sample("logit_rho", dist.Normal(_logit(0.3), 0.5))
        phi = numpyro.sample("phi", dist.Exponential(0.2))

    rho = jax.nn.sigmoid(logit_rho)                               # (R,)

    # Per-region weekly random walk, sharing the pooled volatility sigma_rw.
    eps = numpyro.sample("rw", dist.Normal(jnp.zeros((R, n_weeks)), 1.0))
    log_Rt_weekly = log_R0[:, None] + sigma_rw * jnp.cumsum(eps, axis=1)   # (R, n_weeks)
    # Interpolate each region's weekly grid to daily.
    log_Rt = jax.vmap(lambda row: jnp.interp(day_idx, week_nodes, row))(log_Rt_weekly)
    log_Rt = jnp.clip(log_Rt, jnp.log(0.1), jnp.log(6.0))
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))            # (R, n_steps)

    # --- Vectorised renewal recursion over regions ---------------------------
    init_window = jnp.exp(log_I0)[:, None] * jnp.ones((R, L))     # (R, L)

    def step(window, Rt_col):                                    # window (R,L), Rt_col (R,)
        infness = jnp.sum(window * gen_flip[None, :], axis=1)    # (R,)
        I_t = jnp.clip(Rt_col * infness, 1e-6, 1e12)
        new_window = jnp.concatenate([window[:, 1:], I_t[:, None]], axis=1)
        return new_window, I_t

    _, I_scanned = jax.lax.scan(step, init_window, Rt.T)         # (n_steps, R)
    infections = I_scanned.T                                     # (R, n_steps)

    # --- Observation model (per region) --------------------------------------
    conv = jax.vmap(lambda inf: jnp.convolve(inf, rep)[:n_steps])(infections)  # (R, n_steps)
    weekday = jnp.arange(n_steps) % 7
    expected = jnp.clip(rho[:, None] * conv * jnp.exp(dow[weekday])[None, :], 1e-3, None)
    numpyro.deterministic("expected_cases", expected)

    with numpyro.plate("region_obs", R, dim=-2):
        with numpyro.plate("time_obs", T, dim=-1):
            numpyro.sample(
                "obs",
                dist.GammaPoisson(concentration=phi[:, None], rate=phi[:, None] / expected),
                obs=jnp.asarray(cases),
            )


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def fit_hierarchical(cases, cfg: EpiConfig, num_warmup=500, num_samples=500, chains=2, seed=0):
    gen_pmf = discretise_gamma(cfg.gen_mean, cfg.gen_sd, cfg.gen_max)
    rep_pmf = discretise_lognormal(cfg.rep_mean, cfg.rep_sd, cfg.rep_max)
    kernel = NUTS(hierarchical_model, target_accept_prob=0.95, init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), cases=jnp.asarray(cases, dtype=jnp.float32),
             cfg=cfg, gen_pmf=gen_pmf, rep_pmf=rep_pmf)
    return mcmc


# ---------------------------------------------------------------------------
# Synthetic multi-region data + demo.
# ---------------------------------------------------------------------------
def make_regions(cfg, T=90, seed=2):
    """Three regions with distinct R_t stories (different timing/levels)."""
    rng = np.random.default_rng(seed)
    gen = discretise_gamma(cfg.gen_mean, cfg.gen_sd, cfg.gen_max)
    rep = discretise_lognormal(cfg.rep_mean, cfg.rep_sd, cfg.rep_max)
    L = len(gen); gen_flip = gen[::-1]
    t = np.arange(T)
    stories = [
        np.where(t < 40, 1.6, 0.75),                              # early surge then control
        np.where(t < 25, 0.9, np.where(t < 60, 1.5, 0.8)),        # delayed wave
        1.2 - 0.4 * np.sin(t / 15.0),                             # oscillating
    ]
    rhos = [0.3, 0.2, 0.4]
    cases = np.zeros((3, T)); Rts = np.zeros((3, T))
    for r, (Rt, rho) in enumerate(zip(stories, rhos)):
        Rts[r] = Rt
        window = np.full(L, 40.0); infections = np.zeros(T)
        for i in range(T):
            I_t = Rt[i] * (window @ gen_flip)
            infections[i] = I_t
            window = np.concatenate([window[1:], [I_t]])
        conv = np.convolve(infections, rep)[:T]
        cases[r] = rng.negative_binomial(10.0, 10.0 / (10.0 + rho * conv)).astype(float)
    return cases, Rts


def plot_regions(mcmc, cases, true_Rt, path="hierarchical_Rt.png", burn_in=14):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Rt = np.asarray(mcmc.get_samples()["Rt"])                    # (draws, R, T)
    R = Rt.shape[1]; T = Rt.shape[2]; days = np.arange(T)
    fig, axes = plt.subplots(1, R, figsize=(5 * R, 4.5), sharey=True)
    for r in range(R):
        ax = axes[r]
        lo, mid, hi = (np.quantile(Rt[:, r], q, axis=0) for q in (0.05, 0.5, 0.95))
        ax.fill_between(days, lo, hi, color="purple", alpha=0.2, label="90% CrI")
        ax.plot(days, mid, color="purple", lw=2, label="median $R_t$")
        ax.plot(days, true_Rt[r], "k--", lw=2, label="true $R_t$")
        ax.axhline(1.0, color="red", ls="--", lw=1)
        ax.set_xlim(burn_in, T - 1); ax.set_ylim(0, 2.5)
        ax.set_title(f"Region {r + 1}"); ax.set_xlabel("Day"); ax.grid(alpha=0.3)
        if r == 0:
            ax.set_ylabel("$R_t$"); ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Hierarchical multi-region renewal model — pooled hyperparameters, per-region $R_t$")
    fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"Wrote {path}")


def main():
    cfg = EpiConfig()
    print("Generating 3 synthetic regions with distinct R_t stories...")
    cases, true_Rt = make_regions(cfg, T=90)
    print("Fitting the hierarchical (partially-pooled) model jointly...")
    mcmc = fit_hierarchical(cases, cfg, num_warmup=500, num_samples=500)
    div = int(mcmc.get_extra_fields()["diverging"].sum())
    print(f"  divergences: {div}")
    Rt = np.asarray(mcmc.get_samples()["Rt"])
    for r in range(3):
        med = np.median(Rt[:, r], axis=0)
        rmse = np.sqrt(np.mean((med[14:] - true_Rt[r][14:]) ** 2))
        print(f"  region {r + 1}: R_t RMSE (days 14+) = {rmse:.3f}")
    plot_regions(mcmc, cases, true_Rt)


if __name__ == "__main__":
    main()
