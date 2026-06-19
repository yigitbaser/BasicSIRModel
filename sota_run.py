"""
Driver for the state-of-the-art Bayesian renewal model (``sota_model.py``).

Two modes:

    python sota_run.py --synthetic       # self-test: recover a known R_t path
    python sota_run.py --country Italy --start 2020-08-01 --days 120

The synthetic mode is a *validation*: we simulate data from a known, changing
R_t (mimicking a lockdown), fit the model blind, and check it recovers both the
R_t trajectory and the true infection curve — with calibrated uncertainty.

Outputs: sota_Rt.png, sota_forecast.png and a console summary.
"""

from __future__ import annotations

import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

from sota_model import (
    EpiConfig,
    discretise_gamma,
    discretise_lognormal,
    fit,
    summarise_samples,
    credible_band,
    posterior_predictive_cases,
    save_samples,
    load_samples,
)


# ---------------------------------------------------------------------------
# Synthetic ground-truth generator (for validation).
# ---------------------------------------------------------------------------
def make_synthetic(cfg: EpiConfig, T=120, seed=1):
    rng = np.random.default_rng(seed)
    gen = discretise_gamma(cfg.gen_mean, cfg.gen_sd, cfg.gen_max)
    rep = discretise_lognormal(cfg.rep_mean, cfg.rep_sd, cfg.rep_max)
    death_delay = discretise_lognormal(cfg.death_mean, cfg.death_sd, cfg.death_max)

    # A realistic R_t story: ~2.4 early, sharp lockdown drop below 1, partial
    # relaxation back toward 1.1.
    t = np.arange(T)
    Rt = np.piecewise(
        t.astype(float),
        [t < 25, (t >= 25) & (t < 70), t >= 70],
        [2.4, lambda x: 2.4 - (2.4 - 0.7) * (x - 25) / 45.0, 1.1],
    )

    L = len(gen)
    gen_flip = gen[::-1]
    infections = np.zeros(T)
    seed_level = 50.0
    window = np.full(L, seed_level)
    for i in range(T):
        infectiousness = window @ gen_flip
        I_t = Rt[i] * infectiousness
        infections[i] = I_t
        window = np.concatenate([window[1:], [I_t]])

    # Right-truncation: emulate incomplete reporting of the most recent days, so
    # the synthetic data matches what real surveillance looks like (and exercises
    # the model's reporting-completeness correction). The fraction reported by the
    # end of the window is the delay CDF at (days remaining).
    dist_end = (T - 1 - t)
    rep_cdf = np.cumsum(rep)
    death_cdf = np.cumsum(death_delay)
    comp_c = rep_cdf[np.clip(dist_end, 0, len(rep) - 1)]
    comp_d = death_cdf[np.clip(dist_end, 0, len(death_delay) - 1)]

    # Cases: ascertainment ramps UP over time (testing scaled up), to exercise the
    # time-varying-rho machinery.
    rho_true = 0.15 + 0.25 * (t / T)            # 15% -> 40%
    conv = np.convolve(infections, rep)[:T]
    dow = np.array([0.0, 0.05, 0.05, 0.0, -0.05, -0.15, -0.1])  # weekend dip
    expected = rho_true * conv * np.exp(dow[t % 7]) * comp_c
    phi = 10.0
    cases = rng.negative_binomial(phi, phi / (phi + expected)).astype(float)

    # Deaths: infection-fatality ratio * (infections convolved with death delay).
    ifr_true = 0.0068
    conv_d = np.convolve(infections, death_delay)[:T]
    expected_d = ifr_true * conv_d * comp_d
    phi_d = 15.0
    deaths = rng.negative_binomial(phi_d, phi_d / (phi_d + expected_d)).astype(float)
    return cases, Rt, infections, rho_true, deaths


# ---------------------------------------------------------------------------
# Real data loader (JHU CSSE archived time series; robust, moderate size).
# ---------------------------------------------------------------------------
JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)
JHU_CONFIRMED = JHU_BASE + "time_series_covid19_confirmed_global.csv"
JHU_DEATHS = JHU_BASE + "time_series_covid19_deaths_global.csv"


def _daily_from_jhu(url, country, start, days):
    with urllib.request.urlopen(url, timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    sub = df[df["Country/Region"] == country]
    if sub.empty:
        raise ValueError(f"Country {country!r} not found in JHU data.")
    cum = sub.iloc[:, 4:].sum(axis=0)
    cum.index = pd.to_datetime(cum.index, format="%m/%d/%y")
    daily = cum.diff().clip(lower=0).fillna(0)
    return daily[daily.index >= pd.Timestamp(start)].iloc[:days]


def load_country(country: str, start: str, days: int):
    """Download JHU confirmed + deaths for a country -> daily cases, daily deaths."""
    cases = _daily_from_jhu(JHU_CONFIRMED, country, start, days)
    deaths = _daily_from_jhu(JHU_DEATHS, country, start, days)
    return cases, deaths


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def plot_results(samples, cases, horizon, title, true_Rt=None, true_inf=None,
                 burn_in=14, tag="", deaths=None):
    """Plot R_t and the forecast.

    The first `burn_in` days are a seeding/burn-in region: the renewal window
    needs roughly one generation interval to "forget" the initial seed, so R_t
    and the fit there are artefacts and are excluded from the view (standard
    practice for renewal models). The model still uses them internally.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Rt = np.asarray(samples["Rt"])
    pred = posterior_predictive_cases(samples)
    inf = np.asarray(samples["infections"])
    T = len(cases)
    n_steps = Rt.shape[1]
    days = np.arange(n_steps)
    x0 = burn_in  # first displayed day
    ymax = float(np.max(cases)) * 2.2  # cap forecast y-axis to the data scale

    # --- R_t figure ----------------------------------------------------------
    lo, mid, hi = credible_band(Rt)
    lo50, _, hi50 = credible_band(Rt, 0.25, 0.75)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(days, lo, hi, color="purple", alpha=0.15, label="90% CrI")
    ax.fill_between(days, lo50, hi50, color="purple", alpha=0.25, label="50% CrI")
    ax.plot(days, mid, color="purple", lw=2, label="Posterior median $R_t$")
    if true_Rt is not None:
        ax.plot(np.arange(len(true_Rt)), true_Rt, "k--", lw=2, label="True $R_t$")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="$R_t = 1$")
    ax.axvline(T - 1, color="grey", ls=":", lw=1)
    ax.set_xlim(x0, n_steps - 1)
    rt_top = float(np.quantile(Rt[:, x0:], 0.95)) * 1.15
    ax.set_ylim(0, max(2.6, rt_top))
    ax.text(T - 1, ax.get_ylim()[1] * 0.95, " forecast →", color="grey", fontsize=9)
    ax.set_xlabel("Day")
    ax.set_ylabel("Effective reproduction number $R_t$")
    ax.set_title(f"{title} — time-varying $R_t$ with Bayesian credible intervals")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"sota_Rt{tag}.png", dpi=120)
    plt.close(fig)

    # --- Forecast figure -----------------------------------------------------
    plo, pmid, phi_ = credible_band(pred)
    plo50, _, phi50 = credible_band(pred, 0.25, 0.75)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(days, plo, phi_, color="C0", alpha=0.15, label="90% CrI")
    ax.fill_between(days, plo50, phi50, color="C0", alpha=0.25, label="50% CrI")
    ax.plot(days, pmid, color="C0", lw=2, label="Posterior median")
    ax.scatter(np.arange(T), cases, s=12, color="black", zorder=5, label="Observed cases")
    ax.axvline(T - 1, color="grey", ls=":", lw=1)
    ax.set_xlim(x0, n_steps - 1)
    ax.set_ylim(0, ymax)
    ax.text(T - 1, ymax * 0.9, " forecast →", color="grey", fontsize=9)
    ax.set_xlabel("Day")
    ax.set_ylabel("Reported cases / day")
    ax.set_title(f"{title} — fit + {horizon}-day probabilistic forecast")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"sota_forecast{tag}.png", dpi=120)
    plt.close(fig)
    print(f"Wrote sota_Rt{tag}.png and sota_forecast{tag}.png")

    # --- Deaths panel (only when the model was fit with a deaths stream) ------
    if deaths is not None and "expected_deaths" in samples:
        from sota_model import posterior_predictive_deaths

        dpred = posterior_predictive_deaths(samples)
        dlo, dmid, dhi = credible_band(dpred)
        dlo50, _, dhi50 = credible_band(dpred, 0.25, 0.75)
        dmax = float(np.max(deaths)) * 2.5 + 1
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(days, dlo, dhi, color="C3", alpha=0.15, label="90% CrI")
        ax.fill_between(days, dlo50, dhi50, color="C3", alpha=0.25, label="50% CrI")
        ax.plot(days, dmid, color="C3", lw=2, label="Posterior median")
        ax.scatter(np.arange(len(deaths)), deaths, s=12, color="black", zorder=5,
                   label="Observed deaths")
        ax.axvline(T - 1, color="grey", ls=":", lw=1)
        ax.set_xlim(x0, n_steps - 1)
        ax.set_ylim(0, dmax)
        ax.text(T - 1, dmax * 0.9, " forecast →", color="grey", fontsize=9)
        ax.set_xlabel("Day")
        ax.set_ylabel("Reported deaths / day")
        ax.set_title(f"{title} — deaths fit + {horizon}-day forecast")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"sota_deaths{tag}.png", dpi=120)
        plt.close(fig)
        print(f"Wrote sota_deaths{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="run the self-test on synthetic data")
    ap.add_argument("--country", default="Italy")
    ap.add_argument("--start", default="2020-08-15")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--save", default=None, help="save posterior samples to this .npz path")
    ap.add_argument("--load", default=None, help="skip MCMC and load posterior samples from .npz")
    args = ap.parse_args()

    cfg = EpiConfig()
    true_Rt = true_inf = deaths = None

    if args.synthetic:
        print("Generating synthetic epidemic with a known R_t (lockdown scenario)...")
        cases, true_Rt, true_inf, _, deaths = make_synthetic(cfg, T=args.days)
        title = "SYNTHETIC self-test"
    else:
        print(f"Downloading JHU data for {args.country} from {args.start} ({args.days} days)...")
        try:
            c, d = load_country(args.country, args.start, args.days)
            cases = c.to_numpy(dtype=float)
            deaths = d.to_numpy(dtype=float)
            title = f"{args.country} (from {args.start})"
        except Exception as e:  # noqa: BLE001 - fall back to synthetic if offline
            print(f"  Data download failed ({e!r}); falling back to synthetic mode.")
            cases, true_Rt, true_inf, _, deaths = make_synthetic(cfg, T=args.days)
            title = "SYNTHETIC (network fallback)"

    if args.load:
        print(f"Loading posterior samples from {args.load} (skipping MCMC)...")
        samples = load_samples(args.load)
    else:
        print(f"Fitting renewal model (cases + deaths) on {len(cases)} days with NUTS "
              f"({args.warmup} warmup + {args.samples} samples x2 chains)...")
        mcmc, _, _ = fit(cases, cfg, horizon=args.horizon, deaths=deaths,
                         num_warmup=args.warmup, num_samples=args.samples)
        mcmc.print_summary(exclude_deterministic=True)
        samples = mcmc.get_samples()
        if args.save:
            save_samples(samples, args.save)
            print(f"Saved posterior samples to {args.save}")

    summarise_samples(samples)
    tag = "_synthetic" if args.synthetic else f"_{args.country.lower()}"
    plot_results(samples, cases, args.horizon, title, true_Rt, true_inf, tag=tag, deaths=deaths)


if __name__ == "__main__":
    main()
