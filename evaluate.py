"""
Evaluation harness for the Bayesian renewal model: proper scoring + calibration.

A forecasting model is only as trustworthy as its *measured* skill. This module
adds the evaluation standard used by the US/EU COVID-19 Forecast Hubs:

    * Weighted Interval Score (WIS) — the Hub's headline proper score for
      quantile/interval forecasts (lower is better);
    * Interval coverage — do the nominal 50% / 90% credible intervals actually
      contain the truth 50% / 90% of the time?
    * PIT (Probability Integral Transform) calibration — the histogram should be
      uniform if the predictive distribution is well calibrated;
    * Backtesting — fit on a training window, forecast a held-out horizon, and
      score the forecast against what actually happened.

Run ``python evaluate.py`` for a self-contained backtest on synthetic data.
"""

from __future__ import annotations

import numpy as np

from sota_model import EpiConfig, fit, posterior_predictive_cases


# ---------------------------------------------------------------------------
# Proper scoring.
# ---------------------------------------------------------------------------
def interval_score(y, lower, upper, alpha):
    """Interval score for a central (1-alpha) prediction interval [lower, upper].

    IS = (u - l) + (2/alpha)(l - y) 1{y<l} + (2/alpha)(y - u) 1{y>u}.
    Lower is better: rewards sharpness (narrow interval) but penalises misses.
    """
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    return (
        (upper - lower)
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def weighted_interval_score(y, pred_samples, alphas=(0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Weighted Interval Score from posterior-predictive samples.

    Parameters
    ----------
    y : array (H,)
        Observed values over the forecast horizon.
    pred_samples : array (draws, H)
        Posterior-predictive samples for the same days.
    alphas : sequence
        Central-interval levels; the symmetric quantiles (alpha/2, 1-alpha/2)
        define each interval. The default set matches the COVID-19 Forecast Hub.

    Returns
    -------
    array (H,) of WIS per day (lower = better).
    """
    y = np.asarray(y, dtype=float)
    median = np.median(pred_samples, axis=0)
    K = len(alphas)
    total = 0.5 * np.abs(y - median)
    for alpha in alphas:
        lo = np.quantile(pred_samples, alpha / 2.0, axis=0)
        up = np.quantile(pred_samples, 1.0 - alpha / 2.0, axis=0)
        total = total + (alpha / 2.0) * interval_score(y, lo, up, alpha)
    return total / (K + 0.5)


def coverage(y, pred_samples, level):
    """Empirical coverage of the central `level` (e.g. 0.9) predictive interval."""
    y = np.asarray(y, dtype=float)
    lo = np.quantile(pred_samples, (1 - level) / 2.0, axis=0)
    up = np.quantile(pred_samples, 1 - (1 - level) / 2.0, axis=0)
    return float(np.mean((y >= lo) & (y <= up)))


def pit_values(y, pred_samples, seed=0):
    """Randomised PIT values; uniform on [0,1] iff the forecast is calibrated.

    For discrete (count) predictions we randomise within the probability jump to
    avoid the well-known discreteness artefacts in the PIT histogram.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    draws = pred_samples.shape[0]
    below = np.mean(pred_samples < y[None, :], axis=0)
    equal = np.mean(pred_samples == y[None, :], axis=0)
    u = rng.uniform(size=y.shape[0])
    return below + u * equal


# ---------------------------------------------------------------------------
# Backtesting.
# ---------------------------------------------------------------------------
def backtest(cases, cfg: EpiConfig, train_end: int, horizon: int = 14,
             num_warmup=400, num_samples=400, seed=0):
    """Fit on cases[:train_end], forecast `horizon` days, score against truth.

    Returns a dict with per-day WIS, mean WIS, 50%/90% coverage and PIT values
    for the held-out forecast window.
    """
    train = np.asarray(cases[:train_end], dtype=float)
    truth = np.asarray(cases[train_end:train_end + horizon], dtype=float)
    h = len(truth)
    if h == 0:
        raise ValueError("No held-out data to score; reduce train_end or horizon.")

    mcmc, _, _ = fit(train, cfg, horizon=horizon,
                     num_warmup=num_warmup, num_samples=num_samples, seed=seed)
    pred = posterior_predictive_cases(mcmc.get_samples())     # (draws, train_end+horizon)
    fc = pred[:, train_end:train_end + h]                     # forecast portion

    wis = weighted_interval_score(truth, fc)
    return {
        "train_end": train_end,
        "horizon": h,
        "wis_per_day": wis,
        "wis_mean": float(np.mean(wis)),
        "coverage_50": coverage(truth, fc, 0.50),
        "coverage_90": coverage(truth, fc, 0.90),
        "pit": pit_values(truth, fc),
        "forecast_median": np.median(fc, axis=0),
        "truth": truth,
    }


def rolling_backtest(cases, cfg: EpiConfig, train_ends, horizon=14, **kw):
    """Run several backtests at different origins; aggregate WIS and coverage."""
    results = []
    for te in train_ends:
        if te + horizon > len(cases):
            continue
        r = backtest(cases, cfg, te, horizon=horizon, **kw)
        results.append(r)
        print(f"  origin day {te:>3}: mean WIS {r['wis_mean']:>10.1f}  "
              f"50%cov {r['coverage_50']:.2f}  90%cov {r['coverage_90']:.2f}")
    if results:
        print(f"\n  AGGREGATE: mean WIS {np.mean([r['wis_mean'] for r in results]):.1f}  "
              f"50%cov {np.mean([r['coverage_50'] for r in results]):.2f}  "
              f"90%cov {np.mean([r['coverage_90'] for r in results]):.2f}")
    return results


def plot_calibration(results, path="sota_calibration.png"):
    """PIT histogram + coverage bar from a list of backtest results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pit = np.concatenate([r["pit"] for r in results])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(pit, bins=10, range=(0, 1), density=True, color="C0", edgecolor="white")
    ax1.axhline(1.0, color="red", ls="--", label="uniform (calibrated)")
    ax1.set_title("PIT histogram (flat ⇒ calibrated)")
    ax1.set_xlabel("PIT value")
    ax1.set_ylabel("density")
    ax1.legend()

    levels = [0.5, 0.9]
    emp = [np.mean([r[f"coverage_{int(l*100)}"] for r in results]) for l in levels]
    ax2.bar([f"{int(l*100)}%" for l in levels], emp, color="C0", width=0.5, label="empirical")
    ax2.plot([-0.5, 1.5], [0.5, 0.5], "k:", alpha=0)  # spacer
    for i, l in enumerate(levels):
        ax2.hlines(l, i - 0.25, i + 0.25, color="red", ls="--",
                   label="nominal" if i == 0 else None)
    ax2.set_ylim(0, 1)
    ax2.set_title("Interval coverage (bar ≈ line ⇒ calibrated)")
    ax2.set_ylabel("coverage")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    from sota_run import make_synthetic

    cfg = EpiConfig()
    print("Generating synthetic epidemic and backtesting the forecast...")
    cases, _, _, _ = make_synthetic(cfg, T=140)
    # Forecast from several origins along the epidemic.
    results = rolling_backtest(cases, cfg, train_ends=[70, 90, 110],
                               horizon=14, num_warmup=400, num_samples=400)
    if results:
        plot_calibration(results)


if __name__ == "__main__":
    main()
