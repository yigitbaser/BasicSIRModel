"""
Multi-model ensemble forecasting + evaluation.

Across the COVID-19 Forecast Hubs the single most robust empirical finding was
that a **combination of models** beats essentially every individual model. This
module builds a minimal but honest version of that idea:

    * a cheap statistical **baseline** (log-linear growth extrapolation with
      properly growing prediction intervals);
    * the mechanistic **renewal** model (from `sota_model.py`);
    * an **ensemble** that linearly pools their predictive samples.

All three are scored with the Weighted Interval Score (WIS) from `evaluate.py`
through proper backtesting, so the claim "the ensemble is competitive with / beats
its components" is measured, not asserted.

Run ``python ensemble.py`` for a self-contained backtest comparison.
"""

from __future__ import annotations

import numpy as np

from sota_model import EpiConfig, fit, posterior_predictive_cases
from evaluate import weighted_interval_score, coverage


# ---------------------------------------------------------------------------
# Component 1: cheap statistical baseline.
# ---------------------------------------------------------------------------
def baseline_forecast(train, horizon, window=21, n_samples=2000, seed=0):
    """Log-linear growth-rate extrapolation with growing prediction intervals.

    Fits an OLS line to log(cases) over the last `window` days and projects it
    forward, propagating the slope/intercept sampling uncertainty plus residual
    noise so the intervals widen with the horizon. This is the kind of trivial
    model an ensemble should beat — or at least not lose to.
    """
    rng = np.random.default_rng(seed)
    y = np.log(np.clip(np.asarray(train[-window:], dtype=float), 1.0, None))
    n = len(y)
    x = np.arange(n)
    A = np.vstack([x, np.ones_like(x)]).T              # columns: [slope, intercept]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sigma = float(resid.std(ddof=2)) if n > 2 else 0.3
    sigma = max(sigma, 1e-2)

    cov = sigma**2 * np.linalg.inv(A.T @ A)            # OLS coefficient covariance
    draws = rng.multivariate_normal(coef, cov, size=n_samples)   # (n_samples, 2)
    xf = (n - 1) + np.arange(1, horizon + 1)
    mu = draws[:, 0:1] * xf[None, :] + draws[:, 1:2]   # (n_samples, horizon)
    log_pred = mu + rng.normal(0.0, sigma, size=mu.shape)
    return np.exp(np.clip(log_pred, None, 25))         # clip to avoid overflow


# ---------------------------------------------------------------------------
# Component 2: mechanistic renewal model forecast (held-out horizon only).
# ---------------------------------------------------------------------------
def renewal_forecast(train, cfg, horizon, num_warmup=400, num_samples=400, seed=0):
    mcmc, _, _ = fit(train, cfg, horizon=horizon,
                     num_warmup=num_warmup, num_samples=num_samples, seed=seed)
    pred = posterior_predictive_cases(mcmc.get_samples())   # (draws, len(train)+horizon)
    return pred[:, len(train):len(train) + horizon]         # forecast portion


# ---------------------------------------------------------------------------
# Ensemble: linear pool (equal-weight mixture) of predictive samples.
# ---------------------------------------------------------------------------
def ensemble_pool(*sample_sets):
    """Combine predictive sample sets by stacking them (an equal-weight linear
    pool). Each input is (draws_i, horizon); output is (sum draws_i, horizon)."""
    h = sample_sets[0].shape[1]
    # Resample each component to the same size so weights are equal.
    m = min(s.shape[0] for s in sample_sets)
    rng = np.random.default_rng(0)
    picked = [s[rng.choice(s.shape[0], m, replace=False)] for s in sample_sets]
    return np.concatenate(picked, axis=0)


# ---------------------------------------------------------------------------
# Backtest comparison.
# ---------------------------------------------------------------------------
def compare_at_origin(cases, cfg, train_end, horizon=14, seed=0, **fit_kw):
    train = np.asarray(cases[:train_end], dtype=float)
    truth = np.asarray(cases[train_end:train_end + horizon], dtype=float)
    h = len(truth)
    if h == 0:
        return None

    base = baseline_forecast(train, h, seed=seed)
    renew = renewal_forecast(train, cfg, h, seed=seed, **fit_kw)
    ens = ensemble_pool(base, renew)

    out = {}
    for name, samp in [("baseline", base), ("renewal", renew), ("ensemble", ens)]:
        out[name] = {
            "wis": float(np.mean(weighted_interval_score(truth, samp))),
            "cov90": coverage(truth, samp, 0.90),
        }
    return out


def main():
    from sota_run import make_synthetic

    cfg = EpiConfig()
    print("Backtesting baseline vs renewal vs ensemble (mean WIS, lower is better)...")
    cases, *_ = make_synthetic(cfg, T=140)
    origins = [70, 90, 110]
    agg = {k: [] for k in ("baseline", "renewal", "ensemble")}
    for te in origins:
        res = compare_at_origin(cases, cfg, te, horizon=14)
        if res is None:
            continue
        print(f"\n  origin day {te}:")
        for name in ("baseline", "renewal", "ensemble"):
            print(f"    {name:9s}  WIS {res[name]['wis']:>10.1f}   90% cov {res[name]['cov90']:.2f}")
            agg[name].append(res[name]["wis"])
    print("\n  === mean WIS across origins (lower is better) ===")
    for name in ("baseline", "renewal", "ensemble"):
        if agg[name]:
            print(f"    {name:9s}  {np.mean(agg[name]):>10.1f}")


if __name__ == "__main__":
    main()
