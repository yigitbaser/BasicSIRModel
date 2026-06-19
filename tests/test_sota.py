"""Unit tests for the renewal model's building blocks and the evaluation harness.

These run fast (no MCMC) and guard the parts that are easy to get subtly wrong:
the discretised input distributions, the renewal recursion, and the proper
scoring rules. Run with:  pytest -q
"""

import numpy as np
import pytest

from sota_model import discretise_gamma, discretise_lognormal
import evaluate


# --- input distributions ---------------------------------------------------
def test_generation_interval_is_a_pmf():
    pmf = discretise_gamma(mean=5.2, sd=3.8, length=21)
    assert pmf.shape == (21,)
    assert np.isclose(pmf.sum(), 1.0)
    assert (pmf >= 0).all()


def test_generation_interval_mean_is_reasonable():
    # The discretised mean (days 1..L) should be close to the target mean.
    pmf = discretise_gamma(mean=5.2, sd=3.8, length=30)
    lags = np.arange(1, len(pmf) + 1)
    assert abs((pmf * lags).sum() - 5.2) < 0.5


def test_reporting_delay_is_a_pmf():
    pmf = discretise_lognormal(mean=9.0, sd=4.5, length=30)
    assert np.isclose(pmf.sum(), 1.0)
    assert (pmf >= 0).all()


# --- renewal recursion -----------------------------------------------------
def test_renewal_constant_rt_grows_geometrically():
    """With a one-day generation interval and constant R_t, infections must
    multiply by exactly R_t each day."""
    R = 1.5
    gen = np.array([1.0])  # all weight on lag 1
    window = np.array([10.0])
    series = []
    for _ in range(5):
        I_t = R * np.dot(window, gen[::-1])
        series.append(I_t)
        window = np.array([I_t])
    series = np.array(series)
    # Each step multiplies by exactly R.
    assert np.allclose(series[1:] / series[:-1], R)
    assert np.isclose(series[0], R * 10.0)


# --- proper scoring --------------------------------------------------------
def test_interval_score_zero_width_when_hit_center():
    # Perfect point forecast: interval [y, y] has score 0.
    y = np.array([100.0])
    assert np.allclose(evaluate.interval_score(y, y, y, alpha=0.1), 0.0)


def test_interval_score_penalises_miss():
    # y outside [lo, up] must score worse than y inside.
    y = np.array([100.0])
    inside = evaluate.interval_score(y, np.array([90.0]), np.array([110.0]), 0.1)
    outside = evaluate.interval_score(y, np.array([110.0]), np.array([120.0]), 0.1)
    assert outside > inside


def test_wis_better_for_sharper_calibrated_forecast():
    rng = np.random.default_rng(0)
    y = np.full(50, 100.0)
    sharp = rng.normal(100, 5, size=(500, 50))   # tight, centred
    wide = rng.normal(100, 50, size=(500, 50))   # loose, centred
    assert evaluate.weighted_interval_score(y, sharp).mean() < \
        evaluate.weighted_interval_score(y, wide).mean()


def test_wis_penalises_bias():
    rng = np.random.default_rng(1)
    y = np.full(50, 100.0)
    unbiased = rng.normal(100, 10, size=(500, 50))
    biased = rng.normal(150, 10, size=(500, 50))
    assert evaluate.weighted_interval_score(y, unbiased).mean() < \
        evaluate.weighted_interval_score(y, biased).mean()


def test_coverage_in_unit_interval():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, size=30)
    pred = rng.normal(0, 1, size=(400, 30))
    c = evaluate.coverage(y, pred, 0.9)
    assert 0.0 <= c <= 1.0


def test_pit_values_in_unit_interval():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 100, size=40).astype(float)
    pred = rng.integers(0, 100, size=(300, 40)).astype(float)
    pit = evaluate.pit_values(y, pred)
    assert ((pit >= 0) & (pit <= 1)).all()


def test_synthetic_truncation_lowers_recent_counts():
    """make_synthetic(truncate=True) should report fewer of the most-recent-day
    counts than the complete (truncate=False) version, by construction."""
    from sota_model import EpiConfig
    from sota_run import make_synthetic

    cfg = EpiConfig()
    full = make_synthetic(cfg, T=120, seed=3, truncate=False)[0]
    trunc = make_synthetic(cfg, T=120, seed=3, truncate=True)[0]
    # The last few days are heavily truncated; their sum must drop.
    assert trunc[-5:].sum() < full[-5:].sum()
    # Early days are essentially complete in both.
    assert abs(trunc[:50].sum() - full[:50].sum()) / full[:50].sum() < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
