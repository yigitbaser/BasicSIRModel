"""
Statistics and visualisation layer for the compartmental models in ``models.py``.

The original project only plotted three curves. Real epidemic modelling is about
the *summary statistics* that drive public-health decisions. This module turns a
solved trajectory into the numbers planners actually ask for:

    * Basic reproduction number R0 and the time-varying effective Rt
    * Herd-immunity threshold (1 - 1/R0)
    * Epidemic peak: how many infectious at once, and on which day
    * Final size / attack rate (cumulative fraction ever infected)
    * Total deaths and realised infection-fatality ratio
    * Early-epidemic growth rate and doubling time
    * Epidemic duration (days above a prevalence threshold)

These are computed directly from the simulated curves so they stay consistent
with whatever model structure and parameters were used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import EpidemicResult


def compute_statistics(result: EpidemicResult) -> dict:
    """Derive headline epidemic statistics from a solved trajectory."""
    df = result.frame
    N = result.N
    p = result.params

    I = df["Infected"].to_numpy()
    S = df["Susceptible"].to_numpy()
    day = df["Day"].to_numpy()

    # --- Reproduction numbers -------------------------------------------------
    R0 = p["R0"]
    # Effective reproduction number through time: Rt = R0 * S/N (depletion of
    # susceptibles is what bends the curve over).
    Rt = R0 * S / N

    # --- Peak -----------------------------------------------------------------
    peak_idx = int(np.argmax(I))
    peak_infected = float(I[peak_idx])
    peak_day = float(day[peak_idx])

    # --- Final size / attack rate --------------------------------------------
    # Use the cumulative-infection accumulator so that, in vaccinated scenarios,
    # people who left S only via vaccination are NOT counted as infections.
    if "CumulativeInfections" in df:
        ever_infected = float(df["CumulativeInfections"].iloc[-1])
    else:
        ever_infected = N - S[-1]
    attack_rate = ever_infected / N

    # --- Deaths ---------------------------------------------------------------
    total_dead = float(df["Dead"].iloc[-1])
    realised_ifr = total_dead / ever_infected if ever_infected > 0 else 0.0

    # --- Herd immunity threshold ---------------------------------------------
    hit = 1.0 - 1.0 / R0 if R0 > 1 else 0.0

    # --- Early exponential growth rate & doubling time -----------------------
    # Fit a line to log(I) over the first part of the epidemic where growth is
    # still roughly exponential (before susceptible depletion matters).
    early_mask = (day <= max(1, peak_day * 0.3)) & (I > 0)
    doubling_time = float("nan")
    growth_rate = float("nan")
    if early_mask.sum() >= 2:
        coeffs = np.polyfit(day[early_mask], np.log(I[early_mask]), 1)
        growth_rate = float(coeffs[0])  # per day
        if growth_rate > 0:
            doubling_time = float(np.log(2) / growth_rate)

    # --- Epidemic duration (days with >0.01% of population infectious) --------
    threshold = 0.0001 * N
    above = day[I > threshold]
    duration = float(above[-1] - above[0]) if above.size else 0.0

    return {
        "model": result.name,
        "population": N,
        "R0": R0,
        "herd_immunity_threshold": hit,
        "peak_infected": peak_infected,
        "peak_infected_pct": peak_infected / N,
        "peak_day": peak_day,
        "Rt_final": float(Rt[-1]),
        "total_ever_infected": ever_infected,
        "attack_rate": attack_rate,
        "total_dead": total_dead,
        "realised_ifr": realised_ifr,
        "early_growth_rate_per_day": growth_rate,
        "doubling_time_days": doubling_time,
        "epidemic_duration_days": duration,
        "Rt_series": Rt,
    }


def format_statistics(stats: dict) -> str:
    """Render the statistics dict as a human-readable report block."""
    N = stats["population"]
    lines = [
        f"=== {stats['model']} model — epidemic statistics ===",
        f"  Population (N)............................ {N:>15,.0f}",
        f"  Basic reproduction number R0............. {stats['R0']:>15.2f}",
        f"  Herd-immunity threshold (1 - 1/R0)....... {stats['herd_immunity_threshold']:>14.1%}",
        f"  Early growth rate (per day).............. {stats['early_growth_rate_per_day']:>15.3f}",
        f"  Epidemic doubling time (days)............ {stats['doubling_time_days']:>15.2f}",
        f"  Peak prevalence (infectious at once)..... {stats['peak_infected']:>15,.0f}",
        f"  Peak prevalence (% of population)........ {stats['peak_infected_pct']:>14.1%}",
        f"  Day of peak.............................. {stats['peak_day']:>15.0f}",
        f"  Total ever infected (final size)......... {stats['total_ever_infected']:>15,.0f}",
        f"  Attack rate (% ever infected)............ {stats['attack_rate']:>14.1%}",
        f"  Total deaths............................. {stats['total_dead']:>15,.0f}",
        f"  Realised infection fatality ratio........ {stats['realised_ifr']:>14.2%}",
        f"  Epidemic duration (days >0.01% infected). {stats['epidemic_duration_days']:>15.0f}",
    ]
    return "\n".join(lines)


def plot_trajectory(result: EpidemicResult, stats: dict | None = None, path: str = "seird_trajectory.png"):
    """Plot compartment curves + the effective reproduction number Rt."""
    import matplotlib

    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt

    df = result.frame
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1]})

    colours = {
        "Susceptible": "#1f77b4",
        "Exposed": "#ff7f0e",
        "Infected": "#d62728",
        "Recovered": "#2ca02c",
        "Dead": "#000000",
    }
    for col, colour in colours.items():
        if col in df and df[col].abs().sum() > 0:
            ax1.plot(df["Day"], df[col], color=colour, label=col, linewidth=2)

    if stats is not None:
        ax1.axvline(stats["peak_day"], color="grey", linestyle="--", linewidth=1)
        ax1.annotate(
            f"peak: {stats['peak_infected']:,.0f}\nday {stats['peak_day']:.0f}",
            xy=(stats["peak_day"], stats["peak_infected"]),
            xytext=(stats["peak_day"] + df["Day"].max() * 0.05, stats["peak_infected"]),
            fontsize=9,
        )

    ax1.set_ylabel("Individuals")
    ax1.set_title(f"{result.name} model (R0 = {result.params['R0']:.2f})")
    ax1.legend(loc="center right")
    ax1.grid(alpha=0.3)

    # Effective reproduction number panel.
    R0 = result.params["R0"]
    Rt = R0 * df["Susceptible"].to_numpy() / result.N
    ax2.plot(df["Day"], Rt, color="purple", linewidth=2, label="Effective $R_t$")
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1, label="$R_t = 1$ (epidemic threshold)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("$R_t$")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def compare_models(results: dict[str, EpidemicResult], path: str = "model_comparison.png"):
    """Overlay the Infected curve of several models for side-by-side comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, res in results.items():
        ax.plot(res.frame["Day"], res.frame["Infected"], label=label, linewidth=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious individuals")
    ax.set_title("Infectious-curve comparison across model structures")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
