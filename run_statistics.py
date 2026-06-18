"""
Driver script: run the COVID-19 compartmental models and emit statistics + plots.

Usage
-----
    python run_statistics.py

Outputs
-------
    * console table of headline epidemic statistics for each model structure
    * seird_trajectory.png   - full SEIRD trajectory with the Rt panel
    * model_comparison.png   - infectious curves of SIR vs SEIR vs SEIRD
    * statistics_summary.csv - machine-readable table of all statistics

Every default parameter is the literature-backed ancestral-strain value
documented in PARAMETERS.md.
"""

from __future__ import annotations

import pandas as pd

import models
from statistics import (
    compute_statistics,
    format_statistics,
    plot_trajectory,
    compare_models,
)


def main():
    # A mid-sized city of 1,000,000 people, seeded with 100 infections.
    common = dict(population=1_000_000, I0=100, days=365)

    # Build one model per structure, all with the same ancestral-strain params.
    built = {
        "SIR": models.SIR(**common),
        "SEIR": models.SEIR(**common),
        "SEIRD": models.SEIRD(**common),
    }

    # A vaccinated scenario: 0.5% of susceptibles immunised per day.
    built["SEIRDV (0.5%/day vacc.)"] = models.SEIRDV(vacc_rate=0.005, **common)

    results = {name: m.run() for name, m in built.items()}

    # ------------------------------------------------------------------ stats
    all_stats = []
    for name, res in results.items():
        stats = compute_statistics(res)
        stats["scenario"] = name
        print(format_statistics(stats))
        print()
        # Drop the array before tabulating.
        row = {k: v for k, v in stats.items() if k != "Rt_series"}
        all_stats.append(row)

    summary = pd.DataFrame(all_stats).set_index("scenario")
    summary.to_csv("statistics_summary.csv")
    print("Wrote statistics_summary.csv")

    # ------------------------------------------------------------------ plots
    seird = results["SEIRD"]
    seird_stats = compute_statistics(seird)
    plot_trajectory(seird, seird_stats, path="seird_trajectory.png")
    print("Wrote seird_trajectory.png")

    # Compare the three deterministic structures (exclude the vaccinated run).
    compare_models(
        {k: v for k, v in results.items() if "vacc" not in k},
        path="model_comparison.png",
    )
    print("Wrote model_comparison.png")


if __name__ == "__main__":
    main()
