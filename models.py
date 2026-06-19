"""
Compartmental epidemic models for COVID-19.

This module extends the original primitive `SIR` model (see ``model.py``) with a
family of mechanistically richer, ODE-based compartmental models that are the
standard tools used in the epidemiological literature:

    * SIR    - Susceptible -> Infected -> Recovered
    * SIRD   - adds an explicit Dead compartment (so case fatality can be read off)
    * SEIR   - adds an Exposed (latent, infected-but-not-yet-infectious) compartment
    * SEIRD  - SEIR + explicit Dead compartment  (recommended COVID-19 baseline)
    * SEIRDV - SEIRD + a leaky vaccination flow out of Susceptible

All models are integrated as systems of ordinary differential equations with
``scipy.integrate.solve_ivp`` rather than the hand-rolled forward-Euler loop in
the original code. This is more accurate (adaptive RK45 stepping), faster and
far easier to extend.

Parameters are expressed in *interpretable* epidemiological units:

    * R0      - basic reproduction number (dimensionless)
    * t_inf   - mean infectious period in days        -> gamma = 1 / t_inf
    * t_lat   - mean latent period in days            -> sigma = 1 / t_lat
    * ifr     - infection fatality ratio (fraction of infections that die)

and the transmission rate beta is derived as ``beta = R0 * gamma`` so the user
reasons in terms of quantities that have been measured in the field (see
``PARAMETERS.md``) instead of an abstract ``rateSI``.

The default parameters correspond to the wild-type (ancestral) SARS-CoV-2
strain; every value is documented and referenced in ``PARAMETERS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Default, literature-backed parameters for the ancestral SARS-CoV-2 strain.
# See PARAMETERS.md for the source of every number.
# ---------------------------------------------------------------------------
COVID19_ANCESTRAL = {
    "R0": 2.79,        # pooled/median basic reproduction number, ancestral strain
    "t_latent": 4.0,   # mean latent period (days); ~1 day shorter than incubation
    "t_infectious": 6.5,  # mean infectious period (days)
    "ifr": 0.0068,     # population infection fatality ratio, pre-vaccine era (~0.68%)
}


@dataclass
class EpidemicResult:
    """Container for a solved model trajectory plus derived statistics."""

    name: str
    frame: pd.DataFrame            # time series of every compartment
    params: dict                   # the parameters used
    N: float                       # total population

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<EpidemicResult {self.name}: {len(self.frame)} days, N={self.N:,.0f}>"


class CompartmentalModel:
    """Base class: a configurable SIR / SIRD / SEIR / SEIRD / SEIRDV model.

    Parameters
    ----------
    population : float
        Total population N.
    I0 : float
        Initial number of infectious individuals.
    E0 : float
        Initial number of exposed (latent) individuals (SEIR family only).
    R0_num : float
        Initial number of recovered/immune individuals.
    days : int
        Number of days to simulate.
    R0 : float
        Basic reproduction number (dimensionless).
    t_infectious, t_latent : float
        Mean infectious / latent periods in days.
    ifr : float
        Infection fatality ratio (fraction of infections that end in death).
    vacc_rate : float
        Daily per-capita vaccination rate (SEIRDV only). 0 disables vaccination.
    beta_t : callable or None
        Optional function ``beta_t(t, beta0) -> beta`` to make transmission
        time-varying (e.g. lockdowns / non-pharmaceutical interventions).
    variant : str
        One of {"sir", "sird", "seir", "seird", "seirdv"} selecting structure.
    """

    def __init__(
        self,
        population: float = 1_000_000,
        I0: float = 100,
        E0: float = 0,
        R0_num: float = 0,
        D0: float = 0,
        days: int = 365,
        R0: float = COVID19_ANCESTRAL["R0"],
        t_infectious: float = COVID19_ANCESTRAL["t_infectious"],
        t_latent: float = COVID19_ANCESTRAL["t_latent"],
        ifr: float = COVID19_ANCESTRAL["ifr"],
        vacc_rate: float = 0.0,
        waning_days: float = 0.0,
        beta_t: Callable[[float, float], float] | None = None,
        variant: str = "seird",
    ):
        self.N = float(population)
        self.I0 = float(I0)
        self.E0 = float(E0)
        self.R0_num = float(R0_num)
        self.D0 = float(D0)
        self.days = int(days)

        self.R0 = float(R0)
        self.t_infectious = float(t_infectious)
        self.t_latent = float(t_latent)
        self.ifr = float(ifr)
        self.vacc_rate = float(vacc_rate)
        self.waning_days = float(waning_days) if waning_days else 0.0
        self.beta_t = beta_t
        self.variant = variant.lower()

        if self.variant not in {"sir", "sird", "seir", "seird", "seirdv"}:
            raise ValueError(f"Unknown variant: {variant!r}")

        # Derived rate constants ------------------------------------------------
        self.gamma = 1.0 / self.t_infectious          # recovery rate (1/day)
        self.sigma = 1.0 / self.t_latent              # E -> I progression rate
        self.beta = self.R0 * self.gamma              # transmission rate (1/day)

        self.result: EpidemicResult | None = None

    # -- ODE right-hand side ---------------------------------------------------
    def _rhs(self, t, y):
        # C is a cumulative-infection accumulator (not a real population pool):
        # it integrates the infection flow so the attack rate excludes people
        # who left S only because they were vaccinated.
        S, E, I, R, D, C = y
        beta = self.beta_t(t, self.beta) if self.beta_t else self.beta

        # Force of infection (frequency-dependent transmission).
        infection = beta * S * I / self.N

        if self.variant in {"sir", "sird"}:
            # No latent compartment: S -> I directly.
            progression = infection      # new infections become infectious now
            dS = -infection
            dE = 0.0
        else:
            # SEIR family: S -> E -> I.
            progression = self.sigma * E
            dS = -infection
            dE = infection - progression

        # Split the I outflow into recovery and death using the IFR.
        # A fraction `ifr` of those leaving I die; the rest recover.
        leaving_I = self.gamma * I
        if self.variant in {"sir", "seir"}:
            to_dead = 0.0
        else:
            to_dead = self.ifr * leaving_I
        to_recovered = leaving_I - to_dead

        # New infectious arrivals.
        dI = progression - leaving_I

        # Optional leaky vaccination: moves S directly into R (immune).
        vacc = 0.0
        if self.variant == "seirdv" and self.vacc_rate > 0:
            vacc = self.vacc_rate * S
            dS -= vacc

        # Optional waning immunity (SIRS/SEIRS): recovered individuals lose
        # immunity at rate 1/waning_days and return to Susceptible. This is what
        # turns a single epidemic into recurring waves / an endemic steady state.
        waning = 0.0
        if self.waning_days > 0:
            waning = R / self.waning_days
            dS += waning

        dR = to_recovered + vacc - waning
        dD = to_dead
        dC = infection  # cumulative new infections (true attack-rate numerator)

        return [dS, dE, dI, dR, dD, dC]

    # -- Solver ----------------------------------------------------------------
    def run(self) -> EpidemicResult:
        S0 = self.N - self.I0 - self.E0 - self.R0_num - self.D0
        # Sixth state C0 = people already infected at t=0 (E0 + I0 + R0 + D0).
        C0 = self.E0 + self.I0 + self.R0_num + self.D0
        y0 = [S0, self.E0, self.I0, self.R0_num, self.D0, C0]

        t_eval = np.arange(0, self.days + 1)
        sol = solve_ivp(
            self._rhs,
            t_span=(0, self.days),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-6,
        )

        frame = pd.DataFrame(
            {
                "Day": sol.t,
                "Susceptible": sol.y[0],
                "Exposed": sol.y[1],
                "Infected": sol.y[2],
                "Recovered": sol.y[3],
                "Dead": sol.y[4],
                "CumulativeInfections": sol.y[5],
            }
        )
        # Daily new infections = people leaving S (for plotting incidence curves).
        frame["NewInfections"] = (-frame["Susceptible"].diff()).clip(lower=0).fillna(0)

        self.result = EpidemicResult(
            name=self.variant.upper(),
            frame=frame,
            params={
                "R0": self.R0,
                "beta": self.beta,
                "gamma": self.gamma,
                "sigma": self.sigma,
                "t_infectious": self.t_infectious,
                "t_latent": self.t_latent,
                "ifr": self.ifr,
                "vacc_rate": self.vacc_rate,
            },
            N=self.N,
        )
        return self.result


# Convenience constructors -----------------------------------------------------
def SIR(**kwargs):
    return CompartmentalModel(variant="sir", **kwargs)


def SIRD(**kwargs):
    return CompartmentalModel(variant="sird", **kwargs)


def SEIR(**kwargs):
    return CompartmentalModel(variant="seir", **kwargs)


def SEIRD(**kwargs):
    return CompartmentalModel(variant="seird", **kwargs)


def SEIRDV(**kwargs):
    return CompartmentalModel(variant="seirdv", **kwargs)
