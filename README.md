# BasicSIRModel
This project contains a basic SIR model, with build in plotting. A companion blog post for this project can be found <a href="https://mattravenhall.github.io/2018/01/02/SIR-Model.html">here</a>.

## Running the model
```python
>>> import model
>>> m = model.SIR()
>>> m.run()
>>> m.plot()
```

## Changing Model Parameters
Specific parameters must be indicated when creating an new instance of the model.SIR class. For example, the beta (S to I) rate can be changed as follows:

```python
>>> import model
>>> m = model.SIR(rateSI=0.05)
```

Changable parameters include:
* 'eons' (number of time points to model, default 1000)
* 'Susceptible' (number of susceptible individuals at time 0, default 950)
* 'Infected' (number of infected individuals at time 0, default 50)
* 'Resistant' (number of resistant individuals at time 0, default 0)
* 'rateSI' (base rate 'beta' from S to I, default 0.05)
* 'rateIR' (base rate 'gamma' from I to R, default 0.01)

More complex alterations of the model will require specific engineering of the code. Feel free to dive in.

## Example Output
<img width="600" alt="portfolio_view" src="https://raw.githubusercontent.com/mattravenhall/BasicSIRModel/master/example.png">

---

# COVID-19 Modelling Extension

The original SIR model above is great for teaching but too simple to *predict*
COVID-19. This repository now also includes a richer modelling toolkit and a
written assessment of where SIR falls short and what to use instead.

## New files

| File | Purpose |
|---|---|
| [`REPORT.md`](REPORT.md) | Assessment of the SIR model and a survey of better models (SEIR, SEIRD, age-structured, network, agent-based, Bayesian/ensemble) with a recommendation |
| [`models.py`](models.py) | ODE-based SIR / SIRD / SEIR / SEIRD / SEIRDV models with interpretable, literature-backed parameters and optional time-varying transmission |
| [`statistics.py`](statistics.py) | Computes epidemic statistics: R₀, effective Rₜ(t), peak size & timing, attack rate, deaths, realised IFR, doubling time, herd-immunity threshold, epidemic duration; plus plotting |
| [`run_statistics.py`](run_statistics.py) | Driver that runs every model, prints a statistics table and writes figures + a CSV |
| [`PARAMETERS.md`](PARAMETERS.md) | Scientifically-backed COVID-19 parameter values, each with a peer-reviewed citation |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | List of trusted COVID-19 data repositories for fitting/validation (WHO, OWID, JHU CSSE, CDC, ECDC, …) |

## State-of-the-art model (Bayesian renewal model)

Beyond the compartmental models above, the repository includes a **from-scratch
state-of-the-art model** — a Bayesian semi-mechanistic *renewal* model with a
time-varying reproduction number, fit to real data by gradient-based MCMC
(NumPyro/JAX). This is the model family behind EpiNow2, the Imperial College
*Nature* 2020 study, and the CDC/ECDC Forecast Hubs.

| File | Purpose |
|---|---|
| [`SOTA_MODEL.md`](SOTA_MODEL.md) | Explains the method and why it is state of the art |
| [`sota_model.py`](sota_model.py) | The renewal model + NUTS inference + forecasting |
| [`sota_run.py`](sota_run.py) | Synthetic self-test, real-data loader, plotting |

```bash
pip install numpyro jax arviz numpy scipy pandas matplotlib
python sota_run.py --synthetic                 # self-test: recover a known R_t
python sota_run.py --country Italy --days 120   # fit real JHU data + forecast
python evaluate.py                              # backtest: WIS / coverage / PIT calibration
pytest -q                                       # fast unit tests
```

It infers the time-varying R&#x209C;, nowcasts true infections behind reporting
delays, and produces a probabilistic forecast — all with full Bayesian credible
intervals. See `SOTA_MODEL.md` for details and validation.

## Running the new statistics model

```bash
pip install numpy scipy pandas matplotlib
python run_statistics.py
```

This prints headline epidemic statistics for SIR vs SEIR vs SEIRD vs a
vaccinated SEIRDV scenario, and writes `seird_trajectory.png`,
`model_comparison.png` and `statistics_summary.csv`.

```python
import models
from statistics import compute_statistics, format_statistics

result = models.SEIRD(population=1_000_000, I0=100, days=365).run()
print(format_statistics(compute_statistics(result)))
```
