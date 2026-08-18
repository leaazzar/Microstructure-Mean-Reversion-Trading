# Microstructure Mean-Reversion Trading — Research Notes

Independent simulation study exploring and validating the threshold-trading
results in:

> Lucas Rabechini Amaral, "Optimal Trading of Microstructure Mean
> Reversion," arXiv:2608.00885 [q-fin.TR], 2026.
> https://doi.org/10.48550/arXiv.2608.00885

This repository is **not affiliated with the paper's author**. It is a
personal research/learning project: independently implemented simulations
used to reproduce and stress-test the paper's closed-form results, not a
copy of the paper itself. The paper's PDF is intentionally not included
here — see the DOI/arXiv link above for the original.

## Contents

- `src/simulate_ou.py` — Ornstein-Uhlenbeck gap simulation (Euler-Maruyama).
- `src/trading_strategy.py` — threshold (buy/sell-band) trading rule and
  reward accounting.
- `src/monte_carlo_sweep.py` — Monte Carlo threshold sweeps, the exact
  Dawson-function optimum vs. the leading-order closed form, paired
  significance tests.
- `src/simulate_discrete.py` — discrete tick-jump mid-price model (mid
  moves in ticks, efficient price is continuous Brownian motion).
- `src/discrete_threshold_sweep.py`, `src/discrete_holdout.py`,
  `src/discrete_robustness.py` — threshold optimization on the discrete
  model, an out-of-sample holdout confirmation, and robustness checks
  across tick size and time step.
- `paper/notes/research-log` — full research log: objectives, results,
  and interpretation for every experiment in chronological order.
- `paper/notes/*.png` — generated figures from the simulations above.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

Then run any script directly, e.g. `python src/simulate_ou.py`.
