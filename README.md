[![tests](https://github.com/frankma/Finance/actions/workflows/tests.yml/badge.svg)](https://github.com/frankma/Finance/actions/workflows/tests.yml)

Quantitative finance and related topics in implementation.

## Contents

| Folder | What it is |
|---|---|
| [src/Utils/Valuator](src/Utils/Valuator) | Option pricers: BSM, Black76, Bachelier and normal model, BAW, binomial tree, variance replication, cash flow discounting |
| [src/SABRModel](src/SABRModel) | SABR lognormal and normal approximations, Greeks, calibration |
| [src/HestonModel](src/HestonModel) | Heston stochastic volatility model |
| [src/VolSurface](src/VolSurface) | Equity volatility surfaces |
| [src/SolverFD](src/SolverFD) / [src/SolverMC](src/SolverMC) | Crank-Nicolson finite differences and a Monte Carlo path simulator |
| [src/CDS](src/CDS) | ISDA yield curve |
| [src/Algorithms](src/Algorithms) | Volatility crush around events, asset mixing |
| [src/Utils](src/Utils) | Root finders and regression, interpolation and extrapolation, random sequences, FX quotes, payoffs |
| [src/Assets](src/Assets) / [src/Quote](src/Quote) / [src/DataObject](src/DataObject) | Option, quote and market data objects |
| [src/CardGames](src/CardGames) | A shared card, shoe, counter and game core, with blackjack (basic strategy, hi-lo, Kelly sizing, EV and bankroll simulation), no-limit hold'em and Omaha (side pots, equity, outs, pot odds) and GuanDan (combinations, tribute, levels, a planning bot) |
| [src/MiniProjects](src/MiniProjects) | Delta hedging, Greeks P&L, PCA, panel regression, zero curve, scenarios, heat equation, maze and sudoku solvers |
| [viz](viz) | Interactive pages, e.g. the [SABR smile lab](viz/sabr_smile_lab.html) |
| [doc](doc) | LaTeX notes on the models |

## Running the tests

```bash
pip install -r requirements.txt
MPLBACKEND=Agg python -m pytest test
```

`MPLBACKEND=Agg` keeps the plotting tests from opening windows. CI runs the same suite on Python 3.11, 3.12 and 3.13 for every push to `master` and every pull request.
