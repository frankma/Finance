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
| [src/CardGames](src/CardGames) | A shared card, shoe, counter and game core, with blackjack (basic strategy, hi-lo, Kelly sizing, EV and bankroll simulation), no-limit hold'em and Omaha (side pots, equity, outs, pot odds) and GuanDan (combinations, tribute, levels, a planning bot); [Play](src/CardGames/Play) serves them in the browser with a coach |
| [src/BettingGames](src/BettingGames) | Repeated bets played by many players at once: a bet core with exact distributions, Kelly and pooled staking, and fair-play checks; [CoinFlip](src/BettingGames/CoinFlip) is 1000 players on a +50% / -40% coin, all in, Kelly, and pooled |
| [src/MiniProjects](src/MiniProjects) | Delta hedging, Greeks P&L, PCA, panel regression, zero curve, scenarios, heat equation, maze and sudoku solvers |
| [viz](viz) | Interactive pages: the [SABR smile lab](viz/sabr_smile_lab.html) the [Card Games Lab](viz/card_games_lab.html), which `python -m viz.build_card_games_lab` rebuilds from the CardGames engines, and the [Betting Games Lab](viz/betting_games_lab.html), rebuilt by `python -m viz.build_betting_games_lab` |
| [doc](doc) | LaTeX notes on the models |

### Card games: play and fair play

Play in the browser against the computer, with a coach that suggests each move and says why:

```bash
python -m src.CardGames.Play               # the lobby at http://127.0.0.1:8765
python -m src.CardGames.Play blackjack     # straight to a table: blackjack, holdem or guandan
```

Or double-click [`viz/Play Card Games.command`](viz) in Finder, next to the Card Games Lab: it starts the server and opens the lobby, and the Lab shows Play buttons whenever the server is running. If a card games server is already up on the same code it is reused; if it runs older code, the launcher stops it and starts the current one. All three games are open: blackjack against the dealer, no-limit hold'em or Omaha (heads-up, four- or six-handed) and GuanDan with a partner through a whole match from 2 to A, each against computer seats that play from their own cards only. Each game has a how-to-play page at `/<game>/rules`, linked from the lobby and the Lab, and the same rules open in a panel at the table (the Rules link or `?`). Each game also has guided practice at `/<game>?mode=learn` (six blackjack, four hold'em, four Omaha at `/holdem?mode=learn&variant=omaha` and seven GuanDan lessons): pre-arranged hands that explain each decision and wait for your move. The game runs on a local Python server and the page is only ever sent your seat's view, so hidden cards never reach the browser. Bankrolls, match levels and finished lessons are kept across server restarts in the git-ignored `.card_games/` folder, which holds no cards; a hand in play when the server stops simply does not count (`--no-save` turns this off).


There is also a web version, at [frankma.github.io/Finance/card-games](https://frankma.github.io/Finance/card-games/): the same pages and the same Python, run in the browser by [Pyodide](https://pyodide.org) instead of on a server. It is one project on the repository's GitHub Pages site, whose front page at [frankma.github.io/Finance](https://frankma.github.io/Finance/) links each project in a folder of its own; `python -m viz.build_site _site --base /Finance/` builds the whole site (add a project to `PROJECTS` there), and the `pages` workflow publishes it from `master`. It plays as the local version does, a little slower (the Omaha coach takes about 0.6 s instead of 0.3 s) and after a few seconds' start on each page. The one difference is fair play at the page: the whole game runs in your browser, so the hidden cards sit in the page's memory, where a determined player could find them. The coach and the computer seats still use only what their seat can see, and the pages say so. A hand in play also does not survive reloading the page, as one does not survive restarting the local server.

The card game strategies are there to assist a human player, not to cheat for one. Every assistant sees only what its seat could legitimately know at the table: its own cards, the cards shown, every public action, and what follows from those, such as which cards are still unseen. It never sees another player's hidden cards, a face-down card or the undealt order. Each game's observation is an immutable snapshot, and every game is tested with [`assert_fair_play`](src/CardGames/Core/Fairness.py): two tables that differ only in cards the seat cannot see must give the same view and the same advice.

### Betting games: fair play

The same rule holds for the betting games: a player's stake is set before the coin is flipped, from the public terms of the bet and what that player has seen so far. The outcomes are drawn in advance only so that different strategies face the same luck; a strategy is handed only the rounds already played. [`assert_fair_play`](src/BettingGames/Core/Fairness.py) checks that two tables which agree on everything a player has seen give that player the same stakes.

## Running the tests

```bash
pip install -r requirements.txt
MPLBACKEND=Agg python -m pytest test
```

`MPLBACKEND=Agg` keeps the plotting tests from opening windows. CI runs the same suite on Python 3.11, 3.12 and 3.13 for every push to `master` and every pull request.
