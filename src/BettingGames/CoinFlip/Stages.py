"""
the first betting game: 1000 players, 100 rounds of a bet that pays +50% or -40% on a fair coin

    stage 1  everyone stakes all they have every round; rank the 1000 at the end
    stage 2  one more player stakes the Kelly fraction instead; where would they rank?
    stage 3  everyone is sensible: each stakes Kelly on their own, or the table pools and shares

each stage returns plain numbers for the lab page; every table of outcomes is drawn once from a fixed seed, so
strategies compared on one table face exactly the same luck
"""
import math
from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from src.BettingGames.Core.Bet import Bet
from src.BettingGames.Core.Ranking import rank_in, share_below, summary
from src.BettingGames.Core.Strategy import AllIn, FixedFraction, Kelly
from src.BettingGames.Core.Tournament import play

__author__ = 'frank.ma'

BET = Bet(up=0.5, down=0.4, p_up=0.5)
N_PLAYERS = 1000
N_ROUNDS = 100


def sig(x, digits: int = 5):
    # round to significant figures for the page; keeps 1e-9 and 1e4 alike
    if isinstance(x, (list, tuple, np.ndarray)):
        return [sig(v, digits) for v in x]
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return float('%.*g' % (digits, x))


def rounded_summary(wealth: np.ndarray) -> Dict:
    return {k: ({q: sig(x) for q, x in v.items()} if isinstance(v, dict) else sig(v)) for k, v in summary(wealth).items()}


def table(seed: int, n_players: int = N_PLAYERS, n_rounds: int = N_ROUNDS) -> np.ndarray:
    return BET.draw(np.random.default_rng(seed), (n_rounds, n_players))


def exact(fraction: float, n_rounds: int = N_ROUNDS) -> Dict:
    # the closed form for one player staking a fixed fraction
    q = BET.quantiles(fraction, n_rounds, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {'fraction': fraction, 'growth': BET.growth_rate(fraction), 'mean_return': BET.mean_return(fraction),
            'mean': BET.mean_wealth(fraction, n_rounds), 'median': q[2], 'p01': q[0], 'p10': q[1], 'p90': q[3],
            'p99': q[4], 'above_start': BET.prob_above(fraction, n_rounds)}


def _picked(n_players: int, n_paths: int, seed: int) -> np.ndarray:
    # the players whose paths are drawn: spread across the final ranking, so the lucky and unlucky both show
    return np.sort(np.random.default_rng(seed).choice(n_players, size=min(n_paths, n_players), replace=False))


def all_in(seed: int = 101, n_paths: int = 40) -> Dict:
    """
    stage 1: the table of 1000, everyone all in
    """
    wins = table(seed)
    result = play(BET, AllIn(), wins, keep_paths=True)
    final = result.final
    picked = _picked(N_PLAYERS, n_paths, seed + 1)
    k = wins.sum(axis=0)
    return {
        'seed': seed,
        'summary': rounded_summary(final),
        'ranked': sig(np.sort(final)[::-1], 4),
        'wins_ranked': [int(x) for x in k[np.argsort(-final, kind='stable')]],
        'paths': [sig(result.paths[:, i], 4) for i in picked],
        'mean_path': sig(result.paths.mean(axis=1)),
        'median_path': sig(np.median(result.paths, axis=1)),
        'exact': {key: sig(v) for key, v in exact(1.0).items()},
        'breakeven_wins': int(np.argmax(BET.wealth_after(1.0, N_ROUNDS) > 1.0)),
    }


def kelly_outsider(seed: int = 101, outsider_seed: int = 202, n_trials: int = 2000, trial_seed: int = 303,
                   chunk: int = 100) -> Dict:
    """
    stage 2: one Kelly player with luck of their own, ranked against stage 1's table; then the same contest over
    many fresh tables, to see how that rank is spread; and the Kelly stake on each table player's own luck
    """
    wins = table(seed)
    field = play(BET, AllIn(), wins).final
    own = BET.draw(np.random.default_rng(outsider_seed), (N_ROUNDS, 1))
    outsider = play(BET, Kelly(), own, keep_paths=True)
    final = float(outsider.final[0])
    same_luck = play(BET, Kelly(), wins).final

    ranks: List[int] = []
    rng = np.random.default_rng(trial_seed)
    for start in range(0, n_trials, chunk):
        m = min(chunk, n_trials - start)
        draws = BET.draw(rng, (N_ROUNDS, m * (N_PLAYERS + 1)))  # each trial: 1000 all in, then one Kelly player
        tables = draws.reshape(N_ROUNDS, m, N_PLAYERS + 1)
        allin = play(BET, AllIn(), tables[:, :, :N_PLAYERS].reshape(N_ROUNDS, -1)).final.reshape(m, N_PLAYERS)
        kelly = play(BET, Kelly(), tables[:, :, N_PLAYERS]).final
        ranks.extend(rank_in(kelly[j], allin[j]) for j in range(m))
    ranks_arr = np.array(ranks)

    f = BET.kelly_fraction()
    kelly_exact = exact(f)
    return {
        'seed': seed, 'outsider_seed': outsider_seed,
        'fraction': f,
        'final': sig(final),
        'wins': int(own.sum()),
        'rank': rank_in(final, field),
        'beats': sig(share_below(final, field)),
        'path': sig(outsider.paths[:, 0], 4),
        'same_luck': {
            'kelly_beats_all_in': sig(np.mean(same_luck > field)),
            'ranked': sig(np.sort(same_luck)[::-1], 4),
            'summary': rounded_summary(same_luck),
        },
        'trials': {
            'n': n_trials, 'seed': trial_seed,
            'ranks': [int(r) for r in ranks_arr],
            'median_rank': float(np.median(ranks_arr)),
            'top10': sig(np.mean(ranks_arr <= N_PLAYERS // 10)),
            'top1': sig(np.mean(ranks_arr <= N_PLAYERS // 100)),
            'top_half': sig(np.mean(ranks_arr <= N_PLAYERS // 2)),
        },
        # where the typical Kelly result sits in the exact all-in distribution
        'exact': {key: sig(v) for key, v in kelly_exact.items()},
        'exact_beats': sig(1.0 - BET.prob_above(1.0, N_ROUNDS, kelly_exact['median'])),
    }


def fraction_curve(step: float = 0.01) -> List[Dict]:
    """
    the closed form at every stake from 0 to all in: the page's slider reads it
    """
    return [{key: sig(v) for key, v in exact(round(f, 6)).items()} for f in np.arange(0.0, 1.0 + step / 2, step)]


def pool_growth(n: int, fraction: float) -> float:
    """
    the exact growth a round, the expected log return, of a pool of n that shares every round: each member stakes
    the fraction on their own coin, so the pool's wealth grows by the fraction times the average of n returns, whose
    win count is binomial. n = 1 is a player alone
    """
    j = np.arange(n + 1)
    avg = (j * BET.up - (n - j) * BET.down) / n  # the pool's return per unit staked
    return float(np.dot(BET.wins_pmf(n), np.log1p(fraction * avg)))


def pooled_fraction(n: int) -> float:
    """
    the growth-optimal (Kelly) stake for a pool of n, at most all in. One player alone gets 25%; a big pool, being a
    far steadier bet, stakes everything
    """
    result = minimize_scalar(lambda f: -pool_growth(n, f), bounds=(0.0, 1.0), method='bounded',
                             options={'xatol': 1e-9})
    return 1.0 if pool_growth(n, 1.0) >= -result.fun else float(result.x)


def growth_curves(pool_sizes: Sequence[int] = (1, 2, 5, 1000), step: float = 0.01) -> List[Dict]:
    """
    growth a round against the stake, one curve per pool size, with each curve's Kelly stake: pooling lifts the
    curve and moves its peak towards all in
    """
    fractions = np.arange(0.0, 1.0 + step / 2, step)
    curves = []
    for n in pool_sizes:
        k = pooled_fraction(n)
        curves.append({'size': int(n), 'kelly': sig(k), 'kelly_growth': sig(pool_growth(n, k)),
                       'growth': sig([pool_growth(n, f) for f in fractions])})
    return curves


def everyone_sensible(seed: int = 101, pool_sizes: Sequence[int] = (1, 2, 5, 10, 50, 100, 1000)) -> Dict:
    """
    stage 3 on stage 1's table: everyone all in (for comparison), everyone Kelly on their own, and the whole table
    pooling: each stakes on their own coin and the 1000 share the result every round, either at the lone player's
    Kelly stake or all in, which is the Kelly stake of the pool's own, far steadier bet. Then pools of other sizes,
    each staking the growth-optimal fraction for its size
    """
    wins = table(seed)
    schemes = []
    kelly = BET.kelly_fraction()
    for key, label, stake, pool in (('all_in', 'Everyone all in', 1.0, None),
                                    ('kelly', 'Everyone Kelly, alone', kelly, None),
                                    ('pooled_quarter', 'Everyone pooled, at 25%', kelly, N_PLAYERS),
                                    ('pooled', 'Everyone pooled, all in', 1.0, N_PLAYERS)):
        result = play(BET, FixedFraction(stake), wins, pool_size=pool, keep_paths=True)
        schemes.append({
            'key': key, 'label': label, 'stake': stake,
            'summary': rounded_summary(result.final),
            'ranked': sig(np.sort(result.final)[::-1], 4),
            'median_path': sig(np.median(result.paths, axis=1)),
            'mean_path': sig(result.paths.mean(axis=1)),
        })

    pools = []
    for n in pool_sizes:
        f = pooled_fraction(n)
        final = play(BET, FixedFraction(f), wins, pool_size=n).final
        pools.append({'size': int(n), 'fraction': sig(f), 'median': sig(np.median(final)), 'mean': sig(final.mean()),
                      'growth': sig(np.log(np.median(final)) / N_ROUNDS)})
    return {
        'seed': seed,
        'schemes': schemes,
        'pools': pools,
        'ensemble_growth': sig(math.log1p(BET.mean_return(1.0))),
        'kelly_growth': sig(BET.growth_rate(BET.kelly_fraction())),
    }



MIX = (('pooled', 'Pooled, all in', 1.0, True),
       ('pooled_quarter', 'Pooled, at 25%', BET.kelly_fraction(), True),
       ('all_in', 'All in, alone', 1.0, False),
       ('kelly', 'Kelly, alone', BET.kelly_fraction(), False))


def mixed_table(seed: int = 101) -> Dict:
    """
    stage 4: one table of 1000 on stage 1's coins, a quarter of each kind of player: a pool of 250 all in, a pool of
    250 at the Kelly stake (each pool shares only among its own members), 250 all in alone and 250 Kelly alone.
    Who ends up where in the one ranking, and who holds the wealth?
    """
    wins = table(seed)
    n = N_PLAYERS // len(MIX)
    finals, kinds = [], []
    for g, (key, label, stake, pooled) in enumerate(MIX):
        cols = wins[:, g * n:(g + 1) * n]
        finals.append(play(BET, FixedFraction(stake), cols, pool_size=n if pooled else None).final)
        kinds.append(np.full(n, g))
    final, kind = np.concatenate(finals), np.concatenate(kinds)
    order = np.argsort(-final, kind='stable')
    rank = np.empty(N_PLAYERS, dtype=int)
    rank[order] = np.arange(1, N_PLAYERS + 1)
    total = final.sum()
    groups = []
    for g, (key, label, stake, pooled) in enumerate(MIX):
        w, r = final[kind == g], rank[kind == g]
        groups.append({
            'key': key, 'label': label, 'stake': stake, 'pooled': pooled, 'n': int(n),
            'summary': rounded_summary(w),
            'best_rank': int(r.min()), 'worst_rank': int(r.max()), 'median_rank': float(np.median(r)),
            'top10': int(np.sum(r <= 10)), 'top100': int(np.sum(r <= 100)), 'top250': int(np.sum(r <= 250)),
            'wealth_share': sig(w.sum() / total),
        })
    return {
        'seed': seed,
        'groups': groups,
        'ranked': sig(final[order], 4),
        'kind_ranked': [int(k) for k in kind[order]],
    }


def levers(n_pool: int = N_PLAYERS) -> Dict:
    """
    the two levers on the drag that volatility puts on growth, g(f) ~ f mu - f^2 sigma^2 / 2: Kelly scales the stake
    down (best f ~ mu / sigma^2), pooling divides the variance by the pool's size (best f ~ mu / (sigma^2 / n), capped
    at all in). The approximations beside the exact answers, for the page's explanation
    """
    mu = BET.mean_return(1.0)
    var = BET.p_up * BET.up ** 2 + BET.p_down * BET.down ** 2 - mu ** 2
    return {
        'mu': sig(mu), 'var': sig(var), 'n_pool': int(n_pool),
        'drag_all_in': sig(var / 2),  # what the approximation takes off the average when all in
        'kelly_approx': sig(mu / var), 'kelly_exact': sig(BET.kelly_fraction()),
        'pool_kelly_approx': sig(mu / (var / n_pool)), 'pool_kelly_exact': sig(pooled_fraction(n_pool)),
        'growth_all_in': sig(BET.growth_rate(1.0)), 'growth_kelly': sig(BET.growth_rate(BET.kelly_fraction())),
        'growth_ensemble': sig(math.log1p(mu)),
    }
