"""
build viz/card_games_lab.html: run the CardGames engines once with fixed seeds, then write the numbers into the
page template together with the paper theme, so the page itself carries no game logic

    python -m viz.build_card_games_lab            # full run, several minutes
    python -m viz.build_card_games_lab --quick    # small samples, seconds, for checking the page
    python -m viz.build_card_games_lab --reuse    # keep the page's numbers, refill an edited template
    python -m viz.build_card_games_lab --only guandan    # rerun one game, keep the other numbers

every number on the page comes from src/CardGames; the assistants it describes see only what their seat could know
"""
import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy
from src.CardGames.BlackJack.Counting import FlatBetSizer, KellyBetSizer
from src.CardGames.BlackJack.Round import Observation
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.BlackJack.Table import ev_by_true_count, simulate_bankroll
from src.CardGames.Core.Card import Card, Rank, Suit
from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import RandomStrategy
from src.CardGames.Core.Simulator import MonteCarloResult
from src.CardGames.GuanDan.Combo import BLACK_JOKER, RED_JOKER, is_wild
from src.CardGames.GuanDan.Match import Match
from src.CardGames.GuanDan.Planner import plan_hand
from src.CardGames.GuanDan.Round import N_SEATS, Round as GuanDanRound, deal, team
from src.CardGames.GuanDan.Strategy import LowestPlay, PlannerStrategy
from src.CardGames.Holdem.Equity import outs, simulate_equity

__author__ = 'frank.ma'

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'card_games_lab.template.html')
THEME = os.path.join(HERE, 'theme.css')
OUTPUT = os.path.join(HERE, 'card_games_lab.html')

S17 = Rules()
H17 = Rules(dealer_hits_soft_17=True, surrender=True)
RULES = {'s17': S17, 'h17': H17}
DEALER = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']

# sample sizes for a full build; --quick divides them
FULL = {'ev_rounds': 1000000, 'bankroll_paths': 120, 'bankroll_rounds': 4000, 'hand_sims': 1000, 'spot_sims': 6000,
        'guandan_hands': 3000, 'guandan_versus': 300, 'guandan_matches': 100}


def _rules_dict(rules: Rules) -> dict:
    return {'n_decks': rules.n_decks, 'dealer_hits_soft_17': rules.dealer_hits_soft_17,
            'blackjack_pays': rules.blackjack_pays, 'double_after_split': rules.double_after_split,
            'surrender': rules.surrender, 'penetration': rules.penetration}


def blackjack_ev(rules: Rules, n_rounds: int, seed: int, max_count: int = 6) -> dict:
    """
    flat-bet EV by hi-lo true count, and the line fitted through the buckets: edge = base + slope * true count
    """
    strategy = BasicStrategy(rules)
    buckets = ev_by_true_count(rules, strategy, n_rounds, random.Random(seed), max_count)
    rows = [dict(count=count, share=result.n / n_rounds, **result.to_dict()) for count, result in buckets.items()]
    # weighted least squares over the well-populated buckets, weight 1 / variance of each bucket mean
    fit = [r for r in rows if abs(r['count']) <= 4 and r['n'] > 100]
    x = np.array([r['count'] for r in fit], dtype=float)
    y = np.array([r['mean'] for r in fit])
    w = np.array([1.0 / max(r['stderr'], 1e-9) ** 2 for r in fit])
    slope, base = np.polyfit(x, y, 1, w=np.sqrt(w))
    # the overall return pools the same rounds: mean and variance from each bucket's count, mean and spread
    n = sum(r['n'] for r in rows)
    mean = sum(r['n'] * r['mean'] for r in rows) / n
    second = sum(r['n'] * (r['std'] ** 2 + r['mean'] ** 2) for r in rows) / n
    std = float(np.sqrt(second - mean ** 2))
    overall = {'mean': mean, 'stderr': float(std / np.sqrt(n)), 'std': std, 'n': n}
    return {'rules': _rules_dict(rules), 'buckets': rows, 'fit': {'base': float(base), 'slope': float(slope)},
            'overall': overall}


def basic_strategy_chart(rules: Rules) -> dict:
    """
    the strategy table's entries for every hard total, soft total and pair against each dealer up card
    """
    strategy = BasicStrategy(rules)
    suits = [Suit.SPADE, Suit.HEART]

    def entry(ranks: List[Rank], up: str) -> str:
        cards = tuple(Card(rank, suits[k % 2]) for k, rank in enumerate(ranks))
        observation = Observation(cards, False, Card.parse(up + 'C'), 0, (cards,), rules)
        return strategy.code(observation)

    def rank_of(value: int) -> Rank:
        return Rank.ACE if value == 11 else Rank(value)

    hard = {t: [entry([rank_of(min(10, t - 2)), rank_of(t - min(10, t - 2))], up) for up in DEALER]
            for t in range(8, 18)}
    soft = {t: [entry([Rank.ACE, rank_of(t - 11)], up) for up in DEALER] for t in range(13, 21)}
    pairs = {v: [entry([rank_of(v), rank_of(v)], up) for up in DEALER] for v in range(2, 12)}
    return {'dealer': DEALER, 'hard': hard, 'soft': soft, 'pairs': pairs}


def blackjack_bankroll(fit: dict, variance: float, n_paths: int, n_rounds: int, seed: int) -> dict:
    """
    bankroll paths for flat betting and for fractions of Kelly on the edge fitted from the true count
    """
    bankroll, min_bet, max_bet = 1000.0, 1.0, 100.0
    schemes = [('Flat 1 unit', FlatBetSizer(min_bet))] + [
        (label, KellyBetSizer(min_bet, max_bet, fit['base'], fit['slope'], variance, multiplier))
        for label, multiplier in (('Half Kelly', 0.5), ('Kelly', 1.0), ('Double Kelly', 2.0))]
    steps = np.unique(np.linspace(0, n_rounds, 101).astype(int))
    out = []
    for k, (label, sizer) in enumerate(schemes):
        result = simulate_bankroll(S17, BasicStrategy(S17), sizer, bankroll, n_rounds, n_paths,
                                   random.Random(seed + k))
        bands = np.percentile(result.paths[:, steps], [5, 25, 50, 75, 95], axis=0)
        final = result.final
        out.append({'label': label, 'steps': steps.tolist(), 'bands': bands.round(2).tolist(),
                    'ruin': result.ruin_probability, 'final_mean': float(final.mean()),
                    'final_median': float(np.median(final)), 'p_loss': float((final < bankroll).mean())})
    return {'bankroll': bankroll, 'min_bet': min_bet, 'max_bet': max_bet, 'n_paths': n_paths, 'n_rounds': n_rounds,
            'variance': variance, 'schemes': out}


def starting_hands(n_sims: int, seed: int, max_opponents: int = 5) -> dict:
    """
    equity of every starting hand class (pairs, suited, offsuit) against 1 to max_opponents random hands
    """
    rng = random.Random(seed)
    names = {14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T'}
    ranks = list(range(14, 1, -1))
    grid = {}
    for i, hi in enumerate(ranks):
        for j, lo in enumerate(ranks):
            # upper triangle suited, lower offsuit, the diagonal pairs
            high, low, suited = (hi, lo, True) if j > i else (lo, hi, False) if j < i else (hi, hi, False)
            hole = [Card(Rank(high), Suit.SPADE), Card(Rank(low), Suit.SPADE if suited else Suit.HEART)]
            label = names.get(high, str(high)) + names.get(low, str(low)) + ('s' if suited else 'o' if high != low else '')
            grid[label] = [round(simulate_equity(hole, [], n, n_sims, rng=rng).equity.mean, 4)
                           for n in range(1, max_opponents + 1)]
    return {'ranks': [names.get(r, str(r)) for r in ranks], 'equity': grid, 'n_sims': n_sims,
            'max_opponents': max_opponents}


SPOTS = [('Flush draw and two overcards', 'AS KS', '2S 7S 9H'),
         ('Open-ended straight draw', '8C 9D', '6S 7H KC'),
         ('Gutshot straight draw', '8C 9D', '5S 6H KC'),
         ('Top pair, good kicker', 'AH KD', 'AS 7H 2C'),
         ('Underpair to the board', '5C 5D', 'KS QH 9C')]


def flop_spots(n_sims: int, seed: int) -> List[Dict]:
    """
    common flop situations: outs, the chance to improve by the river, and equity against one random hand
    """
    rng = random.Random(seed)
    rows = []
    for name, hole, board in SPOTS:
        h, b = Card.parse_many(hole), Card.parse_many(board)
        o = outs(h, b)
        equity = simulate_equity(h, b, 1, n_sims, rng=rng)
        rows.append({'name': name, 'hole': hole, 'board': board, 'outs': o.n_outs, 'p_turn': o.p_next_card,
                     'p_river': o.p_by_river(2), 'equity': equity.equity.mean, 'stderr': equity.equity.stderr})
    return rows


def guandan_versus(n_hands: int, seed: int) -> List[Dict]:
    """
    a planning team against a team playing by a simple rule, on independent deals at level 2 with a random leader:
    the share of hands won and the levels gained per hand
    """
    rng = random.Random(seed)
    planner = PlannerStrategy()
    rows = []
    for name, baseline in (('Random legal plays', RandomStrategy(rng)), ('Always the lowest play', LowestPlay())):
        wins, levels = [], []
        seats = [planner, baseline, planner, baseline]
        for _ in range(n_hands):
            game = GuanDanRound(deal(rng), Rank.TWO, rng.randrange(N_SEATS))
            payoffs = play_out(game, seats)
            wins.append(float(game.result.winning_team == 0))
            levels.append(payoffs[0])
        rows.append({'name': name, 'win': MonteCarloResult.from_samples(wins).to_dict(),
                     'levels': MonteCarloResult.from_samples(levels).to_dict()})
    return rows


def _starting_rows(samples: Dict[int, List[float]], total: int, low: int, high: int) -> List[Dict]:
    # one row per value of a feature of the dealt hand, the tails pooled into their end values
    rows = []
    for value in sorted(samples):
        result = MonteCarloResult.from_samples(samples[value])
        label = str(value)
        if value == low and low > 0:
            label = '%i or fewer' % value
        elif value == high:
            label = '%i or more' % value
        rows.append({'value': value, 'label': label, 'share': len(samples[value]) / total, 'win': result.mean,
                     'stderr': result.stderr})
    return rows


def guandan_hands(n_hands: int, seed: int) -> dict:
    """
    planners in all four seats on independent deals at level 2 with a random leader: how often a seat's team wins the
    hand given what that seat could read from its own 27 cards before the first play, and how hands end
    """
    rng = random.Random(seed)
    planner = [PlannerStrategy()] * N_SEATS
    limits = {'turns': (5, 11), 'bombs': (0, 4), 'wilds': (0, 3), 'jokers': (0, 3)}  # no hand holds a third wild
    samples = {key: {} for key in limits}  # type: Dict[str, Dict[int, List[float]]]
    ups = {1: 0, 2: 0, 3: 0}
    leads = []
    for _ in range(n_hands):
        hands = deal(rng)
        leader = rng.randrange(N_SEATS)
        plans = [plan_hand(hand, Rank.TWO) for hand in hands]
        features = [{'turns': plan.turns, 'bombs': plan.bombs, 'wilds': sum(is_wild(c, Rank.TWO) for c in hand),
                     'jokers': sum(c.rank.value in (BLACK_JOKER, RED_JOKER) for c in hand)}
                    for hand, plan in zip(hands, plans)]
        game = GuanDanRound(hands, Rank.TWO, leader)
        play_out(game, planner)
        winners = game.result.winning_team
        ups[game.result.levels_up] += 1
        leads.append(float(team(leader) == winners))
        for seat in range(N_SEATS):
            won = float(team(seat) == winners)
            for key, (low, high) in limits.items():
                value = min(high, max(low, features[seat][key]))
                samples[key].setdefault(value, []).append(won)
    total = n_hands * N_SEATS
    return {
        'starting': {key: _starting_rows(samples[key], total, *limits[key]) for key in limits},
        'levels_up': {str(k): v / n_hands for k, v in ups.items()},
        'leader': MonteCarloResult.from_samples(leads).to_dict(),
        'n_hands': n_hands,
    }


def guandan_matches(n_matches: int, seed: int) -> dict:
    """
    whole matches between planners, tribute included: how long they run, and how often the team that won a hand wins
    the next one, after the tribute it receives
    """
    rng = random.Random(seed)
    planner = [PlannerStrategy()] * N_SEATS
    lengths, repeats = [], []
    for _ in range(n_matches):
        match = Match(random.Random(rng.random()))
        match.play(planner)
        lengths.append(len(match.records))
        winners = [record.result.winning_team for record in match.records]
        repeats += [float(a == b) for a, b in zip(winners, winners[1:])]
    return {'n_matches': n_matches, 'hands_mean': float(np.mean(lengths)),
            'hands_percentiles': [float(p) for p in np.percentile(lengths, [10, 50, 90])],
            'winner_repeats': MonteCarloResult.from_samples(repeats).to_dict()}


def guandan(sizes: dict) -> dict:
    return {'versus': guandan_versus(sizes['guandan_versus'], seed=61),
            'hands': guandan_hands(sizes['guandan_hands'], seed=71),
            'matches': guandan_matches(sizes['guandan_matches'], seed=81)}


def render(data: dict, output: str = OUTPUT) -> None:
    with open(TEMPLATE, encoding='utf-8') as f:
        page = f.read()
    with open(THEME, encoding='utf-8') as f:
        theme = f.read()
    page = page.replace('/*__THEME__*/', theme).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    with open(output, 'w', encoding='utf-8') as f:
        f.write(page)


def read_data(page_path: str = OUTPUT) -> dict:
    # the numbers embedded in a built page
    with open(page_path, encoding='utf-8') as f:
        return json.loads(f.read().split('const DATA = ', 1)[1].split(';\n', 1)[0])


def build(quick: bool = False, output: str = OUTPUT) -> dict:
    sizes = {k: max(1, v // (40 if quick else 1)) for k, v in FULL.items()}
    if quick:
        sizes['bankroll_paths'], sizes['bankroll_rounds'], sizes['hand_sims'] = 8, 300, 60
    start = time.time()
    ev = {key: blackjack_ev(rules, sizes['ev_rounds'], seed=11 + k) for k, (key, rules) in enumerate(RULES.items())}
    variance = ev['s17']['overall']['std'] ** 2
    data = {
        'built': time.strftime('%Y-%m-%d'),
        'quick': quick,
        'sizes': sizes,
        'blackjack': {
            'ev': ev,
            'strategy': {key: basic_strategy_chart(rules) for key, rules in RULES.items()},
            'bankroll': blackjack_bankroll(ev['s17']['fit'], variance, sizes['bankroll_paths'],
                                           sizes['bankroll_rounds'], seed=31),
        },
        'holdem': {
            'hands': starting_hands(sizes['hand_sims'], seed=41),
            'spots': flop_spots(sizes['spot_sims'], seed=51),
        },
        'guandan': guandan(sizes),
    }
    data['seconds'] = round(time.time() - start, 1)
    render(data, output)
    return data


def rerun(game: str, output: str = OUTPUT) -> dict:
    # recompute one game's numbers in a built page, keeping every other number and the page's sample sizes
    began = time.time()
    data = read_data(output)
    data['sizes'] = dict(FULL, **data['sizes'])
    data[game] = {'guandan': guandan}[game](data['sizes'])
    data['built'] = time.strftime('%Y-%m-%d')
    data['seconds'] = round(data['seconds'] + time.time() - began, 1)
    render(data, output)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='small samples, for checking the page')
    parser.add_argument('--reuse', action='store_true', help='keep the numbers in the built page, refill the template')
    parser.add_argument('--only', choices=['guandan'], help='rerun this game, keep the other numbers in the built page')
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    if args.only:
        rerun(args.only, args.output)
        print('reran %s in %s' % (args.only, args.output))
    elif args.reuse:
        render(read_data(args.output), args.output)
        print('refilled %s' % args.output)
    else:
        built = build(args.quick, args.output)
        print('wrote %s in %.1f s' % (args.output, built['seconds']))
