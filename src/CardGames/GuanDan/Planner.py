from dataclasses import dataclass
from typing import List, Tuple

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.GuanDan.Combo import BLACK_JOKER, RED_JOKER, group_value, is_wild, rank_name

__author__ = 'frank.ma'


@dataclass(frozen=True)
class HandPlan(object):
    """
    a rough split of a hand into combinations: how many plays it needs and a cost, lower for a stronger hand
    """
    cost: float
    turns: int
    bombs: int
    sequences: Tuple[str, ...]


def _group_cost(value: int) -> float:
    # a group below the ace costs 10 to get rid of, the level rank and jokers cost less
    return 10.0 - max(0, value - 10) * 1.3


def _leaf(counts: List[int], black: int, red: int, n_wild: int, level: Rank) -> Tuple[float, int, int]:
    # plan the cards left once sequences are taken out: singles, pairs, triples (with pairs as full houses) and bombs
    singles, pairs, trips, bombs = [], [], [], []
    for r in range(2, 15):
        c, v = counts[r], group_value(r, level)
        if c == 1:
            singles.append(v)
        elif c == 2:
            pairs.append(v)
        elif c == 3:
            trips.append(v)
        elif c >= 4:
            bombs.append((c, v))
    joker_bomb = black == 2 and red == 2
    if not joker_bomb:
        for count, v in ((black, BLACK_JOKER), (red, RED_JOKER)):
            if count == 2:
                pairs.append(v)
            elif count == 1:
                singles.append(v)
    singles.sort()
    pairs.sort()
    trips.sort()
    loose = 0
    for _ in range(n_wild):
        # a wild turns a triple into a bomb, else pairs up a low single, else stays a free play
        if trips:
            bombs.append((4, trips.pop()))
        elif singles and singles[0] < 14:
            pairs.append(singles.pop(0))
        else:
            loose += 1
    pairs.sort()
    fulls = min(len(trips), len(pairs))
    pairs = pairs[fulls:]
    cost = sum(_group_cost(v) + 1.0 for v in singles) + sum(_group_cost(v) for v in pairs)
    cost += sum(_group_cost(v) - (2.0 if i < fulls else 1.0) for i, v in enumerate(trips))
    cost -= sum(6.0 + (n - 4) * 1.5 + v / 15.0 for n, v in bombs)
    if joker_bomb:
        cost -= 12.0
    cost -= loose * 7.0
    turns = len(singles) + len(pairs) + len(trips) + loose
    return cost, turns, len(bombs) + (1 if joker_bomb else 0)


_SEQUENCES = (('straight', 5, 1, 9.0), ('tube', 3, 2, 8.0), ('plate', 2, 3, 8.0))


def plan_hand(hand: List[Card], level: Rank) -> HandPlan:
    """
    greedily take out up to four sequences (straights, tubes, plates) wherever that lowers the cost most, then plan
    the rest as groups; it never breaks a bomb into a sequence
    """
    counts = [0] * 15
    black = red = n_wild = 0
    for card in hand:
        if is_wild(card, level):
            n_wild += 1
        elif card.rank.value == BLACK_JOKER:
            black += 1
        elif card.rank.value == RED_JOKER:
            red += 1
        else:
            counts[card.rank.value] += 1

    def ranks(start, length):
        return [14 if q == 1 else q for q in range(start, start + length)]

    best = _leaf(counts, black, red, n_wild, level)
    total, sequence_cost, sequences = best[0], 0.0, []
    for _ in range(4):
        pick = None
        for name, length, per, cost in _SEQUENCES:
            for start in range(1, 16 - length):
                window = ranks(start, length)
                if any(counts[r] < per or counts[r] >= 4 for r in window):
                    continue
                for r in window:
                    counts[r] -= per
                leaf = _leaf(counts, black, red, n_wild, level)
                for r in window:
                    counts[r] += per
                t = sequence_cost + cost + leaf[0]
                if t < total - 0.01 and (pick is None or t < pick[0]):
                    pick = (t, name, window, per, cost, leaf)
        if pick is None:
            break
        total, name, window, per, cost, best = pick
        for r in window:
            counts[r] -= per
        sequence_cost += cost
        sequences.append('%s to %s' % (name, rank_name(window[-1])))
    return HandPlan(total, best[1] + len(sequences), best[2], tuple(sequences))
