import random
from typing import Dict, List, Optional

from src.CardGames.Core.Card import Card
from src.CardGames.Core.Player import Strategy
from src.CardGames.Core.Shoe import Shoe
from src.CardGames.Holdem.Round import Action, Move, Observation, Street
from src.CardGames.Play.HoldemSession import HoldemSession
from src.CardGames.Play.Lesson import Lesson, Step

__author__ = 'frank.ma'


class Scripted(Strategy):
    """
    a lesson's opponent: bets the given fraction of the pot on the given streets when nobody has bet, and otherwise
    checks or calls, so a lesson plays the same way every time; like any seat it sees only its own observation
    """

    def __init__(self, bets: Optional[Dict[Street, float]] = None):
        self.bets = bets or {}

    def decide(self, observation: Observation, legal_moves: List[Move]) -> Move:
        o = observation
        fraction = self.bets.get(o.street)
        if fraction and o.to_call == 0 and o.min_raise_to is not None:
            to = max(o.min_raise_to, min(o.max_raise_to, int(round(fraction * o.pot))))
            return Move(Action.RAISE, to)
        for move in legal_moves:
            if move.action in (Action.CHECK, Action.CALL):
                return move
        return legal_moves[0]


def _heads_up(you: str, them: str, board: str, bets: Optional[Dict[Street, float]] = None):
    """
    a heads-up table with you on the button, these hole cards and board, and a scripted opponent, dealt and waiting
    for your first decision; the next hands are dealt at random
    """
    def setup(rng: random.Random) -> HoldemSession:
        session = HoldemSession({'seats': 2}, rng)
        session.bots[1] = Scripted(bets)
        session.names[1] = 'Opponent'  # it plays a script, not one of the table's styles
        y, t, b = Card.parse_many(you), Card.parse_many(them), Card.parse_many(board)
        # dealt from the top: the opponent's first card, yours, their second, yours, then burn, flop, burn, turn,
        # burn, river; a burn is any other card
        top = [t[0], y[0], t[1], y[1]]
        shoe = Shoe(1, rng=rng)
        rest = [card for card in shoe.cards if card not in top + b]
        burns = rest[:3]
        order = top + [burns[0]] + b[:3] + [burns[1], b[3], burns[2], b[4]]
        shoe.cards = rest[3:] + list(reversed(order))
        session._shoe = lambda: shoe
        session.act({'type': 'deal'})
        del session._shoe  # later hands are shuffled as usual
        return session
    return setup


def _move(name: str) -> dict:
    return {'type': 'move', 'move': name}


def lessons() -> List[Lesson]:
    return [
        Lesson('price', 'Call when the price is right', 'Compare your equity with the pot odds before calling.',
               _heads_up('AH 5H', 'QC 8D', 'KH 9H 2C 4S JD', {Street.FLOP: 0.5}), (
                   Step(_move('call'),
                        'Heads-up you are on the button, so you post the small blind and act first before the flop. '
                        'Ace-five of hearts is a fine hand against one player. It costs 1 more chip to see the flop. '
                        'Fold, call or raise?',
                        'Right: call. Against one random hand, ace-five suited wins more than half the time, and '
                        'the price is only 1 chip.',
                        'Against a single opponent, ace-five suited is too good to fold for 1 chip. Try calling.'),
                   Step(_move('call'),
                        'The flop brings two more hearts: four to a flush, with the ace. Your opponent bets 2 into '
                        'the pot of 4. Look at the coach\'s numbers: your equity against the pot odds. Fold, call or '
                        'raise?',
                        'Right: call. You need 2 ÷ (6 + 2) = 25% to break even, and a nut flush draw with an ace has '
                        'well over that. Nine hearts are left, about 35% to make the flush by the river alone.',
                        'Your equity is far above the 25% the pot odds ask for: folding gives up a winning call. '
                        'Try calling.'),
               ), 'Call when your equity beats the pot odds; the coach shows both at every decision.',
               ('pot-odds', 'outs')),
        Lesson('fold', 'Fold when the price is wrong', 'A big bet into a hand that missed: let it go.',
               _heads_up('3C 2D', 'KS TD', 'AS KH QD 9C 7H', {Street.FLOP: 1.0}), (
                   Step(_move('call'),
                        'Three-two is one of the weakest hands, but it costs only 1 chip to see the flop against '
                        'one player. Fold, call or raise?',
                        'Right: call. A cheap look is worth taking: the pot gives you 3 to 1.',
                        'One chip to win a pot of 3 is a good price even for a weak hand. Try calling.'),
                   Step(_move('fold'),
                        'The flop is ace, king, queen: your three-two has nothing, and your opponent bets the pot, 4 '
                        'chips. Fold, call or raise?',
                        'Right: fold. A call of 4 to win 12 needs 33% equity, and three-high on this board has about '
                        '19%. Folding costs nothing more; calling would lose chips over time.',
                        'Calling 4 into a pot of 8 needs a third of the pot in equity. Three-high with no draw on an '
                        'ace-king-queen board wins far less often than that. Try folding.'),
               ), 'Folding is a move too: it saves the chips a bad call would lose.', ('pot-odds', 'equity')),
        Lesson('value', 'Bet your strong hands', 'Raise for value when you are well ahead.',
               _heads_up('AS AD', 'JC 9C', 'AC 7D 2S 5H KS'), (
                   Step(_move('raise'),
                        'You are dealt a pair of aces, the best starting hand. Fold, call or raise?',
                        'Right: raise. Aces win about 85% against one random hand, so put more chips in now while '
                        'you are a big favourite.',
                        'With the best starting hand, calling lets your opponent see the flop cheaply. Raise to build '
                        'the pot. Use R or the raise buttons.'),
                   Step(_move('raise'),
                        'The flop gives you three aces. Your opponent checks. Check or bet?',
                        'Right: bet. With three of a kind you are almost certain to win, and a bet gets chips from '
                        'any hand that calls.',
                        'Checking gives up the chance to win more. When you are far ahead, bet. Use R or the bet '
                        'buttons.'),
               ), 'Bet for value when your equity is high; the size can grow with your edge.', ('moves', 'equity')),
        Lesson('free', 'Take a free card', 'When nobody bets and your hand is weak, check.',
               _heads_up('8C 3D', 'QS QD', 'KD JS 5C 9H 2H'), (
                   Step(_move('call'),
                        'Eight-three is weak, but against one player 1 chip to see the flop is a fair price. Fold, '
                        'call or raise?',
                        'Right: call, for the cheap look.',
                        'For 1 chip into a pot of 3, take the look at the flop. Try calling.'),
                   Step(_move('check'),
                        'The flop misses you and your opponent checks. Betting would only be called by hands that '
                        'beat you. Check or bet?',
                        'Right: check. With little equity, a bet risks chips mostly against better hands; checking '
                        'sees the next card for free.',
                        'A bet with nothing is called mostly by hands that beat you. Check and take the free card.'),
               ), 'Check when you are behind and nobody has bet: a free card costs nothing.', ('moves', 'equity')),
    ]
