from unittest import TestCase

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy, MimicDealerStrategy
from src.CardGames.BlackJack.Hand import Hand
from src.CardGames.BlackJack.Round import Action, Observation
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Card import Card

__author__ = 'frank.ma'

ALL = [Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT, Action.SURRENDER]
NO_EXTRAS = [Action.STAND, Action.HIT]


def decide(strategy, cards: str, dealer_up: str, legal=ALL) -> Action:
    observation = Observation(Hand(Card.parse_many(cards)), Card.parse(dealer_up), 0, 1)
    return strategy.decide(observation, legal)


class TestBasicStrategy(TestCase):
    def test_hard_totals(self):
        s17 = BasicStrategy(Rules())
        cases = [('5S 3D', '6C', Action.HIT), ('5S 4D', '3C', Action.DOUBLE), ('5S 4D', '7C', Action.HIT),
                 ('6S 5D', 'TC', Action.DOUBLE), ('6S 5D', 'AC', Action.HIT), ('TS 2D', '3C', Action.HIT),
                 ('TS 2D', '4C', Action.STAND), ('TS 6D', '6C', Action.STAND), ('TS 6D', '7C', Action.HIT),
                 ('TS 7D', 'AC', Action.STAND)]
        for cards, up, move in cases:
            self.assertEqual(decide(s17, cards, up, [Action.STAND, Action.HIT, Action.DOUBLE]), move,
                             '%s against %s' % (cards, up))

    def test_soft_totals(self):
        s17 = BasicStrategy(Rules())
        h17 = BasicStrategy(Rules(dealer_hits_soft_17=True))
        self.assertEqual(decide(s17, 'AS 2D', '5C'), Action.DOUBLE)
        self.assertEqual(decide(s17, 'AS 7D', '2C'), Action.STAND)
        self.assertEqual(decide(h17, 'AS 7D', '2C'), Action.DOUBLE)
        self.assertEqual(decide(s17, 'AS 7D', '9C'), Action.HIT)
        self.assertEqual(decide(s17, 'AS 8D', '6C'), Action.STAND)
        self.assertEqual(decide(h17, 'AS 8D', '6C'), Action.DOUBLE)
        self.assertEqual(decide(s17, 'AS 7D', '4C', NO_EXTRAS), Action.STAND, 'soft 18 doubles else stands.')
        self.assertEqual(decide(s17, 'AS 2D', '5C', NO_EXTRAS), Action.HIT, 'soft 13 doubles else hits.')
        self.assertEqual(decide(s17, 'AS 3D 2C 2H', '9C', NO_EXTRAS), Action.HIT, 'a four-card soft 18 still hits.')

    def test_pairs(self):
        das = BasicStrategy(Rules())
        no_das = BasicStrategy(Rules(double_after_split=False))
        self.assertEqual(decide(das, '8S 8D', 'AC'), Action.SPLIT)
        self.assertEqual(decide(das, 'AS AD', 'TC'), Action.SPLIT)
        self.assertEqual(decide(das, 'TS KD', '6C'), Action.STAND)
        self.assertEqual(decide(das, '9S 9D', '7C'), Action.STAND)
        self.assertEqual(decide(das, '5S 5D', '9C'), Action.DOUBLE, 'fives play as a hard 10.')
        self.assertEqual(decide(das, '2S 2D', '2C'), Action.SPLIT)
        self.assertEqual(decide(no_das, '2S 2D', '2C'), Action.HIT)
        self.assertEqual(decide(das, '8S 8D', 'TC', NO_EXTRAS), Action.HIT, 'unsplittable 8 8 is a hard 16.')
        self.assertEqual(decide(das, 'AS AD', '6C', NO_EXTRAS), Action.HIT, 'unsplittable aces are a soft 12.')

    def test_surrender(self):
        h17 = BasicStrategy(Rules(dealer_hits_soft_17=True, surrender=True))
        self.assertEqual(decide(h17, 'TS 6D', 'TC'), Action.SURRENDER)
        self.assertEqual(decide(h17, 'TS 6D', 'TC', NO_EXTRAS), Action.HIT)
        self.assertEqual(decide(h17, 'TS 7D', 'AC'), Action.SURRENDER)
        self.assertEqual(decide(h17, 'TS 7D', 'AC', NO_EXTRAS), Action.STAND)

    def test_mimic_dealer(self):
        self.assertEqual(decide(MimicDealerStrategy(), 'TS 6D', '2C'), Action.HIT)
        self.assertEqual(decide(MimicDealerStrategy(), 'TS 7D', 'AC'), Action.STAND)
