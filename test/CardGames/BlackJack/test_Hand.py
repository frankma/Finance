from unittest import TestCase

from src.CardGames.BlackJack.Dealer import Dealer
from src.CardGames.BlackJack.Hand import Hand
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Card import Card
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'


def hand(cards: str, from_split: bool = False) -> Hand:
    return Hand(Card.parse_many(cards), from_split=from_split)


class TestHand(TestCase):
    def test_totals(self):
        cases = [('AS 6D', 17, True), ('AS 6D TC', 17, False), ('AS AD', 12, True), ('AS AD AH 8C', 21, True),
                 ('KS QD', 20, False), ('9S 7D 5C', 21, False), ('AS AD 9C TH', 21, False)]
        for cards, total, soft in cases:
            self.assertEqual(hand(cards).total, total, cards)
            self.assertEqual(hand(cards).is_soft, soft, cards)

    def test_blackjack_bust_pair(self):
        self.assertTrue(hand('AS KD').is_blackjack)
        self.assertFalse(hand('AS KD', from_split=True).is_blackjack, 'a split 21 is not a blackjack.')
        self.assertFalse(hand('AS 5D 5C').is_blackjack)
        self.assertTrue(hand('KS QD 2C').is_bust)
        self.assertTrue(hand('KS QD').is_pair, 'ten-value cards pair by value.')
        self.assertFalse(hand('KS 9D').is_pair)
        self.assertTrue(hand('9S 9D').is_pair)


class TestDealer(TestCase):
    @staticmethod
    def play(rules: Rules, cards: str, shoe_cards: str) -> Hand:
        shoe = Shoe(cards=Card.parse_many(shoe_cards))
        shoe.cards = list(reversed(Card.parse_many(shoe_cards)))  # deal in the written order
        return Dealer(rules).play(hand(cards), shoe)

    def test_soft_17(self):
        self.assertEqual(self.play(Rules(), 'AS 6D', '5C').total, 17, 'S17 stands on soft 17.')
        self.assertEqual(self.play(Rules(dealer_hits_soft_17=True), 'AS 6D', '3C').total, 20)

    def test_hits_to_17(self):
        self.assertEqual(self.play(Rules(), 'TS 2D', '2C 3H 9S').total, 17)
        self.assertTrue(self.play(Rules(), 'TS 6D', 'KC').is_bust)
