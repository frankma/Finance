from unittest import TestCase

from src.CardGames.Core.Card import Card, Rank, Suit, standard_deck

__author__ = 'frank.ma'


class TestCard(TestCase):
    def test_parse_round_trip(self):
        for card in standard_deck(jokers=True):
            self.assertEqual(Card.parse(str(card)), card, 'incorrect card reading: %s' % card)

    def test_parse_forms(self):
        self.assertEqual(Card.parse('10h'), Card(Rank.TEN, Suit.HEART))
        self.assertEqual(Card.parse(' as '), Card(Rank.ACE, Suit.SPADE))
        self.assertEqual(Card.parse('rj'), Card(Rank.RED_JOKER))
        self.assertEqual(Card.parse_many('AS KD 2C'), [Card(Rank.ACE, Suit.SPADE), Card(Rank.KING, Suit.DIAMOND),
                                                       Card(Rank.TWO, Suit.CLUB)])

    def test_parse_rejects_bad_input(self):
        for text in ('', 'A', '1S', 'AX', 'ASS', 'XJ'):
            with self.assertRaises(ValueError, msg=text):
                Card.parse(text)

    def test_joker_has_no_suit(self):
        with self.assertRaises(ValueError):
            Card(Rank.BLACK_JOKER, Suit.SPADE)
        with self.assertRaises(ValueError):
            Card(Rank.ACE)

    def test_standard_deck(self):
        self.assertEqual(len(standard_deck()), 52)
        self.assertEqual(len(set(standard_deck())), 52)
        self.assertEqual(len(standard_deck(jokers=True)), 54)

    def test_sort_key(self):
        cards = Card.parse_many('RJ AS 2C KD BJ')
        self.assertEqual([str(card) for card in sorted(cards, key=Card.sort_key)], ['2C', 'KD', 'AS', 'BJ', 'RJ'])
