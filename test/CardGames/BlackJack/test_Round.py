from typing import List
from unittest import TestCase

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy
from src.CardGames.BlackJack.Counting import HI_LO, hi_lo_counter
from src.CardGames.BlackJack.Round import Action, Round
from src.CardGames.BlackJack.Rules import Rules
from src.CardGames.Core.Card import Card, Rank, standard_deck
from src.CardGames.Core.Game import play_out
from src.CardGames.Core.Player import Strategy
from src.CardGames.Core.Shoe import Shoe

__author__ = 'frank.ma'


def stacked_shoe(order: str) -> Shoe:
    """
    a shoe dealing the given cards first, in the written order: player, dealer up, player, dealer hole, then draws
    """
    cards = Card.parse_many(order)
    shoe = Shoe(cards=standard_deck() * 2)
    rest = list(shoe.composition)
    for card in cards:
        rest.remove(card)
    shoe.cards = rest + list(reversed(cards))
    return shoe


class Script(Strategy):
    def __init__(self, moves: List[Action]):
        self.moves = list(moves)

    def decide(self, observation, legal_moves):
        return self.moves.pop(0)


class TestRound(TestCase):
    def play(self, order: str, moves: List[Action], rules: Rules = Rules(), bet: float = 1.0) -> float:
        game = Round(stacked_shoe(order), rules, bet)
        return play_out(game, [Script(moves)])[0]

    def test_naturals(self):
        self.assertEqual(self.play('AS 9D KH 7C', []), 1.5)
        self.assertEqual(self.play('AS 9D KH 7C', [], Rules(blackjack_pays=1.2), bet=10.0), 12.0)
        self.assertEqual(self.play('9S AD 8H KC', []), -1.0, 'the dealer peeks under an ace.')
        self.assertEqual(self.play('AS AD KH KC', []), 0.0, 'two naturals push.')

    def test_stand_hit_bust(self):
        self.assertEqual(self.play('TS 9D 9H 8C', [Action.STAND]), 1.0)
        self.assertEqual(self.play('TS 9D 6H 8C 9S', [Action.HIT]), -1.0)
        self.assertEqual(self.play('TS 6D 6H TC 2S 9H', [Action.HIT, Action.STAND]), 1.0, 'dealer 16 draws to 25.')
        self.assertEqual(self.play('TS 7D 7H TC', [Action.STAND]), 0.0)

    def test_double(self):
        self.assertEqual(self.play('6S 5D 5H TC TH 7S', [Action.DOUBLE]), 2.0)
        self.assertEqual(self.play('6S 5D 5H TC 2H 4S', [Action.DOUBLE]), -2.0)

    def test_split(self):
        # 8 8 against a 6: split, each 8 catches a ten, dealer 16 busts
        moves = [Action.SPLIT, Action.STAND, Action.STAND]
        self.assertEqual(self.play('8S 6D 8H TC TH TS TD', moves), 2.0)
        # one card to each split ace, and a split 21 is not a natural
        game = Round(stacked_shoe('AS 9D AH 8C KH KS'), Rules())
        self.assertEqual(game.legal_moves(), [Action.STAND, Action.HIT, Action.DOUBLE, Action.SPLIT])
        game.apply(0, Action.SPLIT)
        self.assertTrue(game.is_terminal())
        self.assertEqual(game.payoffs(), [2.0])

    def test_split_limits(self):
        rules = Rules(max_hands=2, double_after_split=False)
        game = Round(stacked_shoe('8S 6D 8H TC 8C 2D'), rules)
        game.apply(0, Action.SPLIT)
        self.assertEqual(game.legal_moves(), [Action.STAND, Action.HIT], 'no resplit past max_hands nor DAS.')

    def test_surrender(self):
        rules = Rules(surrender=True)
        self.assertEqual(self.play('TS TD 6H 7C', [Action.SURRENDER], rules), -0.5)
        game = Round(stacked_shoe('TS TD 6H 7C'), Rules())
        self.assertNotIn(Action.SURRENDER, game.legal_moves())
        with self.assertRaises(ValueError):
            game.apply(0, Action.SURRENDER)

    def test_hole_card_stays_hidden(self):
        shoe = stacked_shoe('TS 9D 6H 5C')
        counter = hi_lo_counter(shoe.composition)
        shoe.watch(counter)
        game = Round(shoe, Rules(), counter=counter)
        # the player sees T, 9 and 6: -1 + 0 + 1, and not the dealer's 5 until it is turned over
        self.assertEqual(counter.running_count, 0.0)
        observation = game.observation(0)
        self.assertEqual((observation.running_count, observation.true_count), (0.0, 0.0))
        self.assertEqual(dict(observation.unseen)[Rank.FIVE], 8, 'the face-down 5 still counts as unseen.')
        game.apply(0, Action.STAND)
        self.assertEqual(counter.n_seen, len(shoe.face_up))
        self.assertEqual(counter.running_count, sum(HI_LO[card.rank] for card in shoe.face_up))

    def test_play_out_rejects_illegal_moves(self):
        game = Round(stacked_shoe('TS 9D 6H 5C'), Rules())
        with self.assertRaises(ValueError):
            play_out(game, [Script([Action.SURRENDER])])
        with self.assertRaises(ValueError):
            play_out(game, [BasicStrategy(Rules()), BasicStrategy(Rules())])
