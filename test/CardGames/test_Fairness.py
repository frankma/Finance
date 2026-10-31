import random
from unittest import TestCase

from src.CardGames.BlackJack.BasicStrategy import BasicStrategy
from src.CardGames.BlackJack.Counting import hi_lo_counter
from src.CardGames.BlackJack.Round import Action as BlackJackAction, Round as BlackJackRound
from src.CardGames.BlackJack.Rules import Rules as BlackJackRules
from src.CardGames.Core.Card import Rank
from src.CardGames.Core.Fairness import assert_fair_play
from src.CardGames.Core.Player import Strategy
from src.CardGames.GuanDan.Combo import interpret
from src.CardGames.GuanDan.Round import PASS, Round as GuanDanRound, deal
from src.CardGames.GuanDan.Strategy import PlannerStrategy
from src.CardGames.Holdem.Round import Action as HoldemAction, Move, Round as HoldemRound
from src.CardGames.Holdem.Rules import Rules as HoldemRules
from src.CardGames.Holdem.Strategy import MonteCarloStrategy
from test.CardGames.BlackJack.test_Round import stacked_shoe
from test.CardGames.Holdem.test_Round import stacked

__author__ = 'frank.ma'

LEVEL = Rank.TWO

def blackjack_pair():
    # the same two player cards and dealer up card; the hole card and the rest of the shoe differ
    games = []
    for order in ('TS 9D 6H 5C', 'TS 9D 6H KC'):
        shoe = stacked_shoe(order)
        counter = hi_lo_counter(shoe.composition)
        shoe.watch(counter)
        games.append(BlackJackRound(shoe, BlackJackRules(), counter=counter))
    return games


class Peeker(Strategy):
    # a cheat: it was handed the game itself and reads the dealer's hole card
    def __init__(self, game):
        self.game = game

    def decide(self, observation, legal_moves):
        return BlackJackAction.HIT if self.game.hole.rank.value < 10 else BlackJackAction.STAND


class LeakyRound(BlackJackRound):
    # a broken game: its observation carries the face-down card
    def observation(self, seat=0):
        return super().observation(seat), self.hole


class TestFairPlay(TestCase):
    def test_blackjack(self):
        game_a, game_b = blackjack_pair()
        assert_fair_play(game_a, game_b, 0, lambda: BasicStrategy(BlackJackRules()))
        self.assertIn('basic strategy', BasicStrategy(BlackJackRules()).advise(
            game_a.observation(0), game_a.legal_moves(0)).reason)

    def test_detects_a_peeking_strategy(self):
        game_a, game_b = blackjack_pair()
        games = iter([game_a, game_b])
        with self.assertRaises(AssertionError):
            assert_fair_play(game_a, game_b, 0, lambda: Peeker(next(games)))

    def test_detects_a_leaking_observation(self):
        shoes = [stacked_shoe(order) for order in ('TS 9D 6H 5C', 'TS 9D 6H KC')]
        game_a, game_b = (LeakyRound(shoe, BlackJackRules()) for shoe in shoes)
        with self.assertRaises(AssertionError):
            assert_fair_play(game_a, game_b, 0, lambda: BasicStrategy(BlackJackRules()))

    def test_holdem(self):
        # seat 0 holds the same aces; the other seats' hole cards and the burns differ, the board does not
        board = '2C 7D 9H TS 3C'
        games = [HoldemRound([100, 100, 100], HoldemRules(1, 2), shoe=stacked(['AS AH'] + others, board))
                 for others in (['KS KH', 'QS QH'], ['7S 2H', '8C 8D'])]

        def advisor():
            return MonteCarloStrategy(n_sims=200, rng=random.Random(3))

        assert_fair_play(games[0], games[1], 0, advisor)
        for game in games:
            game.apply(0, Move(HoldemAction.CALL, 2))
            game.apply(1, Move(HoldemAction.CALL, 2))
            game.apply(2, Move(HoldemAction.CHECK))
            game.apply(1, Move(HoldemAction.CHECK))
            game.apply(2, Move(HoldemAction.CHECK))
        view = games[0].observation(0)
        self.assertEqual((view.stacks, view.in_hand, len(view.history)), ((98, 98, 98), (True, True, True), 5))
        assert_fair_play(games[0], games[1], 0, advisor)

    def test_guandan(self):
        # the two opponents trade cards, so they hold different hands of the same size
        hands = deal(random.Random(4))
        give = next(card for card in hands[1] if card not in hands[3])
        take = next(card for card in hands[3] if card not in hands[1])
        swapped = [list(hand) for hand in hands]
        swapped[1].remove(give)
        swapped[1].append(take)
        swapped[3].remove(take)
        swapped[3].append(give)
        games = [GuanDanRound(h, LEVEL, 0) for h in (hands, swapped)]
        assert_fair_play(games[0], games[1], 0, PlannerStrategy)
        lead = interpret([min(games[0].hands[0], key=lambda c: c.rank.value)], LEVEL)[0]
        for game in games:
            game.apply(0, lead)
            for seat in (1, 2, 3):
                game.apply(seat, PASS)
        self.assertEqual(len(games[0].observation(0).history), 4)
        assert_fair_play(games[0], games[1], 0, PlannerStrategy)

