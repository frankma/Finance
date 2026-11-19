import json
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
from src.CardGames.Holdem.Range import betting_weights
from src.CardGames.Holdem.Round import Street
from src.CardGames.Holdem.Strategy import MonteCarloStrategy
from src.CardGames.Play.BlackJackSession import BlackJackSession
from src.CardGames.Play.GuanDanSession import GuanDanSession
from src.CardGames.Play.HoldemLessons import _heads_up
from src.CardGames.Play.HoldemSession import HoldemSession
from test.CardGames.BlackJack.test_Round import stacked_shoe
from test.CardGames.Holdem.test_Round import stacked
from test.CardGames.Play.helpers import stack_blackjack, stack_guandan, stack_holdem

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


def holdem_pair():
    # seat 0 holds the same aces; the other seats' hole cards and the burns differ, the board does not
    board = '2C 7D 9H TS 3C'
    return [HoldemRound([100, 100, 100], HoldemRules(1, 2), shoe=stacked(['AS AH'] + others, board))
            for others in (['KS KH', 'QS QH'], ['7S 2H', '8C 8D'])]


def guandan_pair():
    # the two opponents trade a card, so they hold different hands of the same size; with two packs a card may also be
    # in your hand, so they trade cards you do not hold, which never reach your page
    hands = deal(random.Random(4))
    give = next(card for card in hands[1] if card not in hands[3] and card not in hands[0])
    take = next(card for card in hands[3] if card not in hands[1] and card not in hands[0])
    swapped = [list(hand) for hand in hands]
    swapped[1].remove(give)
    swapped[1].append(take)
    swapped[3].remove(take)
    swapped[3].append(give)
    return hands, swapped, (give, take)


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
        games = holdem_pair()

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

    def test_holdem_reading_bets(self):
        # the coach that weighs hands by public bets: the same bets over different hidden cards give the same advice
        games = holdem_pair()

        def advisor():
            return MonteCarloStrategy(n_sims=200, rng=random.Random(3), reads_betting=True)

        for game in games:
            game.apply(0, Move(HoldemAction.CALL, 2))
            game.apply(1, Move(HoldemAction.RAISE, 8))
            game.apply(2, Move(HoldemAction.CALL, 8))
        self.assertIsNotNone(betting_weights(games[0].observation(0)))
        assert_fair_play(games[0], games[1], 0, advisor)

    def test_guandan(self):
        hands, swapped, _ = guandan_pair()
        games = [GuanDanRound(h, LEVEL, 0) for h in (hands, swapped)]
        assert_fair_play(games[0], games[1], 0, PlannerStrategy)
        lead = interpret([min(games[0].hands[0], key=lambda c: c.rank.value)], LEVEL)[0]
        for game in games:
            game.apply(0, lead)
            for seat in (1, 2, 3):
                game.apply(seat, PASS)
        self.assertEqual(len(games[0].observation(0).history), 4)
        assert_fair_play(games[0], games[1], 0, PlannerStrategy)


def blackjack_table(hole: str, seed: int) -> BlackJackSession:
    session = BlackJackSession({}, random.Random(seed))
    stack_blackjack(session, 'TS 9D 6H ' + hole)
    session.table.counter.reset()
    session.act({'type': 'deal', 'bet': 10})
    return session


def holdem_table(other: str, options: dict) -> HoldemSession:
    # heads-up: your aces (and kings in Omaha) against the other seat's cards, dealt alternately from that seat
    session = HoldemSession(dict(options, seats=2), random.Random(5))
    theirs, yours = other.split(), ('AS AH KH TS' if options.get('variant') == 'omaha' else 'AS AH').split()
    stack_holdem(session, ' '.join(c for pair in zip(theirs, yours) for c in pair) + ' 2C 7D 8S 9H')
    session.act({'type': 'deal'})
    return session


def betting_table(them: str) -> HoldemSession:
    # the other seat bets half the pot on the flop over different cards, and you call before it
    session = _heads_up('AH 5H', them, 'KH 9H 2C 4S JD', {Street.FLOP: 0.5})(random.Random(14))
    session.act({'type': 'move', 'move': 'call'})
    return session


def guandan_table(hands) -> GuanDanSession:
    session = GuanDanSession({}, random.Random(8))
    stack_guandan(session, hands)
    session.act({'type': 'deal'})
    return session


class TestTheTablePage(TestCase):
    """
    the rule at the play server: two tables alike in everything your seat can see and different in cards it cannot
    give the page the same view and the same advice, and never one of those cards
    """

    def test_the_page_shows_only_what_your_seat_sees(self):
        hands, swapped, traded = guandan_pair()
        cases = [
            ('blackjack: the hole card and the shoe behind it', [blackjack_table('5C', 5), blackjack_table('KC', 6)],
             ['5C', 'KC'], ('shoe',)),  # the shoes differ in order only, which the page never learns
            ("hold'em: the other seat's cards", [holdem_table(o, {}) for o in ('2D 7C', 'KD KC')],
             ['2D', '7C', 'KD', 'KC'], ()),
            ("hold'em: a bet read from the betting alone", [betting_table(t) for t in ('QC 8D', 'KS KD')],
             ['QC', '8D', 'KS', 'KD'], ()),
            ("Omaha: the other seat's four cards",
             [holdem_table(o, {'variant': 'omaha'}) for o in ('2D 7C 3C 9D', 'KD KC QD QS')],
             ['2D', '7C', '3C', '9D', 'KD', 'KC', 'QD', 'QS'], ()),
            ("GuanDan: the opponents' hands", [guandan_table(h) for h in (hands, swapped)], [str(c) for c in traded],
             ()),
        ]
        for name, tables, hidden, unseen in cases:
            with self.subTest(name):
                pages = []
                for table in tables:
                    view = table.view()
                    for key in unseen:
                        view[key] = None
                    text = json.dumps(view, sort_keys=True)
                    for card in hidden:
                        self.assertNotIn('"%s"' % card, text)
                    pages.append((text, table.advice()))
                self.assertEqual(pages[0], pages[1])
        self.assertTrue(betting_table('QC 8D').advice()['read_betting'], 'the coach did read the bet.')
