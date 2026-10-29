import random
from unittest import TestCase

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.Core.Player import RandomStrategy
from src.CardGames.GuanDan.Combo import ComboType, interpret
from src.CardGames.GuanDan.Match import Match, choose_return, pay_tribute, returnable
from src.CardGames.GuanDan.Round import PASS, HandResult, Round
from src.CardGames.GuanDan.Strategy import PlannerStrategy

__author__ = 'frank.ma'

P = Card.parse_many
L2 = Rank.TWO


class TestTribute(TestCase):
    def test_single_win(self):
        # last out (seat 3) pays its highest card that is not wild, the level 2S over the ace, to the first out
        hands = [P('3S 4S 9D KC'), P('5S'), P('6S'), P('2H 2S AS 8C')]
        leader, tributes = pay_tribute(hands, L2, (0, 1, 2, 3))
        self.assertEqual(len(tributes), 1)
        t = tributes[0]
        self.assertEqual((t.payer, t.receiver, t.card), (3, 0, Card.parse('2S')))
        self.assertLessEqual(t.returned.rank.value, 10, 'the card handed back is a 10 or lower.')
        self.assertIn(Card.parse('2S'), hands[0])
        self.assertIn(t.returned, hands[3])
        self.assertEqual(leader, 3, 'the payer leads.')

    def test_double_win(self):
        # seats 0 and 2 out first: both losers pay, the bigger card to the first out, whose payer leads
        hands = [P('3S 4S'), P('KS 5D'), P('3C 4C'), P('BJ 6D')]
        leader, tributes = pay_tribute(hands, L2, (0, 2, 1, 3))
        self.assertEqual([(t.payer, t.receiver, str(t.card)) for t in tributes], [(3, 0, 'BJ'), (1, 2, 'KS')])
        self.assertEqual(leader, 3)

    def test_resist(self):
        hands = [P('3S'), P('RJ 5D'), P('3C'), P('RJ 6D')]
        leader, tributes = pay_tribute(hands, L2, (0, 2, 1, 3))
        self.assertEqual((leader, tributes), (0, []), 'two red jokers among the payers cancel tribute.')
        self.assertEqual(hands[1], P('RJ 5D'))

    def test_returnable(self):
        self.assertEqual(returnable(P('AS KD 9C 2H'), L2), P('9C'))
        self.assertEqual(returnable(P('AS KD 2H BJ'), L2), P('AS KD'), 'no 10 or lower, so any non-wild non-joker.')
        self.assertIn(choose_return(P('3S 4D 5C 6H 7S 9D'), L2), P('9D 3S 4D 5C 6H 7S'))


class TestScoring(TestCase):
    def result(self, order, up):
        first = order[0]
        return HandResult(tuple(order), first % 2, up)

    def test_climb_and_cap(self):
        match = Match(random.Random(0))
        match._score(self.result((1, 3, 0, 2), 3))
        self.assertEqual((match.levels, match.declarer, match.level), ([2, 5], 1, Rank.FIVE))
        match.levels = [2, 13]
        match._score(self.result((1, 3, 0, 2), 3))
        self.assertEqual(match.levels, [2, 14], 'never past A.')

    def test_winning_and_failing_at_a(self):
        match = Match(random.Random(0))
        match.levels, match.declarer = [14, 5], 0
        match._score(self.result((0, 1, 3, 2), 1))
        self.assertEqual((match.winner, match.a_fails[0], match.levels[0]), (None, 1, 14), '1st and 4th is not enough.')
        match.declarer = 0
        match._score(self.result((1, 0, 3, 2), 2))
        match.declarer = 0
        match._score(self.result((3, 0, 1, 2), 3))
        self.assertEqual((match.a_fails[0], match.levels[0]), (0, 2), 'three failures at A drop back to 2.')
        match.levels, match.declarer = [14, 5], 0
        match._score(self.result((0, 1, 2, 3), 2))
        self.assertTrue(match.is_over)
        self.assertEqual(match.winner, 0)

    def test_full_match(self):
        match = Match(random.Random(5))
        winner = match.play([PlannerStrategy()] * 4)
        self.assertIn(winner, (0, 1))
        self.assertEqual(max(match.records[-1].levels_after), 14)
        self.assertTrue(any(record.tributes for record in match.records))
        self.assertEqual(match.records[0].level, Rank.TWO)
        with self.assertRaises(ValueError):
            match.play_hand([PlannerStrategy()] * 4)

    def test_random_match(self):
        match = Match(random.Random(6))
        match.play([RandomStrategy(random.Random(7))] * 4, max_hands=15)
        self.assertGreater(len(match.records), 0)


class TestPlannerStrategy(TestCase):
    def observe(self, hands, leader, plays):
        game = Round([P(h) for h in hands], L2, leader)
        for seat, cards in plays:
            game.apply(seat, PASS if cards is None else interpret(P(cards), L2)[0])
        seat = game.current_seat()
        return game, seat, game.observation(seat), game.legal_moves(seat)

    def test_lets_partner_win(self):
        # seat 2 leads a king, seat 3 passes, seat 0 could top it with an ace but leaves the trick to partner
        hands = ['AS 3D 5C 7H 9S', '4S 6D 8C TH QS', 'KS 3C 5H 7S 9D', '4D 6C 8H TS JD']
        _, _, observation, legal = self.observe(hands, 2, [(2, 'KS'), (3, None)])
        advice = PlannerStrategy().advise(observation, legal)
        self.assertIs(advice.move, PASS)
        self.assertIn('partner', advice.reason)

    def test_bombs_an_opponent_about_to_go_out(self):
        # seat 1 leads and keeps one card; seat 2 can only top the six by bombing or breaking its bomb
        hands = ['3S 4D 5C 7H 9S', '4S 6D', '9S 9D 9C 9H 3D 4C 5H', 'AS AD 8H TS JD']
        _, seat, observation, legal = self.observe(hands, 1, [(1, '6D')])
        self.assertEqual((seat, observation.counts[1]), (2, 1))
        advice = PlannerStrategy().advise(observation, legal)
        self.assertEqual(advice.move.type, ComboType.BOMB)
        self.assertIn('only 1 card,', advice.reason)

    def test_goes_out_when_it_can(self):
        hands = ['8S 8D', '4S 6D 8C TH QS', 'KS 3C 5H 7S 9D', '4D 6C 8H TS JD']
        _, _, observation, legal = self.observe(hands, 0, [])
        advice = PlannerStrategy().advise(observation, legal)
        self.assertEqual(advice.move.type, ComboType.PAIR)
        self.assertIn('last cards', advice.reason)
