import os
import re
from unittest import TestCase

from src.CardGames.Core.Card import Card, Rank
from src.CardGames.GuanDan.Combo import ComboType, interpret
from src.CardGames.Holdem.HandEvaluator import HandCategory, evaluate
from src.CardGames.Play.Registry import GAMES
from src.CardGames.Play.Server import STATIC

__author__ = 'frank.ma'


def read(name: str) -> str:
    with open(os.path.join(STATIC, name), encoding='utf-8') as f:
        return f.read()


def rows(html: str, section: str):
    # (first cell, the cards in the row) for each table row with example cards in this section
    body = re.search(r'<section id="%s">(.*?)</section>' % section, html, re.S).group(1)
    for row in re.findall(r'<tr>(.*?)</tr>', body, re.S):
        cards = re.search(r'data-cards="([^"]+)"', row)
        if cards:
            yield re.search(r'<td>(.*?)</td>', row).group(1), Card.parse_many(cards.group(1))


class TestRulesText(TestCase):
    def test_every_game_has_rules(self):
        for game in GAMES:
            html = read('rules-%s.html' % game.name)
            ids = re.findall(r'id="([a-z-]+)"', html)
            self.assertEqual(len(ids), len(set(ids)), '%s: section ids are unique.' % game.name)
            self.assertIn('coach', ids, '%s: every game says what the coach knows.' % game.name)
            for code in ' '.join(re.findall(r'data-cards="([^"]+)"', html)).split():
                if code != '_':
                    Card.parse(code)

    def test_table_links_reach_a_section(self):
        # each place a table links into its rules: data-rules attributes, the TERMS map and the context it opens at
        for game in GAMES:
            page, ids = read(game.page), set(re.findall(r'id="([a-z-]+)"', read('rules-%s.html' % game.name)))
            linked = set(re.findall(r'data-rules="([a-z-]+)"', page))
            terms = re.search(r'const TERMS = \{(.*?)\};', page, re.S)
            self.assertIsNotNone(terms, '%s links the coach\'s terms to its rules.' % game.name)
            linked |= set(re.findall(r":\s*'([a-z-]+)'", terms.group(1)))
            for line in re.findall(r"dataset\.rules = (.*)", page):
                linked |= set(re.findall(r"'([a-z-]+)'", line)) - {'return', 'done', 'over', 'deal'}
            self.assertTrue(linked, game.name)
            self.assertEqual(linked - ids, set(), '%s links to sections its rules do not have.' % game.name)

    def test_holdem_rankings_are_in_order(self):
        names = {c.name.replace('_', ' ').capitalize(): c for c in HandCategory}
        found = [(names[name], evaluate(cards).category) for name, cards in rows(read('rules-holdem.html'),
                                                                                  'rankings')]
        self.assertEqual([named for named, _ in found], sorted(HandCategory, reverse=True), 'best first, every one.')
        for named, evaluated in found:
            self.assertEqual(named, evaluated)

    def test_guandan_examples_read_as_named(self):
        html = read('rules-guandan.html')
        level = Rank.TWO
        for name, cards in rows(html, 'combos'):
            self.assertEqual(interpret(cards, level)[0].type, ComboType(name.lower()), name)
        bombs = [interpret(cards, level)[0] for _, cards in rows(html, 'bombs')]
        self.assertTrue(all(bomb.is_bomb for bomb in bombs))
        self.assertEqual([b.power for b in bombs], sorted(b.power for b in bombs), 'weakest to strongest.')
        for i in range(len(bombs) - 1):
            self.assertTrue(bombs[i + 1].beats(bombs[i]))
        # the trick example: each play beats the one before
        trick = re.search(r'<section id="tricks">(.*?)</section>', html, re.S).group(1)
        plays = [interpret(Card.parse_many(c), level)[0] for c in re.findall(r'data-cards="([^"]+)"', trick)]
        self.assertTrue(all(b.beats(a) for a, b in zip(plays, plays[1:])))

    def test_holdem_showdown_example_splits(self):
        board = Card.parse_many('7D 8S 9H TD JS')
        self.assertEqual(evaluate(Card.parse_many('AS AH') + board), evaluate(Card.parse_many('KD KC') + board))
