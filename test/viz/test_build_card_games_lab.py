import json
import os
import tempfile
from unittest import TestCase

from viz.build_card_games_lab import build

__author__ = 'frank.ma'


class TestBuildCardGamesLab(TestCase):
    def test_quick_build(self):
        # small samples: checks the builder runs end to end and fills the template, not the numbers
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, 'lab.html')
            data = build(quick=True, output=output)
            with open(output, encoding='utf-8') as f:
                page = f.read()
        self.assertNotIn('/*__THEME__*/', page)
        self.assertNotIn('/*__DATA__*/', page)
        self.assertIn('--bg: #fff1e5', page, 'the paper theme is inlined.')
        embedded = page.split('const DATA = ', 1)[1].split(';\n', 1)[0]
        self.assertEqual(json.loads(embedded)['sizes'], data['sizes'])
        self.assertTrue(data['quick'])
        self.assertEqual(set(data['blackjack']['ev']), {'s17', 'h17'})
        self.assertEqual(len(data['holdem']['hands']['equity']), 169)
        self.assertEqual(data['blackjack']['strategy']['s17']['pairs'][8], ['P'] * 10, '8,8 always splits.')
        self.assertEqual([len(s['bands']) for s in data['blackjack']['bankroll']['schemes']], [5] * 4)
