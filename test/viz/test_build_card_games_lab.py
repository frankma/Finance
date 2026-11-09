import json
import os
import tempfile
from unittest import TestCase

from viz.build_card_games_lab import build, read_data, render, rerun

__author__ = 'frank.ma'


class TestBuildCardGamesLab(TestCase):
    @classmethod
    def setUpClass(cls):
        # one quick build for every test: small samples check the builder runs end to end, not the numbers
        cls.folder = tempfile.TemporaryDirectory()
        cls.output = os.path.join(cls.folder.name, 'lab.html')
        cls.data = build(quick=True, output=cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_quick_build(self):
        data = self.data
        with open(self.output, encoding='utf-8') as f:
            page = f.read()
        self.assertNotIn('/*__THEME__*/', page)
        self.assertNotIn('/*__DATA__*/', page)
        self.assertIn('--bg: #f4f0e8', page, 'the Kinari theme is inlined.')
        embedded = page.split('const DATA = ', 1)[1].split(';\n', 1)[0]
        self.assertEqual(json.loads(embedded)['sizes'], data['sizes'])
        self.assertTrue(data['quick'])
        self.assertEqual(set(data['blackjack']['ev']), {'s17', 'h17'})
        self.assertEqual(len(data['holdem']['hands']['equity']), 169)
        self.assertEqual(data['blackjack']['strategy']['s17']['pairs'][8], ['P'] * 10, '8,8 always splits.')
        self.assertEqual([len(s['bands']) for s in data['blackjack']['bankroll']['schemes']], [5] * 4)
        gd = data['guandan']
        self.assertEqual(len(gd['versus']), 2)
        self.assertAlmostEqual(sum(gd['hands']['levels_up'].values()), 1.0)
        for key, rows in gd['hands']['starting'].items():
            self.assertAlmostEqual(sum(row['share'] for row in rows), 1.0, msg=key)
            self.assertTrue(all(0.0 <= row['win'] <= 1.0 for row in rows), key)
        self.assertEqual(gd['matches']['n_matches'], data['sizes']['guandan_matches'])

    def test_rerun_keeps_the_other_games(self):
        kept = read_data(self.output)
        kept['blackjack']['marker'] = 1
        render(kept, self.output)
        again = rerun('guandan', self.output)
        self.assertEqual(read_data(self.output), again)
        self.assertEqual(again['blackjack']['marker'], 1, 'blackjack was not recomputed.')
        self.assertEqual(again['guandan']['hands']['n_hands'], self.data['guandan']['hands']['n_hands'],
                         'the same quick sizes.')
