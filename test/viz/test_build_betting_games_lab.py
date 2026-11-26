import json
import os
import tempfile
from unittest import TestCase

from viz.build_betting_games_lab import OUTPUT, build, read_data, render

__author__ = 'frank.ma'


class TestBuildBettingGamesLab(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.output = os.path.join(cls.folder.name, 'lab.html')
        cls.data = build(quick=True, output=cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_quick_build(self):
        with open(self.output, encoding='utf-8') as f:
            page = f.read()
        self.assertNotIn('/*__THEME__*/', page)
        self.assertNotIn('/*__DATA__*/', page)
        self.assertIn('--bg: #f4f0e8', page, 'the Kinari theme is inlined.')
        self.assertEqual(read_data(self.output), json.loads(json.dumps(self.data)))
        cf = self.data['coinflip']
        self.assertTrue(self.data['quick'])
        self.assertEqual(len(cf['kelly']['trials']['ranks']), self.data['sizes']['trials'])
        self.assertEqual(len(cf['curve']), 201)
        self.assertEqual([s['key'] for s in cf['sensible']['schemes']], ['all_in', 'kelly', 'pooled_quarter', 'pooled'])
        self.assertEqual([g['key'] for g in cf['mixed']['groups']], ['pooled', 'pooled_quarter', 'all_in', 'kelly'])
        self.assertEqual(len(cf['mixed']['ranked']), 1000)
        self.assertEqual(cf['levers']['pool_kelly_exact'], 1.0)

    def test_reuse_keeps_the_numbers(self):
        kept = read_data(self.output)
        render(kept, self.output)
        self.assertEqual(read_data(self.output), kept)

    def test_the_committed_page_is_a_full_build(self):
        # the page in the repository is built from the template without --quick
        data = read_data(OUTPUT)
        self.assertFalse(data['quick'])
        self.assertEqual(data['sizes']['trials'], 2000)
