import os
import re
from unittest import TestCase

from src.CardGames.Play.Server import LAB, STATIC

__author__ = 'frank.ma'


def read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


class TestFairChip(TestCase):
    """
    the fair-play note sits behind a 'Fair play' chip on every card games page; common.js puts it in each page's link
    row, so every page needs one, and the note says how this version keeps hidden cards away from the page
    """
    def test_every_page_has_a_link_row_for_the_chip(self):
        for name in ('lobby.html', 'blackjack.html', 'holdem.html', 'guandan.html', 'rules.html'):
            self.assertEqual(read(os.path.join(STATIC, name)).count('<nav class="links">'), 1, name)

    def test_the_note_covers_both_versions(self):
        js = read(os.path.join(STATIC, 'common.js'))
        self.assertIn("fairChip(nav, [FAIR_PROMISE, FAIR_HERE])", js)
        self.assertRegex(js, r"const FAIR_HERE = STATIC\s*\n\s*\? \"This web version runs the whole game in your browser")
        self.assertIn("Hidden cards never leave the server", js)
        self.assertIn('Every game is tested for this', js)

    def test_the_old_notes_are_gone(self):
        self.assertNotIn('fair-note', read(os.path.join(STATIC, 'lobby.html')))
        self.assertNotIn('fair-note', read(os.path.join(STATIC, 'common.css')))
        lab = read(LAB)
        self.assertIn("fairChip(document.getElementById('fair-row')", lab)
        self.assertIsNone(re.search(r'<div class="fair">', lab))
