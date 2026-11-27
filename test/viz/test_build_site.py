import os
import re
from tempfile import TemporaryDirectory
from unittest import TestCase

from viz.build_site import PROJECTS, Project, add_home_link, build, single_page

__author__ = 'frank.ma'


def read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


class TestBuildSite(TestCase):
    def test_the_site(self):
        # a front page at the root linking every project, and each project in its own folder under the base
        with TemporaryDirectory() as folder:
            out = os.path.join(folder, 'site')
            build(out, '/Finance/')
            front = read(os.path.join(out, 'index.html'))
            self.assertNotIn('/*__THEME__*/', front)
            self.assertNotIn('<!--__PROJECTS__-->', front)
            for p in PROJECTS:
                self.assertTrue(os.path.exists(os.path.join(out, p.slug, 'index.html')), p.slug)
                self.assertIn('href="%s/"' % p.slug, front, 'linked relative to the front page.')
                self.assertIn(p.title, front)
            self.assertEqual(re.findall(r'href="/[^"]*"', front), [], 'no link leaves the site base.')
            lobby = read(os.path.join(out, 'card-games', 'index.html'))
            self.assertIn('<base href="/Finance/card-games/">', lobby)
            self.assertNotIn('All projects</a><a', lobby, 'the link home is no longer in the nav row of the games.')
            # every page of every project: one link home, at the top of its column
            pages = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if f == 'index.html' and dp != out]
            self.assertGreater(len(pages), 8)
            for path in pages:
                text = read(path)
                self.assertEqual(text.count('class="site-home"'), 1, path)
                self.assertIn('<div class="wrap">\n  <nav class="site-home"', text, path)
                self.assertIn('<a href="/Finance/" style="color: var(--muted);">All projects</a>', text, path)
            self.assertTrue(os.path.exists(os.path.join(out, 'card-games', 'static', 'cardgames.zip')))
            self.assertTrue(os.path.exists(os.path.join(out, '.nojekyll')))

    def test_projects_need_their_own_folders(self):
        page = single_page(__file__)
        with TemporaryDirectory() as folder:
            for slugs in (['a', 'a'], ['Upper'], ['a/b']):
                with self.assertRaises(ValueError):
                    build(folder, '/', [Project(s, '', '', '', page) for s in slugs])
            with self.assertRaises(ValueError):
                build(folder, 'Finance')

    def test_the_link_home_needs_one_column(self):
        self.assertIn('All projects', add_home_link('<body><div class="wrap"></div></body>', '/'))
        for page in ('<body></body>', '<div class="wrap"></div><div class="wrap"></div>'):
            with self.assertRaises(ValueError):
                add_home_link(page, '/')
