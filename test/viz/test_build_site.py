import os
import re
from tempfile import TemporaryDirectory
from unittest import TestCase

from viz.build_site import PROJECTS, Project, build, single_page

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
            self.assertIn('<meta name="site-home" content="/Finance/">', lobby, 'the games link back to the front page.')
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
