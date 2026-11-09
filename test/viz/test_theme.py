import glob
import os
import re
from unittest import TestCase, skipUnless

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')  # the repository
SKILL_CSS = os.path.expanduser('~/.claude/skills/my-theme/theme.css')
THEME = os.path.join(ROOT, 'viz', 'theme.css')
STATIC = os.path.join(ROOT, 'src', 'CardGames', 'Play', 'static')
PAGES = ['viz/sabr_smile_lab.html', 'viz/card_games_lab.template.html'] + [
    os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(STATIC, '*.html')) + glob.glob(os.path.join(STATIC, '*.css')))]


def read(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def defined(text: str) -> set:
    return set(re.findall(r'(--[a-z0-9-]+)\s*:', text))


def colours(block: str) -> set:
    return set(k for k, v in re.findall(r'(--[a-z0-9-]+)\s*:\s*([^;]+);', block) if v.strip().startswith('#'))


class TestTheme(TestCase):
    def setUp(self):
        self.theme = read('viz/theme.css')

    def test_every_token_a_page_uses_is_defined(self):
        # tokens built at run time (var(--seq${k})) are not checked
        for page in PAGES:
            text = read(page)
            used = set(re.findall(r'var\((--[a-z0-9-]+)\)', text))
            self.assertEqual(set(), used - defined(self.theme) - defined(text), page)
        pass

    def test_dark_mode_overrides_the_same_colours(self):
        light, dark_os, dark_forced = re.findall(r'\{([^{}]*--bg:[^{}]*)\}', self.theme)[:3]
        self.assertEqual(colours(dark_os), colours(dark_forced))
        self.assertEqual(colours(light), colours(dark_os))
        pass

    def test_no_generic_label_class_and_no_bare_svg_rule(self):
        # .label is used for svg text in charts; a bare svg rule makes MathJax formulas huge in Safari
        self.assertIsNone(re.search(r'(^|[\s,}])\.label\b', self.theme))
        self.assertIsNone(re.search(r'(^|\n)\s*svg\s*\{', self.theme))
        pass

    def test_pages_keep_the_type_and_colour_budget(self):
        # Kinari: Arial 400 and 700 (Courier New for code), no italic, no uppercase kickers, no web fonts, no serif,
        # and no raw hex colours in page css: every colour comes from a theme token
        for page in PAGES:
            text = read(page)
            self.assertEqual([], re.findall(r'font-weight:\s*(?!400|700)\d+|font:\s*(?!400|700)\d{3}\b', text), page)
            self.assertIsNone(re.search(r'font-style:\s*italic', text), page)
            self.assertNotIn('text-transform: uppercase', text, page)
            self.assertNotIn('fonts.googleapis', text, page)
            self.assertEqual([], re.findall(r'(?:fill|stroke|color|background)\s*[:=]\s*["\']?#[0-9a-fA-F]{3,6}', text), page)
        pass

    def test_pages_use_the_skill_token_names(self):
        # the skill names its series and fills --ai, --koke, --kuri, --nezumi and --washi; the older --s1..--s5 and
        # --kraft must not come back, and the base CSS must define the names the pages use
        for name in ('--washi', '--ai', '--koke', '--kuri', '--nezumi'):
            self.assertIn(name, defined(self.theme))
        for page in PAGES:
            self.assertEqual([], re.findall(r'--(?:s[1-6]|kraft)\b(?!\$)', read(page).replace('--s${', '')), page)
        pass

    def test_titles_carry_the_accent_and_nothing_else_does_in_base(self):
        # the skill's rule: h1 and h2 are beni, h3 stays ink
        self.assertIn('h1, h2 { color: var(--beni); }', self.theme)
        self.assertNotRegex(self.theme, r'h3[^{]*\{[^}]*color:\s*var\(--beni\)')
        pass

    @skipUnless(os.path.exists(SKILL_CSS), 'my-theme skill not installed here')
    def test_copy_matches_the_skill(self):
        # viz/theme.css starts with the skill's theme.css, then the compatibility aliases; refresh the start with the
        # skill's file and keep the aliases
        with open(SKILL_CSS) as f:
            core = f.read().rstrip('\n')
        self.assertTrue(self.theme.startswith(core), 'viz/theme.css has drifted from %s' % SKILL_CSS)
        for page in ('viz/sabr_smile_lab.html',):
            self.assertIn(core, read(page), '%s carries a stale copy of the theme' % page)
        pass
