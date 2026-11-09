# Finance

Python code for finance and card-game experiments, with the pages and charts that present them. Card games live in
`src/CardGames/` (engines, assistants, and a local play server in `src/CardGames/Play/`). Charts and labs live in `viz/`.

## Theme
Pages, charts and reports use the Kinari theme from the user's `my-theme` skill. `viz/theme.css` is this project's
copy of the skill's `theme.css`, with compatibility aliases for old token names appended; the card-game server and the
lab builder both read it. Page CSS uses only theme tokens, with no raw hex. The full rules, and the colour-budget
check, are in the `my-theme` skill. When the skill changes, refresh the core of `viz/theme.css` from it and keep the
aliases (`test/viz/test_theme.py` fails when the copy drifts from the skill, or a page uses an undefined token, an old
token name, a web font, italic, uppercase or a raw hex colour).

## Generated files
`viz/card_games_lab.html` is built from `viz/card_games_lab.template.html` by `viz/build_card_games_lab.py` (about 10
minutes, runs the simulations). Edit the template, never the generated page, then rebuild:
`PYTHONPATH=. .venv/bin/python viz/build_card_games_lab.py`.

## Card games must play fair
An assistant uses only what its seat can legitimately see. Every game is tested for this: two tables that differ only
in hidden cards must give the same view and the same advice.

## Working conventions
- Python in `.venv`. Run tests with `.venv/bin/python -m pytest test/viz test/CardGames/Play`; tests sit under `test/`
  mirroring `src/`. Add tests with every code change.
- Start the play server with the `card-games` entry in `.claude/launch.json`.
- Ask when a request is unclear rather than guessing.
