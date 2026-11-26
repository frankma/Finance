"""
build viz/betting_games_lab.html: play the BettingGames once with fixed seeds, then write the numbers into the page
template together with the Kinari theme, so the page itself carries no game logic

    python -m viz.build_betting_games_lab            # full run, a few seconds
    python -m viz.build_betting_games_lab --quick    # fewer trials, for checking the page
    python -m viz.build_betting_games_lab --reuse    # keep the page's numbers, refill an edited template

every number on the page comes from src/BettingGames; each player stakes only on what they have seen so far
"""
import argparse
import json
import os
import time

from src.BettingGames.CoinFlip import Stages

__author__ = 'frank.ma'

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'betting_games_lab.template.html')
THEME = os.path.join(HERE, 'theme.css')
OUTPUT = os.path.join(HERE, 'betting_games_lab.html')

FULL = {'trials': 2000}


def render(data: dict, output: str = OUTPUT) -> None:
    with open(TEMPLATE, encoding='utf-8') as f:
        page = f.read()
    with open(THEME, encoding='utf-8') as f:
        theme = f.read()
    page = page.replace('/*__THEME__*/', theme).replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    with open(output, 'w', encoding='utf-8') as f:
        f.write(page)


def read_data(page_path: str = OUTPUT) -> dict:
    # the numbers embedded in a built page
    with open(page_path, encoding='utf-8') as f:
        return json.loads(f.read().split('const DATA = ', 1)[1].split(';\n', 1)[0])


def build(quick: bool = False, output: str = OUTPUT) -> dict:
    sizes = {k: max(1, v // (20 if quick else 1)) for k, v in FULL.items()}
    start = time.time()
    data = {
        'built': time.strftime('%Y-%m-%d'),
        'quick': quick,
        'sizes': sizes,
        'game': {'up': Stages.BET.up, 'down': Stages.BET.down, 'p_up': Stages.BET.p_up,
                 'players': Stages.N_PLAYERS, 'rounds': Stages.N_ROUNDS},
        'coinflip': {
            'all_in': Stages.all_in(),
            'kelly': Stages.kelly_outsider(n_trials=sizes['trials']),
            'curve': Stages.fraction_curve(step=0.005),
            'sensible': Stages.everyone_sensible(),
            'mixed': Stages.mixed_table(),
            'levers': Stages.levers(),
            'curves': Stages.growth_curves(),
        },
    }
    data['seconds'] = round(time.time() - start, 1)
    render(data, output)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='fewer trials, for checking the page')
    parser.add_argument('--reuse', action='store_true', help='keep the numbers in the built page, refill the template')
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args()
    if args.reuse:
        render(read_data(args.output), args.output)
        print('refilled %s' % args.output)
    else:
        built = build(args.quick, args.output)
        print('wrote %s in %.1f s' % (args.output, built['seconds']))
