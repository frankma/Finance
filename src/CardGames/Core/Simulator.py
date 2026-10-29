import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.CardGames.Core.Card import Card
from src.CardGames.Core.CardCounter import CardCounter

__author__ = 'frank.ma'


@dataclass(frozen=True)
class MonteCarloResult(object):
    mean: float
    std: float
    n: int

    @property
    def stderr(self) -> float:
        return self.std / np.sqrt(self.n) if self.n > 0 else float('nan')

    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        return self.mean - z * self.stderr, self.mean + z * self.stderr

    @staticmethod
    def from_samples(samples: Sequence[float]) -> 'MonteCarloResult':
        values = np.asarray(samples, dtype=float)
        if values.size == 0:
            return MonteCarloResult(float('nan'), float('nan'), 0)
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0  # unbiased sample variance
        return MonteCarloResult(float(values.mean()), std, int(values.size))

    def __str__(self):
        return '%.6f +/- %.6f (n = %i)' % (self.mean, self.stderr, self.n)


def run_trials(trial: Callable[[random.Random], float], n_trials: int,
               rng: Optional[random.Random] = None) -> MonteCarloResult:
    """
    repeat an independent random trial and summarize the outcomes
    :param trial: one trial, drawing all its randomness from the rng passed in
    :param n_trials: number of trials
    :param rng: random source shared across trials
    :return: MonteCarloResult
    """
    rng = rng if rng is not None else random.Random()
    return MonteCarloResult.from_samples([trial(rng) for _ in range(n_trials)])


def deal_unseen(counter: CardCounter, n: int, rng: random.Random) -> List[Card]:
    """
    sample n of the cards a seat has not seen, i.e. one hidden deal consistent with what that seat observes
    """
    unseen = counter.unseen()
    if n > len(unseen):
        raise ValueError('cannot deal %i cards, only %i unseen.' % (n, len(unseen)))
    return rng.sample(unseen, n)
