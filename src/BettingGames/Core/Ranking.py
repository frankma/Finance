from typing import Dict, Sequence

import numpy as np

__author__ = 'frank.ma'


def rank_in(value: float, field: np.ndarray) -> int:
    """
    where `value` would place among `field` with 1 the richest: one more than the number strictly richer, so a tie
    shares the better place
    """
    return int(np.sum(np.asarray(field) > value)) + 1


def share_below(value: float, field: np.ndarray) -> float:
    # the share of the field that `value` beats outright
    return float(np.mean(np.asarray(field) < value))


def top_share(wealth: np.ndarray, top: float) -> float:
    """
    the share of all the wealth held by the richest `top` fraction of players (0.01 is the top 1%)
    """
    w = np.sort(np.asarray(wealth, dtype=float))[::-1]
    n = max(1, int(round(top * w.size)))
    total = w.sum()
    return float(w[:n].sum() / total) if total > 0 else 0.0


def summary(wealth: np.ndarray, start: float = 1.0, qs: Sequence[float] = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> Dict:
    w = np.asarray(wealth, dtype=float)
    return {
        'n': int(w.size),
        'mean': float(w.mean()),
        'median': float(np.median(w)),
        'max': float(w.max()),
        'min': float(w.min()),
        'quantiles': {str(q): float(np.quantile(w, q)) for q in qs},
        'above_start': float(np.mean(w > start)),
        'ruined': float(np.mean(w < 0.01 * start)),  # lost more than 99%
        'top1_share': top_share(w, 0.01),
        'top10_share': top_share(w, 0.10),
    }
