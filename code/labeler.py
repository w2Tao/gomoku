"""Symmetric exact labeller: +1 / -1 / unknown.

The one-sided detector from before could only ever produce +1. A value head
trained on that alone learns 'everything is a win'. This adds the dual:

    +1  iff the side to move has a forced win      (one winning line is enough)
    -1  iff EVERY legal move loses                 (all lines must fail)

That asymmetry is not an implementation detail -- it is the shape of proof
search, and it decides what the training set can look like.
"""
from __future__ import annotations

from typing import Optional

from engine import CELLS, has_five
from solvers import _near_stones, five_cells, win_in_1, win_in_3


def proven_win(att: int, dfn: int) -> bool:
    """Side to move has a forced win within 3 plies."""
    if win_in_1(att, att | dfn):
        return True
    return win_in_3(att, dfn)


def proven_loss(att: int, dfn: int, cand=None) -> bool:
    """Side to move loses: every legal move hands the opponent a forced win."""
    if proven_win(att, dfn):
        return False
    occ = att | dfn
    if cand is None:
        cand = _near_stones(occ)
    if not cand:
        # Empty board (or a board with no stones at all): every move is legal but
        # none has been examined, so nothing is proven. Returning True here would
        # label the opening as a forced loss.
        return False
    for m in cand:
        a2 = att | (1 << m)
        if has_five(a2):
            return False
        if not proven_win(dfn, a2):
            return False
    return True


def label(att: int, dfn: int) -> Optional[int]:
    if proven_win(att, dfn):
        return 1
    if proven_loss(att, dfn):
        return -1
    return None


def label_cost(att: int, dfn: int):
    """Returns (label, number of proven_win calls) -- to expose the asymmetry."""
    calls = 1
    if proven_win(att, dfn):
        return 1, calls
    occ = att | dfn
    cand = _near_stones(occ)
    if not cand:
        return None, calls
    for m in cand:
        calls += 1
        a2 = att | (1 << m)
        if has_five(a2) or not proven_win(dfn, a2):
            return None, calls
    return -1, calls
