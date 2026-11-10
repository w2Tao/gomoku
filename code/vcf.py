"""VCF -- victory by continuous fours.

The previous detector (`win_in_3`) only recognises one hand-written pattern:
"a move that creates two or more five-completion cells". That is a *depth 3*
win: my move, your forced block, my five.

A real VCF is a depth-limited AND/OR search over forcing moves:

    OR node (attacker to move)
        I can make five now                        -> win
        for each move that creates >= 1 five cell  (a "four", i.e. a threat)
            if it creates >= 2 five cells          -> win, you cannot block both
            otherwise you are forced to block the single cell
                AND node (defender to move)
                    if my block itself makes five  -> the attack fails
                    recurse with 2 plies consumed

Only moves that create an immediate five threat are ever considered. That is
what keeps the tree narrow: the defender has exactly one legal reply at every
AND node, so the branching is the attacker's four-making moves only.
"""
from __future__ import annotations

from engine import CELLS
from solvers import (
    WINDOWS_AT,
    _five_cells_window_scan,
    _near_stones,
    five_cells,
    win_in_1,
    win_in_3,
)


def forcing_moves(att: int, dfn: int, cand=None):
    """[(move, set_of_five_cells)] for every move that creates a five threat."""
    occ = att | dfn
    if cand is None:
        cand = _near_stones(occ)
    out = []
    for m in cand:
        a2 = att | (1 << m)
        cells = _five_cells_window_scan(a2, occ | (1 << m), WINDOWS_AT[m])
        if cells:
            out.append((m, cells))
    return out


def vcf(att: int, dfn: int, depth: int = 9, stats=None) -> bool:
    """True iff `att` forces five within `depth` plies using only fours."""
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1
    occ = att | dfn
    if win_in_1(att, occ):
        return True
    if depth < 3:
        return False
    for m, cells in forcing_moves(att, dfn, _near_stones(occ)):
        a2 = att | (1 << m)
        occ2 = occ | (1 << m)
        if len(cells) >= 2:
            # a straight four: two five-completion cells, only one can be blocked
            if win_in_1(dfn, occ2):
                continue          # unless the defender fives first
            return True
        b = next(iter(cells))
        d2 = dfn | (1 << b)
        occ3 = occ2 | (1 << b)
        if win_in_1(d2, occ3):
            continue              # the forced block itself makes five
        if vcf(a2, d2, depth - 2, stats):
            return True
    return False


def vcf_move(att: int, dfn: int, depth: int = 7):
    """The first move of a proven VCF win, or None.

    Same search as `vcf`, but it has to report *which* move starts the forced
    sequence -- a heuristic player needs the move, not just the verdict.
    """
    occ = att | dfn
    if win_in_1(att, occ):
        return next(iter(five_cells(att, occ)))
    if depth < 3:
        return None
    for m, cells in forcing_moves(att, dfn, _near_stones(occ)):
        a2 = att | (1 << m)
        occ2 = occ | (1 << m)
        if len(cells) >= 2:
            if win_in_1(dfn, occ2):
                continue
            return m
        b = next(iter(cells))
        d2 = dfn | (1 << b)
        occ3 = occ2 | (1 << b)
        if win_in_1(d2, occ3):
            continue
        if vcf_move(a2, d2, depth - 2) is not None:
            return m
    return None


def vcf_labelled(att: int, dfn: int, depth: int = 9):
    """(label, nodes) where label is +1 if proven win else None."""
    st = {"nodes": 0}
    ok = vcf(att, dfn, depth, st)
    return (1 if ok else None), st["nodes"]


__all__ = ["vcf", "vcf_move", "forcing_moves", "vcf_labelled", "win_in_3"]
