"""Shallow exact solvers: win-in-1 and win-in-3.

These produce *policy-independent, exact* value labels: no matter how strong or
weak the current model is, "side to move has a forced win in <= 3 plies" is a
statement about the game, not about the policy. That is what makes them usable
as a bootstrap signal from a random start.

win-in-1 : the side to move can complete a five immediately.
win-in-3 : the side to move can play a move that creates >= 2 distinct
           five-completion cells (a double four), while the opponent has no
           immediate five of their own -- the opponent can block only one.
"""
from __future__ import annotations

from engine import CELLS, H, W, WINDOWS


def _windows_at():
    at = [[] for _ in range(CELLS)]
    for w in WINDOWS:
        for i in w:
            at[i].append(w)
    return tuple(tuple(x) for x in at)


WINDOWS_AT = _windows_at()


def _five_cells_window_scan(bb_att: int, occ: int, windows):
    out = set()
    for w in windows:
        n = 0
        empty = -1
        for i in w:
            if (bb_att >> i) & 1:
                n += 1
            elif not (occ >> i) & 1:
                empty = i
        if n == 4 and empty >= 0:
            out.add(empty)
    return out


def five_cells(bb_att: int, occ: int) -> set:
    """Empty cells where bb_att completes a five. occ = both bitboards."""
    return _five_cells_window_scan(bb_att, occ, WINDOWS)


def win_in_1(bb_att: int, occ: int) -> bool:
    return bool(five_cells(bb_att, occ))


_FULL = (1 << CELLS) - 1
_NOT_LEFT = _FULL & ~sum(1 << (r * W) for r in range(H))
_NOT_RIGHT = _FULL & ~sum(1 << (r * W + W - 1) for r in range(H))


def _dilate1(bb: int) -> int:
    """Chebyshev dilation by one cell, as 8 masked shifts."""
    return (bb
            | ((bb & _NOT_RIGHT) << 1)
            | ((bb & _NOT_LEFT) >> 1)
            | ((bb << W) & _FULL)
            | (bb >> W)
            | ((bb & _NOT_RIGHT) << (W + 1))
            | ((bb & _NOT_LEFT) >> (W + 1))
            | ((bb & _NOT_LEFT) << (W - 1))
            | ((bb & _NOT_RIGHT) >> (W - 1)))


def near_mask(occ: int, radius: int = 2) -> int:
    m = occ
    for _ in range(radius):
        m = _dilate1(m)
    return m & ~occ & _FULL


def _near_stones(occ: int, radius: int = 2):
    """Empty cells within Chebyshev `radius` of an occupied cell."""
    m = near_mask(occ, radius)
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def _near_stones_ref(occ: int, radius: int = 2):
    """Slow reference version, kept for cross-checks."""
    cells = []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if (occ >> i) & 1:
                continue
            near = False
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and (occ >> (rr * W + cc)) & 1:
                        near = True
                        break
                if near:
                    break
            if near:
                cells.append(i)
    return cells


def win_in_3(bb_att: int, bb_def: int) -> bool:
    """Side to move forces a five in at most 3 plies (its move, block, its move)."""
    occ = bb_att | bb_def
    if win_in_1(bb_att, occ):
        return True
    for m in _near_stones(occ):
        a2 = bb_att | (1 << m)
        occ2 = occ | (1 << m)
        cells = _five_cells_window_scan(a2, occ2, WINDOWS_AT[m])
        if len(cells) < 2:
            continue
        # must survive the opponent's own immediate five
        if five_cells(bb_def, occ2):
            continue
        return True
    return False


def exact_label(bb_att: int, bb_def: int):
    """Returns +1 if the side to move is proven to win, else None (unknown)."""
    occ = bb_att | bb_def
    if win_in_1(bb_att, occ) or win_in_3(bb_att, bb_def):
        return 1
    return None
