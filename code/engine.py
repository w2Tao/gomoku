"""9x9 freestyle gomoku (no forbidden moves): exact rules via bitboard.

Rules
-----
* Board is 15x15, cells indexed row-major: idx = r*15 + c.
* Black (= player 0) moves first.
* Any line of **five or more** stones wins (freestyle: overlines count).
* The only other terminal state is a full board with no five -> draw.

The module exposes two *independent* five-detectors:
  has_five(bb)          -- bitboard, used everywhere for speed
  has_five_ref(board)   -- line-by-line scan of an explicit window list
They are cross-checked at import time and again in tests.py.
"""
from __future__ import annotations

W = H = 15
CELLS = W * H
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
DELTAS = tuple(dr * W + dc for dr, dc in DIRS)


def _windows():
    """Every length-5 collinear window, as explicit cell-index tuples."""
    ws = []
    for r in range(H):
        for c in range(W - 4):
            ws.append([r * W + c + k for k in range(5)])
    for r in range(H - 4):
        for c in range(W):
            ws.append([(r + k) * W + c for k in range(5)])
    for r in range(H - 4):
        for c in range(W - 4):
            ws.append([(r + k) * W + c + k for k in range(5)])
    for r in range(H - 4):
        for c in range(4, W):
            ws.append([(r + k) * W + c - k for k in range(5)])
    return tuple(tuple(w) for w in ws)


WINDOWS = _windows()
N_WINDOWS = len(WINDOWS)          # 140 on 9x9


def _mask_coord(dr, dc):
    """Legal start cells for a 5-in-a-row in direction (dr, dc)."""
    m = 0
    for r in range(H):
        for c in range(W):
            if all(0 <= r + k * dr < H and 0 <= c + k * dc < W for k in range(5)):
                m |= 1 << (r * W + c)
    return m


def _mask_windows(delta):
    """Same set, derived independently from the explicit window list."""
    m = 0
    for w in WINDOWS:
        if w[1] - w[0] == delta:
            m |= 1 << w[0]
    return m


MASKS = tuple(_mask_coord(dr, dc) for dr, dc in DIRS)
assert MASKS == tuple(_mask_windows(d) for d in DELTAS), "bitboard masks disagree"

MASK_H, MASK_V, MASK_D1, MASK_D2 = MASKS


def has_five(bb: int) -> bool:
    """True iff the bitboard `bb` contains 5+ collinear stones."""
    for d, m in zip(DELTAS, MASKS):
        if bb & (bb >> d) & (bb >> (2 * d)) & (bb >> (3 * d)) & (bb >> (4 * d)) & m:
            return True
    return False


def has_five_ref(board) -> bool:
    """Slow, independent reference implementation. board[i] in {0, 1}."""
    for w in WINDOWS:
        if board[w[0]] and board[w[1]] and board[w[2]] and board[w[3]] and board[w[4]]:
            return True
    return False


def bit(cell: int) -> int:
    return 1 << cell


def random_game(rng):
    """One uniform-random playout. Returns (winner_or_None, n_moves)."""
    b = [0, 0]
    empt = list(range(CELLS))
    turn = 0
    n = 0
    while True:
        if not empt:
            return None, n
        k = rng.randrange(len(empt))
        cell = empt[k]
        empt[k] = empt[-1]
        empt.pop()
        b[turn] |= 1 << cell
        n += 1
        if has_five(b[turn]):
            return turn, n
        turn ^= 1


def playout(bb0: int, bb1: int, turn: int, rng, empt=None):
    """Uniform-random continuation from a position. Returns winner or None."""
    b = [bb0, bb1]
    if empt is None:
        occ = bb0 | bb1
        empt = [i for i in range(CELLS) if not (occ >> i) & 1]
    else:
        empt = list(empt)
    while True:
        if not empt:
            return None
        k = rng.randrange(len(empt))
        cell = empt[k]
        empt[k] = empt[-1]
        empt.pop()
        b[turn] |= 1 << cell
        if has_five(b[turn]):
            return turn
        turn ^= 1


def legal_moves(bb0: int, bb1: int):
    occ = bb0 | bb1
    return [i for i in range(CELLS) if not (occ >> i) & 1]
