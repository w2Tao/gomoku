"""Correctness checks for the forced-win detectors.

`win_in_3` argues: if the side to move can create two or more five-completion
cells while the opponent has no five of their own, the opponent can occupy only
one of them, so the other is still open next move. That is a proof, not a
heuristic -- but proofs downstream of buggy code are worthless, so we cross-check
against a brute-force 3-ply minimax over the same candidate set.
"""
import random

from engine import CELLS, W, has_five
from solvers import _near_stones, five_cells, win_in_1, win_in_3


def bb_of(cells):
    o = 0
    for i in cells:
        o |= 1 << i
    return o


def brute_win3(att, dfn, cand):
    occ = att | dfn
    avail = [c for c in cand if not (occ >> c) & 1]
    for m in avail:
        a1 = att | (1 << m)
        if has_five(a1):
            return True
        occ1 = occ | (1 << m)
        all_blocked = True
        for n in avail:
            if n == m:
                continue
            occ2 = occ1 | (1 << n)
            if has_five(dfn | (1 << n)):
                all_blocked = False
                break
            won = False
            for p in avail:
                if p == m or p == n or (occ2 >> p) & 1:
                    continue
                if has_five(a1 | (1 << p)):
                    won = True
                    break
            if not won:
                all_blocked = False
                break
        if all_blocked:
            return True
    return False


def game_positions(rng, n_games):
    out = []
    for _ in range(n_games):
        b = [0, 0]
        empt = list(range(CELLS))
        turn = 0
        while empt:
            out.append((b[0], b[1], turn))
            c = empt[rng.randrange(len(empt))]
            empt[empt.index(c)] = empt[-1]
            empt.pop()
            b[turn] |= 1 << c
            if has_five(b[turn]):
                break
            turn ^= 1
    return out


def main():
    # --- targeted cases -------------------------------------------------
    row = [4 * W + c for c in (2, 3, 4)]

    a = bb_of(row)
    assert not win_in_1(a, a)
    assert win_in_3(a, 0), "open three must be a forced win in 3"

    d = bb_of([4 * W + 1])                      # one end blocked
    assert not win_in_1(a, a | d)
    assert not win_in_3(a, d), "blocked three must not be a forced win"

    d = bb_of([7 * W + c for c in (2, 3, 4, 5)])    # opponent has a four
    assert not win_in_3(a, d), "must not claim a win when the opponent fives first"
    print("targeted cases  ok")

    # --- five_cells -----------------------------------------------------
    a = bb_of([0, 1, 2, 3])
    assert five_cells(a, a) == {4}
    print("five_cells  ok")

    # --- randomized cross-check on positions from real games -----------
    rng = random.Random(31337)
    pos = game_positions(rng, 40)
    checked = claimed = 0
    for att0, dfn0, turn in pos:
        att, dfn = (att0, dfn0) if turn == 0 else (dfn0, att0)
        occ = att | dfn
        if has_five(att) or has_five(dfn):
            continue
        cand = _near_stones(occ, 2)
        if not cand or len(cand) > 40:
            continue
        if checked >= 120:
            break
        fast = win_in_3(att, dfn)
        slow = brute_win3(att, dfn, cand)
        assert fast == slow, f"mismatch fast={fast} slow={slow} att={att:b} dfn={dfn:b}"
        checked += 1
        claimed += int(fast)
    print(f"random cross-check: {checked} positions, {claimed} claimed wins  ok")
    assert checked > 50, "cross-check did not cover enough positions"
    print("ALL SOLVER TESTS PASSED")


if __name__ == "__main__":
    main()
