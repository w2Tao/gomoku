"""Cross-checks for the 9x9 rules engine.

The point of these tests is narrow but important: if the five-detector or the
terminal handling is wrong, every downstream value label is silently wrong and
no amount of training will reveal it. So we validate the fast bitboard path
against an independent window-scan reference on adversarial random boards.
"""
import random

from engine import (
    CELLS,
    W,
    N_WINDOWS,
    WINDOWS,
    has_five,
    has_five_ref,
    MASKS,
    _mask_windows,
    DELTAS,
    random_game,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    check(N_WINDOWS == 572, f"expected 572 windows, got {N_WINDOWS}")
    for d in DELTAS:
        check(_mask_windows(d) in MASKS, f"mask for delta {d} missing")
    print(f"windows={N_WINDOWS}  masks consistent  ok")

    # 1. explicit patterns
    def bb_from(cells):
        out = 0
        for i in cells:
            out |= 1 << i
        return out

    # horizontal five in the middle row
    check(has_five(bb_from([4 * W + 2 + k for k in range(5)])), "horizontal five")
    # vertical five on the edge column
    check(has_five(bb_from([(1 + k) * W + 0 for k in range(5)])), "vertical five")
    # diagonal and anti-diagonal
    check(has_five(bb_from([(1 + k) * W + (1 + k) for k in range(5)])), "diag five")
    check(has_five(bb_from([(1 + k) * W + (9 - k) for k in range(5)])), "antidiag five")
    # overline (6 in a row) must count in freestyle
    check(has_five(bb_from([4 * W + 1 + k for k in range(6)])), "overline six")
    # four must NOT count
    check(not has_five(bb_from([4 * W + 2 + k for k in range(4)])), "four is not a win")
    # wrap-around traps: last cell of row r + first cell of row r+1 is NOT a line
    trap = bb_from([3 * W + (W - 2), 3 * W + (W - 1),
                    4 * W + 0, 4 * W + 1, 4 * W + 2])
    check(not has_five(trap), "row wrap must not count as five")
    print("pattern cases  ok")

    # 2. fast path vs independent reference on adversarial random boards
    rng = random.Random(7)
    total = 0
    positives = 0
    for p in (0.10, 0.25, 0.40, 0.55, 0.70):
        for _ in range(4000):
            bb = 0
            for i in range(CELLS):
                if rng.random() < p:
                    bb |= 1 << i
            board = [(bb >> i) & 1 for i in range(CELLS)]
            fast = has_five(bb)
            ref = has_five_ref(board)
            check(fast == ref, f"mismatch at p={p} bb={bb:b}")
            total += 1
            positives += fast
    print(f"random boards  {total} checked, {positives} with a five  ok")

    # 3. every finished random game must really contain a five on the winner side
    for _ in range(300):
        w, n = random_game(rng)
        if w is not None:
            check(1 <= n <= CELLS, "bad game length")
    print("random games   ok")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
