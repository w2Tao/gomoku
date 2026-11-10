"""Tests for the value-target spec: perspective, terminal, symmetry."""
import random

from engine import CELLS, W, has_five
from value_target import (
    DIHEDRAL8,
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
    apply_perm,
    augment,
    games_needed,
    game_targets,
)


def bb(cells):
    o = 0
    for i in cells:
        o |= 1 << i
    return o


def make_game():
    """Black builds a horizontal five on row 4; white plays down column 8."""
    history = []
    b = [0, 0]
    turn = 0
    last = -1
    moves = [
        (0, 4 * W + 0), (1, 0 * W + 8),
        (0, 4 * W + 1), (1, 1 * W + 8),
        (0, 4 * W + 2), (1, 2 * W + 8),
        (0, 4 * W + 3), (1, 3 * W + 8),
        (0, 4 * W + 4),
    ]
    for p, cell in moves:
        assert p == turn
        history.append((b[0], b[1], turn, last))
        b[p] |= 1 << cell
        last = cell
        turn ^= 1
    return history, 0


def main():
    # --- symmetry group --------------------------------------------------
    assert len(set(DIHEDRAL8)) == 8, "dihedral group must have 8 distinct maps"
    for p in DIHEDRAL8:
        assert sorted(p) == list(range(CELLS)), "perm must be a bijection"
    print("dihedral8: 8 distinct bijections  ok")

    rng = random.Random(5)
    for _ in range(400):
        b = 0
        for i in range(CELLS):
            if rng.random() < 0.35:
                b |= 1 << i
        want = has_five(b)
        for p in DIHEDRAL8:
            assert has_five(apply_perm(b, p)) == want, "symmetry changed a five"
    print("apply_perm preserves fives  ok")

    # --- perspective ------------------------------------------------------
    history, winner = make_game()
    tg = game_targets(history, winner)
    assert len(tg) == len(history)
    # black to move at ply 0, black wins -> win
    assert tg[0].cls == WDL_WIN, tg[0]
    # white to move at ply 1, black wins -> loss
    assert tg[1].cls == WDL_LOSS, tg[1]
    # canonical view: black to move at ply 2 sees its own stone at (4,0)
    assert tg[2].own == bb([4 * W + 0]), bin(tg[2].own)
    assert tg[2].opp == bb([0 * W + 8]), bin(tg[2].opp)
    # white to move at ply 3: "own" is white's single stone at (0,8), and the
    # label is a loss because black is the one with the five
    assert tg[3].own == bb([0 * W + 8]), bin(tg[3].own)
    assert tg[3].cls == WDL_LOSS
    print("perspective / sign  ok")

    # --- draw -------------------------------------------------------------
    tg = game_targets(history, None)
    assert all(t.cls == WDL_DRAW for t in tg)
    print("draw class  ok")

    # --- solver override --------------------------------------------------
    exact = {2: -1, 3: 1}
    tg = game_targets(history, winner, exact=exact)
    assert tg[2].cls == WDL_LOSS and tg[2].source == "solver"
    assert tg[3].cls == WDL_WIN and tg[3].source == "solver"
    assert tg[2].weight > 1.0, "solver labels must outweigh outcome labels"
    print("solver override + weight  ok")

    # --- augmentation -----------------------------------------------------
    base = game_targets(history, winner)[7]
    views = augment(base)
    assert len(views) == 8
    assert all(v.cls == base.cls for v in views)
    assert all(bin(v.own).count("1") == bin(base.own).count("1") for v in views)
    assert len({v.own for v in views}) == 8, "views should be distinct here"
    print("augmentation  ok")

    # --- standard-error helper -------------------------------------------
    assert games_needed(0.5, 0.02) == 625
    assert games_needed(0.5, 0.01) == 2500
    print("sample-size helper  ok")

    print("ALL VALUE-TARGET TESTS PASSED")


if __name__ == "__main__":
    main()
