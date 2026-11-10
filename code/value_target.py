"""The value target for 9x9 freestyle gomoku -- executable spec.

Definition
----------
    V(s) = P(side to move wins | optimal play) - P(side to move loses | optimal play)

    V in [-1, +1], no discounting (gamma = 1).
    Terminal: five-or-more in a row -> +1 for the mover who just played,
              full board without five -> 0.

Representation
--------------
Every position is stored *canonically*: (own stones, opponent stones, last move),
where "own" is the side to move. Colour identity is already abstracted away, so
there is no colour sign flip anywhere in this module -- that whole class of bug
simply cannot occur.

Targets are 3-class WDL: 0 = loss, 1 = draw, 2 = win.

Provenance
----------
Each target carries a source and a weight. The weight exists to make exact
labels dominate noisy ones:

    solver   exact forced win/loss from a VCF/VCT search   w = 3.0
    terminal proven by the rules                           w = 3.0
    search   root value from a deeper search               w = 1.0
    outcome  the single-game result z                      w = 1.0

Measured on 9x9 freestyle: a raw `outcome` label carries ~0.037 bits of
information (1.0 is the ceiling), ~0.01 bits through the first 20 plies. The
weight is a prior on top of that, not a substitute for better labels.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

from engine import CELLS, H, W

WDL_LOSS, WDL_DRAW, WDL_WIN = 0, 1, 2

SOURCE_WEIGHT = {
    "terminal": 3.0,
    "solver": 3.0,
    "search": 1.0,
    "outcome": 1.0,
}


class Target(NamedTuple):
    own: int
    opp: int
    last: int          # cell index of the previous move, -1 for the empty board
    cls: int           # 0 / 1 / 2
    ply: int
    n_moves: int       # total moves in the generating game
    weight: float
    source: str

    @property
    def z(self) -> int:
        return self.cls - 1


def z_to_cls(z: int) -> int:
    assert z in (-1, 0, 1)
    return z + 1


def canonical(bb0: int, bb1: int, turn: int):
    """(black, white, side-to-move) -> (own, opp) from the mover's view."""
    return (bb0, bb1) if turn == 0 else (bb1, bb0)


def game_targets(
    history: Sequence[tuple],
    winner: Optional[int],
    exact: Optional[dict] = None,
    outcome_weight: float = SOURCE_WEIGHT["outcome"],
) -> list[Target]:
    """Turn one finished game into value targets.

    history[i] = (bb_black, bb_white, side_to_move, last_move) *before* ply i.
    winner    = 0 / 1 / None.
    exact     = optional {ply: +1/-1} overrides from a forced-win search.

    Every position is labelled from the perspective of the side to move, which
    is why both players in the same game contribute labels of both signs.
    """
    exact = exact or {}
    out = []
    n = len(history)
    for ply, (bb0, bb1, turn, last) in enumerate(history):
        own, opp = canonical(bb0, bb1, turn)
        if ply in exact:
            cls = z_to_cls(exact[ply])
            src = "solver"
        elif winner is None:
            cls, src = WDL_DRAW, "terminal"
        else:
            cls = WDL_WIN if turn == winner else WDL_LOSS
            src = "outcome"
        out.append(Target(own, opp, last, cls, ply, n, SOURCE_WEIGHT[src],
                          src))
    return out


# --------------------------------------------------------------------------
# 8-fold dihedral symmetry. Value is invariant, so the class never changes --
# only the board and the last-move index are permuted.
# --------------------------------------------------------------------------
def _dihedral8(n: int = W):
    perms = []
    for k in range(4):
        m = []
        for r in range(n):
            for c in range(n):
                rr, cc = r, c
                for _ in range(k):
                    rr, cc = n - 1 - cc, rr
                m.append(rr * n + cc)
        perms.append(tuple(m))
    for k in range(4):
        m = []
        for r in range(n):
            for c in range(n):
                rr, cc = r, c
                for _ in range(k):
                    rr, cc = n - 1 - cc, rr
                rr, cc = cc, rr
                m.append(rr * n + cc)
        perms.append(tuple(m))
    return tuple(perms)


DIHEDRAL8 = _dihedral8()


def apply_perm(bb: int, perm: Sequence[int]) -> int:
    """perm[dest] = source."""
    out = 0
    for dst, src in enumerate(perm):
        if (bb >> src) & 1:
            out |= 1 << dst
    return out


def augment(t: Target) -> list[Target]:
    """All 8 symmetric views of one target. The label is unchanged."""
    res = []
    for p in DIHEDRAL8:
        last = -1 if t.last < 0 else p.index(t.last)
        res.append(t._replace(own=apply_perm(t.own, p),
                              opp=apply_perm(t.opp, p),
                              last=last))
    return res


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def wdl_ce(logits, cls, weight):
    """Weighted 3-class cross entropy. logits: (N,3), cls: (N,), weight: (N,)."""
    import torch.nn.functional as F

    loss = F.cross_entropy(logits, cls.long(), reduction="none")
    return (loss * weight).mean()


def binom_se(p: float, n: int) -> float:
    """Standard error of an empirical win rate, for building calibration sets."""
    return (p * (1.0 - p) / max(n, 1)) ** 0.5


def games_needed(p: float, target_se: float) -> int:
    """How many games are needed to estimate a rate p to +-target_se."""
    return int((p * (1 - p) / (target_se ** 2)) + 0.999)
