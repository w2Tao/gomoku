"""Turn the value network into a player, and measure it.

The network alone outputs no moves. A decision rule is needed:

    1-ply : play the move that leaves the opponent in the worst position
            value(m) = -V(child)          -> argmin_a V(child)
    2-ply : respect the opponent's best reply
            value(m) = min_j V(grandchild_mj) -> argmax_m

Both are exact up to the evaluated horizon; terminal states inside the horizon
are scored +-1 directly, which is what gives the agent basic tactics (finish
your five, block theirs) without any hand-written rule.

Everything for one move is a single batched forward pass (~k^2 boards).
"""
from __future__ import annotations

import random

import numpy as np
import torch

from engine import CELLS, H, W, has_five
from model import ValueNet, ValuePolicyNet
from vcf import vcf_move
from solvers import WINDOWS_AT, _five_cells_window_scan

BITS = np.arange(CELLS)


def bb_to_plane(bb: int) -> np.ndarray:
    v = np.zeros(CELLS, dtype=np.float32)
    i = 0
    while bb:
        if bb & 1:
            v[i] = 1.0
        bb >>= 1
        i += 1
    return v


class SearchAgent:
    """Value network + fixed-depth search. No training, no policy head."""

    def __init__(self, net, dev="cpu", plies=2, temp=0.0, topk=1,
                 agg="threat", q_quantile=0.2):
        self.net = net
        self.dev = dev
        self.plies = plies
        self.temp = temp
        self.topk = topk
        self.agg = agg            # "min" | "quantile" | "threat"
        self.q_quantile = q_quantile
        self.last_q = None        # last root Q vector, for diagnostics

    def _pick(self, q, legal, rng):
        """argmax, top-k sampling, or a temperature sample.

        A deterministic agent makes "120 games" one game replayed 120 times; the
        win-rate statistics are then meaningless. Any evaluation match must use
        temp > 0 or topk > 1.
        """
        if self.topk > 1:
            order = np.argsort(-q)
            return legal[int(rng.choice(order[:self.topk].tolist()))]
        if self.temp <= 0:
            return legal[int(np.argmax(q))]
        z = (q - q.max()) / self.temp
        p = np.exp(z)
        p /= p.sum()
        return legal[int(rng.choices(range(len(q)), weights=p.tolist())[0])]

    @torch.no_grad()
    def _values(self, planes):
        x = torch.from_numpy(planes).to(self.dev)
        out = self.net(x)
        # a ValuePolicyNet returns (value_logits, policy_logits); the search only
        # wants the value, so a tuple has to be unwrapped here
        if isinstance(out, tuple):
            out = out[0]
        p = torch.softmax(out, dim=1)
        return (p[:, 2] - p[:, 0]).cpu().numpy()      # expected value in [-1,1]

    def move(self, my: int, their: int, last: int, rng):
        own = bb_to_plane(my)
        opp = bb_to_plane(their)
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        rng.shuffle(legal)
        k = len(legal)
        oh = np.zeros((k, CELLS), dtype=np.float32)
        oh[np.arange(k), legal] = 1.0

        # my moves that already win, and their replies that already win
        i_win = np.array([has_five(my | (1 << c)) for c in legal])
        t_win = np.array([has_five(their | (1 << c)) for c in legal])
        self._threat = (self._replies_create_four(legal, their, occ)
                        if self.plies > 1 else [])

        if self.plies == 1:
            # Child position, canonical for the opponent to move:
            #   own  = their stones, unchanged
            #   opp  = my stones plus the stone I just played
            #   last = the stone I just played
            # Adding `oh` to the own plane as well put the same stone on both
            # sides, so the network was scoring a position that cannot exist.
            own_g = np.broadcast_to(opp[None, :], (k, CELLS))
            opp_g = np.broadcast_to(own[None, :], (k, CELLS)) + oh
            last_g = oh
            v = self._values(np.stack([own_g, opp_g, last_g], 1).reshape(k, 3, H, W))
            q = -v
            q[i_win] = 1.0
            self.last_q = q
            return self._pick(q, legal, rng)

        # 2-ply: grandchild has *me* to move again
        own_g = own[None, None, :] + oh[:, None, :]           # (k,1,81)
        opp_g = opp[None, None, :] + oh[None, :, :]           # (1,k,81)
        last_g = np.broadcast_to(oh[None, :, :], (k, k, CELLS))
        planes = np.stack([
            np.broadcast_to(own_g, (k, k, CELLS)),
            np.broadcast_to(opp_g, (k, k, CELLS)),
            last_g,
        ], axis=1).reshape(k * k, 3, H, W)
        v = self._values(planes).reshape(k, k)

        # their winning reply is a loss for me
        bad = np.broadcast_to(t_win[None, :], (k, k))
        v = np.where(bad, -1.0, v)
        # my own legal move is not available to them; keep it out of the min
        v[np.arange(k), np.arange(k)] = np.inf
        v = np.where(i_win[:, None], 1.0, v)                  # I win immediately

        v_child = None
        if self.agg == "selective":
            own_c = np.broadcast_to(opp[None, :], (k, CELLS))
            opp_c = np.broadcast_to(own[None, :], (k, CELLS)) + oh
            v_child = self._values(
                np.stack([own_c, opp_c, oh], 1).reshape(k, 3, H, W))
        q = self._aggregate(v, k, i_win, t_win, legal, their, occ, v_child)
        self.last_q = q
        return self._pick(q, legal, rng)

    def _aggregate(self, v, k, i_win, t_win, legal, their, occ, v_child=None):
        """Collapse the k opponent replies of each move into one number.

        `min` over all replies is an extreme order statistic: with ~45 noisy
        value estimates its result is set by the error tail, not by the
        opponent's actual best reply. Measured effect: 45 moves collapse to 7.7
        distinct Q values and 7.2 of them tie at the top, so argmax picks
        essentially at random.

        Two fixes:
          * a reply that creates a four is genuinely forcing, so those (usually
            0-5 of them) are still combined with a hard min;
          * when the opponent has no forcing reply, the quantile is used instead,
            which is robust to a single unlucky estimate.
        """
        n_winning = int(t_win.sum())
        q = np.empty(k)
        for i in range(k):
            if i_win[i]:
                q[i] = 1.0
                continue
            # replies that five immediately, excluding the cell I just took
            if n_winning - (1 if t_win[i] else 0) > 0:
                q[i] = -1.0
                continue
            cols = [j for j in range(k) if j != i]
            if not cols:
                q[i] = 0.0
                continue
            if self.agg == "selective":
                # Base on the reliable 1-ply estimate and go a ply deeper ONLY
                # where the opponent has a forcing reply. Taking a min over all
                # ~45 replies is what destroys the ordering: it is an extreme
                # order statistic, so it reports the worst value error rather
                # than the opponent's best move.
                q[i] = -float(v_child[i])
                th = [j for j in cols if self._threat[j]]
                if th:
                    q[i] = min(q[i], float(v[i, th].min()))
            elif self.agg == "min":
                q[i] = v[i, cols].min()
            elif self.agg == "quantile":
                q[i] = np.quantile(v[i, cols], self.q_quantile)
            else:                                   # "threat"
                th = [j for j in cols if self._threat[j]]
                if th:
                    q[i] = v[i, th].min()
                else:
                    q[i] = np.quantile(v[i, cols], self.q_quantile)
        return q

    def _replies_create_four(self, legal, their, occ):
        """For each legal reply, does the opponent make a four by playing it?"""
        out = []
        for c in legal:
            cells = _five_cells_window_scan(their | (1 << c), occ | (1 << c),
                                            WINDOWS_AT[c])
            out.append(bool(cells))
        return out


class GreedyAgent:
    """Complete my five, else block theirs, else random."""

    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c
        return rng.choice(legal)


class GreedyPlusAgent:
    """Win > block five > make a double threat > block theirs > random."""

    def _double(self, att, occ, c):
        cells = _five_cells_window_scan(att | (1 << c), occ | (1 << c),
                                        WINDOWS_AT[c])
        return len(cells) >= 2

    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c
        for c in legal:
            if self._double(my, occ, c):
                return c
        for c in legal:
            if self._double(their, occ, c):
                return c
        return rng.choice(legal)


def creates_double_three(me, them, c):
    """After playing `c`, are there two cells that each make a straight four?

    A straight four has two five-completion cells. Any such cell must lie on a
    line through `c`, because the four would include the new stone, so only the
    four lines through `c` need scanning.
    """
    a2 = me | (1 << c)
    occ = a2 | them
    r, col = divmod(c, W)
    cand = set()
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for k in (-4, -3, -2, -1, 1, 2, 3, 4):
            rr, cc = r + dr * k, col + dc * k
            if 0 <= rr < H and 0 <= cc < W:
                x = rr * W + cc
                if not (occ >> x) & 1:
                    cand.add(x)
    n = 0
    for x in cand:
        cells = _five_cells_window_scan(a2 | (1 << x), occ | (1 << x),
                                        WINDOWS_AT[x])
        if len(cells) >= 2:
            n += 1
            if n >= 2:
                return True
    return False


# --------------------------------------------------------------------------
# Frozen rules-based player
#
# FROZEN means frozen. The rule set was tuned to convergence against itself:
# every added rule either ties or loses (see manual.md section 8), and a
# self-play sweep of 1,056 mid-game positions found no position it answers
# wrongly. Do not add rules here. If you believe you have found a real gap, the
# acceptance bar is in AGENT.md section 3 and it is not "it looks better".
# --------------------------------------------------------------------------
FROZEN_HEURISTIC = dict(
    vcf_depth=7,       # coverage saturates at depth 5; 7 costs 3.8 ms and is safe
    use_vcf=True,      # removing VCF drops it from 97.5% to 18% against greedy+
    use_score=True,
    topk=1,            # >1 randomises only among the top quiet moves
    double_three=True, # +3.0 sigma over the version without it
    dt_top=8,
    dt_defence=True,   # unproven (+1.8 sigma) but free; rarely applicable
    dt_top_def=8,
)


class HeuristicAgent:
    """The frozen rules-based player. Build it with `frozen_heuristic()`.

    Rule ladder, highest priority first:

      1. make five
      2. block their five
      3. VCF attack   (continuous fours, depth `vcf_depth`)
      4. VCF defence  (occupy the first cell of their winning line)
      5. my double three   (two independent straight-four threats)
      6. their double three (occupy the cell that would give them one)
      7. static window score

    Rules 1-3 and 5 are exact: they prove a win or they stay silent. Rules 4, 6
    and 7 are heuristic.
    """

    # Stone counts in a 5-window, summed over every window through the cell. The
    # sum is what makes this work: a cell that extends several windows at once
    # accumulates, which a single-pattern-per-direction table loses. Measured:
    # replacing this with a line-pattern table costs 22.7 points (-6.4 sigma).
    PATTERN = {4: 1000, 3: 100, 2: 10, 1: 1}

    def __init__(self, vcf_depth: int = 7, use_vcf: bool = True,
                 use_score: bool = True, topk: int = 1,
                 double_three: bool = True, dt_top: int = 8,
                 dt_defence: bool = True, dt_top_def: int | None = None):
        self.vcf_depth = vcf_depth
        self.use_vcf = use_vcf
        self.use_score = use_score
        # topk > 1 randomises only among the best-scoring *quiet* moves. Forced
        # moves (five, block, VCF) stay deterministic, so rollouts get diversity
        # without ever blundering tactically.
        self.topk = topk
        # double_three: check the top `dt_top` candidates for a move creating two
        # independent straight-four threats. VCF cannot see these, because a live
        # three creates no five cell. +3.0 sigma over the version without it.
        self.double_three = double_three
        self.dt_top = dt_top
        # dt_defence: also occupy the cell that would give the opponent a double
        # three. UNPROVEN (+1.8 sigma over 700 paired games, interval includes
        # 50%) and in practice almost never reachable: whenever the opponent can
        # make one, so can we, and we move first. Kept because it is free.
        self.dt_defence = dt_defence
        self.dt_top_def = dt_top if dt_top_def is None else dt_top_def

    def _cell_score(self, me, them, occ, c):
        """What playing at `c` is worth to `me`, ignoring `them` as blockers.

        Purely offensive, and that is deliberate. A balanced version
        (`my gain - defence * their gain`) looks more correct but measures worse:
        43.3% head to head (-2.9 sigma). With defence > 1 it also penalises
        central cells, because a cell with more windows offers more to both sides.
        """
        a2 = me | (1 << c)
        s = 0
        for win in WINDOWS_AT[c]:
            k = 0
            for i in win:
                if (a2 >> i) & 1:
                    k += 1
                elif (them >> i) & 1:
                    k = 0
                    break
            s += self.PATTERN.get(k, 0)
        return s

    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c
        if self.use_vcf:
            m = vcf_move(my, their, self.vcf_depth)
            if m is not None:
                return m
            m = vcf_move(their, my, self.vcf_depth)
            if m is not None:
                return m
        if self.use_score:
            scored = sorted((self._cell_score(my, their, occ, c), c)
                            for c in legal)
            if self.double_three:
                for _, c in reversed(scored[-self.dt_top:]):
                    if creates_double_three(my, their, c):
                        return c
                if self.dt_defence:
                    their_scored = sorted(
                        (self._cell_score(their, my, occ, c), c)
                        for c in legal)
                    for _, c in reversed(their_scored[-self.dt_top_def:]):
                        if creates_double_three(their, my, c):
                            return c
            if self.topk > 1:
                return rng.choice([c for _, c in scored[-self.topk:]])
            return scored[-1][1]
        return rng.choice(legal)


def frozen_heuristic():
    """The frozen rules-based player. Prefer this over constructing by hand."""
    return HeuristicAgent(**FROZEN_HEURISTIC)


class RandomAgent:
    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        return rng.choice(legal) if legal else None


def play(black, white, rng, opening=0):
    """Returns 0 (black wins), 1 (white wins) or None (draw).

    The first `opening` plies are uniformly random. Without this a deterministic
    agent replays the same game every time, so "N games" is really one game and
    the win-rate statistics are fiction.
    """
    bb = [0, 0]
    last = -1
    turn = 0
    for ply in range(CELLS):
        if ply < opening:
            occ = bb[0] | bb[1]
            legal = [c for c in range(CELLS) if not (occ >> c) & 1]
            m = rng.choice(legal)
        else:
            agent = black if turn == 0 else white
            m = agent.move(bb[turn], bb[turn ^ 1], last, rng)
        if m is None:
            return None
        bb[turn] |= 1 << m
        last = m
        if has_five(bb[turn]):
            return turn
        turn ^= 1
    return None


def match(a, b, games, seed=0, opening=0):
    """a vs b, alternating colours. Returns win rate of `a`."""
    rng = random.Random(seed)
    wins = draws = 0
    for i in range(games):
        if i % 2 == 0:
            r = play(a, b, rng, opening)
            wins += r == 0
            draws += r is None
        else:
            r = play(b, a, rng, opening)
            wins += r == 1
            draws += r is None
    return wins / games, draws / games


def load_net(path, dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = ValueNet(**ck["config"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


class PolicyAgent:
    """One forward pass: argmax over the policy logits. No search, no value.

    The two safety overrides (finish my own five, block theirs) run first and are
    identical to the ones inside SearchAgent, so a comparison against the 1-ply
    search isolates the policy head rather than the ability to spot an immediate
    five.
    """

    def __init__(self, net, dev="cpu"):
        self.net = net
        self.dev = dev
        self.last_logits = None

    @torch.no_grad()
    def move(self, my: int, their: int, last: int, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c

        x = np.zeros((1, 3, H, W), dtype=np.float32)
        x[0, 0] = bb_to_plane(my).reshape(H, W)
        x[0, 1] = bb_to_plane(their).reshape(H, W)
        if last >= 0:
            x[0, 2].flat[last] = 1.0
        out = self.net(torch.from_numpy(x).to(self.dev))
        logits = (out[1] if isinstance(out, tuple) else out)[0].cpu().numpy()
        self.last_logits = logits
        return max(legal, key=lambda c: float(logits[c]))


class HybridAgent:
    """The heuristic's tactical ladder in front of the learned policy.

    Each half does what the other cannot. VCF and double-three detection are exact
    -- they prove a win or stay silent -- and the policy net misses them entirely.
    In quiet positions the heuristic falls back on a crude window score, while the
    policy was trained on what strong engines actually played.

    Order is deliberate: an exact fact beats any evaluation, so the ladder runs
    first and the policy only decides where no forcing line exists.
    """

    def __init__(self, net, dev="cpu", vcf_depth: int = 7, double_three: bool = True):
        self.net = net
        self.dev = dev
        self.vcf_depth = vcf_depth
        self.double_three = double_three
        self.policy = PolicyAgent(net, dev)
        self.last_source = None

    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None

        for c in legal:                              # 1. make five
            if has_five(my | (1 << c)):
                self.last_source = "five"
                return c
        for c in legal:                              # 2. block five
            if has_five(their | (1 << c)):
                self.last_source = "block"
                return c

        m = vcf_move(my, their, self.vcf_depth)      # 3. VCF attack
        if m is not None:
            self.last_source = "vcf"
            return m
        m = vcf_move(their, my, self.vcf_depth)      # 4. VCF defence
        if m is not None:
            self.last_source = "vcf_def"
            return m

        if self.double_three:
            scored = sorted(((self.policy.last_logits[c] if
                              self.policy.last_logits is not None else 0.0, c)
                             for c in legal))
            order = [c for _, c in reversed(scored)]
            for c in order[:8]:                      # 5. my double three
                if creates_double_three(my, their, c):
                    self.last_source = "double_three"
                    return c
            for c in order[:8]:                      # 6. block theirs
                if creates_double_three(their, my, c):
                    self.last_source = "block_dt"
                    return c

        self.last_source = "policy"                  # 7. learned policy
        return self.policy.move(my, their, last, rng)


def load_policy_net(path, dev="cpu"):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = ValuePolicyNet(**ck["config"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


class PolicySearchAgent:
    """Negamax to a fixed depth, narrowed by the policy, evaluated by the value head.

    This is the "real search" answer to a 1-ply agent that lost 23.5 points: taking
    argmax of -V(child) over every legal cell makes one value error decisive for
    every candidate. Here the opponent gets a reply, the candidate set is ordered
    and pruned by the policy prior (which is the head trained on engine moves), and
    the value head only has to score a handful of positions per move.

    Depth 2 means: my move, their best reply, and I take the value of the resulting
    position. `width` and `replies` control how much of the tree survives the prior.

    The tactical layer stays: an immediate five is played, an immediate threat is
    blocked, because no evaluation beats an exact fact.
    """

    def __init__(self, net, dev="cpu", width=12, replies=8):
        self.net = net
        self.dev = dev
        self.width = width
        self.replies = replies
        self.last_q = None

    @torch.no_grad()
    def _forward(self, planes):
        out = self.net(torch.from_numpy(planes).to(self.dev))
        if isinstance(out, tuple):
            v, p = out
        else:
            v, p = out, None
        v = torch.softmax(v, 1)
        value = (v[:, 2] - v[:, 0]).cpu().numpy()
        pol = None if p is None else p.cpu().numpy()
        return value, pol

    @staticmethod
    def _planes(own, opp, last_cells):
        own = np.atleast_2d(np.asarray(own, dtype=np.float32))
        opp = np.atleast_2d(np.asarray(opp, dtype=np.float32))
        n = own.shape[0]
        x = np.zeros((n, 3, H, W), dtype=np.float32)
        x[:, 0] = own.reshape(n, H, W)
        x[:, 1] = opp.reshape(n, H, W)
        for i, c in enumerate(last_cells):
            if c >= 0:
                x[i, 2].flat[c] = 1.0
        return x

    def move(self, my, their, last, rng):
        occ = my | their
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None

        # exact facts first
        for c in legal:
            if has_five(my | (1 << c)):
                return c
        for c in legal:
            if has_five(their | (1 << c)):
                return c

        own_p = bb_to_plane(my)
        opp_p = bb_to_plane(their)
        _, pol = self._forward(self._planes(own_p, opp_p, [last]))
        if pol is None:
            return rng.choice(legal)
        pol = pol[0]

        mine = sorted(legal, key=lambda c: -float(pol[c]))[:self.width]
        oh = np.zeros((1, CELLS), dtype=np.float32)
        roots = []
        for c in mine:
            oh[0, c] = 1.0
            roots.append((own_p + oh[0], opp_p, c))
            oh[0, c] = 0.0

        # their replies, one batched forward per root position
        own_r = np.stack([r[0] for r in roots])
        opp_r = np.stack([r[1] for r in roots])
        last_r = np.array([r[2] for r in roots])
        _, rpol = self._forward(self._planes(own_r, opp_r, last_r))

        child_own, child_opp, child_last, owner = [], [], [], []
        for i, c in enumerate(mine):
            occ2 = occ | (1 << c)
            reply_legal = [x for x in range(CELLS) if not (occ2 >> x) & 1]
            if not reply_legal:
                continue
            if rpol is not None:
                order = sorted(reply_legal, key=lambda x: -float(rpol[i][x]))
            else:
                order = reply_legal
            for x in order[:self.replies]:
                # grandchild, canonical for me to move:
                #   own = my stones including c   opp = their stones including x
                own_g = own_r[i].copy()
                opp_g = opp_r[i].copy()
                opp_g[x] = 1.0
                child_own.append(own_g)
                child_opp.append(opp_g)
                child_last.append(x)
                owner.append(i)

        if not child_own:
            return mine[0]

        n = len(owner)
        planes = np.zeros((n, 3, H, W), dtype=np.float32)
        planes[:, 0] = np.stack(child_own).reshape(n, H, W)
        planes[:, 1] = np.stack(child_opp).reshape(n, H, W)
        for k, c in enumerate(child_last):
            planes[k, 2].flat[c] = 1.0
        value, _ = self._forward(planes)

        q = np.full(len(mine), 1.0, dtype=np.float32)
        seen = np.zeros(len(mine), dtype=bool)
        for k, i in enumerate(owner):
            if not seen[i]:
                q[i] = value[k]
                seen[i] = True
            else:
                q[i] = min(q[i], value[k])
        self.last_q = q
        return mine[int(np.argmax(q))]
