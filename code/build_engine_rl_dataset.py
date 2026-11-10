"""Collect outcome-labelled positions from games against a Gomocup engine.

This is the reinforcement-learning alternative to self-play. Self-play failed because
its labels come from two copies of the same weak model -- the winner is whoever
blundered last. Here the opponent is stronger, so a win means the move was actually
good.

The catch is measured, not assumed: against TITO14 the model wins 0% at 300 ms/move
and 10% at 50 ms. Policy gradient on a 10%-win stream is nearly no gradient, so this
records every position and lets the training side weight by outcome instead
(exponentiated, see `--temperature`). Games are cheap -- roughly 3 s each at 50 ms --
so the real limit is how many can be generated in the time available.

    python code/build_engine_rl_dataset.py --engine TITO14 --games 600 \
        --time-ms 50 --out rl1.npz

Records only OUR moves: the engine's moves are already in the Gomocup dataset and
are stronger than anything we would learn from here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np

import baselines
import paths
from dataset import pack_boards
from engine import CELLS, has_five
from fetch_engines import ENGINE
from play import PolicyAgent
from pbrain import EngineError, Pbrain, find_exe

OPENING = 2          # random plies before our model takes over


def onehot(cls_arr):
    out = np.zeros((len(cls_arr), 3), dtype=np.float32)
    out[np.arange(len(cls_arr)), cls_arr] = 1.0
    return out


def play_game(engine, agent, rng, timeout, engine_black):
    """-> (rows, result) where result is 0 our win, 1 engine win, None draw.

    rows are (own, opp, last, cell, mover_won) at our positions only, recorded
    BEFORE the move is played.
    """
    our_colour = 1 if engine_black else 0
    engine_colour = 1 - our_colour
    bb = [0, 0]
    last = -1
    rows = []
    random_left = OPENING

    def record(cell, mover):
        own, opp = (bb[0], bb[1]) if mover == 0 else (bb[1], bb[0])
        rows.append((own, opp, last, cell, mover))

    mv = engine.begin(engine_black)
    turn = 0

    for _ in range(CELLS):
        occ = bb[0] | bb[1]
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return rows, None

        if turn == engine_colour:
            if mv is None:
                raise EngineError("engine owed a move but sent nothing")
            x, y = mv
            cell = y * 15 + x
            if not (0 <= cell < CELLS) or (occ >> cell) & 1:
                raise EngineError(f"engine returned an illegal cell {(x, y)}")
        elif random_left > 0:
            cell = rng.choice(legal)
            random_left -= 1
        else:
            cell = agent.move(bb[turn], bb[turn ^ 1], last, rng)
            if cell is None:
                return rows, None
            record(cell, turn)          # before the move is applied

        bb[turn] |= 1 << cell
        last = cell
        if has_five(bb[turn]):
            return rows, (0 if turn == our_colour else 1)
        turn ^= 1
        if turn == engine_colour:
            mv = engine.play(last % 15, last // 15, timeout)
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="TITO14")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--time-ms", type=int, default=50,
                    help="engine budget; shorter means we can actually win some")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="advantage weighting: weight = exp(outcome / T)")
    ap.add_argument("--out", default="rl1.npz")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    out = args.out if os.path.dirname(args.out) else paths.data(args.out)

    exe = find_exe(os.path.join("data", "engines", args.engine),
                   ENGINE().get(args.engine, {}).get("prefer"))
    meta = ENGINE().get(args.engine, {})

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, man = baselines.load_baseline(args.model, dev)
    agent = PolicyAgent(net, dev)
    rng = random.Random(args.seed)

    eng = Pbrain(exe, time_ms=args.time_ms, stall_s=meta.get("stall_s", 10.0),
                 begin_white=meta.get("begin_white", False))
    all_rows = []
    wins = losses = errors = 0
    for g in range(1, args.games + 1):
        engine_black = g % 2 == 0
        try:
            eng.start()
            rows, result = play_game(eng, agent, rng, args.time_ms * 10,
                                     engine_black)
        except EngineError:
            errors += 1
            continue
        finally:
            try:
                eng.close()
            except Exception:
                pass
        if result is None:
            continue
        if result == 0:
            wins += 1
        else:
            losses += 1
        # outcome from the mover's perspective at each recorded position
        for own, opp, prev, cell, mover in rows:
            won = (result == 0 and mover == 0) or (result == 1 and mover == 1)
            all_rows.append((own, opp, prev, cell, won))
        if g % 50 == 0:
            print(f"  {g} games  win {wins} loss {losses} "
                  f"| {len(all_rows):,} positions", flush=True)
    eng.close()

    total = wins + losses
    print(f"\ngames {total}  win {wins} ({wins/max(total,1):.1%})  "
          f"loss {losses}  aborted {errors}")
    if not all_rows:
        sys.exit("no positions collected")
    print(f"positions {len(all_rows):,}")

    weights = [math.exp((1.0 if r[4] else -1.0) / args.temperature)
               for r in all_rows]
    clss = [2 if r[4] else 0 for r in all_rows]
    # no per-game grouping is recorded; chunk so the val split is not per-position
    gid = np.arange(len(all_rows)) // 64
    rng2 = np.random.default_rng(args.seed)
    games = np.unique(gid)
    val = set(rng2.choice(games, size=max(1, int(len(games) * 0.1)),
                          replace=False).tolist())

    data = dict(
        own=pack_boards([r[0] for r in all_rows]),
        opp=pack_boards([r[1] for r in all_rows]),
        last=np.asarray([r[2] for r in all_rows], dtype=np.int16),
        ply=np.zeros(len(all_rows), dtype=np.uint8),
        cls=np.asarray(clss, dtype=np.uint8),
        game_id=gid,
        is_val=np.isin(gid, list(val)),
        move=np.asarray([r[3] for r in all_rows], dtype=np.int16),
        value_target=onehot(np.asarray(clss, dtype=np.uint8)),
        value_weight=np.asarray(weights, dtype=np.float32),
        meta=np.array(json.dumps({
            "source": f"games vs {args.engine} at {args.time_ms} ms",
            "positions": len(all_rows),
            "our_win_rate": wins / max(total, 1),
            "weighting": f"exp(outcome / {args.temperature})",
        })),
    )
    np.savez_compressed(out, **data)
    print(f"saved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
