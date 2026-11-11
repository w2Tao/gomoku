"""Play our agent against a Gomocup engine: the external yardstick.

The engines here are written by other people and are far stronger than anything in
this repository. The point is a fixed external coordinate system, not a fair fight.

    python code/fetch_engines.py EMBRYO26
    python code/duel_engine.py EMBRYO26 baseline 20 --time-ms 300

Games start from an empty board and use only `BEGIN` / `TURN`. `BOARD` is avoided
on purpose: its semantics differ between engines and the resulting board diverged
from the engine's own, producing "opponent's move [...]" errors. Opening variety
comes from our side instead -- our agent plays `--opening` random moves before it
starts thinking. Colour alternates every game, so the randomisation applies
equally to both colours.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import baselines
from engine import CELLS, has_five
from fetch_engines import ENGINE
from play import PolicyAgent
from pbrain import EngineError, Pbrain, find_exe


TRACE = []


def play_game(engine, our_agent, rng, timeout, engine_black, opening):
    """-> 0 if our agent won, 1 if the engine won, None for a draw."""
    our_colour = 1 if engine_black else 0
    engine_colour = 1 - our_colour
    TRACE.clear()
    bb = [0, 0]
    last = -1
    random_left = opening

    mv = engine.begin(engine_black)
    turn = 0

    for _ in range(CELLS):
        occ = bb[0] | bb[1]
        legal = [c for c in range(CELLS) if not (occ >> c) & 1]
        if not legal:
            return None

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
            cell = our_agent.move(bb[turn], bb[turn ^ 1], last, rng)
            if cell is None:
                return None

        TRACE.append((len(TRACE), "E" if turn == engine_colour else "U", cell))
        bb[turn] |= 1 << cell
        last = cell
        if has_five(bb[turn]):
            return 0 if turn == our_colour else 1
        turn ^= 1
        if turn == engine_colour:
            mv = engine.play(last % 15, last // 15, timeout)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("baseline", nargs="?", default="baseline")
    ap.add_argument("games", nargs="?", type=int, default=40)
    ap.add_argument("--prefer", default=None, help="substring of the exe to use")
    ap.add_argument("--time-ms", type=int, default=500, help="per move")
    ap.add_argument("--stall-s", type=float, default=None,
                    help="kill the engine after this long with no output "
                         "(default: the engine's registered value, else 10)")
    ap.add_argument("--opening", type=int, default=2,
                    help="random moves our agent plays before thinking")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shard", type=int, default=0,
                    help="which strided slice of --games this process plays")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--json", default=None,
                    help="write {wins,losses,draws,errors} here for aggregation")
    args = ap.parse_args()

    prefer = args.prefer or ENGINE().get(args.engine, {}).get("prefer")
    exe = find_exe(os.path.join("data", "engines", args.engine), prefer)
    print(f"engine   : {os.path.basename(exe)}  ({args.time_ms} ms/move)")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, manifest = baselines.load_baseline(args.baseline, dev)
    agent = PolicyAgent(net, dev)
    print(f"opponent : {args.baseline}  "
          f"({manifest['model_type']}, {manifest['parameters']:,} params)")
    print(f"{args.games} games, colours alternating, "
          f"{args.opening} random opening moves from us\n", flush=True)

    rng = random.Random(7)
    wins = losses = draws = errors = 0
    meta = ENGINE().get(args.engine, {})
    stall = args.stall_s if args.stall_s is not None else meta.get("stall_s", 10.0)
    print(f"stall    : {stall:.0f}s of silence before a kill\n", flush=True)
    eng = Pbrain(exe, time_ms=args.time_ms, verbose=args.verbose,
                 stall_s=stall,
                 begin_white=meta.get("begin_white", False))
    # games are independent, so a shard takes a strided slice and the driver sums
    order = list(range(args.games))[args.shard::args.shards]
    for i in order:
        engine_black = i % 2 == 1
        try:
            # A fresh process per game. Reusing one leaks stale output into the
            # next game -- the engine keeps talking after the game ends, and the
            # next `begin()` then reads a leftover coordinate as its own move.
            eng.start()
            r = play_game(eng, agent, rng, args.time_ms * 10, engine_black,
                          args.opening)
        except EngineError as e:
            errors += 1
            print(f"  game {i + 1} "
                  f"({'engine black' if engine_black else 'engine white'}): {e}",
                  flush=True)
            tail = " ".join(f"{ply}{who}{cell}"
                            for ply, who, cell in TRACE[-14:])
            print(f"    last plies ({len(TRACE)} total): {tail}", flush=True)
            continue
        finally:
            try:
                eng.close()
            except Exception:
                pass
        if r is None:
            draws += 1
        elif r == 0:
            wins += 1
        else:
            losses += 1
        done = wins + losses + draws
        if done and done % 5 == 0:
            print(f"  {done:>3} games  win {wins/done:5.1%}  "
                  f"loss {losses/done:5.1%}", flush=True)
    eng.close()

    done = wins + losses + draws
    print()
    if done:
        p = wins / done
        se = (0.25 / done) ** 0.5
        print(f"score vs {args.engine}: {wins}/{done} = {p:.1%}  "
              f"(loss {losses}, draw {draws})")
        print(f"  95% interval {max(0, p-1.96*se):.1%} .. {min(1, p+1.96*se):.1%}")
    else:
        print("no games completed")
    if errors:
        print(f"  {errors} games aborted by engine errors")
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"engine": args.engine, "model": args.baseline,
                       "wins": wins, "losses": losses, "draws": draws,
                       "errors": errors}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
