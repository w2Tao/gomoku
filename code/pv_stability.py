"""Is the distilled label stable with search depth?

Distillation never used the Eval -- build_distill_dataset discards it and sets every
value_weight to zero, so the labels are the principal variation only. But the PV is
also a product of search depth, and the collection ran at 300 ms. If the engine picks
a different move when it thinks longer, then the labels are a 300 ms teacher's opinion
rather than a strong one -- which is not necessarily wrong (the phase-1 comparison
that produced v1 (training tag d3) was also run at 300 ms) but should be known rather than assumed.

Measures the engine's reply to a fixed opening at a shallow and a deep budget and
reports how often they agree.

    python code/pv_stability.py --limit 10 --shallow 300 --deep 5000
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "code")

from engine import CELLS, has_five                     # noqa: E402
from fetch_engines import ENGINE                       # noqa: E402
from opening_eval import corpus_openings               # noqa: E402
from pbrain import EngineError, Pbrain, find_exe       # noqa: E402


def engine_reply(exe, line, time_ms):
    """Play `line` as Black, return the engine's first reply cell (or None)."""
    eng = Pbrain(exe, time_ms=time_ms, stall_s=40.0)
    eng.start()
    try:
        eng.begin(False)            # engine is White, we move first
        bb = [0, 0]
        last = -1
        turn = 0
        ours = list(line)
        mv = None
        for _ in range(2 * len(line) + 4):
            occ = bb[0] | bb[1]
            if turn == 0:
                if not ours:
                    break
                cell = ours.pop(0)
                if (occ >> cell) & 1:
                    return None
            else:
                if mv is None:
                    break
                x, y = mv
                cell = y * 15 + x
                if not (0 <= cell < CELLS) or (occ >> cell) & 1:
                    return None
            bb[turn] |= 1 << cell
            last = cell
            if has_five(bb[turn]):
                return None
            turn ^= 1
            if turn == 1:
                mv = eng.play(last % 15, last // 15, time_ms * 10)
                if not ours:
                    ox, oy = mv
                    return oy * 15 + ox      # reply to the completed line
        return None
    except EngineError:
        return None
    finally:
        try:
            eng.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--shallow", type=int, default=300)
    ap.add_argument("--deep", type=int, default=5000)
    args = ap.parse_args()

    exe = find_exe("data/engines/RAPFI25", ENGINE()["RAPFI25"]["prefer"])
    openings = corpus_openings(args.limit)
    print(f"  shallow {args.shallow} ms vs deep {args.deep} ms, one reply each\n")

    same = compared = 0
    for line, games in openings:
        seq = " ".join(f"({c % 15},{c // 15})" for c in line)
        a = engine_reply(exe, line, args.shallow)
        b = engine_reply(exe, line, args.deep)
        if a is None or b is None:
            print(f"  {seq:<44} unreadable ({a}, {b})")
            continue
        compared += 1
        agree = a == b
        same += agree
        print(f"  {seq:<44} shallow=({a % 15},{a // 15}) "
              f"deep=({b % 15},{b // 15})  {'same' if agree else 'DIFFERENT'}")

    if compared:
        print(f"\n  agreement {same}/{compared} = {same/compared:.0%}")
        print("  (one sample per line; this bounds the disagreement rate, "
              "it does not pin it down)")


if __name__ == "__main__":
    main()
