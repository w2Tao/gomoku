"""Score the known openings with the champion's own evaluation.

The corpus cannot rank openings: the tournament gives every pairing every designated
opening, colours swapped, and the popular lines all score ~50% for black. So the book
idea has no premise there. RAPFI's MESSAGE stream does carry a verdict, the `Eval`
field, but only for positions it reaches by playing -- pbrain has no reliable way to
hand it an arbitrary board.

The way round it: play the opening as OUR moves and let RAPFI answer. Its first turn
comes right after the line we want judged, and the Eval it prints there is the value
of that line from its side. Flip the sign when it is the second player and the number
is our advantage.

    python code/opening_eval.py --games-per-line 1 --time-ms 300

Only openings seen in the corpus are scored, since no other candidate is known to be
worth asking about.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "code")

from build_gomocup_dataset import parse, replay      # noqa: E402
from engine import CELLS, has_five                   # noqa: E402
from fetch_engines import ENGINE                     # noqa: E402
from pbrain import EngineError, Pbrain, find_exe, rapfi_targets   # noqa: E402

DEPTH = 4


def corpus_openings(limit=None):
    """Distinct opening lines from the corpus with their game counts."""
    counts = defaultdict(int)
    for path in sorted(glob.glob("data/gomocup/Freestyle15_*/*.psq")):
        moves = parse(path)
        if replay(moves) is None or len(moves) < DEPTH:
            continue
        counts[tuple((y - 1) * 15 + (x - 1) for x, y in moves[:DEPTH])] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return ranked[:limit] if limit else ranked


def score_line(eng, line, time_ms, trace=None):
    """Play `line` as our moves (we are Black) and return RAPFI's verdict on it.

    We must play first. Corpus lines usually start at the centre, and an engine given
    Black takes the centre immediately, so asking it to open makes our first move
    collide with its stone and score nothing -- that is why every "we are White" case
    returned None. Scoring only as Black also makes the number directly comparable
    across lines: it is Black's advantage after the line.

    `trace`, when given, collects the eval after each of our moves so the scale can be
    checked rather than assumed.
    """
    eng.start()
    try:
        eng.begin(False)                       # engine is White, so we move first
        turn = 0
        engine_colour = 1
        bb = [0, 0]
        last = -1
        ours = list(line)
        mv = None
        for _ in range(64):
            occ = bb[0] | bb[1]
            if not any(not (occ >> c) & 1 for c in range(225)):
                break
            if turn == engine_colour:
                if mv is None:
                    break
                x, y = mv
                cell = y * 15 + x
                if not (0 <= cell < 225) or (occ >> cell) & 1:
                    break
                ev, _pv = rapfi_targets(eng.last_messages)
                if ev is not None:
                    # Eval is from the engine's own point of view; we are Black
                    if trace is not None:
                        trace.append(-ev)
                    if not ours:
                        return -ev          # verdict on the completed opening
            else:
                legal = [c for c in range(225) if not (occ >> c) & 1]
                if not ours:
                    return None
                cell = ours.pop(0)
                if (occ >> cell) & 1 or cell not in legal:
                    return None
            bb[turn] |= 1 << cell
            last = cell
            if has_five(bb[turn]):
                return None
            turn ^= 1
            if turn == engine_colour:
                mv = eng.play(last % 15, last // 15, time_ms * 10)
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
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--time-ms", type=int, default=300)
    ap.add_argument("--out", default="opening_scores.json")
    args = ap.parse_args()

    exe = find_exe(os.path.join("data", "engines", "RAPFI25"),
                   ENGINE()["RAPFI25"]["prefer"])
    openings = corpus_openings(args.limit)
    print(f"  scoring {len(openings)} opening lines with RAPFI25 @ "
          f"{args.time_ms} ms\n")

    results = []
    for idx, (line, games) in enumerate(openings):
        eng = Pbrain(exe, time_ms=args.time_ms, stall_s=25.0)
        trace = [] if idx == 0 else None
        v = score_line(eng, line, args.time_ms, trace)
        if trace:
            print(f"  calibration (eval after each of our moves, line 1): {trace}",
                  flush=True)
        mean = v
        seq = " ".join(f"({c % 15},{c // 15})" for c in line)
        results.append({"line": line, "seq": seq, "games": games, "mean": mean})
        print(f"  {idx+1:>2}. {seq:<44} n={games:>5} "
              f"our advantage={mean}", flush=True)

    good = [r for r in results if r["mean"] is not None]
    good.sort(key=lambda r: -r["mean"])
    print("\n  ranked by our advantage (higher is better for us):")
    for r in good:
        print(f"    {r['mean']:>+8.1f}  {r['seq']}")

    out = args.out if os.path.dirname(args.out) else os.path.join("data", args.out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
