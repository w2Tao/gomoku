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


def score_line(eng, line, we_are_black, time_ms):
    """Play `line` as our moves; return RAPFI's eval after it, from our side."""
    eng.start()
    try:
        mv = eng.begin(not we_are_black)       # engine is the other colour
        turn = 0
        bb = [0, 0]
        last = -1
        engine_colour = 1 if we_are_black else 0
        our_colour = 1 - engine_colour
        ours = list(line)
        evals = []
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
                # Only trust an eval once the whole line is on the board. Reading it
                # at the engine's first turn judged a single move, which is why seven
                # different openings reported the same number.
                if ev is not None and not ours:
                    evals.append(-ev if we_are_black else ev)
                if evals:
                    break                # verdict on the completed opening
            else:
                legal = [c for c in range(225) if not (occ >> c) & 1]
                cell = ours.pop(0) if ours else legal[0]
                if (occ >> cell) & 1 or cell not in legal:
                    break
            bb[turn] |= 1 << cell
            last = cell
            from engine import has_five
            if has_five(bb[turn]):
                break
            turn ^= 1
            if turn == engine_colour:
                mv = eng.play(last % 15, last // 15, time_ms * 10)
        return evals[0] if evals else None
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
        per_colour = {}
        for we_black in (True, False):
            v = score_line(eng, line, we_black, args.time_ms)
            per_colour["black" if we_black else "white"] = v
            eng = Pbrain(exe, time_ms=args.time_ms, stall_s=25.0)
        vals = [v for v in per_colour.values() if v is not None]
        mean = statistics.mean(vals) if vals else None
        seq = " ".join(f"({c % 15},{c // 15})" for c in line)
        results.append({"line": line, "seq": seq, "games": games,
                        "black": per_colour["black"],
                        "white": per_colour["white"], "mean": mean})
        print(f"  {idx+1:>2}. {seq:<44} n={games:>5} "
              f"black={per_colour['black']} white={per_colour['white']} "
              f"mean={mean}", flush=True)

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
