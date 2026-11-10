"""Drive a Gomocup engine over the pbrain protocol.

This is how the project measures itself against something it did not write. The
engines are far stronger than anything here -- the point is a fixed, external
coordinate system, not a fair fight.

Protocol notes, all learned the hard way against RAPFI/EMBRYO/TITO/MONAS:

* `START <size>` -> `OK`. Coordinates are 1-indexed, `x,y`.
* Without `INFO timeout_turn` the engine keeps deepening forever; the first probe
  looked like a hang because of it.
* `BOARD` / `x,y,who` list / `DONE` sets up an arbitrary prefix, where `who` is
  1 for black and 2 for white. This is what lets the paired-opening design work:
  every game can start from the same random prefix instead of from move one.
* After `DONE` the engine answers at once if it is its turn, and stays silent
  otherwise. Sending `TURN` when it is not waiting is an error.
* Output interleaves `MESSAGE`/`DEBUG`/`UNKNOWN` diagnostics with the move; only
  a bare `x,y` (or `ERROR ...`) is the answer.
* The engine's working directory must be its own folder -- weights are relative.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

STARTUP_WAIT = 3.0     # banner + info exchange before the first command


class EngineError(RuntimeError):
    pass


class Pbrain:
    def __init__(self, exe, time_ms: int = 500, match_ms: int = 60000,
                 verbose: bool = False, stall_s: float = 10.0):
        self.exe = os.path.abspath(exe)
        self.time_ms = time_ms
        self.match_ms = match_ms
        self.stall_s = stall_s
        self.verbose = verbose
        self.proc = None
        self._q = None
        self._pump = None

    # ---- lifecycle ----------------------------------------------------
    def start(self):
        if not os.path.isfile(self.exe):
            raise EngineError(f"engine not found: {self.exe}")
        self.proc = subprocess.Popen(
            [self.exe], cwd=os.path.dirname(self.exe),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._q = queue.Queue()

        def pump():
            for line in self.proc.stdout:
                self._q.put(line.rstrip())

        self._pump = threading.Thread(target=pump, daemon=True)
        self._pump.start()
        self._send("START 15")
        # Wait for OK before sending INFO: lines sent before the engine is ready
        # get dropped, and then it searches with its own default limit (EMBRYO
        # used 20 s per move even though timeout_turn was set).
        self._await(lambda s: s == "OK", STARTUP_WAIT, required=False)
        self._send(f"INFO timeout_turn {self.time_ms}")
        self._send(f"INFO timeout_match {self.match_ms}")
        return self

    def close(self):
        if self.proc is None:
            return
        try:
            self._send("END")
        except Exception:
            pass
        time.sleep(0.05)
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ---- protocol -----------------------------------------------------
    def _send(self, cmd):
        if self.proc is None:
            raise EngineError("engine not started")
        if self.verbose:
            print(f"    > {cmd}")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _await(self, pred, timeout, required=True):
        """Read until `pred(line)`, skipping diagnostics.

        Two independent limits, because they catch different failures:

        * `stall_s` -- no output at all for this long. The engine is wedged (bad
          board state, waiting on something that will never come). The process is
          killed so the caller can restart it; a wedged engine never recovers.
        * `timeout` -- total time for this answer. Kept much larger than the move
          budget so a slow move is not mistaken for a hang.
        """
        start = time.time()
        last_output = start
        while True:
            now = time.time()
            if now - last_output > self.stall_s:
                self._kill()
                raise EngineError(f"engine silent for {self.stall_s:.0f}s, killed")
            if now - start > timeout:
                break
            try:
                line = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            last_output = time.time()
            if self.verbose:
                print(f"    < {line}")
            if line.startswith("ERROR"):
                raise EngineError(line)
            if line.startswith(("MESSAGE", "DEBUG", "UNKNOWN")) or not line:
                continue
            if pred(line):
                return line
            if line[0].isdigit() and "," in line:
                # Looks like a coordinate but is not a legal one. Fail now instead
                # of skipping it and blocking: an out-of-range move means the
                # engine is misbehaving (MONAS returns x=0 along the left edge).
                raise EngineError(f"illegal move from engine: {line!r}")
        if required:
            raise EngineError(f"no answer within {timeout}s")
        return None

    def _kill(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    @staticmethod
    def _is_move(line):
        """Strictly a board coordinate, 1..15.

        Engines emit other comma-separated lines that look like a move at a
        glance; accepting them produced nonsense cells such as (13, 0). pbrain
        coordinates are 1-indexed, so anything out of range is not a move.
        """
        a, _, b = line.partition(",")
        if not (a.isdigit() and b.isdigit()):
            return False
        return 1 <= int(a) <= 15 and 1 <= int(b) <= 15

    def _read_move(self, timeout):
        line = self._await(self._is_move, timeout)
        x, y = line.split(",")
        return int(x), int(y)

    # ---- game setup ---------------------------------------------------
    def begin(self, engine_black: bool):
        """Start a game from an empty board. Returns the engine's move or None.

        `BOARD` is deliberately not used. Its semantics differ between engines --
        `DONE` makes some search immediately regardless of whose turn it is, the
        `who` field is not uniform, and setting up a prefix reliably diverged from
        the engine's own board (subsequent `TURN`s hit occupied cells). The plain
        `BEGIN` + `TURN` flow is what every engine actually implements.
        """
        self._send("BEGIN")
        if engine_black:
            return self._read_move(max(5000, self.time_ms * 20))
        return None

    def play(self, x, y, timeout=None):
        """Tell the engine the opponent just played here; return its answer."""
        self._send(f"TURN {x},{y}")
        return self._read_move(timeout or self.match_ms)


def find_exe(engine_dir, prefer=None):
    """Pick the freestyle build: engines ship rule-specific variants."""
    cands = []
    for root, _, files in os.walk(engine_dir):
        for f in files:
            if f.lower().endswith(".exe"):
                cands.append(os.path.join(root, f))
    if not cands:
        raise EngineError(f"no .exe under {engine_dir}")
    cands.sort()
    if prefer:
        for c in cands:
            if prefer.lower() in os.path.basename(c).lower():
                return c
    return cands[0]
