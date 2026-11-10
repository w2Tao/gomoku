"""Local web UI: play 9x9 gomoku against a frozen baseline.

    python app.py             # from the project root: starts the server and
                              # opens the browser (the normal entry point)
    python code/server.py     # server only, e.g. for a headless run

Deliberately dependency-free: the standard library HTTP server plus one static
HTML file. The server owns the rules (it validates every move and detects the
end of the game), the browser only renders and sends the move list.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import baselines
import paths
from engine import CELLS, H, W, has_five
from play import RandomAgent, SearchAgent, frozen_heuristic

HOST = "127.0.0.1"
PORT = int(os.environ.get("GOMOKU_PORT", "8765"))
WEB = str(paths.WEB)

AGENTS: dict[str, object] = {}
RNG = random.Random(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def model_catalogue():
    items = []
    for b in baselines.list_baselines():
        items.append({
            "id": b["id"],
            "label": f"{b['name']}  (vs 贪心 {b['strength_vs_greedy']:.0%})"
                     if b["strength_vs_greedy"] is not None else b["name"],
            "kind": "baseline",
            "detail": {
                "params": b["params"],
                "step": b["source_step"],
                "vs_greedy": b["strength_vs_greedy"],
                "search_depth": b["inference"].get("search_depth_plies"),
            },
        })
    items.append({"id": "heuristic",
                  "label": "启发式（VCF 深度 7 + 威胁评分）",
                  "kind": "heuristic", "detail": {}})
    items.append({"id": "random", "label": "纯随机", "kind": "heuristic",
                  "detail": {}})
    return items


def get_agent(model_id: str):
    if model_id in AGENTS:
        return AGENTS[model_id]
    if model_id == "heuristic":
        a = frozen_heuristic()
    elif model_id == "random":
        a = RandomAgent()
    else:
        net, _manifest = baselines.load_baseline(model_id, DEV)
        a = SearchAgent(net, DEV, plies=1)
    AGENTS[model_id] = a
    return a


def replay(moves):
    """Rebuild (black, white) and validate that the move list is a legal game."""
    if len(moves) > CELLS:
        raise ValueError("着法过多")
    bb = [0, 0]
    seen = set()
    for i, m in enumerate(moves):
        if not isinstance(m, int) or not (0 <= m < CELLS):
            raise ValueError("非法坐标")
        if m in seen:
            raise ValueError("该点已有子")
        seen.add(m)
        turn = i % 2
        bb[turn] |= 1 << m
        if has_five(bb[turn]) and i != len(moves) - 1:
            raise ValueError("对局已经结束")
    return bb


def status_of(bb, moves):
    if moves:
        last_turn = (len(moves) - 1) % 2
        if has_five(bb[last_turn]):
            return True, last_turn
    if len(moves) >= CELLS:
        return True, None
    return False, None


def handle_play(payload):
    moves = list(payload.get("moves", []))
    model_id = payload.get("model", "baseline0")
    bb = replay(moves)
    over, winner = status_of(bb, moves)
    if over:
        return {"moves": moves, "over": True, "winner": winner, "ms": 0.0,
                "modelMove": None}

    turn = len(moves) % 2
    my, their = bb[turn], bb[turn ^ 1]
    last = moves[-1] if moves else -1
    agent = get_agent(model_id)

    t0 = time.perf_counter()
    move = agent.move(my, their, last, RNG)
    ms = (time.perf_counter() - t0) * 1000.0
    if move is None:
        return {"moves": moves, "over": True, "winner": None, "ms": ms,
                "modelMove": None}

    moves.append(int(move))
    bb[turn] |= 1 << int(move)
    over, winner = status_of(bb, moves)
    return {"moves": moves, "over": over, "winner": winner, "ms": ms,
            "modelMove": int(move)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            fp = os.path.join(WEB, "index.html")
            if not os.path.isfile(fp):
                return self._send(500, {"error": "index.html 缺失"}, "text/plain")
            with open(fp, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/models":
            return self._send(200, {"models": model_catalogue(),
                                    "default": (model_catalogue()[0]["id"]
                                                if model_catalogue() else "greedy"),
                                    "device": DEV})
        return self._send(404, {"error": "not found"}, "application/json")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/play":
            return self._send(404, {"error": "not found"}, "application/json")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            return self._send(200, handle_play(payload))
        except Exception as exc:                       # noqa: BLE001
            return self._send(400, {"error": str(exc)})


def prepare():
    """Load and warm every baseline before the first request arrives.

    A lazy first request costs ~10 s on CUDA (device context + cuDNN autotuning)
    and looks exactly like a hang, so pay that cost at startup instead.
    """
    catalogue = model_catalogue()
    print(f"device : {DEV}")
    print(f"models : {[m['id'] for m in catalogue]}")
    for item in catalogue:
        if item["kind"] != "baseline":
            continue
        t0 = time.perf_counter()
        agent = get_agent(item["id"])
        if isinstance(agent, SearchAgent):
            # warm up at a realistic batch size: cuDNN autotunes per shape, so a
            # batch-1 probe would leave the first real move slow
            with torch.no_grad():
                agent.net(torch.zeros(CELLS, 3, H, W, device=DEV))
        print(f"loaded : {item['id']} ({time.perf_counter()-t0:.1f}s)")
    return catalogue


def create_server(preferred: int = PORT):
    """Bind `preferred`, falling back to any free port if it is taken."""
    try:
        return ThreadingHTTPServer((HOST, preferred), Handler)
    except OSError:
        return ThreadingHTTPServer((HOST, 0), Handler)


def main():
    prepare()
    httpd = create_server()
    print(f"serving: http://{HOST}:{httpd.server_address[1]}")
    print("ctrl-c to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
