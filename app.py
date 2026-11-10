"""One-click launcher for the play UI.

    Windows : double-click app.bat
    anywhere: python app.py

Starts the local server, then opens the browser. Everything else (weights, data,
scripts) lives under ckpt/, data/ and code/; this file is only a launcher.
"""
from __future__ import annotations

import os
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))

try:                                  # keep the banner readable in a cp936 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                     # noqa: BLE001
    pass


def main() -> int:
    import paths
    import server

    paths.ensure()

    print("=" * 46)
    print("  五子棋 15x15 · 人机对弈")
    print("=" * 46)
    print("正在加载模型…", flush=True)
    server.prepare()

    httpd = server.create_server()
    port = httpd.server_address[1]
    url = f"http://{server.HOST}:{port}/"

    print()
    print(f"对弈界面 : {url}")
    print("浏览器会自动打开；关闭本窗口或按 Ctrl-C 退出。")
    print()

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
