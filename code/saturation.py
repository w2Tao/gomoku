"""Is the gain explained by reduced saturation rather than by information?

Compares the output distribution of the three checkpoints on arbitrary (not
solver-selected) mid-game positions. If the shuffled-label placebo is just as
de-saturated as the real one, the mechanism is calibration, not information.
"""

import paths
import random

import numpy as np
import torch

from engine import CELLS, H, W, has_five
from play import bb_to_plane, load_net


def samples(n_games=30, seed=1):
    rng = random.Random(seed)
    pos = []
    for _ in range(n_games):
        bb = [0, 0]
        empt = list(range(CELLS))
        turn = 0
        while empt:
            if rng.random() < 0.2:
                pos.append((bb[0], bb[1], turn))
            c = empt[rng.randrange(len(empt))]
            empt[empt.index(c)] = empt[-1]
            empt.pop()
            bb[turn] |= 1 << c
            if has_five(bb[turn]):
                break
            turn ^= 1
    return pos


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pos = samples()
    x = []
    for b0, b1, t in pos:
        own, opp = (b0, b1) if t == 0 else (b1, b0)
        x.append(np.stack([bb_to_plane(own), bb_to_plane(opp),
                           np.zeros(CELLS, np.float32)]))
    x = torch.tensor(np.array(x).reshape(-1, 3, H, W), dtype=torch.float32).to(dev)
    print(f"arbitrary positions (no solver selection): {len(x)}")
    print()
    print(f"{'checkpoint':<10}{'mean':>8}{'std':>8}{'median':>9}{'|V|>0.5':>9}")
    print("-" * 44)
    for tag in ("v1", "v2"):
        net, _ = load_net(paths.run_ckpt(tag), dev)
        with torch.no_grad():
            p = torch.softmax(net(x), 1)
        v = (p[:, 2] - p[:, 0]).cpu().numpy()
        print(f"{tag:<10}{v.mean():>+8.3f}{v.std():>8.3f}"
              f"{np.median(v):>+9.3f}{(np.abs(v) > 0.5).mean():>8.0%}")


if __name__ == "__main__":
    main()
