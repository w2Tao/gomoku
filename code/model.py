"""Value network for 9x9 freestyle gomoku.

Input  : (N, 3, 9, 9) float32
         plane 0 = own stones, plane 1 = opponent stones, plane 2 = the
         opponent's last move.  Everything is canonical (own/opponent relative
         to the side to move), so colour never appears in the representation.

Output : (N, 3) logits over [loss, draw, win] for the side to move.

Design notes
------------
* Receptive field is 3 + 4*blocks cells. Four blocks already give 19, which is
  more than the 9-cell board, so every output unit can see the whole board.
  Deeper than ~4 blocks buys connectivity, not sight.
* Stride is always 1 and there is no pooling: gomoku value lives in exact cell
  geometry (a three is worthless one cell off), so the spatial resolution is
  never thrown away.
* The head is 1x1 conv -> flatten -> linear. The 1x1 conv compresses channels
  while keeping the board layout, which is much cheaper than flattening the
  full-width body (a 64-wide flatten would be 5184 x 128 = 663k parameters on
  a 45k-sample dataset).
* BatchNorm is used instead of LayerNorm because batches are large and the
  running statistics make inference deterministic and cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import CELLS, H, W

IN_CHANNELS = 3
BOARD = W
N_CLASSES = 3


class ResidualBlock(nn.Module):
    """post-activation residual block: relu(bn(conv(relu(bn(conv(x)))))) + x"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 4,
                 head_channels: int = 24, n_classes: int = N_CLASSES):
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks,
                           head_channels=head_channels, n_classes=n_classes)
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(head_channels * CELLS, n_classes),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x)))

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)

    @property
    def receptive_field(self) -> int:
        return 3 + 4 * self.config["blocks"]


class ValuePolicyNet(ValueNet):
    """The same trunk, plus a policy head over the 81 cells.

    The trunk is shared, so `baseline0` can warm-start it. The policy head is a
    1x1 convolution that scores each cell plus a learned 81-dim positional bias
    -- a move *is* a cell, so a per-cell score is the natural parameterisation,
    and it adds only ~130 parameters.

    `forward` returns (value_logits, policy_logits). The policy logits still
    need masking: occupied cells must be set to -inf before any softmax.
    """

    has_policy = True

    def __init__(self, channels: int = 48, blocks: int = 4,
                 head_channels: int = 24, n_classes: int = N_CLASSES):
        super().__init__(channels, blocks, head_channels, n_classes)
        self.policy_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.policy_bias = nn.Parameter(torch.zeros(CELLS))

    def forward(self, x):
        trunk = self.body(self.stem(x))
        value = self.head(trunk)
        policy = self.policy_conv(trunk).flatten(1) + self.policy_bias
        return value, policy

    def predict_proba(self, x):
        value, _ = self.forward(x)
        return torch.softmax(value, dim=1)


# ---------------------------------------------------------------------------
def describe(model: ValueNet) -> None:
    """Layer-by-layer shapes and parameter counts."""
    cfg = model.config
    rows = []

    def add(name, mod, shape):
        n = sum(p.numel() for p in mod.parameters())
        rows.append((name, shape, n))

    add("stem conv 3x3", model.stem[0], f"{IN_CHANNELS}->{cfg['channels']} 9x9")
    add("stem bn", model.stem[1], f"{cfg['channels']} 9x9")
    for i, blk in enumerate(model.body):
        add(f"block{i} conv 3x3", blk.conv1, f"{cfg['channels']}->{cfg['channels']} 9x9")
        add(f"block{i} bn", blk.bn1, "")
        add(f"block{i} conv 3x3", blk.conv2, f"{cfg['channels']}->{cfg['channels']} 9x9")
        add(f"block{i} bn", blk.bn2, "")
    add("head conv 1x1", model.head[0], f"{cfg['channels']}->{cfg['head_channels']} 9x9")
    add("head bn", model.head[1], "")
    add("head linear", model.head[4],
        f"{cfg['head_channels']*CELLS}->{cfg['n_classes']}")

    total = sum(r[2] for r in rows)
    print(f"ValueNet(channels={cfg['channels']}, blocks={cfg['blocks']}, "
          f"head={cfg['head_channels']})")
    print(f"  receptive field : {model.receptive_field} cells "
          f"(board is {BOARD}, so {'global' if model.receptive_field >= BOARD else 'LOCAL'})")
    print(f"  parameters      : {total:,}")
    print()
    print(f"  {'layer':<18} {'params':>10}  shape")
    print("  " + "-" * 56)
    for name, shape, n in rows:
        if n == 0:
            continue
        print(f"  {name:<18} {n:>10,}  {shape}")


if __name__ == "__main__":
    print("=" * 60)
    describe(ValueNet(48, 4, 24))
    print()
    print("=" * 60)
    describe(ValueNet(64, 6, 32))
    print()
    print("=" * 60)
    print("data-to-parameter ratio (44,722 training positions):")
    for ch, bl, hd in ((32, 3, 16), (48, 4, 24), (64, 6, 32), (96, 8, 48)):
        m = ValueNet(ch, bl, hd)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {ch:>2}ch x{bl} blocks : {n:>8,} params   "
              f"{44722/n:6.2f} samples/param")
