"""Value network for 15x15 freestyle gomoku.

Input  : (N, 3, 15, 15) float32
         plane 0 = own stones, plane 1 = opponent stones, plane 2 = the
         opponent's last move.  Everything is canonical (own/opponent relative
         to the side to move), so colour never appears in the representation.

Output : (N, 3) logits over [loss, draw, win] for the side to move.

Design notes
------------
* Receptive field is 3 + 4*blocks cells. Four blocks already give 19, which is
  more than the 15-cell board, so every output unit can see the whole board.
  Deeper than ~4 blocks buys connectivity, not sight.
* Stride is always 1 and there is no pooling: gomoku value lives in exact cell
  geometry (a three is worthless one cell off), so the spatial resolution is
  never thrown away.
* The head is 1x1 conv -> flatten -> linear. The 1x1 conv compresses channels
  while keeping the board layout, which is much cheaper than flattening the
  full-width body (a 48-wide flatten would be 48*225*128 = 1.38M parameters on a
  few-hundred-thousand-sample dataset).
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
    """The same trunk, plus a policy head over the 225 cells.

    The trunk is shared with the value-only model, so `baseline` can warm-start
    it. The policy head is a 1x1 convolution that scores each cell plus a learned
    225-dim positional bias -- a move *is* a cell, so a per-cell score is the
    natural parameterisation, and the whole head adds only 273 parameters.

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


class ValueQNet(ValuePolicyNet):
    """Adds a per-move win probability on top of the shared trunk.

    The value head answers one question per position -- "am I winning" -- and that
    proved too coarse to rank ~150 candidate moves: both 1-ply and 2-ply search
    over it lost to the plain policy head. This head answers the question directly,
    one number per cell: if I play here, do I win.

    The target is the game outcome at the cell that was actually played, so the
    label is sparse (one cell per position). The policy head next door is what makes
    that workable -- sharing a trunk with a densely supervised head keeps the
    per-cell embedding meaningful, so the sparse signal has something to shape.

    `forward` returns (value_logits, policy_logits, q). `q` is an unbounded score
    per cell in [-1, 1]: no softmax, only masking of occupied cells before argmax.
    """

    has_q = True

    def __init__(self, channels: int = 48, blocks: int = 4,
                 head_channels: int = 24, n_classes: int = N_CLASSES):
        super().__init__(channels, blocks, head_channels, n_classes)
        self.q_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(CELLS))

    def forward(self, x):
        trunk = self.body(self.stem(x))
        value = self.head(trunk)
        policy = self.policy_conv(trunk).flatten(1) + self.policy_bias
        q = self.q_conv(trunk).flatten(1) + self.q_bias
        return value, policy, q

    def predict_proba(self, x):
        value, _, _ = self.forward(x)
        return torch.softmax(value, dim=1)


class ValuePolicyEvalNet(ValuePolicyNet):
    """Policy + WDL value + a scalar eval head trained on RAPFI's `Eval`.

    RAPFI narrates a continuous score for the position it searched. The WDL head
    is trained on game outcomes, which is a much coarser signal (one number per
    game); the scalar head is trained on `tanh(Eval / T)` and gives search a
    continuous value to order leaves with.

    `forward` returns `(value_logits, policy_logits, eval_scalar)`, with
    `eval_scalar` in [-1, 1] from the side-to-move perspective.
    """

    has_eval = True

    def __init__(self, channels: int = 48, blocks: int = 4,
                 head_channels: int = 24, n_classes: int = N_CLASSES):
        super().__init__(channels, blocks, head_channels, n_classes)
        self.eval_head = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(head_channels * CELLS, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        trunk = self.body(self.stem(x))
        value = self.head(trunk)
        policy = self.policy_conv(trunk).flatten(1) + self.policy_bias
        ev = self.eval_head(trunk).squeeze(1)
        return value, policy, ev

    def predict_proba(self, x):
        value, _, _ = self.forward(x)
        return torch.softmax(value, dim=1)


class ValueEvalNet(ValueNet):
    """Pure scalar eval network: no policy head, no WDL head.

    The trunk is the same value-network trunk. The only output is a scalar
    `tanh` value in [-1, 1], trained on `tanh(RAPFI Eval / 400)` (RAPFI's default
    scaling factor is 200, so `2*sigmoid(Eval/200)-1 = tanh(Eval/400)`).
    """

    has_eval = True

    def __init__(self, channels: int = 48, blocks: int = 4,
                 head_channels: int = 24, n_classes: int = N_CLASSES):
        super().__init__(channels, blocks, head_channels, n_classes)
        # Drop the WDL head created by ValueNet; this model is eval-only.
        self.head = None
        self.eval_head = nn.Sequential(
            nn.Conv2d(channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(head_channels * CELLS, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        trunk = self.body(self.stem(x))
        return self.eval_head(trunk).squeeze(1)


# ---------------------------------------------------------------------------
def describe(model: ValueNet) -> None:
    """Layer-by-layer shapes and parameter counts."""
    cfg = model.config
    rows = []

    def add(name, mod, shape):
        n = sum(p.numel() for p in mod.parameters())
        rows.append((name, shape, n))

    add("stem conv 3x3", model.stem[0], f"{IN_CHANNELS}->{cfg['channels']} {W}x{H}")
    add("stem bn", model.stem[1], f"{cfg['channels']} {W}x{H}")
    for i, blk in enumerate(model.body):
        add(f"block{i} conv 3x3", blk.conv1, f"{cfg['channels']}->{cfg['channels']} {W}x{H}")
        add(f"block{i} bn", blk.bn1, "")
        add(f"block{i} conv 3x3", blk.conv2, f"{cfg['channels']}->{cfg['channels']} {W}x{H}")
        add(f"block{i} bn", blk.bn2, "")
    add("head conv 1x1", model.head[0], f"{cfg['channels']}->{cfg['head_channels']} {W}x{H}")
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
    print("data-to-parameter ratio (344,989 training positions):")
    for ch, bl, hd in ((32, 3, 16), (48, 4, 24), (64, 6, 32), (96, 8, 48)):
        m = ValueNet(ch, bl, hd)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {ch:>2}ch x{bl} blocks : {n:>8,} params   "
              f"{44722/n:6.2f} samples/param")
