"""CCTC: Commit-Core-Then-Confirm, the local half of C4 (paper Sec. 4.3).

Inside each accelerated block, a masked position is *eligible* at step t when its top-1
probability clears tau_CCTC. The reference decoder's scheduled top-k commit (branch (c) of
Eq. 7) is always kept, so every step still makes progress. On top of it:

- mode "off" commits every eligible position at once: Fast-dLLM's confidence-threshold rule.
- mode "window" commits only the boundary-anchored *core* (Eq. 5): eligible positions with at
  most H unresolved holes (masked, not eligible) to their left. Committed positions are
  transparent, so the core starts at the resolved boundary. Eligible positions outside the core
  are *deferred*.
- with `confirm`, a deferred position is *confirmed* (Eq. 6) one step later if it is still
  masked, still eligible and still predicts the same token now that the core is in its context.
  Confirmation reuses the next scheduled forward pass, so it costs nothing extra. Confirmations
  bypass the hole budget, as in Eq. 7.

Everything in `CoreWindow` runs on device without host syncs.
"""
from dataclasses import dataclass

import torch

STRUCT_MODES = ("off", "window")


@dataclass(frozen=True)
class CCTCConfig:
    """CCTC settings. Deployed: tau=0.8, mode="window", holes=1, confirm=True, final_block=True.

    `final_block` extends acceleration to the last block (A = {0..N-1} in the paper); without it
    the block holding the answer keeps the reference schedule. `stall_escalate`: if eligible
    positions exist but the window admits none for this many consecutive steps, commit all of them
    once. `top1_floor` replaces the scheduled top-k quota in accelerated blocks with a single
    highest-confidence commit (Fast-dLLM's own floor). `protect_span` / `protect_block` are the
    control arms of Appendix C: inside the candidate span CVEE is reading (or the
    whole block containing it, from the first extractable candidate until the gate fires) only the
    scheduled top-k may commit. Both need CVEE."""
    tau: float = 0.8
    mode: str = "window"
    holes: int = 1
    confirm: bool = True
    final_block: bool = True
    stall_escalate: int = 3
    top1_floor: bool = False
    protect_span: bool = False
    protect_block: bool = False

    def __post_init__(self):
        if self.mode not in STRUCT_MODES:
            raise ValueError(f"CCTC mode must be one of {STRUCT_MODES}, got {self.mode!r}")


class CoreWindow:
    """Per-block state of the windowed rule: the proposal buffer of deferred positions and the
    stall counter. Created at block entry, so nothing carries across a block boundary (block-local
    offsets would otherwise be confirmed against another block's positions)."""

    def __init__(self, config, block_length, device, counters=None):
        self.holes, self.confirm = config.holes, config.confirm
        self.stall_escalate = config.stall_escalate
        self.counters = counters
        self.proposed = torch.zeros(block_length, dtype=torch.bool, device=device)
        self.proposed_tok = torch.full((block_length,), -1, dtype=torch.long, device=device)
        self.stall = torch.zeros((), dtype=torch.long, device=device)

    def _count(self, key, value):
        if self.counters is not None:
            self.counters[key] += value

    def select(self, eligible, alive, x0_block, last_step):
        """Positions of this block to commit beyond the schedule.

        `eligible`: confidence >= tau. `alive`: still masked when this step started.
        `x0_block`: this step's predictions. On the block's last local step every eligible
        position is committed."""
        eligible = eligible & alive
        # holes[j] counts alive-but-ineligible positions at or left of j; for an eligible j that
        # equals the number strictly left of it.
        unresolved = alive & ~eligible
        holes_left = torch.cumsum(unresolved.long(), 0)
        commit = eligible & (holes_left <= self.holes)

        if self.confirm:
            confirmed = self.proposed & alive & eligible & (x0_block == self.proposed_tok)
            self._count("confirmed", confirmed.sum())
            self._count("confirm_failed", (self.proposed & alive & ~confirmed).sum())
            commit = commit | confirmed

        # Stall escalation: demonstrated no-progress, not elapsed steps, triggers it.
        blocked = eligible.any() & ~commit.any()
        self.stall = torch.where(blocked, self.stall + 1, torch.zeros_like(self.stall))
        if self.stall_escalate > 0:
            escalate = self.stall >= self.stall_escalate
            commit = torch.where(escalate, eligible, commit)
            self.stall = torch.where(escalate, torch.zeros_like(self.stall), self.stall)
            self._count("escalations", escalate.long())

        if last_step:
            commit = eligible

        deferred = eligible & ~commit
        self._count("deferred", deferred.sum())
        self._count("truncated_steps", deferred.any().long())
        if self.confirm:
            self.proposed = deferred
            self.proposed_tok = torch.where(deferred, x0_block,
                                            torch.full_like(self.proposed_tok, -1))
        return commit
