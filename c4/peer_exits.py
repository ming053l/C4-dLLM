"""Peer early-exit rules run inside the same decoding loop as CVEE (paper Table 1 rows).

Both read the top-1/top-2 logit gap rather than an extracted answer, and both fill every
remaining mask when they fire, exactly as CVEE does, so the three differ only in the stopping
rule. Pass one of these as `generate(..., exit_rule=...)` in place of a `CVEEConfig`.

- Prophet (Li et al., 2026): mean gap over a monitored region against a three-stage threshold
  keyed to decoding progress (< 0.33, < 0.67, rest). The region is the one CVEE searches for the
  same task (trailing `tail_frac` for "last", whole generation for "code", first five positions
  otherwise), which gives both rules the same information.
- SchED (Mohamed et al., 2025): mean gap over the still-masked positions of a region against a
  progress-dependent threshold, guarded by a patience counter on the fraction of argmax changes.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from .cvee import TAIL_FRAC, search_window_start


@dataclass
class ProphetExit:
    """`thresholds`: {"early", "mid", "late"} gap thresholds; Prophet's published defaults are
    7.5 / 5.0 / 2.5."""
    thresholds: dict = field(default_factory=lambda: {"early": 7.5, "mid": 5.0, "late": 2.5})
    search_mode: str = "last"
    tail_frac: float = TAIL_FRAC

    def start(self, prompt_len, gen_length, steps):
        start = search_window_start(gen_length, self.search_mode, self.tail_frac)
        end = gen_length if self.search_mode in ("last", "code") else min(5, gen_length)
        lo, hi = prompt_len + start, prompt_len + end
        th = self.thresholds

        def should_exit(logits, x0, mask_index, global_step):
            hi_clamped = min(hi, logits.shape[1])
            if hi_clamped <= lo:
                return False
            top2 = torch.topk(logits[0, lo:hi_clamped, :], k=2, dim=-1).values
            gaps = (top2[:, 0] - top2[:, 1]).tolist()
            avg_gap = sum(gaps) / len(gaps)
            progress = global_step / steps
            if progress < 0.33:
                thresh = th["early"]
            elif progress < 0.67:
                thresh = th["mid"]
            else:
                thresh = th["late"]
            return avg_gap >= thresh

        return should_exit


@dataclass
class SchEDExit:
    """SchED's live gate. Keys of `config` and their defaults: answer_region ("all" | "last"),
    tail_frac (0.3, used with "last"), tau_mode ("cosine" | "linear" | "exp"), tau_high (7.5),
    tau_low (2.5), tau_k (4.0, "exp" only), min_progress (0.0), patience_steps (0),
    max_change_ratio (1.0). Progress is the committed fraction of the region."""
    config: dict = field(default_factory=dict)

    def start(self, prompt_len, gen_length, steps):
        cfg = self.config
        if cfg.get("answer_region", "all") == "last":
            r_start = max(0, int(gen_length * (1 - cfg.get("tail_frac", 0.3))))
        else:
            r_start = 0
        lo, hi = prompt_len + r_start, prompt_len + gen_length
        region_len = gen_length - r_start
        tau_mode = cfg.get("tau_mode", "cosine")
        tau_high, tau_low = cfg.get("tau_high", 7.5), cfg.get("tau_low", 2.5)
        tau_k = cfg.get("tau_k", 4.0)
        min_progress = cfg.get("min_progress", 0.0)
        patience_steps = cfg.get("patience_steps", 0)
        max_change_ratio = cfg.get("max_change_ratio", 1.0)
        state = {"prev_x0": None, "patience": 0}

        def should_exit(logits, x0, mask_index, global_step):
            region_mask = mask_index[:, lo:hi]
            n_masked = int(region_mask.sum().item())
            progress = 1.0 - (n_masked / region_len)
            region_x0 = x0[:, lo:hi]
            if n_masked == 0:
                return True
            top2 = torch.topk(logits[:, lo:hi, :], k=2, dim=-1).values
            gaps = (top2[..., 0] - top2[..., 1])[region_mask]
            mean_gap = float(gaps.mean().item()) if gaps.numel() > 0 else float("inf")

            prev = state["prev_x0"]
            if prev is not None and prev.shape == region_x0.shape:
                changed = (region_x0 != prev)[region_mask]
                changed_ratio = (float(changed.float().mean().item())
                                 if changed.numel() > 0 else 1.0)
            else:
                changed_ratio = 1.0
            state["prev_x0"] = region_x0.clone()

            if tau_mode == "linear":
                tau = tau_high + (tau_low - tau_high) * progress
            elif tau_mode == "exp":
                tau = tau_low + (tau_high - tau_low) * float(np.exp(-tau_k * progress))
            else:
                w = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
                tau = tau_low + (tau_high - tau_low) * w

            if progress >= min_progress and changed_ratio <= max_change_ratio:
                state["patience"] += 1
            else:
                state["patience"] = 0
            return (progress >= min_progress and mean_gap >= tau
                    and state["patience"] > patience_steps)

        return should_exit
