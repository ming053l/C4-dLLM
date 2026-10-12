"""The C4 decoding loop: the reference block-wise sampler of LLaDA with CCTC and CVEE spliced in.

`generate` runs one sequence. With no CCTC and no exit rule it is the reference decoder
(paper Sec. 3): the generation region is split into N blocks of length B, each given S = T/N
steps; every step runs one forward pass, predicts x0 by argmax, and commits the scheduled top-k
most confident masked positions of the current block. On top of that schedule:

- CCTC (`cctc.py`) commits extra positions inside a block, so blocks finish early;
- CVEE (`cvee.py`) decides when the whole sequence may stop; every still-masked position is then
  filled with its current prediction in one shot.

Both only read quantities the step already computes, so neither adds a forward pass (the optional
span-reconstruction probe is the one exception, and it is a diagnostic).

`generate_live` is the keyword-argument interface of earlier releases, kept for existing callers.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .cctc import CCTCConfig, CoreWindow
from .cvee import AnswerTracker, CVEEConfig, decode_ids, span_confidence

LLADA_MASK_ID = 126336
MASK_ID = LLADA_MASK_ID  # name used by earlier releases

# Structural counters reported through `stats`. Accumulated on device; synced once at the end.
COUNTER_KEYS = ("deferred", "confirmed", "confirm_failed", "truncated_steps", "escalations",
                "commit_width", "span_suppressed")


@dataclass
class DecodeResult:
    text: str
    steps: int                     # forward passes spent (decoding steps), the paper's "Step"
    step_budget: int               # T, the reference schedule's total
    block_steps: list = field(default_factory=list)  # steps actually spent in each block run


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling as in the LLaDA reference code; identity at temperature 0."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Reference schedule: spread a block's masked count evenly over its steps, earlier steps
    taking the remainder. Returns (batch, steps) int64."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                                      dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def block_layout(steps, gen_length, block_length):
    """(num_blocks, steps_per_block) for a generation shape, checking it divides evenly."""
    if gen_length % block_length != 0:
        raise ValueError(f"gen_length={gen_length} is not a multiple of block_length={block_length}")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(f"steps={steps} is not a multiple of the block count {num_blocks}")
    return num_blocks, steps // num_blocks


def shift_logits(logits):
    """Dream predicts token i from position i-1; shift right by one so logits[i] scores token i."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


class _BlockForward:
    """One forward pass per step, optionally with Fast-dLLM's DualCache.

    DualCache runs one full pass at block entry and then recomputes only the live block, reusing
    the KV of everything else. In a bidirectional model this is an approximation (positions
    outside the block keep the logits they had at block entry), so it is a separate reported arm
    (Appendix G), not the same computation made faster.

    DualCache needs Fast-dLLM's modified LLaDA, whose forward accepts `replace_position` and
    returns `past_key_values`. It does not work with the stock LLaDA-8B-Instruct or
    Dream-v0-Instruct-7B checkpoints, whose remote code supports neither."""

    def __init__(self, model, dual_cache):
        self.model, self.dual_cache = model, dual_cache

    def begin_block(self, x, block_start, block_end):
        if not self.dual_cache:
            return
        out = self.model(x, use_cache=True)
        self.kv, self.logits = out.past_key_values, out.logits
        self.replace = torch.zeros_like(x, dtype=torch.bool)
        self.replace[:, block_start:block_end] = True
        self.span = (block_start, block_end)

    def __call__(self, x, local_step):
        if not self.dual_cache:
            return self.model(x).logits
        if local_step == 0:
            return self.logits
        bs, be = self.span
        block_logits = self.model(x[:, bs:be], past_key_values=self.kv, use_cache=True,
                                  replace_position=self.replace).logits
        self.logits = self.logits.clone()
        self.logits[:, bs:be] = block_logits
        return self.logits


@torch.no_grad()
def span_reconstruction(model, x, x0, mask_index, span, prompt_len, mask_id, logit_shift):
    """Read-only probe (Appendix C): re-mask the accepted span on a copy of the
    buffer and count how many of its tokens one extra forward pass writes back identically.
    Returns (matches, span_length); never touches `x`."""
    probe_x = torch.where(mask_index, x0, x)
    span_abs = [prompt_len + p for p in span if prompt_len + p < x.shape[1]]
    if not span_abs:
        return 0, 0
    original = probe_x[0, span_abs].tolist()
    probe_x = probe_x.clone()
    probe_x[0, span_abs] = mask_id
    logits = model(probe_x).logits
    if logit_shift:
        logits = shift_logits(logits)
    rebuilt = torch.argmax(logits, dim=-1)[0, span_abs].tolist()
    return sum(a == b for a, b in zip(original, rebuilt)), len(original)


class _CVEEExit:
    """CVEE inside the loop: candidate tracking, the gate, exit momentum and the span audit."""

    def __init__(self, config, tokenizer, prompt_len, gen_length):
        self.config = config
        self.answers = AnswerTracker(tokenizer, config.extractor, prompt_len, gen_length,
                                     config.search_mode, config.tail_frac)
        self.prompt_len = prompt_len
        self.code = config.search_mode == "code"

    def observe(self, x0, conf):
        """Update r_t / v_t from this step's prediction. Returns (span, c_t): the candidate's
        generation-relative token range (or None) and its mean confidence (or None)."""
        answer, answer_ids, pos = self.answers.candidate(x0)
        span, conf_at = None, None
        if answer is not None and pos is not None and answer_ids:
            span = range(pos, pos + len(answer_ids))
            start = self.prompt_len + pos
            conf_at = span_confidence(conf, start, start + len(answer_ids),
                                      reduce_on_device=self.code)
        observed = answer is not None
        if observed and self.code:
            # A code span reaching past the current block reads -inf confidence there: those are
            # provisional guesses no block has decoded yet, and counting them would inflate v_t.
            observed = conf_at is not None and np.isfinite(conf_at)
        if observed:
            self.answers.observe(answer)
        return span, conf_at

    def gate(self, conf_at):
        return self.config.gate(run=self.answers.run, changes=self.answers.changes, conf=conf_at,
                                **(self.config.gate_params or {}))


@torch.no_grad()
def generate(model, tokenizer, prompt, *, gen_length, steps, block_length,
             mask_id=LLADA_MASK_ID, logit_shift=False, temperature=0.0,
             cctc: Optional[CCTCConfig] = None, cvee: Optional[CVEEConfig] = None,
             exit_rule=None, dual_cache=False, stats=None, traj_recorder=None) -> DecodeResult:
    """Decode one sequence.

    `prompt`: (1, P) token ids. `cctc=None` disables CCTC (no block rule); `cvee=None` disables
    CVEE. `exit_rule` takes a peer stopping rule from `peer_exits` instead of CVEE.
    `dual_cache=True` needs Fast-dLLM's modified LLaDA and fails on the stock checkpoints
    (see `_BlockForward`). `stats`, if a
    dict, receives the CCTC counters, the step count and the CVEE span audit at the end.
    `traj_recorder`, if set, is called once per step with keyword arguments x0, true_conf, x,
    prompt_len, gen_length, global_step, num_block, local_step (observational only)."""
    if cvee is not None and exit_rule is not None:
        raise ValueError("pass at most one of `cvee` and `exit_rule`")
    if cctc is not None and (cctc.protect_span or cctc.protect_block) and cvee is None:
        raise ValueError("protect_span/protect_block need CVEE: they protect the span it reads")

    num_blocks, steps_per_block = block_layout(steps, gen_length, block_length)
    prompt_len = prompt.shape[1]
    seq_len = prompt_len + gen_length
    device = model.device
    x = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()

    forward = _BlockForward(model, dual_cache)
    cvee_exit = _CVEEExit(cvee, tokenizer, prompt_len, gen_length) if cvee is not None else None
    peer_exit = exit_rule.start(prompt_len, gen_length, steps) if exit_rule is not None else None
    windowed = cctc is not None and cctc.mode == "window"
    counters = ({k: torch.zeros((), dtype=torch.long, device=device) for k in COUNTER_KEYS}
                if stats is not None else None)
    # Positions committed by CCTC rather than by the schedule, for the span audit: it separates
    # "the candidate stopped changing" from "our own sampler froze it".
    by_cctc = torch.zeros(seq_len, dtype=torch.bool, device=device)
    audit = {}

    global_step, stopped, block_steps = 0, False, []
    momentum_deadline, momentum_fired_at = None, None
    span = None                      # candidate span CVEE read on its latest evaluation
    candidate_seen, last_span = False, None

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length
        quota = get_num_transfer_tokens(x[:, block_start:block_end] == mask_id,
                                        steps_per_block)[0].tolist()
        accelerated = cctc is not None and (num_block < num_blocks - 1 or cctc.final_block)
        window = CoreWindow(cctc, block_length, device, counters) if windowed else None
        forward.begin_block(x, block_start, block_end)

        for i in range(steps_per_block):
            global_step += 1
            mask_index = x == mask_id
            logits = forward(x, i)
            if logit_shift:
                logits = shift_logits(logits)
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            # Top-1 softmax probability: CCTC's eligibility score and CVEE's c_t. Positions past
            # the current block are -inf, so neither can act on blocks not yet entered.
            p = F.softmax(logits, dim=-1)
            conf = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            conf[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            if traj_recorder is not None:
                traj_recorder(x0=x0, true_conf=conf, x=x, prompt_len=prompt_len,
                              gen_length=gen_length, global_step=global_step,
                              num_block=num_block, local_step=i)

            # ---- global exit ------------------------------------------------------------
            fill = False
            if momentum_deadline is not None and global_step >= momentum_deadline:
                fill = True
                audit["momentum_extra"] = global_step - momentum_fired_at
            elif cvee_exit is not None and momentum_deadline is None:
                span, conf_at = cvee_exit.observe(x0, conf)
                if span is not None:
                    candidate_seen, last_span = True, span
                if cvee_exit.gate(conf_at):
                    if span is not None:
                        _audit_span(audit, span, prompt_len, seq_len, mask_index, by_cctc)
                        if cvee.span_recon_probe:
                            audit["recon_match"], audit["recon_total"] = span_reconstruction(
                                model, x, x0, mask_index, span, prompt_len, mask_id, logit_shift)
                    if cvee.momentum_alpha > 0.0:
                        momentum_fired_at = global_step
                        momentum_deadline = global_step + max(
                            1, int(np.ceil(cvee.momentum_alpha * np.sqrt(global_step))))
                    else:
                        fill = True
            elif peer_exit is not None:
                fill = peer_exit(logits, x0, mask_index, global_step)
            if fill:
                x[mask_index] = x0[mask_index]
                stopped = True

            # ---- local commit -----------------------------------------------------------
            confidence = torch.where(mask_index, conf, torch.tensor(-np.inf, device=conf.device))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            k = 1 if (accelerated and cctc.top1_floor) else quota[i]
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer[0, select_index] = True
            scheduled = transfer.clone()
            if accelerated:
                eligible = confidence[0] >= cctc.tau
                if window is None:
                    transfer[0] |= eligible
                else:
                    transfer[0, block_start:block_end] |= window.select(
                        eligible[block_start:block_end], mask_index[0, block_start:block_end],
                        x0[0, block_start:block_end], last_step=(i == steps_per_block - 1))
            if counters is not None:
                counters["commit_width"] += transfer[0].sum()

            if cctc is not None and cctc.protect_block and candidate_seen and last_span is not None:
                s0 = prompt_len + last_span.start
                s1 = min(prompt_len + last_span.stop, seq_len)
                if min(s1, block_end) > max(s0, block_start):
                    _restrict_to_schedule(transfer, scheduled, block_start, block_end, counters)
            if cctc is not None and cctc.protect_span and span is not None:
                a = prompt_len + span.start
                b = min(prompt_len + span.stop, seq_len)
                if b > a:
                    _restrict_to_schedule(transfer, scheduled, a, b, counters)

            by_cctc |= transfer[0] & ~scheduled[0]
            x[transfer] = x0[transfer]

            block_done = accelerated and not (x[0, block_start:block_end] == mask_id).any()
            if stopped or block_done:
                break
        block_steps.append(i + 1)
        if stopped:
            break

    text = decode_ids(tokenizer, x[0, prompt_len:seq_len].cpu().numpy())
    if stats is not None:
        stats.update({k: int(v) for k, v in counters.items()})
        stats["steps"] = global_step
        stats.update(audit)
    return DecodeResult(text, global_step, steps, block_steps)


def _restrict_to_schedule(transfer, scheduled, a, b, counters):
    """Within [a, b) keep only the scheduled top-k commits (protect-span / protect-block arms)."""
    keep = scheduled[0, a:b]
    if counters is not None:
        counters["span_suppressed"] += (transfer[0, a:b] & ~keep).sum()
    transfer[0, a:b] &= keep


def _audit_span(audit, span, prompt_len, seq_len, mask_index, by_cctc):
    """Record, for the span the gate just accepted, how much of it was already committed when the
    step began (`frozen`) and how much of that CCTC rather than the schedule wrote (`by_cctc`)."""
    a = prompt_len + span.start
    b = min(prompt_len + span.stop, seq_len)
    audit["span_len"] = b - a
    audit["frozen"] = int((~mask_index[0, a:b]).sum().item())
    audit["by_cctc"] = int(by_cctc[a:b].sum().item())


# ------------------------------------------------------------------------------------------------
# Keyword interface of earlier releases.
# ------------------------------------------------------------------------------------------------

def generate_live(model, tokenizer, prompt, steps, gen_length, block_length, temperature=0.,
                  remasking="low_confidence", mask_id=LLADA_MASK_ID,
                  policy_fn=None, policy_params=None, extractor=None, search_mode="last",
                  tail_frac=None, prophet_thresholds=None, sched_config=None,
                  block_accel_tau=None, logit_shift=False,
                  block_accel_final_too=False, block_accel_top1_floor=False, dual_cache=False,
                  block_struct_mode="off", block_struct_holes=1, block_struct_confirm=False,
                  block_struct_stall_escalate=3, block_struct_stats=None,
                  block_accel_protect_span=False, block_accel_protect_block=False,
                  span_recon_probe=False, exit_momentum_alpha=0.0, traj_recorder=None):
    """`generate` behind the flat keyword names used by `run_task.py` before the refactor and by
    the README. Returns (text, steps_used, steps_budget, per_block_steps).

    Mapping: `block_accel_tau` (None = no CCTC) with `block_struct_*` and `block_accel_*` build a
    `CCTCConfig`; `policy_fn` (None = no CVEE) with `policy_params`, `extractor`, `search_mode`,
    `tail_frac`, `exit_momentum_alpha` and `span_recon_probe` build a `CVEEConfig`;
    `prophet_thresholds` / `sched_config` select a peer exit rule when `policy_fn` is None."""
    if remasking != "low_confidence":
        raise ValueError(f"only remasking='low_confidence' is supported, got {remasking!r}")
    cctc = None
    if block_accel_tau is not None:
        cctc = CCTCConfig(tau=block_accel_tau, mode=block_struct_mode, holes=block_struct_holes,
                          confirm=block_struct_confirm, final_block=block_accel_final_too,
                          stall_escalate=block_struct_stall_escalate,
                          top1_floor=block_accel_top1_floor,
                          protect_span=block_accel_protect_span,
                          protect_block=block_accel_protect_block)
    elif block_struct_mode not in ("off", "window"):
        raise ValueError(f"block_struct_mode must be 'off' or 'window', got {block_struct_mode!r}")
    cvee, exit_rule = None, None
    if policy_fn is not None:
        cvee = CVEEConfig(extractor=extractor, search_mode=search_mode, tail_frac=tail_frac,
                          gate=policy_fn, gate_params=policy_params or {},
                          momentum_alpha=exit_momentum_alpha, span_recon_probe=span_recon_probe)
    elif prophet_thresholds is not None or sched_config is not None:
        from .peer_exits import ProphetExit, SchEDExit
        exit_rule = (ProphetExit(prophet_thresholds, search_mode, tail_frac)
                     if prophet_thresholds is not None else SchEDExit(sched_config))
    result = generate(model, tokenizer, prompt, gen_length=gen_length, steps=steps,
                      block_length=block_length, mask_id=mask_id, logit_shift=logit_shift,
                      temperature=temperature, cctc=cctc, cvee=cvee, exit_rule=exit_rule,
                      dual_cache=dual_cache, stats=block_struct_stats,
                      traj_recorder=traj_recorder)
    return result.text, result.steps, result.step_budget, result.block_steps
