"""Core LLaDA denoising loop, adapted from `Prophet/analysis/generate.py::generate()` (itself the
official LLaDA sampling loop with confidence-history tracking added for the paper's own trajectory
analysis). Copied rather than imported so this repo is self-contained (SPEC.md). Kept byte-similar to
the upstream version deliberately -- this must produce the same generation as a full (non-early-exit)
Prophet/TAEC run so that trajectories collected here are a faithful substrate for offline replay
(SPEC.md §2 Stage A).

Two entry points:
- `generate_full(...)`: full decode, always returns the complete step-by-step history
  (x0_history/true_indices_history/conf_history) needed for offline replay. This is what
  `traj_collect.py` calls.
- `generate_live(...)`: same loop, but with a pluggable `commit_gate` callback that can stop early
  and fill remaining masks -- used by `live_generate.py` (Stage B) to validate a policy for real.
"""
import inspect
import numpy as np
import torch
import torch.nn.functional as F

MASK_ID = 126336  # LLaDA's [MASK] token id, same constant used throughout Prophet/


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                                       dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_full(model, prompt, steps, gen_length, block_length, temperature=0.,
                   remasking="low_confidence", mask_id=MASK_ID, logit_shift=False,
                   constraints=None):
    """Full (no early exit) decode. Returns
    (x, x0_history, true_indices_history, conf_history, gap_history).
    x0_history[block] is a tensor [steps_per_block, seq_len] of the model's full denoised guess at
    every step of that block. conf_history[pos] is a list of per-step softmax confidence for every
    generated position pos (0-indexed within the generated region), concatenated across all blocks
    in decode order -- length == total steps actually run (== steps, since no early exit here).
    gap_history[pos] mirrors conf_history's shape but stores the top1-top2 LOGIT gap at that
    position/step (Prophet's own confidence-gap statistic, `generate_live`'s `prophet_thresholds`
    branch) -- letting Prophet's staged-threshold gate be replayed offline against this same cached
    decode, exactly like CVEE's gate (`replay.py`), instead of requiring a separate live run.

    `constraints`: optional dict {gen-relative position (0-indexed): token_id}, ported from
    `Prophet/analysis/generate.py`'s own `constraints` mechanism -- pins the given token at that
    absolute position from step 0 onward (never masked, re-enforced every step) rather than
    generating it. This is the suffix-prompt anchor ablation (e.g. forcing the literal word
    "Answer" at a fixed late position): our own zero-shot-CoT protocol deliberately does not use
    this by default (see `generate_live`'s docstring), but this parameter lets us reproduce
    Prophet's own ablation offline for comparison."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    if constraints:
        for rel_pos, token_id in constraints.items():
            x[0, prompt.shape[1] + rel_pos] = token_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    conf_history = {pos: [] for pos in range(gen_length)}
    gap_history = {pos: [] for pos in range(gen_length)}
    x0_history, true_indices_history = [], []

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        block_x0_hist, block_true_idx_hist = [], []
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits
            if logit_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            block_x0_hist.append(x0.detach().cpu())

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "topk_margin":
                # Prophet's own confidence-gap statistic (top1-top2 logit margin), reused here as a
                # transfer-selection criterion instead of a stopping criterion (paper Table 3b
                # reproduction) -- ranks positions by how decisively separated the top prediction is
                # from the runner-up, rather than by raw softmax mass.
                top2 = torch.topk(logits, k=2, dim=-1).values
                x0_p = top2[..., 0] - top2[..., 1]
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)

            gap_top2 = torch.topk(logits, k=2, dim=-1).values
            gap_vals = gap_top2[..., 0] - gap_top2[..., 1]
            for pos in range(gen_length):
                abs_pos = prompt.shape[1] + pos
                if abs_pos < x0_p.shape[1]:
                    conf_history[pos].append(float(x0_p[0, abs_pos].item()))
                    gap_history[pos].append(float(gap_vals[0, abs_pos].item()))

            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0_p.device))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = int(num_transfer_tokens[0, i].item())
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer_index[0, select_index] = True
            block_true_idx_hist.append(torch.nonzero(transfer_index, as_tuple=False).detach().cpu())

            x[transfer_index] = x0[transfer_index]

        x0_history.append(torch.cat(block_x0_hist, dim=0))
        true_indices_history.append(block_true_idx_hist)

    return x, x0_history, true_indices_history, conf_history, gap_history


@torch.no_grad()
def generate_live(model, tokenizer, prompt, steps, gen_length, block_length, temperature=0.,
                   remasking="low_confidence", mask_id=MASK_ID,
                   policy_fn=None, policy_params=None, extractor=None, search_mode="last",
                   tail_frac=None, prophet_thresholds=None, sched_config=None,
                   sgpd_enabled=False, sgpd_tau=0.9, sgpd_tau_lo=0.55, sgpd_run=3,
                   block_accel_tau=None, logit_shift=False, bmc_verify=False,
                   block_accel_persist=1, block_accel_volatility_k=0.0, constraints=None,
                   block_accel_no_schedule=False, block_accel_final_too=False,
                   block_accel_top1_floor=False, dual_cache=False,
                   block_accel_cooldown_radius=0, block_accel_cooldown_steps=0,
                   block_accel_cooldown_boost=0.04,
                   block_struct_mode="off", block_struct_holes=1,
                   block_struct_confirm=False, block_struct_confirm_holes=None,
                   block_struct_settle_run=0,
                   block_struct_risk_budget=None, block_struct_risk_scope="step",
                   block_struct_stall_escalate=3,
                   block_struct_log=None, block_struct_stats=None,
                   block_accel_protect_span=False, block_accel_protect_block=False,
                   span_recon_probe=False,
                   exit_momentum_alpha=0.0,
                   traj_recorder=None):
    """Same denoising loop as `generate_full`, but with a real online stop/commit gate (SPEC.md §2
    Stage B) -- mirrors `Prophet/ours_generate.py`'s pattern of splicing a gate into the official
    loop. `policy_fn=None` and `prophet_thresholds=None` means baseline (always run the full
    `steps`). Returns (text, exit_step, total_steps).

    `prophet_thresholds`: if set (dict with 'early'/'mid'/'late' keys, matching
    `Prophet/generate_earlyexit.py::should_early_exit`'s own defaults 7.5/5.0/2.5), runs the
    OFFICIAL Prophet gate instead of `policy_fn` -- top1-top2 logit gap over the answer region,
    averaged, checked against a staged threshold keyed by decoding progress
    (<0.33/<0.67/else). Monitors the SAME answer region our own `tail_frac`/`search_mode`
    convention would use (rather than requiring a forced `constraints_text` anchor like the
    official repo's GSM8K setup, which our zero-shot-CoT protocol deliberately doesn't use) -- same
    information access as `policy_fn`, different decision rule, for a fair peer comparison
    (paper Table 1's Prophet row).

    `sched_config`: if set (dict with 'tau_mode', 'tau_high', 'tau_low', 'tau_k', 'min_progress',
    'patience_steps', 'max_change_ratio'), runs the live SchED gate (Mohamed et al., 2025,
    arXiv:2512.02892) -- same top1-top2 logit-margin statistic as `prophet_thresholds`, but
    aggregated over the ENTIRE currently-masked generation region (their own paper-default
    `answer_region="all"`, not our tail/first5 convention) against a smooth progress-dependent
    threshold `tau(progress)`, gated additionally by a `changed_ratio`/`patience_steps` stability
    guard: the fraction of masked-region positions whose argmax changed since the previous step must
    stay at or below `max_change_ratio` for more than `patience_steps` consecutive steps before the
    gate is allowed to fire. Mutually exclusive with `policy_fn`/`prophet_thresholds`. Exists so the
    live run and `replay.py::run_sched`'s offline replay can be checked against each other
    (Appendix~sched's parity check) before either is trusted for a reported number.

    `sgpd_enabled`: staleness-gated parallel emit (ported from `Prophet/ours_components.py::
    sgpd_extra`), a SEPARATE axis from `policy_fn`/`prophet_thresholds` -- it does not decide when
    to STOP, it emits extra tokens per step (beyond the scheduled confidence-topk transfer) for
    positions that are either very confident OR moderately confident AND cross-step argmax-stable
    for >= `sgpd_run` steps. This does not change any commit-gate correctness reasoning (the gate
    still fires on the same signals), it just empties the masked buffer faster, which indirectly
    reduces NFE. NOT valid for offline trajectory replay (RISKS.md R0): it alters what gets
    computed at future steps, so a cached full-decode trajectory can't stand in for it -- must
    always be measured live.

    `block_accel_tau`: task-aware BLOCK-SCOPED acceleration. Unlike `policy_fn` (which only ever
    judges the FINAL answer block) or `sgpd_enabled` (which applies uniformly to every block),
    this applies a pure confidence-threshold parallel-decode rule (Fast-dLLM-style, no answer
    semantics involved) ONLY to non-final blocks: any position with confidence >= this threshold
    is transferred immediately regardless of the step's scheduled top-k budget. The FINAL block
    (where GSM8K/MATH's answer actually lives) is deliberately excluded -- it still goes through
    `policy_fn`'s full corroborated-commit reasoning. Rationale: intermediate reasoning tokens
    tolerate being fixed slightly early far better than the final answer does (a CoT token being
    confidently-but-not-yet-verified-correct rarely changes the final numeric answer the way a
    premature final-block commit would), so it is safe to be far more aggressive there. Also NOT
    offline-replayable (same reason as `sgpd_enabled`).

    `block_accel_persist` (2026-07 exploration, default 1 = off, reproduces the original one-shot
    rule): require a position's confidence to clear the (possibly volatility-adaptive) threshold for
    this many CONSECUTIVE steps, not just the current instant, before Advance-and-Hold force-commits
    it -- pure temporal persistence, no extra forward pass.

    `block_accel_volatility_k` (2026-07 exploration, default 0.0 = off): scales the effective
    per-position threshold up by this much times that position's own observed argmax-flip count
    (`pos_flips`) -- a position that has been flip-flopping needs to clear a stricter bar than one
    that has been quiet, an online per-example signal rather than a fixed global constant.

    `block_accel_no_schedule` (default False): ablation removing the scheduled top-k floor (Eq. 4's
    `i in K(t)` term) from non-final blocks entirely -- only the confidence-threshold rule can
    commit positions there, with a deadline fallback on each block's last local step so the block
    still terminates within its step budget. Isolates whether the schedule's guaranteed per-step
    floor is doing real work, independent of `block_accel_tau` itself.

    `block_accel_final_too` (default False): ablation extending CCTC's non-final-block treatment
    (scheduled top-k + confidence-threshold force-commit, `block_accel_tau`) to the FINAL block as
    well, in place of CVEE's corroborated-commit gate -- i.e. CCTC applied uniformly to every block,
    no run-length-verified answer-span gate anywhere. Only meaningful with `policy_fn=None`
    (otherwise the final block would be governed by both mechanisms at once, not a clean ablation of
    "final block scoping" in isolation). Isolates whether restricting CCTC to non-final blocks is
    load-bearing for safety, or whether the final block could have been accelerated the same
    confidence-threshold way without CVEE's extra verification.

    `block_struct_mode` (2026-08 exploration, default "off" = the upstream one-line rule, which is
    then reproduced bit-for-bit): decides membership over the SET of positions eligible this step
    instead of judging each position on its own scalar. Every gate tried before this one
    (`block_accel_persist`, `block_accel_volatility_k`, the neighbourhood cooldown, top1/top2
    margin) re-ranked the same positions by a different per-position statistic, and none moved the
    accuracy/NFE frontier; the one failure this project traced to a token was a commit made ACROSS
    a hole, four slots to the right of the frontier, which fixed the end of an argument list before
    the argument list existed -- invisible to any statistic read at that position alone. Two modes:
      "window" -- commit only the frontier-anchored run of eligible positions that admits at most
        `block_struct_holes` still-masked-but-ineligible positions to its left. Already-committed
        positions are transparent, so a block whose whole slice clears the threshold has no holes
        and still finishes in ONE step. That is what makes this affordable: about half of all
        non-final blocks here one-shot, so any rule costing an extra pass per block would cost
        ~5 NFE, more than the entire headroom under the known frontier.
      "risk" -- commit the longest confidence-descending prefix of the eligible set whose cumulative
        (1 - p) stays under `block_struct_risk_budget`, i.e. membership by rank under a budget on
        expected mis-commits rather than by threshold, so the verdict for a position depends on
        every position ranked above it (three at 0.9999 cost 0.0003 between them; one at 0.9001
        costs 0.1). Meant to be run with `block_accel_tau` lowered to ~0.5 as a mere pre-filter:
        at tau 0.9 the eligible sets are already narrow enough that any budget loose enough to be
        free is inert. Assumes `remasking="low_confidence"`, since it reads x0_p as a probability.

    `block_struct_confirm` ("window" only): the propose/confirm round. An eligible position the
    window withholds is RECORDED with the token it predicted rather than written; on the next step,
    once the committed core sits in its context, it is committed only if it is still eligible and
    still predicts the same token. The context change is manufactured deliberately, which is what
    separates this from `block_accel_persist` -- that variant waited for the same context to say the
    same thing twice, which costs steps and tests nothing new. `block_struct_confirm_holes=None`
    lets confirmations bypass the window entirely; an int applies a second (normally looser) hole
    budget to them, for the case where confirming a proposal one step later merely re-creates the
    across-a-hole commit the window had blocked.

    `block_struct_settle_run` ("window" only, default 0 = off): a position may anchor the core only
    if its raw argmax has held for this many consecutive steps. An eligible-but-unsettled position
    then does not merely fail to commit, it consumes a hole and closes the window on everything to
    its right however confident -- the inversion of `block_accel_volatility_k`, which spent the same
    per-position history on that position's own bar and so only re-ranked a set it could not change.

    `block_struct_risk_scope`: "step" resets the budget each step; "block" accumulates it across the
    block (reset at block entry), which front-loads -- an avalanche on the block's first local step
    exhausts the budget and forces the remainder to trickle. The scheduled top-k transfer is never
    charged to the budget (it commits regardless); its risk is by construction the minimum over the
    masked set, so the leakage is bounded, but it is leakage and is not corrected here.

    `block_struct_stall_escalate` (default 3, 0 = never escalate): if positions are eligible but the
    structural rule admits none of them for this many CONSECUTIVE steps, commit every eligible one
    for a single step, then reset. Keyed to demonstrated no-progress rather than to elapsed steps on
    purpose: the traced failure happens at block-local step 9 of a 14-step block, so an
    elapsed-step valve would switch the mechanism off before it could ever act. Termination never
    depends on it -- the scheduled top-k floor still commits at least one position per step, so a
    block drains inside its step budget no matter how much the structural rule withholds, and the
    block's last local step commits all eligible positions unconditionally anyway.

    `block_struct_log` / `block_struct_stats`: optional sinks, both None by default. The first is a
    list receiving one record per non-final-block step (alive and eligible in-block offsets, their
    confidences, their argmax run lengths, and the whole block's run vector at block entry); it
    costs a host sync per step and is for the pre-flight probe only. It is honoured with
    `block_struct_mode="off"` as well, where the loop logs those diagnostics but commits exactly as
    today, so the logged trajectory is the deployed one and the measurements transfer. The second is
    a dict receiving the structural counters once at the end of the generation, kept out of the
    return tuple so callers written against the upstream signature are unaffected; it accumulates on
    device and syncs once, and the one op it adds to a disabled run (summing the step's commit
    width) reads `transfer_index` without being able to affect it.

    `traj_recorder`: optional per-step hook for visualization/debugging. If set, it is called once
    per live step with keyword args carrying the current `x0`, `true_conf`, committed buffer `x`,
    and a few indices (`prompt_len`, `gen_length`, `global_step`, `num_block`, `local_step`).
    It is observational only and ignored when None, so existing reported arms are unchanged.

    Scoping note: where the upstream rule ORs its verdict over the whole sequence, the structural
    branch ORs only over [block_start, block_end). Nothing outside that slice can be eligible under
    the invariants above, so the two agree, but if a stale masked position ever did survive to the
    left of block_start it would be force-committed by the upstream rule and left to the scheduled
    top-k here."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if constraints:
        # Reserved-span construction (2026-07 exploration, testing whether Prophet's own
        # reserved-answer-span prompt structure -- Appendix C.1 of li2026diffusion -- is what
        # actually makes their reported numbers safe): pins the given tokens (e.g. the literal
        # anchor "The answer is") at fixed positions from step 0, never masked, so they are never
        # eligible for transfer_index and stay fixed for the whole decode without needing
        # per-step re-application (mirrors generate_full's identical mechanism).
        for rel_pos, token_id in constraints.items():
            x[0, prompt.shape[1] + rel_pos] = token_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    ans_prev, run, changes = None, 0, 0
    committed_so_far = set()
    sched_prev_x0 = None
    sched_patience_counter = 0
    spectral_window = []  # rolling window of conf_at_answer, for entropy_verified/spectral_verified
    entropy_min_so_far = None  # greedy running-best entropy, for greedy_trajectory_verified
    entropy_recent_window = []  # recency-bounded running-best, for greedy_recent_trajectory_verified
    entropy_trend_window = []  # local windowed entropy readings, for the trend variant
    stopped = False
    # Per-step CVEE bookkeeping is Python-side (detokenize/regex/re-encode/re-locate), not GPU work,
    # and was identified as a real wall-clock tax distinct from the gate's own safety logic (the
    # deployed `conf_floor` policy only ever reads run/changes/conf -- everything else below is
    # computed unconditionally regardless of which policy is active). Two lossless speedups, neither
    # changes any gate decision, NFE, or accuracy:
    #  (a) `_search_cache`: the answer text/span is a deterministic function of the search-window's
    #      token ids alone: if those ids are byte-identical to last step's (common -- most steps
    #      don't change the tail window at all), skip decode+extract+encode+locate and reuse the
    #      cached (ans, ans_ids, pos); confidence/entropy are still recomputed fresh every step since
    #      those genuinely change even when the argmax text doesn't.
    #  (b) `_needed`: which of {entropy*, hf_ratio, settle_frac, ctx_conf} the active policy_fn
    #      actually consumes, via signature introspection (every policy takes the full kwarg set,
    #      unused ones absorbed by **_) -- skips the corresponding list comprehensions entirely when
    #      not needed, computed once here rather than re-inspecting every step.
    _search_cache = {"ids": None, "ans": None, "ans_ids": None, "pos": None}
    _needed = set(inspect.signature(policy_fn).parameters) if policy_fn is not None else set()
    _need_entropy = bool({"entropy", "entropy_min_so_far", "entropy_min_recent",
                          "entropy_trend"} & _needed)
    _need_hf_ratio = "hf_ratio" in _needed
    _need_settle_frac = "settle_frac" in _needed
    _need_ctx_conf = "ctx_conf" in _needed
    seq_len = prompt.shape[1] + gen_length
    sgpd_last_am = torch.full((seq_len,), -1, dtype=torch.long, device=model.device)
    sgpd_stable_run = torch.zeros((seq_len,), dtype=torch.long, device=model.device)
    # Per-position bookkeeping for the two block_accel_tau extensions below (2026-07 exploration):
    # accel_persist_run counts CONSECUTIVE steps a position's confidence has stayed >= threshold
    # (temporal persistence, not argmax-identity -- distinct from sgpd_stable_run, and computed
    # unconditionally so it's available whenever block_accel_tau is set, independent of sgpd_enabled).
    # pos_last_am/pos_flips track per-position argmax flips, feeding the volatility-adaptive
    # threshold variant.
    accel_persist_run = torch.zeros((seq_len,), dtype=torch.long, device=model.device)
    pos_last_am = torch.full((seq_len,), -1, dtype=torch.long, device=model.device)
    pos_flips = torch.zeros((seq_len,), dtype=torch.long, device=model.device)
    # Neighbourhood cooldown (2026-08 exploration), a third optional block_accel extension, OFF by
    # default so that radius=0 or steps=0 reproduces the upstream rule exactly. Rationale: the one
    # CCTC failure this project traced to a single token was not a weak commit. Its confidence,
    # top1/top2 ratio, margin and argmax run length all sat within the range of correct commits, so
    # no per-position decisiveness threshold separates it. What distinguished it was a burst, seven
    # adjacent positions written across two steps while their argmax was still fresh, one of which
    # bundled an opening parenthesis with an argument and a comma and foreclosed the call. Raising
    # the bar near positions written in the last few steps makes a contested neighbourhood settle
    # over several steps instead of at once.
    cooldown_on = block_accel_cooldown_radius > 0 and block_accel_cooldown_steps > 0
    cool_last = torch.full((seq_len,), -(10 ** 9), dtype=torch.long, device=model.device)
    cool_clock = 0
    cool_suppressed = 0
    # Structural (set-level) commit rules, 2026-08 exploration -- see docstring. With mode "off" and
    # no probe log, not one tensor op below executes and the loop takes the upstream
    # `transfer_index[0] |= accel_gate` line unchanged; temperature 0 consumes no RNG, so that path
    # is bit-identical rather than merely statistically equivalent.
    if block_struct_mode not in ("off", "window", "risk"):
        raise ValueError(f"block_struct_mode must be off/window/risk, got {block_struct_mode!r}")
    if block_struct_risk_scope not in ("step", "block"):
        raise ValueError(f"block_struct_risk_scope must be step/block, got {block_struct_risk_scope!r}")
    _struct_on = block_struct_mode != "off" and block_accel_tau is not None
    if _struct_on:
        # Each of these three failed on its own, and a structural rule stacked on any of them is a
        # different experiment than the one being reported -- refuse the combination rather than
        # publish a confounded arm. block_accel_no_schedule is worse than confounded: it removes the
        # scheduled per-step floor that is the only reason a withholding rule still drains a block
        # inside its step budget.
        if (block_accel_persist > 1 or block_accel_volatility_k or cooldown_on or sgpd_enabled
                or block_accel_no_schedule):
            raise ValueError("block_struct_mode requires block_accel_persist=1, "
                             "block_accel_volatility_k=0, cooldown off, sgpd_enabled=False and "
                             "block_accel_no_schedule=False")
        if block_struct_mode == "risk" and block_struct_risk_budget is None:
            raise ValueError("block_struct_mode='risk' needs block_struct_risk_budget")
    _need_stable_run = _struct_on or block_struct_log is not None
    # Consecutive steps a position has kept the same RAW argmax. Per generation, deliberately never
    # reset at a block boundary, and updated above the -inf masking below: positions accumulate
    # argmax history for the whole decode before their own block is ever entered, which is the only
    # reason the settledness term has a signal at a block's first local step (measured: pure-padding
    # positions are unanimous ~100 steps before their block starts).
    pos_stable_run = torch.ones((seq_len,), dtype=torch.long, device=model.device)
    struct_counters = ({k: torch.zeros((), dtype=torch.long, device=model.device)
                        for k in ("deferred", "confirmed", "confirm_failed", "truncated_steps",
                                  "escalations", "risk_truncated", "commit_width",
                                  "span_suppressed")}
                       if (_struct_on or block_struct_stats is not None) else None)
    block_steps_used = []  # real (not offline-approximated) per-block step count, in block order
    # Provenance of every generated position: True once it was committed by a CCTC rule (core,
    # confirmed deferral, or the bare threshold branch) rather than by the schedule's own top-k.
    # This is what separates "the candidate stopped changing" from "the candidate was frozen by
    # our own sampler", so the exit gate's stability claim can be audited on the same run.
    extra_commit_pos = torch.zeros((seq_len,), dtype=torch.bool, device=model.device)
    span_audit = {}
    # Exit momentum. The gate decides when the answer has settled, but the step that acts on that
    # decision fills every position still masked at once, and the text it writes there is worse the
    # more of it there is. Arming a deadline instead of filling immediately keeps the decoder
    # running for a short window, so ordinary decoding commits some of those positions properly and
    # the one-shot fill is left with fewer to guess. The window is measured from the firing step
    # itself, which makes it cheap exactly where the gate fires early and nearly free where the
    # gate rarely fires at all.
    momentum_deadline, momentum_fired_at = None, None
    # Protected-block control. Barring CCTC from the candidate span alone leaves it free to commit
    # the positions AROUND that span in the same block, which can carry the path to a value the
    # gate then reads as settled. This c4 covers the window the objection is actually about,
    # opening the first time a candidate is extractable and closing when the gate fires, and it
    # protects the block the candidate sits in rather than the span alone.
    candidate_seen, last_span = False, None

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        if _struct_on:
            # Proposals and the block-scoped risk account are indexed by BLOCK-LOCAL offset, so they
            # must die at the boundary: a proposal carried across would be confirmed against a
            # different block's position, which fails silently as a mild accuracy regression.
            prop_mask = torch.zeros(block_end - block_start, dtype=torch.bool, device=x.device)
            prop_tok = torch.full((block_end - block_start,), -1, dtype=torch.long, device=x.device)
            blk_risk_spent = torch.zeros((), dtype=torch.float32, device=x.device)
            stall = torch.zeros((), dtype=torch.long, device=x.device)

        if dual_cache:
            # Fast-dLLM's dual cache, in this loop's terms: one full forward per block builds the
            # KV, then each later step recomputes only the live block and reuses the rest. In a
            # bidirectional model this is an approximation, not a re-parameterisation -- every
            # position outside the block keeps the logits it had when the block opened, which is
            # what makes it cheap and also what can move the answer -- so the arm using it is
            # reported separately rather than treated as the same computation made faster.
            _out = model(x, use_cache=True)
            _kv = _out.past_key_values
            _cached_logits = _out.logits
            _replace = torch.zeros_like(x, dtype=torch.bool)
            _replace[:, block_start:block_end] = True

        for i in range(steps_per_block):
            global_step += 1
            mask_index = (x == mask_id)
            if dual_cache:
                if i == 0:
                    logits = _cached_logits
                else:
                    blk = model(x[:, block_start:block_end], past_key_values=_kv,
                                use_cache=True, replace_position=_replace).logits
                    _cached_logits = _cached_logits.clone()
                    _cached_logits[:, block_start:block_end] = blk
                    logits = _cached_logits
            else:
                logits = model(x).logits
            if logit_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # `true_conf`: genuine softmax top-1 probability, ALWAYS computed this way regardless
            # of `remasking` -- this is what the gate's own conf_at_answer/ctx_conf must read.
            # `x0_p`, by contrast, is the remasking strategy's own transfer-selection score, which
            # for "random"/"topk_margin" is NOT a probability (uniform noise / unbounded logit
            # margin) -- reusing it as the gate's confidence signal was a bug (Table 3b
            # reproduction originally did this, making the gate read random noise as "confidence"
            # under random remasking -- an artifact, not a real safety gap).
            p = F.softmax(logits, dim=-1)
            true_conf = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            # Full-distribution predictive entropy (nats), for entropy_verified/spectral_verified
            # (2026-07 exploration) -- unlike true_conf, deliberately NOT masked to -inf beyond
            # block_end: the model still emits a real (if premature) distribution there, and this
            # matches figures/collect_diagnostics.py's entropy_history convention exactly, so
            # offline-replay-tuned hyperparameters transfer unchanged to this live loop.
            true_entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)

            if remasking == "low_confidence":
                x0_p = true_conf.clone()
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "topk_margin":
                # Prophet's own confidence-gap statistic (top1-top2 logit margin), reused here as a
                # transfer-selection criterion instead of a stopping criterion (paper Table 3b
                # reproduction) -- ranks positions by how decisively separated the top prediction is
                # from the runner-up, rather than by raw softmax mass.
                top2 = torch.topk(logits, k=2, dim=-1).values
                x0_p = top2[..., 0] - top2[..., 1]
            else:
                raise NotImplementedError(remasking)

            if sgpd_enabled:
                is_same = (x0[0] == sgpd_last_am)
                sgpd_stable_run = torch.where(is_same, sgpd_stable_run + 1,
                                               torch.zeros_like(sgpd_stable_run))
                sgpd_last_am = x0[0].clone()

            if block_accel_tau is not None:
                # Unconditional (independent of sgpd_enabled): per-position argmax-flip count,
                # feeding the volatility-adaptive block_accel_tau variant below.
                flipped = (pos_last_am != -1) & (x0[0] != pos_last_am)
                pos_flips = pos_flips + flipped.long()
                if _need_stable_run:
                    # Same tensor, opposite reading: run length rather than flip count, and it must
                    # be taken here, above the -inf masking, so that it is already populated when a
                    # block is entered.
                    same_am = (pos_last_am != -1) & (x0[0] == pos_last_am)
                    pos_stable_run = torch.where(same_am, pos_stable_run + 1,
                                                 torch.ones_like(pos_stable_run))
                pos_last_am = x0[0].clone()

            x0_p[:, block_end:] = -np.inf
            true_conf[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            if traj_recorder is not None:
                traj_recorder(x0=x0, true_conf=true_conf, x=x, prompt_len=prompt.shape[1],
                              gen_length=gen_length, global_step=global_step,
                              num_block=num_block, local_step=i)

            # tail_start is STATIC relative to the total gen_length -- deliberately UNCHANGED from
            # the version that produced the validated Table 1 numbers. RISKS.md R5 documents an
            # investigation into clipping this window to block_end_rel (excluding not-yet-reached
            # blocks' structurally-(-inf) confidence): it turned out this window's "pollution" was
            # accidentally acting as an implicit safety floor (it suppresses any answer extraction
            # until deep into the generation); removing it made GSM8K's deployed `ours`
            # hyperparameters unsafe, and a re-tuned corrected version needed a much narrower
            # tail_frac and did not clearly beat what's already validated here. Reverted rather than
            # risk silently changing the numbers `REPRODUCE.md` promises these commands regenerate.
            # `block_end_rel` is still used below for `ctx_conf` (the context-corroboration
            # ablation only, never part of the deployed `ours`/`taec_adaptive` configs).
            block_end_rel = block_end - prompt.shape[1]
            tail_start = 0
            if tail_frac is not None and search_mode == "last":
                tail_start = max(0, int(gen_length * (1 - tail_frac)))
            search_end = gen_length

            if momentum_deadline is not None and global_step >= momentum_deadline:
                x[mask_index] = x0[mask_index]
                stopped = True
                span_audit["momentum_extra"] = global_step - momentum_fired_at
            elif policy_fn is not None and momentum_deadline is None:
                gen_buf = x0[0, prompt.shape[1]:prompt.shape[1] + gen_length]
                search_ids_t = gen_buf[tail_start:search_end]
                # (a) ans/ans_ids/pos are a deterministic function of the search window's token ids
                # alone -- if unchanged since last step (the common case), skip the Python-side
                # detokenize + regex-extract + re-encode + re-locate entirely and reuse the cache.
                if _search_cache["ids"] is not None and torch.equal(search_ids_t, _search_cache["ids"]):
                    ans, ans_ids, pos = _search_cache["ans"], _search_cache["ans_ids"], _search_cache["pos"]
                else:
                    search_text = tokenizer.decode(search_ids_t.tolist(), skip_special_tokens=True)
                    ans = extractor(search_text)
                    ans_ids, pos = None, None
                    if ans is not None and search_mode == "code":
                        # Code tasks: the candidate is a multi-line function definition, so the
                        # bare re-encode-and-subsequence-search below is both wrong (a detokenize/
                        # retokenize round trip does not preserve indentation token boundaries) and
                        # quadratic in a 384-token buffer. Locate it in CHARACTER space instead --
                        # where the extractor itself works -- and convert the two offsets to token
                        # indices by binary search over prefix-decode lengths (exact, ~2*log2(n)
                        # detokenizations, no concatenativity assumption about the tokenizer).
                        ids = search_ids_t.tolist()
                        c0 = search_text.find(ans)
                        if c0 >= 0:
                            i0 = _char_offset_to_token_idx(tokenizer, ids, c0)
                            i1 = _char_offset_to_token_idx(tokenizer, ids, c0 + len(ans))
                            if i1 > i0:
                                pos, ans_ids = tail_start + i0, ids[i0:i1]
                    elif ans is not None:
                        buf_ids = gen_buf.tolist()
                        # BPE tokenizers give bare "B" and mid-text " B" DIFFERENT token ids --
                        # extractor output is the bare regex match, which rarely matches how the
                        # token actually sits in context. Try both encodings (replay.py mirrors this).
                        def _locate(cand_ids):
                            if search_mode == "last":
                                p = _find_last_subseq(buf_ids[tail_start:search_end], cand_ids)
                                return p + tail_start if p is not None else None
                            return _find_first_subseq_within(buf_ids, cand_ids, limit=5)

                        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
                        pos = _locate(ans_ids) if ans_ids else None
                        if pos is None:
                            ans_ids_sp = tokenizer.encode(" " + ans, add_special_tokens=False)
                            if ans_ids_sp != ans_ids:
                                pos_sp = _locate(ans_ids_sp)
                                if pos_sp is not None:
                                    pos, ans_ids = pos_sp, ans_ids_sp
                    _search_cache["ids"] = search_ids_t.clone()
                    _search_cache["ans"], _search_cache["ans_ids"], _search_cache["pos"] = ans, ans_ids, pos

                # (b) confidence/entropy at the (possibly cached) span are read fresh every step --
                # these genuinely change even when the argmax text doesn't -- but only for the
                # signals the active policy_fn actually consumes (signature-introspected once above).
                conf_at_answer, settle_frac, entropy_at_answer = None, None, None
                span = None
                if ans is not None and pos is not None and ans_ids:
                    span = range(pos, pos + len(ans_ids))
                    candidate_seen, last_span = True, span
                    if search_mode == "code":
                        # A code span runs to hundreds of tokens, where the per-token `.item()`
                        # below would issue one GPU->CPU sync per token per step and dominate the
                        # decode's own wall clock. The span is contiguous by construction, so slice
                        # once and reduce on device instead: same mean, one sync. (Kept separate
                        # from the scalar-answer path rather than replacing it, so the arithmetic
                        # behind every already-validated number stays bit-for-bit untouched.)
                        a = prompt.shape[1] + pos
                        b = min(a + len(ans_ids), true_conf.shape[1])
                        if b > a:
                            conf_at_answer = float(true_conf[0, a:b].float().mean().item())
                            if _need_entropy:
                                entropy_at_answer = float(true_entropy[0, a:b].float().mean().item())
                    else:
                        confs = [float(true_conf[0, prompt.shape[1] + p2].item()) for p2 in span
                                 if prompt.shape[1] + p2 < true_conf.shape[1]]
                        if confs:
                            conf_at_answer = sum(confs) / len(confs)
                        if _need_entropy:
                            ents = [float(true_entropy[0, prompt.shape[1] + p2].item()) for p2 in span
                                    if prompt.shape[1] + p2 < true_entropy.shape[1]]
                            if ents:
                                entropy_at_answer = sum(ents) / len(ents)
                    if _need_settle_frac:
                        settle_frac = sum(1 for p2 in span if p2 in committed_so_far) / len(ans_ids)

                # A code candidate is only a real observation once the block schedule has actually
                # reached the positions it occupies. Before that, the span reaches past `block_end`,
                # where confidence is structurally -inf and the text is the model's provisional
                # guess at positions no block has begun decoding; counting those steps would inflate
                # `changes` into the hundreds before the first meaningful one is ever seen. Scalar
                # answers get this for free (their extractor finds nothing in an undecoded region),
                # so this is the same "no extractable candidate leaves run/changes unchanged" rule
                # the gate already specifies, applied where a whole-program span makes it explicit.
                observed = ans is not None
                if observed and search_mode == "code":
                    observed = conf_at_answer is not None and np.isfinite(conf_at_answer)
                if observed:
                    if ans == ans_prev:
                        run += 1
                    else:
                        if ans_prev is not None:
                            changes += 1
                        run = 1
                    ans_prev = ans

                ctx_conf = None
                if _need_ctx_conf:
                    still_masked = [p2 for p2 in range(block_end_rel) if p2 not in committed_so_far]
                    if not still_masked:
                        ctx_conf = 1.0
                    else:
                        ctx_confs = [float(true_conf[0, prompt.shape[1] + p2].item()) for p2 in still_masked
                                     if prompt.shape[1] + p2 < true_conf.shape[1]]
                        ctx_conf = sum(ctx_confs) / len(ctx_confs) if ctx_confs else None

                hf_ratio = 1.0
                if _need_hf_ratio:
                    sig = conf_at_answer if (conf_at_answer is not None and
                                              np.isfinite(conf_at_answer)) else 0.0
                    spectral_window.append(sig)
                    if len(spectral_window) > 8:
                        spectral_window.pop(0)
                    hf_ratio = (_spectral_hf_ratio(spectral_window) if len(spectral_window) >= 8
                               else 1.0)

                entropy_min_recent, entropy_trend = None, None
                if _need_entropy:
                    if entropy_at_answer is not None and np.isfinite(entropy_at_answer):
                        entropy_min_so_far = (entropy_at_answer if entropy_min_so_far is None
                                              else min(entropy_min_so_far, entropy_at_answer))
                        entropy_recent_window.append(entropy_at_answer)
                        if len(entropy_recent_window) > 20:
                            entropy_recent_window.pop(0)
                        entropy_trend_window.append(entropy_at_answer)
                        if len(entropy_trend_window) > 5:
                            entropy_trend_window.pop(0)
                    entropy_min_recent = min(entropy_recent_window) if entropy_recent_window else None
                    entropy_trend = (entropy_trend_window[-1] - entropy_trend_window[0]
                                     if len(entropy_trend_window) >= 5 else None)

                if policy_fn(run=run, changes=changes, conf=conf_at_answer,
                                 entropy=entropy_at_answer, entropy_min_so_far=entropy_min_so_far,
                                 entropy_min_recent=entropy_min_recent,
                                 entropy_trend=entropy_trend, hf_ratio=hf_ratio,
                                 settle_frac=settle_frac, ctx_conf=ctx_conf,
                                 progress=global_step / steps, **(policy_params or {})):
                        verified = True
                        if bmc_verify and span is not None:
                            verified = _bmc_verify(model, x, x0, mask_index, span, prompt.shape[1],
                                                    mask_id, logit_shift)
                        if verified:
                            if span is not None:
                                # Audit the span the gate just accepted, using the state the gate
                                # actually saw. `frozen` counts positions already unmasked when
                                # this step began; `by_cctc` counts the subset CCTC committed on
                                # its own rather than through the schedule's top-k quota. A span
                                # that is largely `by_cctc` means the run counter was reading our
                                # own commits back, which is exactly what the protected-span arm
                                # removes.
                                _a = prompt.shape[1] + span.start
                                _b = min(prompt.shape[1] + span.stop, seq_len)
                                span_audit["span_len"] = _b - _a
                                span_audit["frozen"] = int((~mask_index[0, _a:_b]).sum().item())
                                span_audit["by_cctc"] = int(extra_commit_pos[_a:_b].sum().item())
                                if span_recon_probe:
                                    # Diagnostic only, and it never touches `x`: re-mask the
                                    # accepted span on a throwaway clone and ask whether the model
                                    # still writes the same tokens there from the surrounding
                                    # context alone. A span that reconstructs itself was converged
                                    # whoever committed it; one that does not was being held in
                                    # place by the commits themselves.
                                    _m, _n = _span_recon_match(
                                        model, x, x0, mask_index, span, prompt.shape[1],
                                        mask_id, logit_shift)
                                    span_audit["recon_match"] = _m
                                    span_audit["recon_total"] = _n
                            if exit_momentum_alpha > 0.0:
                                momentum_fired_at = global_step
                                momentum_deadline = global_step + max(
                                    1, int(np.ceil(exit_momentum_alpha * np.sqrt(global_step))))
                            else:
                                x[mask_index] = x0[mask_index]
                                stopped = True
                        # else: candidate failed self-consistency reconstruction this step; fall
                        # through to the normal scheduled top-k transfer below and re-evaluate next
                        # step once run/changes naturally update again (does not reset run/changes
                        # itself -- a transient reconstruction disagreement isn't treated as "the
                        # answer changed", only as "not yet safe to one-shot-fill the rest").

            elif prophet_thresholds is not None:
                # Official Prophet gate (generate_earlyexit.py::should_early_exit), ported to
                # monitor the same answer region our own tail_frac/search_mode convention uses.
                # search_mode='code' monitors the WHOLE generated buffer, which is both what
                # Prophet's own code-generation setup does (it has no trailing answer span to
                # anchor on there) and the only defensible region for a task whose answer is the
                # entire program. It must not fall through to the 'first5' branch below, which
                # would leave the gate reading five positions of a 384-token program.
                region_start = tail_start if search_mode == "last" else 0
                region_end = (search_end if search_mode in ("last", "code")
                              else min(5, gen_length))
                region = range(prompt.shape[1] + region_start, prompt.shape[1] + region_end)
                gaps = []
                for p2 in region:
                    if p2 < logits.shape[1]:
                        top2 = torch.topk(logits[0, p2, :], k=2).values
                        gaps.append(float((top2[0] - top2[1]).item()))
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    progress = global_step / steps
                    if progress < 0.33:
                        thresh = prophet_thresholds["early"]
                    elif progress < 0.67:
                        thresh = prophet_thresholds["mid"]
                    else:
                        thresh = prophet_thresholds["late"]
                    if avg_gap >= thresh:
                        x[mask_index] = x0[mask_index]
                        stopped = True

            elif sched_config is not None:
                # Live SchED gate (see docstring). answer_region="all" (default, their own
                # paper-default) uses the entire gen_length; "last" restricts to the trailing
                # sched_config['tail_frac'] fraction, matching CVEE/Prophet's own tail convention.
                # Either way, `progress` is fraction-of-REGION committed (their own
                # `progress = 1 - remaining/total_region_positions`), not fraction-of-gen_length.
                sched_region = sched_config.get("answer_region", "all")
                if sched_region == "last":
                    sched_tail_frac = sched_config.get("tail_frac", 0.3)
                    r_start = max(0, int(gen_length * (1 - sched_tail_frac)))
                else:
                    r_start = 0
                r_end = gen_length
                r_len = r_end - r_start

                gen_mask_index = mask_index[:, prompt.shape[1] + r_start:prompt.shape[1] + r_end]
                n_masked = int(gen_mask_index.sum().item())
                progress = 1.0 - (n_masked / r_len)
                gen_x0 = x0[:, prompt.shape[1] + r_start:prompt.shape[1] + r_end]
                if n_masked == 0:
                    x[mask_index] = x0[mask_index]
                    stopped = True
                else:
                    gen_logits = logits[:, prompt.shape[1] + r_start:prompt.shape[1] + r_end, :]
                    top2 = torch.topk(gen_logits, k=2, dim=-1).values
                    gaps = (top2[..., 0] - top2[..., 1])[gen_mask_index]
                    gbar = float(gaps.mean().item()) if gaps.numel() > 0 else float("inf")

                    if sched_prev_x0 is not None and sched_prev_x0.shape == gen_x0.shape:
                        changed = (gen_x0 != sched_prev_x0)[gen_mask_index]
                        changed_ratio = (float(changed.float().mean().item())
                                         if changed.numel() > 0 else 1.0)
                    else:
                        changed_ratio = 1.0
                    sched_prev_x0 = gen_x0.clone()

                    tau_mode = sched_config.get("tau_mode", "cosine")
                    tau_high = sched_config.get("tau_high", 7.5)
                    tau_low = sched_config.get("tau_low", 2.5)
                    tau_k = sched_config.get("tau_k", 4.0)
                    min_progress = sched_config.get("min_progress", 0.0)
                    patience_steps = sched_config.get("patience_steps", 0)
                    max_change_ratio = sched_config.get("max_change_ratio", 1.0)

                    if tau_mode == "linear":
                        tau = tau_high + (tau_low - tau_high) * progress
                    elif tau_mode == "exp":
                        tau = tau_low + (tau_high - tau_low) * float(np.exp(-tau_k * progress))
                    else:
                        w = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
                        tau = tau_low + (tau_high - tau_low) * w

                    if progress >= min_progress and changed_ratio <= max_change_ratio:
                        sched_patience_counter += 1
                    else:
                        sched_patience_counter = 0

                    if (progress >= min_progress and gbar >= tau
                            and sched_patience_counter > patience_steps):
                        x[mask_index] = x0[mask_index]
                        stopped = True

            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0_p.device))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            is_nonfinal_block = num_block < num_blocks - 1
            if block_accel_no_schedule and block_accel_tau is not None and is_nonfinal_block:
                # Ablation (Eq. 4's "i in K(t)" scheduled top-k term removed): non-final blocks get
                # NO scheduled top-k floor at all -- progress within the block comes entirely from
                # the confidence-threshold rule below. A deadline fallback on the block's LAST local
                # step still force-commits whatever remains masked in this block, so generation
                # always terminates correctly within budget even if confidence never crosses
                # tau_blk anywhere (a block-scoped version of the one-shot-fill safety valve every
                # semi-autoregressive schedule needs).
                if i == steps_per_block - 1:
                    cur_block_mask = (x[0, block_start:block_end] == mask_id)
                    k = int(cur_block_mask.sum().item())
                else:
                    k = 0
            elif (block_accel_top1_floor and block_accel_tau is not None
                  and (num_block < num_blocks - 1 or block_accel_final_too)):
                # Fast-dLLM's own floor: when nothing clears the confidence threshold this step,
                # commit exactly the single highest-confidence masked position ("always unmask
                # max c^i") instead of the schedule's k-by-rank quota. Isolates the floor rule from
                # everything else, since the threshold branch below is untouched.
                k = 1
            else:
                k = int(num_transfer_tokens[0, i].item())
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                transfer_index[0, select_index] = True
            # Snapshot of what the schedule alone would have written this step. Everything added
            # after this line is a CCTC commit, which is what makes commit provenance separable.
            sched_index = transfer_index.clone()
            if sgpd_enabled:
                # Extra emission beyond the scheduled top-k: very confident, OR moderately
                # confident AND cross-step argmax-stable for >= sgpd_run steps. `confidence` is
                # already -inf outside mask_index and beyond the current block (block_end_rel),
                # so this naturally respects both without extra masking.
                sgpd_gate = (confidence[0] >= sgpd_tau) | (
                    (confidence[0] >= sgpd_tau_lo) & (sgpd_stable_run >= sgpd_run))
                transfer_index[0] |= sgpd_gate
            if block_accel_tau is not None and (num_block < num_blocks - 1 or block_accel_final_too):
                # Non-final-block-only pure confidence-threshold acceleration -- see docstring.
                # (block_accel_final_too extends this to the final block too, ablation only.)
                # Two independent, optional extensions (2026-07 exploration), both OFF by default
                # (block_accel_persist=1, block_accel_volatility_k=0.0 reproduces the original
                # one-shot-instant-threshold rule exactly):
                if block_accel_volatility_k:
                    # Volatility-adaptive threshold: a position that has flipped its own argmax
                    # more often gets a STRICTER bar before Advance-and-Hold trusts it, using the
                    # per-position flip count as an online, per-example volatility signal (no
                    # per-task tuning) -- distinct from a fixed global tau_blk.
                    tau_eff = torch.clamp(block_accel_tau + block_accel_volatility_k * pos_flips.float(),
                                          max=0.99)
                else:
                    tau_eff = block_accel_tau
                if cooldown_on:
                    recent = (cool_clock - cool_last) <= block_accel_cooldown_steps
                    hot = torch.zeros_like(recent)
                    for g in recent.nonzero(as_tuple=True)[0].tolist():
                        hot[max(0, g - block_accel_cooldown_radius):
                            g + block_accel_cooldown_radius + 1] = True
                    bar = torch.where(hot, torch.as_tensor(tau_eff, device=x.device) +
                                      block_accel_cooldown_boost,
                                      torch.as_tensor(tau_eff, device=x.device).expand_as(hot)
                                      if not torch.is_tensor(tau_eff) or tau_eff.dim() == 0
                                      else tau_eff)
                    above = confidence[0] >= bar
                    cool_suppressed += int(((confidence[0] >= tau_eff) & ~above).sum().item())
                else:
                    above = confidence[0] >= tau_eff
                if block_accel_persist > 1:
                    # Persistent-confidence requirement: a position must clear tau_eff for
                    # `block_accel_persist` CONSECUTIVE steps, not just the current instant, before
                    # Advance-and-Hold force-commits it -- a pure temporal-persistence check (no
                    # extra forward pass, no re-masking/reconstruction -- distinct in mechanism from
                    # BMC-lite's re-mask-and-reconstruct self-consistency verification).
                    accel_persist_run = torch.where(above, accel_persist_run + 1,
                                                    torch.zeros_like(accel_persist_run))
                    accel_gate = accel_persist_run >= block_accel_persist
                else:
                    accel_gate = above
                if not _struct_on and block_struct_log is None:
                    transfer_index[0] |= accel_gate
                else:
                    # Set-level commit rule (see docstring). Everything here is block-local: nothing
                    # outside [block_start, block_end) can be eligible anyway (x0_p is -inf past
                    # block_end, and every position left of block_start is already committed because
                    # the scheduled top-k floor guarantees at least one commit per step), so the
                    # slice loses no candidate and lets the whole rule run on 32 elements.
                    bs, be = block_start, block_end
                    alive = mask_index[0, bs:be]  # exact "still masked when this step decided"
                    elig = accel_gate[bs:be] & alive
                    x0b = x0[0, bs:be]
                    confb = confidence[0, bs:be]
                    if block_struct_log is not None:
                        elig_off = torch.nonzero(elig, as_tuple=True)[0]
                        rec = {"blk": num_block, "i": i, "gstep": global_step,
                               "alive_off": torch.nonzero(alive, as_tuple=True)[0].tolist(),
                               "elig_off": elig_off.tolist(),
                               "p": [round(v, 5) for v in confb[elig_off].float().tolist()],
                               "run": pos_stable_run[bs:be][elig_off].tolist()}
                        if i == 0:
                            rec["entry_run"] = pos_stable_run[bs:be].tolist()
                        block_struct_log.append(rec)
                    if not _struct_on:
                        transfer_index[0] |= accel_gate  # probe run: log only, commit as upstream
                    else:
                        ok = (elig & (pos_stable_run[bs:be] >= block_struct_settle_run)
                              if block_struct_settle_run > 0 else elig)
                        if block_struct_mode == "window":
                            # holes[j] is the number of alive-but-not-ok positions strictly left of
                            # j: the cumsum is inclusive but `bad` is False wherever `ok` holds.
                            # Committed positions are ~alive and therefore transparent, which is
                            # what makes this a frontier-anchored run without an explicit frontier
                            # index and without a host sync.
                            bad = alive & ~ok
                            holes = torch.cumsum(bad.long(), 0)
                            sel = ok & (holes <= block_struct_holes)
                        else:
                            sel = _struct_risk_select(
                                confb, elig, block_struct_risk_budget,
                                blk_risk_spent if block_struct_risk_scope == "block" else None)
                            if struct_counters is not None:
                                struct_counters["risk_truncated"] += (elig & ~sel).any().long()
                        if block_struct_confirm:
                            # A proposal survives only if its position is still masked, still
                            # eligible, and still predicts the same token now that last step's core
                            # is in its context. A proposal whose block drained meanwhile simply
                            # expires with the per-block state.
                            surv = prop_mask & alive & elig & (x0b == prop_tok)
                            if block_struct_confirm_holes is not None:
                                hb = torch.cumsum((alive & ~(sel | surv)).long(), 0)
                                surv = surv & (hb <= block_struct_confirm_holes)
                            if struct_counters is not None:
                                struct_counters["confirmed"] += surv.sum()
                                struct_counters["confirm_failed"] += (prop_mask & alive & ~surv).sum()
                            sel = sel | surv
                        if block_struct_mode == "window" and block_struct_risk_budget is not None:
                            risked = _struct_risk_select(confb, sel, block_struct_risk_budget, None)
                            if struct_counters is not None:
                                struct_counters["risk_truncated"] += (sel & ~risked).any().long()
                            sel = risked
                        blocked = elig.any() & ~sel.any()
                        stall = torch.where(blocked, stall + 1, torch.zeros_like(stall))
                        if block_struct_stall_escalate > 0:
                            escal = stall >= block_struct_stall_escalate
                            sel = torch.where(escal, elig, sel)
                            stall = torch.where(escal, torch.zeros_like(stall), stall)
                            if struct_counters is not None:
                                struct_counters["escalations"] += escal.long()
                        if i == steps_per_block - 1:
                            sel = elig
                        if struct_counters is not None:
                            held = elig & ~sel
                            struct_counters["deferred"] += held.sum()
                            struct_counters["truncated_steps"] += held.any().long()
                        if block_struct_mode == "risk" and block_struct_risk_scope == "block":
                            # The running account only means anything if what was actually written
                            # is charged to it; _struct_risk_select merely reads the total.
                            contrib = (1.0 - confb.float()).clamp_min(0.0)
                            blk_risk_spent = blk_risk_spent + torch.where(
                                sel, contrib, torch.zeros_like(contrib)).sum()
                        if block_struct_confirm:
                            prop_mask = elig & ~sel
                            prop_tok = torch.where(prop_mask, x0b, torch.full_like(prop_tok, -1))
                        transfer_index[0, bs:be] |= sel
            if struct_counters is not None:
                struct_counters["commit_width"] += transfer_index[0].sum()
            newly = set()
            for pos2 in transfer_index[0].nonzero(as_tuple=True)[0].tolist():
                rel = pos2 - prompt.shape[1]
                if 0 <= rel < gen_length:
                    newly.add(rel)
            committed_so_far |= newly

            if block_accel_protect_block and candidate_seen and last_span is not None:
                # Whichever block the candidate currently sits in, only the schedule may write in it
                # until the gate fires. Blocks that hold no part of the candidate keep CCTC, so this
                # isolates the interaction rather than switching the commit rule off.
                _s0 = prompt.shape[1] + last_span.start
                _s1 = min(prompt.shape[1] + last_span.stop, seq_len)
                if min(_s1, block_end) > max(_s0, block_start):
                    _keep = sched_index[0, block_start:block_end]
                    if struct_counters is not None:
                        struct_counters["span_suppressed"] += (
                            transfer_index[0, block_start:block_end] & ~_keep).sum()
                    transfer_index[0, block_start:block_end] &= _keep

            if block_accel_protect_span and span is not None:
                # Control arm for the exit gate's stability test. CCTC keeps accelerating every
                # block, but inside the span the gate is currently reading, only the schedule's
                # own top-k may write. The candidate therefore settles at the baseline's pace
                # there, so a run counter that still reaches the gate's threshold is reporting
                # convergence rather than replaying this sampler's commits.
                _a = prompt.shape[1] + span.start
                _b = min(prompt.shape[1] + span.stop, seq_len)
                if _b > _a:
                    _keep = sched_index[0, _a:_b]
                    if struct_counters is not None:
                        struct_counters["span_suppressed"] += (
                            transfer_index[0, _a:_b] & ~_keep).sum()
                    transfer_index[0, _a:_b] &= _keep

            extra_commit_pos |= transfer_index[0] & ~sched_index[0]

            if cooldown_on:
                cool_clock += 1
                cool_last[transfer_index[0]] = cool_clock

            x[transfer_index] = x0[transfer_index]

            block_done = (block_accel_tau is not None and (num_block < num_blocks - 1 or block_accel_final_too) and
                          not (x[0, block_start:block_end] == mask_id).any())
            if stopped or block_done:
                break
        block_steps_used.append(i + 1)  # real per-block usage: early-skip, gate stop, or full budget
        if stopped:
            break

    text = tokenizer.decode(x[0, prompt.shape[1]:prompt.shape[1] + gen_length].tolist(),
                             skip_special_tokens=True)
    if block_struct_stats is not None and struct_counters is not None:
        # One sync for the whole generation: the counters are accumulated on device precisely so
        # that a reported arm pays nothing per step for its own instrumentation.
        block_struct_stats.update({k: int(v) for k, v in struct_counters.items()})
        block_struct_stats["steps"] = global_step
        block_struct_stats.update(span_audit)
    return text, global_step, steps, block_steps_used


@torch.no_grad()
def _bmc_verify(model, x, x0, mask_index, span, prompt_len, mask_id, logit_shift):
    """Bidirectional Manifold Consistency-lite (2026-07 exploration, inspired by the forward-mask +
    backward-reconstruction cycle of arXiv:2604.16565's BMC metric): before trusting a candidate
    commit, build the WOULD-BE post-commit buffer on a throwaway clone, re-mask exactly the
    identity-tracked answer span on it, and run one extra forward pass to see whether the model
    reconstructs the SAME tokens there from their own (now-committed) surrounding context. A
    genuinely converged answer should sit in a stable, high-density region of the model's own
    distribution and reconstruct itself; a coincidentally-stable-looking guess that is still
    actually drifting typically will not. Costs exactly one extra forward pass, and only on steps
    where every other gate condition has already passed (not every step) -- does NOT touch the real
    `x` buffer, so a failed verification has zero side effects on the live decode state."""
    verify_x = torch.where(mask_index, x0, x)
    span_abs = [prompt_len + p2 for p2 in span]
    orig_ids = verify_x[0, span_abs].tolist()
    verify_x = verify_x.clone()
    verify_x[0, span_abs] = mask_id
    verify_logits = model(verify_x).logits
    if logit_shift:
        verify_logits = torch.cat([verify_logits[:, :1], verify_logits[:, :-1]], dim=1)
    recon = torch.argmax(verify_logits, dim=-1)
    recon_ids = recon[0, span_abs].tolist()
    return recon_ids == orig_ids


@torch.no_grad()
def _span_recon_match(model, x, x0, mask_index, span, prompt_len, mask_id, logit_shift):
    """Token-level version of `_bmc_verify`, used as a read-only probe rather than as a gate: it
    returns how many of the span's positions the model reproduces after the span is re-masked, and
    how many there are. Same single extra forward pass, same throwaway clone, no side effects."""
    verify_x = torch.where(mask_index, x0, x)
    span_abs = [prompt_len + p2 for p2 in span if prompt_len + p2 < x.shape[1]]
    if not span_abs:
        return 0, 0
    orig_ids = verify_x[0, span_abs].tolist()
    verify_x = verify_x.clone()
    verify_x[0, span_abs] = mask_id
    verify_logits = model(verify_x).logits
    if logit_shift:
        verify_logits = torch.cat([verify_logits[:, :1], verify_logits[:, :-1]], dim=1)
    recon_ids = torch.argmax(verify_logits, dim=-1)[0, span_abs].tolist()
    return sum(a == b for a, b in zip(orig_ids, recon_ids)), len(orig_ids)


def _struct_risk_select(confb, cand, budget, spent):
    """Longest confidence-descending prefix of `cand` whose cumulative (1 - p) stays at or under
    `budget` (plus `spent`, the block's running total, when the budget is block-scoped). The point
    is that membership depends on the rest of the set rather than on a per-position threshold: a
    position is admitted only if everything more confident than it was cheap enough to leave room.

    Sync-free and shape-stable: positions outside `cand` are given +inf risk, which sorts them
    behind every candidate and poisons the cumulative sum from there on, so no boolean indexing (and
    hence no device-to-host size query) is needed. Cast to float32 because the confidences arrive in
    bf16, whose ~3 significant digits would make a budget of 0.15 meaningless once summed."""
    contrib = torch.where(cand, (1.0 - confb.float()).clamp_min(0.0),
                          torch.full_like(confb, float("inf"), dtype=torch.float32))
    order = torch.argsort(contrib, stable=True)
    cum = torch.cumsum(contrib[order], 0)
    if spent is not None:
        cum = cum + spent
    keep = torch.zeros_like(cand)
    keep[order] = cum <= budget
    return keep & cand


def _spectral_hf_ratio(window):
    """Fraction of spectral power in the upper 2/3 of frequency bins of a short, mean-centered
    window -- see replay.py::_spectral_hf_ratio (identical logic, duplicated here so common/ stays
    import-self-contained rather than reaching across to the repo-root replay.py)."""
    x = np.asarray(window, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    total = power.sum()
    if total <= 1e-12:
        return 0.0
    cut = max(1, len(power) // 3)
    return float(power[cut:].sum() / total)


def _char_offset_to_token_idx(tokenizer, ids, char_off):
    """Smallest token index j such that decoding ids[:j] yields at least `char_off` characters --
    the token-space image of a character offset into `tokenizer.decode(ids)`. Binary search is valid
    because prefix-decode length is non-decreasing in j; it is used instead of summing per-token
    decode lengths so no assumption is made that the tokenizer's decode is concatenative."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid], skip_special_tokens=True)) >= char_off:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _find_last_subseq(haystack, needle):
    if not needle:
        return None
    last = None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            last = i
    return last


def _find_first_subseq_within(haystack, needle, limit):
    if not needle:
        return None
    for i in range(min(limit, len(haystack) - len(needle) + 1)):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None
