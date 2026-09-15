"""Stop/commit-gate policies, evaluated per-step against the running state `replay.py` maintains
(ans, run, changes, conf, refine_count, progress). Style mirrors `Prophet/ours_components.py`: pure,
stateless step functions -- all state lives in the caller. See SPEC.md §1 for the design rationale.

Every policy signature is `(ans, run, changes, conf, refine_count, progress, **params) -> bool`.
`ans is not None` is guaranteed by the caller before these are invoked.
"""
import math


def taec_adaptive(run, changes, progress, gamma=16.0, p_min=3, min_progress=0.10, **_):
    """Current default from `Prophet/ours_components.py::taec_ready_adaptive` + the
    `taec_min_progress` blanket time floor (`Prophet/RISKS.md` R1) -- ported verbatim as the
    baseline this project must reproduce (PLAN.md P4 sanity gate) before trusting new policies."""
    if progress < min_progress:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def conf_floor(run, changes, conf, gamma=2.0, p_min=3, tau=0.7, **_):
    """CVEE's joint exit gate (paper Eq.~4): the candidate span's confidence must clear `tau` AND
    its argmax run must reach `max(p_min, ceil(gamma * changes))`. The defaults are the deployed
    values, so calling it without parameters gives the gate the paper reports."""
    if conf is None or conf < tau:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def confidence_only(conf, tau=0.7, **_):
    """CVEE component ablation, "confidence only" arm: drop the adaptive run-length/stability
    requirement entirely -- commit the instant the candidate span's confidence clears `tau`, with
    no check that the candidate has stopped changing. Isolates whether `conf_floor`'s stability
    term (`run >= max(p_min, ceil(gamma*changes))`) is load-bearing, or whether confidence alone
    already suffices."""
    return conf is not None and conf >= tau


def stability_only(run, changes, gamma=2.0, p_min=3, **_):
    """CVEE component ablation, "stability only" arm: drop the confidence requirement entirely --
    commit once the adaptive run-length/stability bar clears, regardless of how confident the model
    is in the candidate. Isolates whether `conf_floor`'s confidence term is load-bearing, or whether
    argmax stability alone already suffices (this is exactly `taec_adaptive` without its blanket
    `min_progress` time floor)."""
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def fixed_patience(run, conf, tau=0.7, p_min=3, **_):
    """CVEE component ablation, "fixed patience" arm: keep both the confidence gate and a run-length
    requirement, but make the run-length bar a FIXED constant (`p_min`) instead of adaptive to how
    often the candidate has already flipped (`gamma * changes`). Isolates whether the
    changes-adaptive term itself matters, or whether a flat patience floor is equally safe against
    trajectories that repeatedly flip candidates."""
    if conf is None or conf < tau:
        return False
    return run >= p_min


def nontrivial_floor(run, changes, settle_frac, gamma=16.0, p_min=3, m=1.0, **_):
    """R1 candidate 2: replace the time floor with a "genuinely committed, not just a masked
    position's repeated guess" requirement -- >= m fraction of the answer span must already be
    permanently unmasked (settle_frac in [0,1], see replay.py::prepare_trajectory)."""
    if settle_frac is None or settle_frac < m:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def conf_x_nontrivial(run, changes, conf, settle_frac, gamma=16.0, p_min=3, tau=0.9, m=1.0, **_):
    """Ours (primary): both signals must corroborate -- no time floor at all."""
    if conf is None or conf < tau or settle_frac is None or settle_frac < m:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def volatility_scaled_conf(run, changes, conf, settle_frac, gamma=16.0, p_min=3,
                            tau_base=0.7, k=0.1, m=0.5, **_):
    """Ours (stretch): confidence threshold scales with this run's OWN observed volatility
    (`changes`, how many times the committed answer has already flipped) instead of a per-task
    constant -- online, task-agnostic (R1 candidate 3 rejected hardcoding this by task; this makes
    the equivalent signal emerge per-sample instead)."""
    if settle_frac is None or settle_frac < m or conf is None:
        return False
    tau = min(0.999, tau_base + k * math.log1p(changes))
    if conf < tau:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def context_corroborated(run, changes, conf, settle_frac, ctx_conf, gamma=16.0, p_min=3,
                          tau_hi=0.95, tau_lo=0.6, tau_ctx=0.5, m=1.0, **_):
    """Ours (extension, inspired by Dustin's hybrid historical+lookahead fusion --
    lee2026dustin): a SECOND corroboration signal, `ctx_conf` (mean confidence over positions NOT
    YET committed anywhere else in the generation, not just the answer span), opens a LOWER-
    confidence commit path instead of just adding another strict AND on top of `conf_x_nontrivial`
    -- an extra required-AND condition can only ever reduce trigger rate, never recover speedup.
    Structure mirrors this codebase's own prior winning SGPC "scp" gate (see memory
    sgpc-fast-decoding.md): commit if the answer span is confident enough on its own (`tau_hi`), OR
    if it's only moderately confident (`tau_lo`) but the surrounding still-undetermined context
    ALSO looks settled (`ctx_conf >= tau_ctx`) -- two weaker, complementary signals substituting for
    one strict one, same intuition as SubSpec's self-substitute corroboration (wang2025subspec):
    corroborate a cheap/narrow signal with an independent one before trusting it, rather than
    requiring one estimator to clear a high bar alone."""
    if conf is None or settle_frac is None or settle_frac < m:
        return False
    strong = conf >= tau_hi
    corroborated = (conf >= tau_lo) and (ctx_conf is not None) and (ctx_conf >= tau_ctx)
    if not (strong or corroborated):
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def entropy_nontrivial(run, changes, entropy, settle_frac, gamma=2.0, p_min=3, tau_h=0.5, m=1.0, **_):
    """Entropy + "genuinely committed span" corroboration (2026-07 exploration): combines
    `entropy_verified`'s full-distribution bound with `nontrivial_floor`'s settle_frac requirement
    -- both signals must agree the answer span is low-uncertainty AND already permanently
    unmasked, not just a masked hole the model is re-guessing the same low-entropy filler for."""
    if entropy is None or entropy > tau_h:
        return False
    if settle_frac is None or settle_frac < m:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def volatility_scaled_entropy(run, changes, entropy, gamma=2.0, p_min=3, tau_h_base=0.5, k=0.15, **_):
    """Entropy analog of `volatility_scaled_conf` (2026-07 exploration): the entropy CEILING
    tightens (demands lower residual uncertainty) the more this run's own answer has already
    flipped, instead of a fixed per-task constant -- an online, task-agnostic way to make the bar
    adaptive to how noisy this specific trajectory has shown itself to be so far."""
    if entropy is None:
        return False
    tau_h = max(0.02, tau_h_base - k * math.log1p(changes))
    if entropy > tau_h:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def entropy_verified(run, changes, entropy, gamma=2.0, p_min=3, tau_h=0.5, **_):
    """Entropy-based confidence verification (2026-07 exploration): replaces `conf >= tau` (top-1
    softmax probability) with a full-distribution predictive-entropy bound `H_t <= tau_h` (nats, see
    figures/collect_diagnostics.py::entropy_history). Strictly more informative than top-1
    probability alone: for a FIXED top-1 probability, entropy is minimized when the residual mass
    concentrates on a single runner-up and maximized when it spreads over many live alternatives --
    so a case that clears a lenient `conf>=tau` bar purely because top-1 happens to be ahead, while
    several other tokens remain plausible, is still correctly refused here (H_t stays high), which
    is exactly the "argmax is only ever the current best guess among several live candidates"
    failure mode Prophet's pure-gap statistic is blind to (Section 4.2)."""
    if entropy is None or entropy > tau_h:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def spectral_verified(run, changes, conf, hf_ratio, gamma=2.0, p_min=3, tau=0.7, tau_spec=0.3, **_):
    """Spectral/volatility-based stability verification (2026-07 exploration): augments the
    confidence bar with a windowed high-frequency-power-ratio bound on the recent confidence
    trajectory (`hf_ratio`, see replay.py::_spectral_hf_ratio/run_policy) instead of (or alongside)
    raw consecutive run-length. Catches "genuinely still oscillating, happened to repeat once"
    trajectories plain run-length can misread as stable, and is more forgiving of "softly
    converging, never landed on two bit-identical argmax picks in a row" trajectories plain
    run-length unfairly penalizes -- a continuous relaxation of the same underlying idea."""
    if conf is None or conf < tau:
        return False
    if hf_ratio > tau_spec:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def entropy_spectral_verified(run, changes, entropy, hf_ratio, gamma=2.0, p_min=3, tau_h=0.5,
                               tau_spec=0.3, **_):
    """Combines both 2026-07 signals: entropy-based confidence (`entropy_verified`) + spectral-
    based stability (`spectral_verified`'s `hf_ratio`), dropping reliance on exact categorical
    run-length altogether -- the most information-rich member of this family, at the cost of two
    hyperparameters instead of one."""
    if entropy is None or entropy > tau_h:
        return False
    if hf_ratio > tau_spec:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def greedy_trajectory_verified(run, changes, entropy, entropy_min_so_far, gamma=2.0, p_min=3,
                                margin=0.1, **_):
    """Greedy trajectory-relative gate (2026-07 exploration): instead of an absolute entropy bound
    calibrated on one task family (`entropy_verified`'s `tau_h`, which was found NOT to transfer
    cleanly to MMLU), commit only once H_t is within `margin` nats of `entropy_min_so_far`, the
    LOWEST entropy this specific trajectory has shown so far.
    `entropy_min_so_far` is a classic greedy running-best update: an O(1)-per-step, no-lookahead
    summary of the GLOBAL trajectory shape (computed online, never requiring hindsight over future
    steps), while $H_t$ itself is the LOCAL instantaneous signal -- the gate fires the first time the
    local reading is (near-)optimal relative to everything the trajectory has shown so far, which is
    self-normalizing per example and per task by construction (no absolute nats scale to mis-tune),
    directly targeting the portability failure absolute-threshold entropy gating showed on MMLU."""
    if entropy is None or entropy_min_so_far is None:
        return False
    if entropy > entropy_min_so_far + margin:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def greedy_trajectory_trend_verified(run, changes, entropy, entropy_min_so_far, entropy_trend,
                                      gamma=2.0, p_min=3, margin=0.1, tau_trend=0.05, **_):
    """Extends `greedy_trajectory_verified` with a second, complementary GLOBAL+LOCAL check:
    `entropy_trend` is the local windowed slope of the last 5 real entropy readings (very negative =
    still actively descending; near zero = the local neighborhood has flattened). Requiring
    |trend| <= tau_trend on top of the near-global-minimum condition asks not just "are we near the
    best point seen so far" but "has the trajectory actually STOPPED descending here" -- guards
    against greedily committing at a point that merely dips briefly before continuing to fall
    further, which `entropy_min_so_far` alone (a pure running minimum) cannot distinguish from a
    genuine plateau."""
    if entropy is None or entropy_min_so_far is None or entropy_trend is None:
        return False
    if entropy > entropy_min_so_far + margin:
        return False
    if abs(entropy_trend) > tau_trend:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


def greedy_recent_trajectory_verified(run, changes, entropy, entropy_min_recent, gamma=2.0, p_min=3,
                                       margin=0.1, **_):
    """Fixed version of `greedy_trajectory_verified` (2026-07 exploration, round 2): the all-time
    running minimum was found EXPLOITABLE offline -- a single anomalous early low-entropy reading
    (e.g. the model briefly, wrongly, very confident about a spurious early digit) sets a floor that
    is never revisited, so "close to the best ever seen" stops meaning "converged". This version
    compares against `entropy_min_recent`, a running minimum over only a trailing window of real
    readings (`replay.py::run_policy`'s `recent_window`, default 20 steps) -- old anomalies age out
    once they leave the window, so the greedy reference reflects the trajectory's CURRENT regime,
    not a possibly-stale early fluke, while still being self-normalizing per example/task (no
    absolute nats scale) rather than an absolute-threshold bound."""
    if entropy is None or entropy_min_recent is None:
        return False
    if entropy > entropy_min_recent + margin:
        return False
    need = max(p_min, math.ceil(gamma * changes))
    return run >= need


POLICIES = {
    "taec_adaptive": taec_adaptive,
    "conf_floor": conf_floor,
    "confidence_only": confidence_only,
    "stability_only": stability_only,
    "fixed_patience": fixed_patience,
    "nontrivial_floor": nontrivial_floor,
    "conf_x_nontrivial": conf_x_nontrivial,
    "volatility_scaled_conf": volatility_scaled_conf,
    "context_corroborated": context_corroborated,
    "entropy_verified": entropy_verified,
    "spectral_verified": spectral_verified,
    "entropy_spectral_verified": entropy_spectral_verified,
    "entropy_nontrivial": entropy_nontrivial,
    "volatility_scaled_entropy": volatility_scaled_entropy,
    "greedy_trajectory_verified": greedy_trajectory_verified,
    "greedy_trajectory_trend_verified": greedy_trajectory_trend_verified,
    "greedy_recent_trajectory_verified": greedy_recent_trajectory_verified,
}
