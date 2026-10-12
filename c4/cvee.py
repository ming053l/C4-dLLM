"""CVEE: candidate-verified early exit, the global half of C4 (paper Sec. 4.1).

At every step CVEE re-extracts the candidate answer a_t from the current prediction x0, tracks
how long it has stayed the same (the run r_t) and how often it has changed (v_t), and reads the
mean top-1 probability c_t over the candidate's own token span. The sequence may stop once the
gate (Eq. 4) holds:

    c_t >= tau_CVEE   and   r_t >= max(p_min, ceil(gamma * v_t))

Deployed values: tau_CVEE = 0.7, gamma = 2.0, p_min = 3, searched over the trailing 30% of the
generation region for answers that close a response.

This module holds the gate conditions, the per-sequence candidate tracker used by `generate`, and
`CVEETracker`, the same gate packaged for a decoding loop we do not own (e.g. WINO, Appendix
"Early Exit with WINO").
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Deployed CVEE configuration (paper Sec. 5.1), frozen across all tasks and both backbones.
TAU, GAMMA, P_MIN, TAIL_FRAC = 0.7, 2.0, 3, 0.3


# --------------------------------------------------------------------------------------------
# Gate conditions. Each takes the tracker state and the gate parameters and returns True when
# the sequence may stop. Extra keyword arguments are ignored so that one parameter dict serves
# all three.
# --------------------------------------------------------------------------------------------

def required_run(changes, gamma, p_min):
    """Revision-adaptive patience: every observed change of the candidate raises the run CVEE
    demands by gamma steps, never below p_min."""
    return max(p_min, math.ceil(gamma * changes))


def conf_floor(run, changes, conf, gamma=GAMMA, p_min=P_MIN, tau=TAU, **_):
    """The joint CVEE gate (Eq. 4): confidence AND revision-adaptive stability."""
    if conf is None or conf < tau:
        return False
    return run >= required_run(changes, gamma, p_min)


def confidence_only(conf, tau=TAU, **_):
    """Ablation ("+ confidence gate only"): stop as soon as c_t >= tau, ignoring stability."""
    return conf is not None and conf >= tau


def stability_only(run, changes, gamma=GAMMA, p_min=P_MIN, **_):
    """Ablation ("+ stability gate only"): stop once the run is long enough, ignoring confidence.
    This arm can fire on a candidate whose span was never located, since it reads no c_t."""
    return run >= required_run(changes, gamma, p_min)


EXIT_GATES = {
    "conf_floor": conf_floor,
    "confidence_only": confidence_only,
    "stability_only": stability_only,
}
POLICIES = EXIT_GATES  # name used by earlier releases and the README


@dataclass
class CVEEConfig:
    """Everything `generate` needs to run CVEE.

    `extractor(text) -> str | None` returns the candidate answer from decoded text. `search_mode`
    says where to look: "last" takes the last match in the trailing `tail_frac` of the generation
    region (answers that close a response), "first5" the first match starting within the first
    five tokens (multiple-choice letters), "code" the whole region (a function definition, located
    in character space). `momentum_alpha` > 0 delays the exit by ceil(alpha * sqrt(t)) steps after
    the gate fires at step t (Appendix "fill quality", Table "momentum"). `span_recon_probe` adds
    the read-only reconstruction diagnostic of Appendix "span control"; it costs one extra forward
    pass per example and must not be enabled for timing runs."""
    extractor: Callable[[str], Optional[str]]
    search_mode: str = "last"
    tail_frac: Optional[float] = TAIL_FRAC
    gate: Callable[..., bool] = conf_floor
    gate_params: dict = field(default_factory=lambda: {"tau": TAU, "gamma": GAMMA, "p_min": P_MIN})
    momentum_alpha: float = 0.0
    span_recon_probe: bool = False


# --------------------------------------------------------------------------------------------
# Candidate extraction and span location.
# --------------------------------------------------------------------------------------------

def search_window_start(gen_length, search_mode, tail_frac):
    """First generation-relative position CVEE decodes. Only "last" restricts the window; the
    window is fixed relative to the whole generation length, not to the current block, which is
    what keeps an extraction from firing on the provisional text of blocks not yet reached."""
    if tail_frac is not None and search_mode == "last":
        return max(0, int(gen_length * (1 - tail_frac)))
    return 0


def _find_last_subseq(haystack, needle):
    if not needle:
        return None
    n, last = len(needle), None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            last = i
    return last


def _find_first_subseq_within(haystack, needle, limit):
    if not needle:
        return None
    n = len(needle)
    for i in range(min(limit, len(haystack) - n + 1)):
        if haystack[i:i + n] == needle:
            return i
    return None


def decode_ids(tokenizer, ids):
    """`tokenizer.decode(ids, skip_special_tokens=True)` for a numpy id array.

    Same text as decoding the equivalent Python list: `decode` converts its input with
    `to_py_obj` either way, but a numpy array converts in one `tolist()` call where a list is
    converted element by element, which made decoding the dominant per-step CPU cost."""
    return tokenizer.decode(np.asarray(ids), skip_special_tokens=True)


def _char_offset_to_token_idx(tokenizer, ids, char_off):
    """Smallest j such that decode(ids[:j]) has at least `char_off` characters. Binary search over
    prefix decodes, so no assumption is made that decoding is concatenative per token."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode_ids(tokenizer, ids[:mid])) >= char_off:
            hi = mid
        else:
            lo = mid + 1
    return lo


def locate_candidate(tokenizer, extractor, gen_ids, window_start, search_mode):
    """Extract the candidate from `gen_ids[window_start:]` and find its token span.

    `gen_ids` is a 1-D numpy array of the generation region's token ids. Returns
    (answer, answer_ids, pos) with `answer_ids` a list and `pos` relative to the generation
    region; both are None when the candidate cannot be located in token space."""
    window = gen_ids[window_start:]
    text = decode_ids(tokenizer, window)
    answer = extractor(text)
    if answer is None:
        return None, None, None

    if search_mode == "code":
        # A function definition does not survive a decode/re-encode round trip token for token
        # (indentation merges differently), so locate it in character space and map both ends
        # back to token indices.
        c0 = text.find(answer)
        if c0 >= 0:
            i0 = _char_offset_to_token_idx(tokenizer, window, c0)
            i1 = _char_offset_to_token_idx(tokenizer, window, c0 + len(answer))
            if i1 > i0:
                return answer, window[i0:i1].tolist(), window_start + i0
        return answer, None, None

    gen_list = gen_ids.tolist()

    def find(ids):
        if search_mode == "last":
            p = _find_last_subseq(gen_list[window_start:], ids)
            return p + window_start if p is not None else None
        return _find_first_subseq_within(gen_list, ids, limit=5)

    # BPE tokenizers encode a bare "B" and a mid-sentence " B" differently; the extractor returns
    # the bare form, so retry with a leading space before giving up.
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    pos = find(answer_ids) if answer_ids else None
    if pos is None:
        spaced_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if spaced_ids != answer_ids:
            spaced_pos = find(spaced_ids)
            if spaced_pos is not None:
                return answer, spaced_ids, spaced_pos
    return answer, answer_ids, pos


def span_confidence(conf, start, end, reduce_on_device):
    """Mean of `conf[0, start:end]`, or None for an empty span.

    Two reductions exist for historical reasons and both are kept bit-for-bit, since c_t is
    compared against tau: `generate` averages short answer spans in Python double precision and
    whole-program code spans on device in float32 (one sync instead of one per token);
    `CVEETracker` always reduces on device."""
    end = min(end, conf.shape[1])
    if end <= start:
        return None
    if reduce_on_device:
        return float(conf[0, start:end].float().mean().item())
    values = conf[0, start:end].tolist()
    return sum(values) / len(values)


class AnswerTracker:
    """Per-sequence CVEE state: the current candidate, its run r_t and its change count v_t.

    Extraction is cached on the token ids of the search window, which are unchanged on most
    steps; confidence is re-read every step because it moves even when the text does not."""

    def __init__(self, tokenizer, extractor, prompt_len, gen_length, search_mode="last",
                 tail_frac=TAIL_FRAC):
        self.tokenizer, self.extractor = tokenizer, extractor
        self.prompt_len, self.gen_length = int(prompt_len), int(gen_length)
        self.search_mode = search_mode
        self.window_start = search_window_start(gen_length, search_mode, tail_frac)
        self.run, self.changes, self.previous = 0, 0, None
        self._cached_window, self._cached = None, (None, None, None)

    def candidate(self, x0):
        """(answer, answer_ids, pos) for the current prediction `x0` of shape (1, seq_len)."""
        gen_ids = x0[0, self.prompt_len:self.prompt_len + self.gen_length].cpu().numpy()
        window = gen_ids[self.window_start:]
        if self._cached_window is None or not np.array_equal(window, self._cached_window):
            self._cached = locate_candidate(self.tokenizer, self.extractor, gen_ids,
                                            self.window_start, self.search_mode)
            self._cached_window = window
        return self._cached

    def observe(self, answer):
        """Advance r_t and v_t with an extracted candidate. Steps with no candidate are not
        passed here, so they leave both counters untouched rather than resetting them."""
        if answer == self.previous:
            self.run += 1
        else:
            if self.previous is not None:
                self.changes += 1
            self.run = 1
        self.previous = answer


class CVEETracker:
    """CVEE as a stop signal for a decoder we do not own. One instance per sequence; call `step`
    once per denoising step with the argmax `x0` (1, L) and the softmax probability of that
    argmax `conf` (1, L). Returns True when the sequence may be filled and stopped.

    CCTC is deliberately absent: it decides which positions a step commits, which a foreign
    sampler (WINO's accept-and-revoke rule, for instance) already decides itself."""

    def __init__(self, tokenizer, extractor, prompt_len, gen_length, search_mode="last",
                 tail_frac=TAIL_FRAC, gamma=GAMMA, p_min=P_MIN, tau=TAU):
        self.answers = AnswerTracker(tokenizer, extractor, prompt_len, gen_length, search_mode,
                                     tail_frac)
        self.gamma, self.p_min, self.tau = gamma, p_min, tau
        self.steps = 0

    # Read-only views kept for callers of the earlier attribute names.
    run = property(lambda self: self.answers.run)
    changes = property(lambda self: self.answers.changes)
    prompt_len = property(lambda self: self.answers.prompt_len)
    gen_length = property(lambda self: self.answers.gen_length)
    tail_start = property(lambda self: self.answers.window_start)

    def step(self, x0, conf, x_block=None):
        """`x_block` is accepted and ignored so a recorder and this gate can share one hook."""
        self.steps += 1
        answer, answer_ids, pos = self.answers.candidate(x0)
        conf_at = None
        if answer is not None and pos is not None and answer_ids:
            start = self.answers.prompt_len + pos
            conf_at = span_confidence(conf, start, start + len(answer_ids), reduce_on_device=True)
        if answer is not None:
            self.answers.observe(answer)
        return bool(conf_floor(run=self.answers.run, changes=self.answers.changes, conf=conf_at,
                               gamma=self.gamma, p_min=self.p_min, tau=self.tau))
