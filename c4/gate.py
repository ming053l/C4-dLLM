"""CVEE as a stop signal for a decoder we do not own.

`generate_live` implements CVEE inline, because it also owns the loop. WINO owns its own loop and we
are not going to reimplement their sampler to put a gate in it, so the gate has to travel instead:
this is the same rule, same hyperparameters, reading the same two tensors any diffusion step already
computes (`x0` and the softmax probability of its argmax), exposed as a callable a foreign loop can
invoke once per step.

The bookkeeping mirrors `common/decode.py::generate_live` deliberately and is worth stating, because
a stop rule that drifted from the one the paper describes would be a different method under the same
name: the candidate is re-extracted every step from the trailing `tail_frac` of the generation
region of `x0`; a step with no extractable candidate leaves the run and change counters untouched
rather than resetting them; confidence is the plain mean over the candidate's own span, located by
re-encoding the extracted string and searching for it as the last matching subsequence, with a
leading-space retry because most tokenizers encode a mid-sentence word differently from a bare one.

What this composition is for. WINO paces commits inside a block and cannot stop a sequence at all --
its loop runs until every position is filled. CVEE decides only whether the sequence may stop, and
reads nothing WINO does not already compute, so the two occupy different decisions and the second
costs no additional forward pass. CCTC is deliberately absent: it decides which positions a step may
commit, which is the decision WINO's accept-and-revoke rule already makes, and stacking them would
be two rules fighting over one choice rather than two rules composing.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "c4"))

from .decode import (_char_offset_to_token_idx,  # noqa: E402
                           _find_first_subseq_within, _find_last_subseq)
from .policies import POLICIES  # noqa: E402

# The deployed CVEE configuration, imported from nowhere on purpose: these are the frozen values
# quoted in the paper, and a composition that re-tuned them would not be evidence about CVEE.
GAMMA, P_MIN, TAU, TAIL_FRAC = 2.0, 3, 0.7, 0.3


class CVEETracker:
    """One instance per generated sequence. Call `step` once per denoising step."""

    def __init__(self, tokenizer, extractor, prompt_len, gen_length, search_mode="last",
                 tail_frac=TAIL_FRAC, gamma=GAMMA, p_min=P_MIN, tau=TAU):
        self.tok, self.extract = tokenizer, extractor
        self.prompt_len, self.gen_length = int(prompt_len), int(gen_length)
        self.search_mode = search_mode
        # `last` reads a trailing window, the convention for answers that close a generation;
        # `first5` reads the whole region and takes the earliest match, the convention for
        # multiple-choice prompts whose answer letter opens it. Both mirror generate_live.
        self.tail_start = (max(0, int(gen_length * (1 - tail_frac)))
                           if search_mode == "last" else 0)
        self.gamma, self.p_min, self.tau = gamma, p_min, tau
        self.policy = POLICIES["conf_floor"]
        self.run, self.changes, self.ans_prev = 0, 0, None
        self.steps = 0
        self._cache_ids, self._cache = None, (None, None, None)

    def step(self, x0, conf, x_block=None):
        """`x0`: (1, L) argmax token ids over the whole buffer. `conf`: (1, L) softmax probability
        of those argmax tokens. `x_block` is accepted and ignored, so that a recorder and this gate
        share one hook signature. Returns True when the sequence may be filled and stopped."""
        self.steps += 1
        gen = x0[0, self.prompt_len:self.prompt_len + self.gen_length]
        window = gen[self.tail_start:]

        if self._cache_ids is not None and torch.equal(window, self._cache_ids):
            ans, ans_ids, pos = self._cache
        else:
            text = self.tok.decode(window.tolist(), skip_special_tokens=True)
            ans = self.extract(text)
            ans_ids, pos = None, None
            if ans is not None and self.search_mode == "code":
                # A whole-program candidate runs to hundreds of tokens and a detokenize/retokenize
                # round trip does not preserve indentation token boundaries, so generate_live
                # locates it in character space and converts the two offsets by binary search over
                # prefix decodes. Mirrored here rather than approximated.
                ids = window.tolist()
                c0 = text.find(ans)
                if c0 >= 0:
                    i0 = _char_offset_to_token_idx(self.tok, ids, c0)
                    i1 = _char_offset_to_token_idx(self.tok, ids, c0 + len(ans))
                    if i1 > i0:
                        pos, ans_ids = self.tail_start + i0, ids[i0:i1]
            elif ans is not None:
                buf = gen.tolist()

                def locate(ids):
                    if self.search_mode == "last":
                        q = _find_last_subseq(buf[self.tail_start:], ids)
                        return q + self.tail_start if q is not None else None
                    return _find_first_subseq_within(buf, ids, limit=5)

                ans_ids = self.tok.encode(ans, add_special_tokens=False)
                pos = locate(ans_ids) if ans_ids else None
                if pos is None:
                    alt = self.tok.encode(" " + ans, add_special_tokens=False)
                    if alt != ans_ids:
                        pos_alt = locate(alt)
                        if pos_alt is not None:
                            pos, ans_ids = pos_alt, alt
            self._cache_ids, self._cache = window.clone(), (ans, ans_ids, pos)

        conf_at = None
        if ans is not None and pos is not None and ans_ids:
            a = self.prompt_len + pos
            b = min(a + len(ans_ids), conf.shape[1])
            if b > a:
                conf_at = float(conf[0, a:b].float().mean().item())

        if ans is not None:
            if ans == self.ans_prev:
                self.run += 1
            else:
                if self.ans_prev is not None:
                    self.changes += 1
                self.run = 1
            self.ans_prev = ans

        return bool(self.policy(run=self.run, changes=self.changes, conf=conf_at,
                                gamma=self.gamma, p_min=self.p_min, tau=self.tau))
