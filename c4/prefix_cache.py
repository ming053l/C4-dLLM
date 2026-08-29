"""A prefix KV cache for our decode loop, as a model proxy rather than a second decode path.

The loop calls `model(x).logits` once per step over the whole buffer. The prompt occupies a fixed
left segment of that buffer and is never re-masked, so its keys and values are identical at every
step of every block: computing them once and reusing them changes no logit anywhere in the
generation region. This proxy does exactly that and nothing else, so the decision sequence, the
committed tokens and the step count are bit-identical to the uncached run -- only the per-step cost
falls. `verify_prefix_cache.py` checks that claim against real decodes rather than asserting it.

Why not the dual cache Fast-dLLM reports with. The dual cache additionally freezes everything to the
right of the live block, recomputing only that block. Our commit rule is block-local and would not
notice, but CVEE is not: it re-extracts the candidate from a trailing search region of the buffer that
spans `tail_frac` of the generation length, which at the deployed shapes is wider than one block, so
under a dual cache the confidences it reads outside the live block would be stale and the gate would
be measuring something other than what it measures everywhere else in this paper. The prefix cache
has no such interaction because the prompt is not part of any search region.

The proxy returns full-width logits with the prompt segment zero-filled. Nothing downstream reads
those positions -- `x0` is only consumed inside the current block and inside the generation-side
search region -- and keeping the shape identical means the loop needs no knowledge of the cache.
"""
import torch


class PrefixCachedModel:
    """Wraps a cache-capable LLaDA module so repeated full-buffer calls reuse the prompt's KV.

    Counts forward passes like the peer runners do, since the cache must not change that number:
    if it ever does, the cached arm is not the same computation and the comparison is void.
    """

    def __init__(self, real, prompt_len):
        self._m = real
        self.prompt_len = int(prompt_len)
        self.calls = 0
        self._kv = None

    def _prime(self, x):
        out = self._m(x[:, :self.prompt_len], use_cache=True)
        self._kv = out.past_key_values

    def __call__(self, x, **kw):
        # Anything that passes its own attention mask or cache is asking for a different
        # computation than this proxy models; run it uncached rather than silently mixing.
        if kw or x.shape[1] <= self.prompt_len:
            self.calls += 1
            return self._m(x, **kw)
        if self._kv is None:
            self._prime(x)
        self.calls += 1
        out = self._m(x[:, self.prompt_len:], past_key_values=self._kv, use_cache=True)
        suffix = out.logits
        full = torch.zeros((suffix.shape[0], x.shape[1], suffix.shape[2]),
                           dtype=suffix.dtype, device=suffix.device)
        full[:, self.prompt_len:] = suffix
        return _Out(full)

    def __getattr__(self, name):
        return getattr(self._m, name)


class _Out:
    __slots__ = ("logits",)

    def __init__(self, logits):
        self.logits = logits
