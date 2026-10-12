"""Prompt KV cache, as a model proxy (`run_task.py --prefix-cache`).

The prompt is a fixed left segment of the buffer that is never re-masked. This proxy computes its
keys and values once, from the prompt alone, and runs every later step on the generation region
only, so the decoding loop needs no knowledge of the cache.

This is an approximation, not a re-parameterisation: LLaDA and Dream attend bidirectionally, so
in an uncached pass the prompt's hidden states (and hence its keys and values above the first
layer) also depend on the generation region. Treat the cached arm as its own configuration.
It also needs a backbone whose forward accepts `use_cache` / `past_key_values`; the stock
LLaDA-8B-Instruct remote code rejects both ("The kvcache is not suppotred for MDM").

Unlike Fast-dLLM's DualCache (`generate(..., dual_cache=True)`), nothing right of the live block
is frozen, so the confidences CVEE reads across its search window stay current.

The proxy returns full-width logits with the prompt segment zero-filled; nothing downstream reads
those positions.
"""
import torch


class PrefixCachedModel:
    """Wraps a cache-capable model so repeated full-buffer calls reuse the prompt's KV.
    `calls` counts forward passes, which caching must not change."""

    def __init__(self, model, prompt_len):
        self._model = model
        self.prompt_len = int(prompt_len)
        self.calls = 0
        self._kv = None

    def __call__(self, x, **kwargs):
        # A caller passing its own cache or mask wants a different computation; run it uncached.
        if kwargs or x.shape[1] <= self.prompt_len:
            self.calls += 1
            return self._model(x, **kwargs)
        if self._kv is None:
            self._kv = self._model(x[:, :self.prompt_len], use_cache=True).past_key_values
        self.calls += 1
        suffix = self._model(x[:, self.prompt_len:], past_key_values=self._kv,
                             use_cache=True).logits
        full = torch.zeros((suffix.shape[0], x.shape[1], suffix.shape[2]),
                           dtype=suffix.dtype, device=suffix.device)
        full[:, self.prompt_len:] = suffix
        return _Output(full)

    def __getattr__(self, name):
        return getattr(self._model, name)


class _Output:
    __slots__ = ("logits",)

    def __init__(self, logits):
        self.logits = logits
