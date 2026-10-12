"""Full reference decode with its per-step history, for offline analysis of a trajectory.

`generate_full` runs the reference schedule to the end (no CCTC, no exit) and returns what every
step saw, so stopping rules can be replayed against one cached decode instead of one live run
each: the raw argmax, the committed indices, and per generated position the top-1 probability and
the top-1/top-2 logit gap (Prophet's statistic). Its final buffer equals `generate` with both
gates off.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .decode import (LLADA_MASK_ID, add_gumbel_noise, block_layout, get_num_transfer_tokens,
                     shift_logits)


@torch.no_grad()
def generate_full(model, prompt, steps, gen_length, block_length, temperature=0.,
                  mask_id=LLADA_MASK_ID, logit_shift=False):
    """Returns (x, x0_history, transfer_history, conf_history, gap_history).

    x0_history[b]: (steps_per_block, seq_len) raw argmax at each step of block b, on CPU.
    transfer_history[b][s]: (k, 2) indices committed at step s of block b, on CPU.
    conf_history[pos] / gap_history[pos]: lists over all steps, for generation-relative `pos`;
    confidence is -inf past the current block, as the commit rule sees it."""
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_len] = prompt.clone()
    num_blocks, steps_per_block = block_layout(steps, gen_length, block_length)

    conf_history = {pos: [] for pos in range(gen_length)}
    gap_history = {pos: [] for pos in range(gen_length)}
    x0_history, transfer_history = [], []

    for num_block in range(num_blocks):
        block_end = prompt_len + (num_block + 1) * block_length
        quota = get_num_transfer_tokens(x[:, block_end - block_length:block_end] == mask_id,
                                        steps_per_block)[0].tolist()
        block_x0, block_transfer = [], []
        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x).logits
            if logit_shift:
                logits = shift_logits(logits)
            x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
            block_x0.append(x0.detach().cpu())

            p = F.softmax(logits, dim=-1)
            conf = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            conf[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)

            top2 = torch.topk(logits, k=2, dim=-1).values
            gaps = top2[..., 0] - top2[..., 1]
            gen_conf = conf[0, prompt_len:prompt_len + gen_length].tolist()
            gen_gaps = gaps[0, prompt_len:prompt_len + gen_length].tolist()
            for pos in range(gen_length):
                conf_history[pos].append(gen_conf[pos])
                gap_history[pos].append(gen_gaps[pos])

            confidence = torch.where(mask_index, conf, torch.tensor(-np.inf, device=conf.device))
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            if quota[i] > 0:
                _, select_index = torch.topk(confidence[0], k=quota[i])
                transfer[0, select_index] = True
            block_transfer.append(torch.nonzero(transfer, as_tuple=False).detach().cpu())
            x[transfer] = x0[transfer]

        x0_history.append(torch.cat(block_x0, dim=0))
        transfer_history.append(block_transfer)

    return x, x0_history, transfer_history, conf_history, gap_history
