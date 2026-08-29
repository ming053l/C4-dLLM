# C⁴ — Commit Locally, Exit Globally

Accelerated decoding for masked diffusion language models, from *Commit Locally, Exit Globally:
Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models*.

A diffusion LM exposes a provisional prediction at every denoising step, and on many tasks the
candidate answer inside it stops changing well before the step schedule is exhausted. C⁴ turns that
into two separate decisions, each with its own gate.

- **CVEE** decides *when* the sequence may stop. It re-extracts the candidate answer at every step
  and permits termination only once confidence and sustained argmax stability hold together over it.
- **CCTC** decides *which token positions* a step may commit. It borrows an autoregressive freezing
  order inside each block, committing a boundary-anchored core and confirming deferred positions one
  step later, without an extra forward pass.

One frozen configuration removes 64–95% of decoding steps across 12 zero-shot tasks on LLaDA-8B and
Dream-7B, at 2.6×–18.6× measured end-to-end speedup.

## Install

```bash
pip install -r requirements.txt
```

The backbones are loaded from the Hub on first use and are not vendored here, since both ship their
own modeling code that `trust_remote_code=True` pulls in:

| `--model` | checkpoint |
|---|---|
| `llada` | [`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) |
| `dream` | [`Dream-org/Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |

One 40 GB card is enough for either at the shapes used here. Dream needs a one-position logit shift
that `run_task.py` applies for you; that and the mask id are the only model-dependent pieces.

## Run it

Accelerated:

```bash
python run_task.py --task mmlu --model llada \
    --struct-mode window --holes 1 --confirm --block-accel-tau 0.8 --final-too \
    --out out/mmlu_c4.json
```

The same model with both gates off, which is the baseline every speedup is measured against:

```bash
python run_task.py --task mmlu --model llada --no-cvee --no-block-rule --out out/mmlu_base.json
```

On LLaDA/MMLU the two print `acc=0.6400  nfe=4.82/64` and `acc=0.6400  nfe=64.00/64`, so the
accuracy is unchanged and 92.5% of the steps are gone.

## Using the gates on your own decoder

`c4.generate_live` is the accelerated loop. Both gates are off unless their arguments are given, so
the same call reproduces an unaccelerated schedule.

```python
from c4 import generate_live, POLICIES

text, steps_used, steps_budget, per_block = generate_live(
    model, tokenizer, prompt,
    steps=64, gen_length=64, block_length=16,
    block_accel_tau=0.8, block_struct_mode="window",     # CCTC
    block_struct_holes=1, block_struct_confirm=True,
    block_accel_final_too=True,
    policy_fn=POLICIES["conf_floor"],                     # CVEE
    policy_params={"tau": 0.7, "gamma": 2.0, "p_min": 3},
    extractor=my_extractor, search_mode="last",
)
```

If you already have a decoder you do not want to replace, `c4.CVEETracker` is the exit gate alone.
Feed it the argmax and its softmax probability once per step and it tells you when to stop.

## Configuration

The deployed setting is one frozen tuple, used unchanged for every task and both backbones.

| | value | what it controls |
|---|---|---|
| `--block-accel-tau` | `0.8` | confidence a position must clear before CCTC will commit it |
| `--holes` | `1` | unresolved gaps the boundary-anchored core may cross |
| `--confirm` | on | recheck deferred positions one step later |
| `--final-too` | on | accelerate every block, including the one holding the answer |
| CVEE `tau/gamma/p_min` | `0.7 / 2.0 / 3` | confidence floor, extra stable steps demanded per flip, minimum run |

Other flags select the arms rather than change the deployed system. `--gate` picks the joint exit
gate or one of its two single-condition versions, `--gen-length` and `--block-length` sweep the
generation shape, `--exit-momentum` delays the exit by `ceil(ALPHA*sqrt(t))` steps, and
`--protect-span` and `--protect-block` bar CCTC from the span the exit gate reads.

## Layout

```
run_task.py   entry point: one task, one backbone, accuracy and step count
c4/
  decode.py       the accelerated decoding loop
  gate.py         CVEE as a standalone stop signal
  policies.py     the gate conditions
  extract.py      answer extraction and grading
  prompts.py      task loading and generation shapes
  prefix_cache.py prompt KV reuse, exact rather than approximate
```

## Citation

```bibtex
@article{lee2026c4,
  title  = {Commit Locally, Exit Globally: Coordinating Adaptive Sampling and Early Exit in Diffusion Language Models},
  author = {Lee, Chia-Ming and Liu, Shao-Kai and Chang, Ming-Ching and Li, Xin and Liu, Yu-Lun and Hsu, Chih-Chung},
  journal= {arXiv preprint arXiv:2607.28166},
  year   = {2026}
}
```

MIT License.
