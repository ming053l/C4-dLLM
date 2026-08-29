"""Run C4 on one task and report accuracy and the decoding steps it spent.

The two gates are independent and either can be turned off, so this one entry point covers the
deployed system, each gate on its own, and the unaccelerated baseline. CVEE decides when the
sequence may stop and is selected with `--gate`; CCTC decides which token positions a step may
commit and is turned on with `--struct-mode window --confirm`. Passing `--no-cvee --no-block-rule`
disables both and reproduces the model's own schedule, which is the baseline every speedup is
measured against.

Everything task-dependent is looked up rather than passed in. The generation shape comes from
GEN_SHAPE, the answer extractor and its search region from the task-family tables, and the sample
size and offset from the convention each family is evaluated under. Getting any of those wrong
produces a number that looks comparable and is not.

Accuracy and step count are reported here. Throughput is measured separately, since a shared
machine cannot give a trustworthy wall-clock number.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from c4.codegen_grade import code_candidate  # noqa: E402
from c4.decode import generate_live  # noqa: E402
from c4.extract import (extract_answer, extract_boxed, extract_letter,  # noqa: E402
                             extract_number, grade)
from c4.prompts import GEN_SHAPE, LOADERS, build_chat_prompt  # noqa: E402
from c4.policies import POLICIES  # noqa: E402

from c4.prefix_cache import PrefixCachedModel  # noqa: E402

MODEL_PATH = {"llada": "GSAI-ML/LLaDA-8B-Instruct", "dream": "Dream-org/Dream-v0-Instruct-7B"}
# Mask id and the one-position logit shift are the only model-dependent pieces of the decode loop,
# taken from live_eval.py so a Dream run here is the same decode Table 1 already reports.
MASK_ID = {"llada": 126336, "dream": 151666}
# the harness's `ours` variant, verbatim from its live_eval.py VARIANTS table. Frozen across every task,
# exactly as C4 freezes it, so nothing here is retuned per task.
OURS = {"policy": "conf_floor", "params": {"gamma": 2.0, "p_min": 3, "tau": 0.7}, "tail_frac": 0.3}

_MC = ["mmlu", "arc_c", "hellaswag", "winogrande", "piqa", "truthfulqa"]
_NUM = ["gsm8k", "svamp", "asdiv", "gsmhard"]
_CODE = ["humaneval", "mbpp"]
EXTRACTORS = {**{t: extract_number for t in _NUM}, "math": extract_boxed,
              **{t: extract_letter for t in _MC}, **{t: code_candidate for t in _CODE}}
SEARCH_MODE = {**{t: "last" for t in _NUM}, "math": "last",
               **{t: "first5" for t in _MC}, **{t: "code" for t in _CODE}}
# the harness's own per-family sample convention: 200 for short-answer, 100 for long-reasoning at a
# held-out offset, and the full set for code, which contributes to no calibration pool.
DEFAULT_N = {**{t: 200 for t in _MC}, **{t: 100 for t in _NUM}, "math": 100,
             "humaneval": 164, "mbpp": 257}
DEFAULT_SKIP = {**{t: 40 for t in _MC}, **{t: 40 for t in _NUM}, "math": 40,
                "humaneval": 0, "mbpp": 0}


def _span_summary(rows):
    """Aggregate the per-example answer-span audit into the three numbers the control arm turns
    on. `frozen_frac` is how much of the accepted span was already unmasked when the gate looked
    at it, and `cctc_frac` is how much of it CCTC had committed on its own rather than through the
    schedule's top-k quota. A small `cctc_frac` means the stability the gate measured was not
    manufactured by our own sampler."""
    if not rows:
        return {"n_fired": 0}
    tot = sum(r["span_len"] for r in rows)
    frozen = sum(r.get("frozen", 0) for r in rows)
    cctc = sum(r.get("by_cctc", 0) for r in rows)
    out = {"n_fired": len(rows), "span_tokens": tot,
           "frozen_frac": frozen / tot if tot else 0.0,
           "cctc_frac": cctc / tot if tot else 0.0,
           "suppressed_commits": sum(r.get("span_suppressed", 0) for r in rows)}
    rt = sum(r.get("recon_total", 0) for r in rows)
    if rt:
        out["recon_frac"] = sum(r.get("recon_match", 0) for r in rows) / rt
        out["recon_exact_frac"] = sum(
            r.get("recon_match", 0) == r.get("recon_total", -1) for r in rows) / len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(EXTRACTORS))
    ap.add_argument("--model", choices=["llada", "dream"], default="llada")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=None)
    ap.add_argument("--block-accel-tau", type=float, default=0.9)
    ap.add_argument("--struct-mode", choices=["off", "window", "risk"], default="off")
    ap.add_argument("--holes", type=int, default=1)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--confirm-holes", type=int, default=None)
    ap.add_argument("--settle-run", type=int, default=0)
    ap.add_argument("--risk-budget", type=float, default=None)
    ap.add_argument("--risk-scope", choices=["step", "block"], default="step")
    ap.add_argument("--stall-escalate", type=int, default=3)
    ap.add_argument("--gen-length", type=int, default=None,
                    help="override the task's generation length; steps follow it, since this\n"
                         "protocol keeps one token per step on average at baseline")
    ap.add_argument("--block-length", type=int, default=None)
    ap.add_argument("--prefix-cache", action="store_true",
                    help="reuse the prompt's KV across steps; exact, since the prompt is\n"
                         "never re-masked, so only per-step cost changes")
    ap.add_argument("--final-too", action="store_true",
                    help="widen the paced set to every block including the final one "
                         "(Eq. eq:uniform's A={0..N-1}); the answer-bearing block is then "
                         "committed on the same evidence CVEE re-judges afterwards")
    ap.add_argument("--protect-span", action="store_true",
                    help="control arm for the exit gate: CCTC may not commit inside the span the\n"
                         "gate is currently reading, so only the schedule's own top-k writes\n"
                         "there and candidate stability cannot be produced by our own commits")
    ap.add_argument("--protect-block", action="store_true",
                    help="control for the CCTC/CVEE interaction: from the first step a candidate\n"
                         "is extractable until the gate fires, bar CCTC from committing anywhere\n"
                         "in the block the candidate sits in, so it cannot carry the path there.\n"
                         "Other blocks keep CCTC, which is what separates this from disabling it")
    ap.add_argument("--exit-momentum", type=float, default=0.0, metavar="ALPHA",
                    help="delay the exit by ceil(ALPHA*sqrt(t)) steps past the step t the gate\n"
                         "fires on, so ordinary decoding commits some of what the one-shot fill\n"
                         "would otherwise have to write. 0 disables it")
    ap.add_argument("--save-text", action="store_true",
                    help="store each example's full generated text, which grading discards after\n"
                         "extracting the answer. Needed to score generative perplexity offline,\n"
                         "since the one-shot fill that follows an exit writes token positions no\n"
                         "accuracy metric ever looks at")
    ap.add_argument("--span-recon-probe", action="store_true",
                    help="read-only diagnostic: at the step the gate fires, re-mask the accepted\n"
                         "span on a throwaway clone and record how much of it the model rewrites\n"
                         "identically from context alone. One extra forward pass per example,\n"
                         "no effect on the decode, so it must not be used for timing runs")
    ap.add_argument("--gate", choices=["conf_floor", "confidence_only", "stability_only"],
                    default="conf_floor",
                    help="which CVEE arm to run: the deployed joint gate, or one of its\n"
                         "two single-condition ablations")
    ap.add_argument("--no-cvee", action="store_true")
    ap.add_argument("--no-block-rule", action="store_true",
                    help="disable the block-level commit rule entirely, leaving the\n"
                         "baseline scheduled top-k and no early block exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the shape and sample and exit before loading the model, so a\n"
                         "shape override can be checked on a machine with no free memory")
    ap.add_argument("--record-traj", default=None,
                    help="if set, save per-step x0/conf/filled tensors for each example in a\n"
                         "WINO-compatible .pt format for visualization or offline inspection")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = args.n if args.n is not None else DEFAULT_N[args.task]
    skip = args.skip if args.skip is not None else DEFAULT_SKIP[args.task]
    gen_length, steps, block_length = GEN_SHAPE[args.task]
    if args.gen_length is not None:
        gen_length = steps = args.gen_length
    if args.block_length is not None:
        block_length = args.block_length
    assert gen_length % block_length == 0 and steps % (gen_length // block_length) == 0, \
        f"steps={steps} block_length={block_length} incompatible with gen_length={gen_length}"
    docs = LOADERS[args.task](n=n, seed=args.seed, skip=skip)
    if (args.protect_span or args.protect_block) and args.no_cvee:
        raise SystemExit("--protect-span needs the exit gate: the protected span is the one the "
                         "gate reads, and --no-cvee never extracts it")
    policy_fn = None if args.no_cvee else POLICIES[args.gate]
    counters = {}
    span_audit = []
    print(f"[{args.task}/{args.model}] n={len(docs)} seed={args.seed} skip={skip} "
          f"shape={gen_length}/{steps}/{block_length} tau_blk={args.block_accel_tau} "
          f"block_rule={not args.no_block_rule} struct={args.struct_mode} holes={args.holes} confirm={args.confirm} final_too={args.final_too} "
          f"cvee={not args.no_cvee} gate={args.gate}", flush=True)

    if args.dry_run:
        return
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args.model], trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH[args.model], trust_remote_code=True,
                                           torch_dtype=torch.bfloat16).to("cuda").eval()

    correct, nfes, per_example = [], [], []
    traj = []
    # Per-block step usage, averaged over examples: Figure 5's panel is drawn from this
    # and no other run records it.
    block_hist = []
    for idx, doc in enumerate(docs):
        prompt = torch.tensor(
            tokenizer(build_chat_prompt(tokenizer, doc["prompt_text"]))["input_ids"],
            device="cuda").unsqueeze(0)
        t0 = time.time()
        run_model = (PrefixCachedModel(model, prompt.shape[1]) if args.prefix_cache
                     else model)
        rec = [] if args.record_traj else None
        counters.clear()  # the span audit is per generation; a stale key would be read as this one's

        def _traj_recorder(x0, true_conf, x, prompt_len, gen_length, **_):
            filled = torch.where(x == MASK_ID[args.model], x0, x)
            rec.append((x0[0, prompt_len:prompt_len + gen_length].to("cpu", torch.int32).clone(),
                        true_conf[0, prompt_len:prompt_len + gen_length].to("cpu", torch.float32).clone(),
                        filled[0, prompt_len:prompt_len + gen_length].to("cpu", torch.int32).clone()))

        text, exit_step, total_steps, block_steps = generate_live(
            run_model, tokenizer, prompt, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0.0, remasking="low_confidence",
            policy_fn=policy_fn, policy_params=OURS["params"],
            extractor=EXTRACTORS[args.task], search_mode=SEARCH_MODE[args.task],
            tail_frac=OURS["tail_frac"], mask_id=MASK_ID[args.model],
            logit_shift=(args.model == "dream"),
            block_accel_tau=None if args.no_block_rule else args.block_accel_tau,
            block_struct_mode=args.struct_mode, block_struct_holes=args.holes,
            block_struct_confirm=args.confirm, block_struct_confirm_holes=args.confirm_holes,
            block_struct_settle_run=args.settle_run,
            block_struct_risk_budget=args.risk_budget, block_struct_risk_scope=args.risk_scope,
            block_struct_stall_escalate=args.stall_escalate, block_struct_stats=counters,
            block_accel_final_too=args.final_too,
            block_accel_protect_span=args.protect_span,
            block_accel_protect_block=args.protect_block,
            span_recon_probe=args.span_recon_probe,
            exit_momentum_alpha=args.exit_momentum,
            traj_recorder=_traj_recorder if rec is not None else None)
        ok = grade(args.task, text, doc["gt"])
        correct.append(bool(ok))
        nfes.append(exit_step)
        block_hist.append(list(block_steps))
        # `counters` is rewritten by every generation, so read this example's span audit now.
        # A gate that never fired leaves no span keys behind, which is itself the record.
        span_rec = ({k: counters[k] for k in ("span_len", "frozen", "by_cctc", "span_suppressed",
                                             "recon_match", "recon_total")
                     if k in counters} if "span_len" in counters else {})
        if span_rec:
            span_audit.append({"idx": idx, **span_rec})
        per_example.append({"idx": idx, "correct": bool(ok), "nfe": exit_step,
                            "block_steps": list(block_steps),
                            "pred": extract_answer(args.task, text),
                            **({"text": text} if args.save_text else {}),
                            **{f"span_{k}" if not k.startswith("span") else k: v
                               for k, v in span_rec.items()},
                            "wall_s": time.time() - t0})
        if rec is not None:
            traj.append({"idx": idx, "gt": doc["gt"], "prompt_len": int(prompt.shape[1]),
                         "x0": torch.stack([r[0] for r in rec]),
                         "conf": torch.stack([r[1] for r in rec]),
                         "filled": torch.stack([r[2] for r in rec])})
        if (idx + 1) % 25 == 0 or idx == len(docs) - 1:
            print(f"  [{idx + 1}/{len(docs)}] acc={sum(correct) / len(correct):.4f} "
                  f"nfe={sum(nfes) / len(nfes):.2f}", flush=True)

    acc, nfe = sum(correct) / len(correct), sum(nfes) / len(nfes)
    summary = {"task": args.task, "model": args.model, "n": len(docs), "seed": args.seed,
               "skip": skip,
               "gen_length": gen_length, "steps": steps, "block_length": block_length,
               "block_accel_tau": None if args.no_block_rule else args.block_accel_tau, "struct_mode": args.struct_mode,
               "holes": args.holes, "confirm": args.confirm, "final_too": args.final_too,
               "protect_span": args.protect_span,
               "protect_block": args.protect_block,
               "exit_momentum": args.exit_momentum,
               "span_audit": _span_summary(span_audit),
               "prefix_cache": args.prefix_cache,
               "settle_run": args.settle_run,
               "risk_budget": args.risk_budget, "cvee": not args.no_cvee, "gate": args.gate,
               "acc": acc, "avg_nfe": nfe, "step_ratio": steps / nfe,
               "steps_per_block": steps // (gen_length // block_length),
               "total_steps": steps,
               "avg_block_steps": [sum(b[i] if i < len(b) else 0 for b in block_hist)
                                   / len(block_hist)
                                   for i in range(max(len(b) for b in block_hist))],
               "struct_counters": {k: (v.item() if hasattr(v, "item") else v)
                                   for k, v in counters.items()}}
    print(f"\n[{args.task}] acc={acc:.4f} ({sum(correct)}/{len(correct)})  "
          f"nfe={nfe:.2f}/{steps}  ratio={steps / nfe:.2f}x")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2)
    print(f"[{args.task}] wrote {args.out}")
    if args.record_traj:
        os.makedirs(os.path.dirname(args.record_traj) or ".", exist_ok=True)
        torch.save({"task": args.task, "gen_length": gen_length, "block_length": block_length,
                    "examples": traj}, args.record_traj)
        print(f"[{args.task}] wrote trajectory {args.record_traj} "
              f"({sum(len(e['x0']) for e in traj)} recorded steps)")


if __name__ == "__main__":
    main()
