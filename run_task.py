"""Run C4 on one task and report accuracy and decoding steps.

The two gates are independent, so this one entry point covers the deployed system, each gate on
its own, and the full-decoding baseline:

    deployed C4:  --struct-mode window --holes 1 --confirm --block-accel-tau 0.8 --final-too
    baseline:     --no-cvee --no-block-rule

The generation shape, answer parser, search mode, sample size and offset are looked up per task
(`c4/tasks.py`). Accuracy and step counts are reported here; throughput needs a dedicated card and
is measured separately.
"""
import argparse
import json
import os
import time

import torch

from c4.cctc import CCTCConfig
from c4.cvee import EXIT_GATES, CVEEConfig, GAMMA, P_MIN, TAU, TAIL_FRAC
from c4.decode import generate
from c4.grading import extract_answer, grade
from c4.prefix_cache import PrefixCachedModel
from c4.tasks import (DEFAULT_N, DEFAULT_SKIP, EXTRACTORS, GEN_SHAPE, LOADERS, SEARCH_MODE, TASKS,
                      build_chat_prompt)

MODEL_PATH = {"llada": "GSAI-ML/LLaDA-8B-Instruct", "dream": "Dream-org/Dream-v0-Instruct-7B"}
# The mask id and Dream's one-position logit shift are the only model-dependent pieces.
MASK_ID = {"llada": 126336, "dream": 151666}
# Deployed CVEE gate parameters, frozen across tasks and backbones.
OURS = {"params": {"gamma": GAMMA, "p_min": P_MIN, "tau": TAU}, "tail_frac": TAIL_FRAC}
SPAN_KEYS = ("span_len", "frozen", "by_cctc", "span_suppressed", "recon_match", "recon_total")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", choices=sorted(MODEL_PATH), default="llada")
    ap.add_argument("--n", type=int, default=None,
                    help="number of examples (default: the task family's evaluation size)")
    ap.add_argument("--seed", type=int, default=0, help="dataset shuffle seed")
    ap.add_argument("--skip", type=int, default=None,
                    help="offset into the shuffled split (default: the family's held-out offset)")
    ap.add_argument("--out", required=True, help="output JSON path")

    cctc = ap.add_argument_group("CCTC (local commitment)")
    cctc.add_argument("--block-accel-tau", type=float, default=0.8,
                      help="tau_CCTC: confidence a masked position needs to be eligible")
    cctc.add_argument("--no-block-rule", action="store_true",
                      help="disable CCTC: only the scheduled top-k commits, no early block exit")
    cctc.add_argument("--struct-mode", choices=["off", "window"], default="off",
                      help="'window': commit the boundary-anchored core; 'off': commit every\n"
                           "eligible position (bare confidence threshold)")
    cctc.add_argument("--holes", type=int, default=1,
                      help="H: unresolved holes the core may cross (window mode)")
    cctc.add_argument("--confirm", action="store_true",
                      help="confirm deferred positions one step later (window mode)")
    cctc.add_argument("--stall-escalate", type=int, default=3,
                      help="commit all eligible positions after this many consecutive steps in\n"
                           "which the window admitted none (0 = never)")
    cctc.add_argument("--final-too", action="store_true",
                      help="accelerate the final block too (A = {0..N-1}, deployed)")
    cctc.add_argument("--protect-span", action="store_true",
                      help="control arm: inside the span CVEE is reading, only the schedule commits")
    cctc.add_argument("--protect-block", action="store_true",
                      help="control arm: in the block holding the candidate, from its first\n"
                           "extraction until the gate fires, only the schedule commits")

    cvee = ap.add_argument_group("CVEE (global exit)")
    cvee.add_argument("--no-cvee", action="store_true", help="disable the exit gate")
    cvee.add_argument("--gate", choices=sorted(EXIT_GATES), default="conf_floor",
                      help="the joint gate or one of its single-condition ablations")
    cvee.add_argument("--exit-momentum", type=float, default=0.0, metavar="ALPHA",
                      help="fill ceil(ALPHA*sqrt(t)) steps after the gate fires at step t\n"
                           "instead of immediately (0 = off)")
    cvee.add_argument("--span-recon-probe", action="store_true",
                      help="diagnostic: when the gate fires, re-mask the accepted span on a copy\n"
                           "and record how much one extra forward pass rebuilds (not for timing)")

    shape = ap.add_argument_group("generation shape and caching")
    shape.add_argument("--gen-length", type=int, default=None,
                       help="override L; T follows it (one token per step at baseline)")
    shape.add_argument("--block-length", type=int, default=None, help="override B")
    shape.add_argument("--prefix-cache", action="store_true",
                       help="reuse the prompt's KV across steps (needs a cache-capable backbone)")

    out = ap.add_argument_group("output")
    out.add_argument("--save-text", action="store_true",
                     help="store each example's full generated text")
    out.add_argument("--record-traj", default=None, metavar="PATH",
                     help="save per-step x0 / confidence / filled buffer of every example (.pt)")
    out.add_argument("--dry-run", action="store_true",
                     help="resolve shape and sample, print them, and exit before loading the model")
    return ap


def resolve_shape(args):
    gen_length, steps, block_length = GEN_SHAPE[args.task]
    if args.gen_length is not None:
        gen_length = steps = args.gen_length
    if args.block_length is not None:
        block_length = args.block_length
    if gen_length % block_length or steps % (gen_length // block_length):
        raise SystemExit(f"steps={steps} block_length={block_length} incompatible with "
                         f"gen_length={gen_length}")
    return gen_length, steps, block_length


def decode_configs(args):
    """(CCTCConfig | None, CVEEConfig | None) for the parsed command line."""
    if (args.protect_span or args.protect_block) and args.no_cvee:
        raise SystemExit("--protect-span/--protect-block need the exit gate: the protected span "
                         "is the one the gate reads, and --no-cvee never extracts it")
    cctc = None if args.no_block_rule else CCTCConfig(
        tau=args.block_accel_tau, mode=args.struct_mode, holes=args.holes, confirm=args.confirm,
        final_block=args.final_too, stall_escalate=args.stall_escalate,
        protect_span=args.protect_span, protect_block=args.protect_block)
    cvee = None if args.no_cvee else CVEEConfig(
        extractor=EXTRACTORS[args.task], search_mode=SEARCH_MODE[args.task],
        tail_frac=OURS["tail_frac"], gate=EXIT_GATES[args.gate], gate_params=OURS["params"],
        momentum_alpha=args.exit_momentum, span_recon_probe=args.span_recon_probe)
    return cctc, cvee


def span_summary(rows):
    """Aggregate the per-example audit of the span the gate accepted (Appendix "span control"):
    `frozen_frac` is how much of it was already committed when the gate read it, `cctc_frac` how
    much of it CCTC rather than the schedule had committed."""
    if not rows:
        return {"n_fired": 0}
    total = sum(r["span_len"] for r in rows)
    out = {"n_fired": len(rows), "span_tokens": total,
           "frozen_frac": sum(r.get("frozen", 0) for r in rows) / total if total else 0.0,
           "cctc_frac": sum(r.get("by_cctc", 0) for r in rows) / total if total else 0.0,
           "suppressed_commits": sum(r.get("span_suppressed", 0) for r in rows)}
    recon_total = sum(r.get("recon_total", 0) for r in rows)
    if recon_total:
        out["recon_frac"] = sum(r.get("recon_match", 0) for r in rows) / recon_total
        out["recon_exact_frac"] = sum(
            r.get("recon_match", 0) == r.get("recon_total", -1) for r in rows) / len(rows)
    return out


def load_model(name):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[name], trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH[name], trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    return model, tokenizer


class TrajectoryRecorder:
    """Per-step hook storing the generation region's x0, confidence and filled buffer on CPU."""

    def __init__(self, mask_id):
        self.mask_id, self.steps = mask_id, []

    def __call__(self, x0, true_conf, x, prompt_len, gen_length, **_):
        region = slice(prompt_len, prompt_len + gen_length)
        filled = torch.where(x == self.mask_id, x0, x)
        self.steps.append((x0[0, region].to("cpu", torch.int32).clone(),
                           true_conf[0, region].to("cpu", torch.float32).clone(),
                           filled[0, region].to("cpu", torch.int32).clone()))

    def stacked(self):
        return {key: torch.stack([s[j] for s in self.steps])
                for j, key in enumerate(("x0", "conf", "filled"))}


def main():
    # Some environments get 401s from the Hub's Xet backend for code datasets; plain HTTP works.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = build_parser().parse_args()
    n = args.n if args.n is not None else DEFAULT_N[args.task]
    skip = args.skip if args.skip is not None else DEFAULT_SKIP[args.task]
    gen_length, steps, block_length = resolve_shape(args)
    cctc, cvee = decode_configs(args)
    docs = LOADERS[args.task](n=n, seed=args.seed, skip=skip)
    print(f"[{args.task}/{args.model}] n={len(docs)} seed={args.seed} skip={skip} "
          f"shape={gen_length}/{steps}/{block_length} tau_blk={args.block_accel_tau} "
          f"block_rule={not args.no_block_rule} struct={args.struct_mode} holes={args.holes} "
          f"confirm={args.confirm} final_too={args.final_too} "
          f"cvee={not args.no_cvee} gate={args.gate}", flush=True)
    if args.dry_run:
        return

    model, tokenizer = load_model(args.model)
    mask_id = MASK_ID[args.model]
    correct, step_counts, per_example, block_hist, span_audit, traj = [], [], [], [], [], []
    counters = {}
    for idx, doc in enumerate(docs):
        prompt = torch.tensor(
            tokenizer(build_chat_prompt(tokenizer, doc["prompt_text"]))["input_ids"],
            device="cuda").unsqueeze(0)
        t0 = time.time()
        run_model = PrefixCachedModel(model, prompt.shape[1]) if args.prefix_cache else model
        recorder = TrajectoryRecorder(mask_id) if args.record_traj else None
        counters.clear()  # per generation: a stale span key would be read as this example's
        result = generate(run_model, tokenizer, prompt, gen_length=gen_length, steps=steps,
                          block_length=block_length, mask_id=mask_id,
                          logit_shift=(args.model == "dream"), cctc=cctc, cvee=cvee,
                          stats=counters, traj_recorder=recorder)
        ok = bool(grade(args.task, result.text, doc["gt"]))
        correct.append(ok)
        step_counts.append(result.steps)
        block_hist.append(list(result.block_steps))
        # A gate that never fired on a located span leaves no span keys behind.
        span_rec = ({k: counters[k] for k in SPAN_KEYS if k in counters}
                    if "span_len" in counters else {})
        if span_rec:
            span_audit.append({"idx": idx, **span_rec})
        per_example.append({"idx": idx, "correct": ok, "nfe": result.steps,
                            "block_steps": list(result.block_steps),
                            "pred": extract_answer(args.task, result.text),
                            **({"text": result.text} if args.save_text else {}),
                            **{k if k.startswith("span") else f"span_{k}": v
                               for k, v in span_rec.items()},
                            "wall_s": time.time() - t0})
        if recorder is not None:
            traj.append({"idx": idx, "gt": doc["gt"], "prompt_len": int(prompt.shape[1]),
                         **recorder.stacked()})
        if (idx + 1) % 25 == 0 or idx == len(docs) - 1:
            print(f"  [{idx + 1}/{len(docs)}] acc={sum(correct) / len(correct):.4f} "
                  f"nfe={sum(step_counts) / len(step_counts):.2f}", flush=True)

    acc, nfe = sum(correct) / len(correct), sum(step_counts) / len(step_counts)
    longest = max(len(b) for b in block_hist)
    summary = {"task": args.task, "model": args.model, "n": len(docs), "seed": args.seed,
               "skip": skip, "gen_length": gen_length, "steps": steps,
               "block_length": block_length,
               "block_accel_tau": None if args.no_block_rule else args.block_accel_tau,
               "struct_mode": args.struct_mode, "holes": args.holes, "confirm": args.confirm,
               "final_too": args.final_too, "protect_span": args.protect_span,
               "protect_block": args.protect_block, "exit_momentum": args.exit_momentum,
               "span_audit": span_summary(span_audit), "prefix_cache": args.prefix_cache,
               "cvee": not args.no_cvee, "gate": args.gate,
               "acc": acc, "avg_nfe": nfe, "step_ratio": steps / nfe,
               "steps_per_block": steps // (gen_length // block_length), "total_steps": steps,
               # mean steps spent per block, over examples (0 for blocks never reached)
               "avg_block_steps": [sum(b[i] if i < len(b) else 0 for b in block_hist)
                                   / len(block_hist) for i in range(longest)],
               "struct_counters": dict(counters)}
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
