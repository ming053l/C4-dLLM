"""Regression check: does this checkout decode exactly like a reference commit?

For each configuration (a `run_task.py` flag string) and task sample, both implementations decode
the same prompts on the same model with greedy decoding, and the final text, the total step count,
the per-block step list, the graded prediction and the CCTC/CVEE counters are compared exactly.
Each implementation runs in its own process (both packages are called `c4`), then the two dumps
are diffed:

    git worktree add /tmp/c4_ref b6913bf          # the reference: the release the paper used
    python tools/check_equivalence.py run --impl reference --root /tmp/c4_ref --out ref.jsonl
    python tools/check_equivalence.py run --impl current --root . --out cur.jsonl
    python tools/check_equivalence.py compare ref.jsonl cur.jsonl

`--plan` picks the configurations and samples (see PLANS); each record also carries the wall
time of its decode, so the "timing" plan doubles as a before/after speed comparison when both
runs use the same idle card. The reference side parses each flag
string with the reference's own argument parser and calls its `generate_live` exactly as its
`run_task.py` did, so a changed default on either side shows up as a difference.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time

DEPLOYED = "--struct-mode window --holes 1 --confirm --block-accel-tau 0.8 --final-too"
ALL_TASKS_DEPLOYED = [("mmlu", 8), ("gsm8k", 4), ("humaneval", 3), ("math", 2), ("hellaswag", 4),
                      ("arc_c", 2), ("winogrande", 2), ("piqa", 2), ("svamp", 2), ("asdiv", 2),
                      ("gsmhard", 2), ("mbpp", 2)]
ARM_TASKS = [("mmlu", 4), ("gsm8k", 2), ("humaneval", 2)]

# name -> (flags, samples). Flags prefixed "api:" bypass the CLI and call `generate_live` with
# keyword arguments the CLI does not expose (peer exit rules, the DualCache code path, ...).
LLADA_PLAN = {
    "deployed": (DEPLOYED, ALL_TASKS_DEPLOYED),
    "baseline": ("--no-cvee --no-block-rule", ARM_TASKS),
    "gate_confidence_only": ("--gate confidence_only --no-block-rule", ARM_TASKS),
    "gate_stability_only": ("--gate stability_only --no-block-rule", ARM_TASKS),
    "cvee_only": ("--no-block-rule", ARM_TASKS),
    "cctc_core_only": ("--struct-mode window --holes 1 --block-accel-tau 0.8", ARM_TASKS),
    "cctc_core_confirm": ("--struct-mode window --holes 1 --confirm --block-accel-tau 0.8",
                          ARM_TASKS),
    "threshold_rule": ("--block-accel-tau 0.8", ARM_TASKS),
    "protect_span": (DEPLOYED + " --protect-span", ARM_TASKS),
    "protect_block": (DEPLOYED + " --protect-block", ARM_TASKS),
    "exit_momentum_1.0": (DEPLOYED + " --exit-momentum 1.0", ARM_TASKS),
    "span_recon_probe": (DEPLOYED + " --span-recon-probe", ARM_TASKS),
    "tau_0.9": (DEPLOYED.replace("0.8", "0.9"), ARM_TASKS),
    "holes_0": (DEPLOYED.replace("--holes 1", "--holes 0"), ARM_TASKS),
    "holes_2": (DEPLOYED.replace("--holes 1", "--holes 2"), ARM_TASKS),
    "shape_L128_B32": (DEPLOYED + " --gen-length 128 --block-length 32", ARM_TASKS),
    "prefix_cache": (DEPLOYED + " --prefix-cache", [("mmlu", 1)]),
    "api:prophet": ("--no-cvee --no-block-rule", ARM_TASKS),
    "api:sched": ("--no-cvee --no-block-rule", ARM_TASKS),
    "api:fastdllm_floor": ("--no-cvee --block-accel-tau 0.9 --final-too", ARM_TASKS),
    "api:dual_cache_emulated": (DEPLOYED, [("mmlu", 2), ("gsm8k", 1)]),
    "api:generate_full": ("--no-cvee --no-block-rule", [("mmlu", 2), ("gsm8k", 1)]),
}
DREAM_PLAN = {
    "deployed": (DEPLOYED, [("mmlu", 4), ("gsm8k", 2), ("humaneval", 1)]),
    "baseline": ("--no-cvee --no-block-rule", [("mmlu", 2), ("gsm8k", 1)]),
    "prefix_cache": (DEPLOYED + " --prefix-cache", [("mmlu", 2), ("gsm8k", 1)]),
}
SMOKE_PLAN = {"deployed": (DEPLOYED, [("mmlu", 2), ("gsm8k", 1)]),
              "baseline": ("--no-cvee --no-block-rule", [("mmlu", 1)])}
TIMING_PLAN = {"deployed": (DEPLOYED, [("mmlu", 8), ("gsm8k", 4), ("humaneval", 3)])}
PLANS = {"llada": LLADA_PLAN, "dream": DREAM_PLAN, "smoke": SMOKE_PLAN, "timing": TIMING_PLAN}
PROPHET = {"early": 7.5, "mid": 5.0, "late": 2.5}
SCHED = {"tau_mode": "cosine", "tau_high": 7.5, "tau_low": 2.5, "patience_steps": 0}


class _DualCacheEmulator:
    """Accepts Fast-dLLM's DualCache calling convention on a stock model by recomputing the full
    buffer, so the loop's DualCache bookkeeping can be compared without Fast-dLLM's modeling
    code. It checks the code path, not the approximation."""

    def __init__(self, model):
        self._m = model

    def __call__(self, x, past_key_values=None, use_cache=False, replace_position=None, **kw):
        if past_key_values is None:
            out = self._m(x)
            return _Out(out.logits, x.clone())
        full = past_key_values.clone()
        full[replace_position] = x.reshape(-1)
        logits = self._m(full).logits
        return _Out(logits[replace_position].reshape(x.shape[0], x.shape[1], -1), full)

    def __getattr__(self, name):
        return getattr(self._m, name)


class _Out:
    def __init__(self, logits, kv):
        self.logits, self.past_key_values = logits, kv


def _parse_reference_args(rt, argv):
    """Parse `argv` with the reference run_task's own parser (built inside its main())."""
    class Captured(Exception):
        pass

    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise Captured(real_parse(self, argv, namespace))

    argparse.ArgumentParser.parse_args = capture
    try:
        rt.main()
    except Captured as c:
        return c.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    raise RuntimeError("reference main() returned without parsing arguments")


def _digest(obj):
    import torch
    h = hashlib.sha256()
    if isinstance(obj, str):
        h.update(obj.encode())
    elif isinstance(obj, torch.Tensor):
        h.update(obj.cpu().numpy().tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj):
            h.update(repr((k, obj[k])).encode())
    else:
        for item in obj:
            h.update(_digest(item).encode())
    return h.hexdigest()[:16]


class _Impl:
    """The pieces of one implementation the harness needs, behind one interface."""

    def __init__(self, kind, root):
        sys.path.insert(0, os.path.abspath(root))
        os.chdir(os.path.abspath(root))
        self.kind = kind
        self.rt = importlib.import_module("run_task")
        if kind == "reference":
            self.prompts = importlib.import_module("c4.prompts")
            self.extract = importlib.import_module("c4.extract")
            self.decode = importlib.import_module("c4.decode")
            self.cache = importlib.import_module("c4.prefix_cache")
        else:
            self.tasks = importlib.import_module("c4.tasks")
            self.grading = importlib.import_module("c4.grading")
            self.decode = importlib.import_module("c4.decode")
            self.cache = importlib.import_module("c4.prefix_cache")

    def loaders(self):
        return (self.prompts if self.kind == "reference" else self.tasks).LOADERS

    def chat(self, tok, text):
        return (self.prompts if self.kind == "reference" else self.tasks).build_chat_prompt(tok,
                                                                                          text)

    def shape(self, task, args):
        shapes = (self.prompts if self.kind == "reference" else self.tasks).GEN_SHAPE
        gen_length, steps, block_length = shapes[task]
        if args.gen_length is not None:
            gen_length = steps = args.gen_length
        if args.block_length is not None:
            block_length = args.block_length
        return gen_length, steps, block_length

    def grade(self, task, text, gt):
        mod = self.extract if self.kind == "reference" else self.grading
        return bool(mod.grade(task, text, gt)), mod.extract_answer(task, text)

    def parse(self, flags, task):
        argv = ["--task", task, "--out", "/dev/null"] + flags.split()
        if self.kind == "reference":
            return _parse_reference_args(self.rt, argv)
        return self.rt.build_parser().parse_args(argv)

    def decode_one(self, model, tok, prompt, task, args, shape, api, model_name):
        gen_length, steps, block_length = shape
        stats = {}
        mask_id = self.rt.MASK_ID[model_name]
        run_model = (self.cache.PrefixCachedModel(model, prompt.shape[1]) if args.prefix_cache
                     else model)
        extra = {}
        if api == "api:prophet":
            extra["prophet_thresholds"] = PROPHET
        elif api == "api:sched":
            extra["sched_config"] = SCHED
        elif api == "api:fastdllm_floor":
            extra["block_accel_top1_floor"] = True
        elif api == "api:dual_cache_emulated":
            extra["dual_cache"] = True
            run_model = _DualCacheEmulator(model)
        elif api == "api:generate_full":
            return self._full_trajectory(model, tok, prompt, shape, mask_id, model_name)
        if self.kind == "reference":
            rt = self.rt
            text, used, budget, blocks = self.decode.generate_live(
                run_model, tok, prompt, steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0, remasking="low_confidence",
                policy_fn=None if args.no_cvee else rt.POLICIES[args.gate],
                policy_params=rt.OURS["params"], extractor=rt.EXTRACTORS[task],
                search_mode=rt.SEARCH_MODE[task], tail_frac=rt.OURS["tail_frac"],
                mask_id=mask_id, logit_shift=(model_name == "dream"),
                block_accel_tau=None if args.no_block_rule else args.block_accel_tau,
                block_struct_mode=args.struct_mode, block_struct_holes=args.holes,
                block_struct_confirm=args.confirm, block_struct_confirm_holes=args.confirm_holes,
                block_struct_settle_run=args.settle_run,
                block_struct_risk_budget=args.risk_budget,
                block_struct_risk_scope=args.risk_scope,
                block_struct_stall_escalate=args.stall_escalate, block_struct_stats=stats,
                block_accel_final_too=args.final_too,
                block_accel_protect_span=args.protect_span,
                block_accel_protect_block=args.protect_block,
                span_recon_probe=args.span_recon_probe,
                exit_momentum_alpha=args.exit_momentum, **extra)
            stats.pop("risk_truncated", None)  # counter of the removed "risk" mode, always 0
        elif api is None:
            cctc, cvee = self.rt.decode_configs(args)
            r = self.decode.generate(run_model, tok, prompt, gen_length=gen_length, steps=steps,
                                     block_length=block_length, mask_id=mask_id,
                                     logit_shift=(model_name == "dream"), cctc=cctc, cvee=cvee,
                                     stats=stats)
            text, used, budget, blocks = r.text, r.steps, r.step_budget, r.block_steps
        else:
            # The keyword interface, exactly as an external caller of `generate_live` uses it.
            rt = self.rt
            text, used, budget, blocks = self.decode.generate_live(
                run_model, tok, prompt, steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=0.0,
                policy_fn=None if args.no_cvee else rt.EXIT_GATES[args.gate],
                policy_params=rt.OURS["params"], extractor=rt.EXTRACTORS[task],
                search_mode=rt.SEARCH_MODE[task], tail_frac=rt.OURS["tail_frac"],
                mask_id=mask_id, logit_shift=(model_name == "dream"),
                block_accel_tau=None if args.no_block_rule else args.block_accel_tau,
                block_struct_mode=args.struct_mode, block_struct_holes=args.holes,
                block_struct_confirm=args.confirm,
                block_struct_stall_escalate=args.stall_escalate, block_struct_stats=stats,
                block_accel_final_too=args.final_too,
                block_accel_protect_span=args.protect_span,
                block_accel_protect_block=args.protect_block,
                span_recon_probe=args.span_recon_probe,
                exit_momentum_alpha=args.exit_momentum, **extra)
        return text, used, budget, list(blocks), stats

    def _full_trajectory(self, model, tok, prompt, shape, mask_id, model_name):
        """`generate_full` compared through digests of everything it returns."""
        gen_length, steps, block_length = shape
        module = importlib.import_module("c4.decode" if self.kind == "reference"
                                         else "c4.trajectory")
        x, x0_hist, idx_hist, conf_hist, gap_hist = module.generate_full(
            model, prompt, steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, mask_id=mask_id, logit_shift=(model_name == "dream"))
        text = tok.decode(x[0, prompt.shape[1]:].tolist(), skip_special_tokens=True)
        stats = {"x0": _digest(x0_hist), "transfers": _digest([_digest(b) for b in idx_hist]),
                 "conf": _digest(conf_hist), "gap": _digest(gap_hist)}
        return text, steps, steps, [], stats


def run(opts):
    import torch
    from transformers import AutoModel, AutoTokenizer

    impl = _Impl(opts.impl, opts.root)
    plan = PLANS[opts.plan]
    model_name = opts.model
    tok = AutoTokenizer.from_pretrained(impl.rt.MODEL_PATH[model_name], trust_remote_code=True)
    model = AutoModel.from_pretrained(impl.rt.MODEL_PATH[model_name], trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    only = set(opts.configs.split(",")) if opts.configs else None
    # One untimed decode first, so CUDA/cuBLAS initialisation does not land in the first
    # example's wall time.
    warm = impl.parse("--no-cvee --no-block-rule", "mmlu")
    warm_doc = impl.loaders()["mmlu"](n=1, seed=0, skip=0)[0]
    warm_prompt = torch.tensor(tok(impl.chat(tok, warm_doc["prompt_text"]))["input_ids"],
                               device="cuda").unsqueeze(0)
    impl.decode_one(model, tok, warm_prompt, "mmlu", warm, (16, 4, 16), None, model_name)
    with open(opts.out, "w") as out:
        for name, (flags, samples) in plan.items():
            if only and name not in only:
                continue
            api = name if name.startswith("api:") else None
            for task, n in samples:
                args = impl.parse(flags, task)
                skip = args.skip if args.skip is not None else impl.rt.DEFAULT_SKIP[task]
                docs = impl.loaders()[task](n=n, seed=args.seed, skip=skip)
                shape = impl.shape(task, args)
                for idx, doc in enumerate(docs):
                    prompt_text = impl.chat(tok, doc["prompt_text"])
                    prompt = torch.tensor(tok(prompt_text)["input_ids"],
                                          device="cuda").unsqueeze(0)
                    rec = {"config": name, "flags": flags, "task": task, "idx": idx,
                           "model": model_name,
                           "prompt_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:16]}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    try:
                        text, used, budget, blocks, stats = impl.decode_one(
                            model, tok, prompt, task, args, shape, api, model_name)
                    except Exception as e:  # recorded, so both sides can be compared on it
                        rec["error"] = f"{type(e).__name__}: {e}"
                    else:
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - t0
                        correct, pred = impl.grade(task, text, doc["gt"])
                        rec.update({"text": text, "steps": used, "budget": budget,
                                    "block_steps": blocks, "stats": stats, "correct": correct,
                                    "pred": pred, "wall_s": wall})
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"[{opts.impl}] {name} {task}#{idx}: "
                          f"{rec.get('error') or (str(rec['steps']) + ' steps')}", flush=True)


COMPARED = ("prompt_sha", "text", "steps", "budget", "block_steps", "correct", "pred", "stats",
            "error")


def compare(opts):
    def load(path):
        return {(r["model"], r["config"], r["task"], r["idx"]): r
                for r in map(json.loads, open(path))}

    ref, cur = load(opts.reference), load(opts.current)
    rows, n_bad = {}, 0
    for key in sorted(set(ref) | set(cur)):
        a, b = ref.get(key), cur.get(key)
        group = rows.setdefault(key[:3], {"n": 0, "same": 0, "diff": [], "errors": 0})
        group["n"] += 1
        if a is None or b is None:
            group["diff"].append(f"#{key[3]} missing on one side")
            continue
        diffs = [f for f in COMPARED if a.get(f) != b.get(f)]
        if "error" in a:
            group["errors"] += 1
        if diffs:
            group["diff"].append(f"#{key[3]}: {','.join(diffs)}")
            n_bad += 1
        else:
            group["same"] += 1
    print(f"{'model':6} {'config':26} {'task':11} {'n':>3} {'identical':>9}  notes")
    for (model, config, task), g in rows.items():
        note = "; ".join(g["diff"]) or ("both raise the same error" if g["errors"] else "")
        print(f"{model:6} {config:26} {task:11} {g['n']:>3} "
              f"{'yes' if not g['diff'] else 'NO':>9}  {note}")
    print(f"\n{sum(g['n'] for g in rows.values())} decodes, {n_bad} with differences")
    return 1 if n_bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="decode the plan with one implementation")
    r.add_argument("--impl", choices=["reference", "current"], required=True)
    r.add_argument("--root", required=True, help="repository root of that implementation")
    r.add_argument("--model", choices=["llada", "dream"], default="llada")
    r.add_argument("--plan", choices=sorted(PLANS), default="smoke")
    r.add_argument("--configs", default=None, help="comma-separated subset of the plan")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="diff two dumps")
    c.add_argument("reference")
    c.add_argument("current")
    opts = ap.parse_args()
    opts.out = os.path.abspath(opts.out) if opts.cmd == "run" else None
    if opts.cmd == "run":
        run(opts)
    else:
        sys.exit(compare(opts))


if __name__ == "__main__":
    main()
