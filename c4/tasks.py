"""The 12 evaluation tasks: prompts, dataset loaders and the per-task protocol.

Every task is zero-shot and wrapped in the backbone's chat template. Each loader returns a list of
{"prompt_text", "gt"} dicts from a seeded shuffle of the split, starting at `skip`; the per-family
sample size and offset in DEFAULT_N / DEFAULT_SKIP are the ones the paper's tables use.

Families (Appendix "hyperparams"):
- choice:  MMLU, ARC-C, HellaSwag, WinoGrande, PIQA. Answer = first letter ("first5").
- numeric: GSM8K, SVAMP, ASDiv, GSM-Hard. Answer = last number in the trailing 30% ("last").
- math:    MATH-500. Answer = last \\boxed{} ("last").
- code:    HumanEval, MBPP. Answer = the function definition ("code"); graded by execution.
"""
import re

from datasets import load_dataset

from .answers import code_candidate, extract_boxed, extract_letter, extract_number

CHOICE_TASKS = ("mmlu", "arc_c", "hellaswag", "winogrande", "piqa")
NUMERIC_TASKS = ("gsm8k", "svamp", "asdiv", "gsmhard")
CODE_TASKS = ("humaneval", "mbpp")
TASKS = CHOICE_TASKS + NUMERIC_TASKS + ("math",) + CODE_TASKS

# (gen_length L, steps T, block_length B). One step per token at baseline.
GEN_SHAPE = {
    **{t: (64, 64, 16) for t in CHOICE_TASKS},
    **{t: (256, 256, 32) for t in NUMERIC_TASKS},
    "math": (256, 256, 32),
    **{t: (384, 384, 32) for t in CODE_TASKS},
}
# CVEE candidate parser and search mode per task.
EXTRACTORS = {**{t: extract_letter for t in CHOICE_TASKS},
              **{t: extract_number for t in NUMERIC_TASKS},
              "math": extract_boxed,
              **{t: code_candidate for t in CODE_TASKS}}
SEARCH_MODE = {**{t: "first5" for t in CHOICE_TASKS},
               **{t: "last" for t in NUMERIC_TASKS},
               "math": "last",
               **{t: "code" for t in CODE_TASKS}}
# Evaluation sample per family: 200 short-answer and 100 long-reasoning examples after a held-out
# offset of 40 (the calibration pool comes from before it), and the full set for code.
DEFAULT_N = {**{t: 200 for t in CHOICE_TASKS}, **{t: 100 for t in NUMERIC_TASKS}, "math": 100,
             "humaneval": 164, "mbpp": 257}
DEFAULT_SKIP = {**{t: 40 for t in CHOICE_TASKS + NUMERIC_TASKS}, "math": 40,
                "humaneval": 0, "mbpp": 0}

COT_TEMPLATE = "Q: {question}\nA: Let's think step by step."  # lm-eval gsm8k_cot_zeroshot
MMLU_TEMPLATE = "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
MATH_TEMPLATE = ("Q: {problem}\nA: Let's think step by step. "
                 "Put your final answer within \\boxed{{}}.")
HUMANEVAL_TEMPLATE = ("Complete the following Python function. Respond with ONLY the full function "
                      "implementation (signature and body), no explanation, no markdown code "
                      "fences.\n\n{prompt}")
MBPP_TEMPLATE = ("Write a Python function for the following task. Respond with ONLY the full "
                 "function implementation, no explanation, no markdown code fences. Your function "
                 "must satisfy this example usage:\n{example_test}\n\nTask: {text}")
LETTERS = "ABCDEFGHIJKLM"

_GSM8K_GT_RE = re.compile(r"####\s*(-?\d+[\d,]*)")
_ASDIV_ANSWER_RE = re.compile(r"^\s*(-?[\d.,]+)")


def build_chat_prompt(tokenizer, user_content):
    """Wrap the task prompt in the instruct model's chat template."""
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def choice_prompt(question, choices):
    """MMLU-style prompt for any number of choices: question, lettered options, "Answer:"."""
    lines = [question.strip()]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def _sample(path, name=None, *, split, n, seed, skip, keep=None, **kwargs):
    """`n` items starting at `skip` from the split shuffled with `seed` (no shuffle if None)."""
    ds = load_dataset(path, name, split=split, **kwargs)
    if keep is not None:
        ds = ds.filter(keep)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    return ds.select(range(skip, min(skip + n, len(ds))))


def load_gsm8k(n, seed=0, skip=0, split="test"):
    out = []
    for item in _sample("gsm8k", "main", split=split, n=n, seed=seed, skip=skip):
        m = _GSM8K_GT_RE.search(item["answer"])
        out.append({"prompt_text": COT_TEMPLATE.format(question=item["question"]),
                    "gt": m.group(1).replace(",", "") if m else None})
    return out


def load_mmlu(n, seed=0, skip=0, split="test"):
    out = []
    for item in _sample("cais/mmlu", "all", split=split, n=n, seed=seed, skip=skip):
        c = item["choices"]
        out.append({"prompt_text": MMLU_TEMPLATE.format(question=item["question"].strip(),
                                                        a=c[0], b=c[1], c=c[2], d=c[3]),
                    "gt": "ABCD"[item["answer"]]})
    return out


def load_math(n, seed=0, skip=0, split="test"):
    return [{"prompt_text": MATH_TEMPLATE.format(problem=item["problem"]), "gt": item["answer"]}
            for item in _sample("HuggingFaceH4/MATH-500", split=split, n=n, seed=seed, skip=skip)]


def load_arc_c(n, seed=0, skip=0, split="test"):
    """ARC-Challenge, restricted to the four-choice A-D items (1144 of 1172). Loaded under the
    legacy `ai2_arc` name, which datasets==2.17.1 reads where `allenai/ai2_arc` fails."""
    items = _sample("ai2_arc", "ARC-Challenge", split=split, n=n, seed=seed, skip=skip,
                    keep=lambda x: tuple(x["choices"]["label"]) == ("A", "B", "C", "D"))
    return [{"prompt_text": choice_prompt(item["question"], item["choices"]["text"]),
             "gt": item["answerKey"]} for item in items]


def load_hellaswag(n, seed=0, skip=0, split="validation"):
    out = []
    for item in _sample("Rowan/hellaswag", split=split, n=n, seed=seed, skip=skip):
        ctx = ((item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip() if item["ctx_b"]
               else item["ctx"])
        out.append({"prompt_text": choice_prompt(f"{item['activity_label']}: {ctx}",
                                                 item["endings"]),
                    "gt": LETTERS[int(item["label"])]})
    return out


def load_winogrande(n, seed=0, skip=0, split="validation"):
    out = []
    for item in _sample("winogrande", "winogrande_xl", split=split, n=n, seed=seed, skip=skip,
                        trust_remote_code=True):
        question = f"{item['sentence']}\nWhich option correctly fills in the blank (_)?"
        out.append({"prompt_text": choice_prompt(question, [item["option1"], item["option2"]]),
                    "gt": LETTERS[int(item["answer"]) - 1]})
    return out


def load_piqa(n, seed=0, skip=0, split="validation"):
    """PIQA from `baber/piqa`, a script-free Parquet mirror of the original validation split."""
    return [{"prompt_text": choice_prompt(item["goal"], [item["sol1"], item["sol2"]]),
             "gt": LETTERS[int(item["label"])]}
            for item in _sample("baber/piqa", split=split, n=n, seed=seed, skip=skip)]


def load_svamp(n, seed=0, skip=0, split="test"):
    return [{"prompt_text": COT_TEMPLATE.format(question=item["question_concat"]),
             "gt": item["Answer"].strip()}
            for item in _sample("ChilleD/SVAMP", split=split, n=n, seed=seed, skip=skip)]


def load_asdiv(n, seed=0, skip=0, split="validation"):
    """ASDiv answers carry a unit ("9 (apples)"); only the leading number is the ground truth."""
    out = []
    for item in _sample("EleutherAI/asdiv", split=split, n=n, seed=seed, skip=skip):
        m = _ASDIV_ANSWER_RE.match(item["answer"])
        out.append({"prompt_text": COT_TEMPLATE.format(
                        question=f"{item['body'].strip()} {item['question'].strip()}"),
                    "gt": m.group(1).replace(",", "") if m else item["answer"]})
    return out


def load_gsmhard(n, seed=0, skip=0, split="train"):
    """GSM-Hard targets are floats: whole numbers become bare integers, the rest are rounded to two
    decimals (about a quarter carry floating-point noise such as 2047414.7999999996)."""
    out = []
    for item in _sample("reasoning-machines/gsm-hard", split=split, n=n, seed=seed, skip=skip):
        t = float(item["target"])
        out.append({"prompt_text": COT_TEMPLATE.format(question=item["input"]),
                    "gt": str(int(t)) if t.is_integer() else str(round(t, 2))})
    return out


def load_humaneval(n, seed=0, skip=0, split="test"):
    return [{"prompt_text": HUMANEVAL_TEMPLATE.format(prompt=item["prompt"]),
             "gt": {"test": item["test"], "entry_point": item["entry_point"]}}
            for item in _sample("openai/openai_humaneval", split=split, n=n, seed=seed, skip=skip)]


def load_mbpp(n, seed=0, skip=0, split="test"):
    """MBPP (sanitized); the prompt shows the first test assertion so the model learns the
    expected function name, and grading runs all of `test_list`."""
    out = []
    for item in _sample("google-research-datasets/mbpp", "sanitized", split=split, n=n, seed=seed,
                        skip=skip):
        tests = item["test_list"]
        out.append({"prompt_text": MBPP_TEMPLATE.format(text=item["prompt"],
                                                        example_test=tests[0] if tests else ""),
                    "gt": {"test_list": tests,
                           "test_setup_code": item.get("test_setup_code", "")}})
    return out


LOADERS = {
    "mmlu": load_mmlu, "arc_c": load_arc_c, "hellaswag": load_hellaswag,
    "winogrande": load_winogrande, "piqa": load_piqa,
    "gsm8k": load_gsm8k, "svamp": load_svamp, "asdiv": load_asdiv, "gsmhard": load_gsmhard,
    "math": load_math, "humaneval": load_humaneval, "mbpp": load_mbpp,
}
