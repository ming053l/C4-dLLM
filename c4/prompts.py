"""Zero-shot CoT prompt templates + small-sample dataset loaders for GSM8K / MMLU / MATH.

GSM8K: exact `doc_to_text` string from lm-eval's own `gsm8k_cot_zeroshot` task (verified against
`Prophet/.venv/.../lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml`), so numbers stay comparable to what's
already in `Prophet/STATUS.md`.

MMLU: exact `doc_to_text` string from `Prophet/custom_tasks/mmlu_lenient/_default_template_yaml`.
Dataset source here is `cais/mmlu` (config "all", canonical HF mirror) rather than Prophet's
`hails/mmlu_no_train` fork -- same underlying test split, different HF repo id; noted here since it's
a deliberate deviation, not an oversight (this project doesn't need lm-eval's fewshot-leak-safe fork
since we run genuinely zero-shot).

MATH: new task, not present anywhere else in this codebase. Source `HuggingFaceH4/MATH-500` (500-q
curated test subset, standard recent-paper choice). Prompt mirrors GSM8K's own zero-shot CoT style
plus a one-line format nudge asking for `\\boxed{}` -- MATH answers are often non-numeric (fractions,
expressions), so some formatting nudge is necessary for the answer to be extractable at all zero-shot;
this is a deliberate, minimal deviation from the "identical style" goal, documented rather than hidden
(see SPEC.md §5 protocol table).
"""
import os
import random
import re

# HF's xet CAS backend 401s in this environment for some dataset repos (openai_humaneval, mbpp) --
# falls back to plain HTTP download instead. Set as a default (not override) so a caller with a
# working xet setup elsewhere is unaffected.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset

from .planning import load_countdown, load_sudoku

GSM8K_TEMPLATE = "Q: {question}\nA: Let's think step by step."
# Strict-format variant (RISKS.md / paper discussion: does Prophet's Lossless claim depend on tightly
# formatted output rather than the gate itself being safe?) -- same free-form CoT instruction, plus a
# one-line answer-format nudge, mirroring simple-evals-style "Answer:" cues and MATH_TEMPLATE's own
# existing \boxed{} nudge below. Deliberately a natural-language instruction, not a hard token/position
# constraint (that was already tried separately, suffix_anchor_collect.py, and found to hurt accuracy
# by itself since it drops the accompanying "keep reasoning short" instruction Prophet also used).
GSM8K_TEMPLATE_STRICT = ("Q: {question}\nA: Let's think step by step, and end your response with "
                         "\"The answer is X.\" where X is the final numeric answer.")
MMLU_TEMPLATE = "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
MATH_TEMPLATE = ("Q: {problem}\nA: Let's think step by step. "
                  "Put your final answer within \\boxed{{}}.")
_LETTERS = "ABCDEFGHIJKLM"  # TruthfulQA MC1 has up to 13 choices (confirmed empirically) -- exactly
                             # matches extract.py's _LETTER_RE range (A-M); do not extend past M
                             # here without also extending that regex, or grading would silently
                             # never match letters beyond it


def _mc_template(question, choices):
    """Generalized MMLU-style MC prompt for any number of choices (2 for WinoGrande/PIQA, 4 for
    ARC-C/HellaSwag, variable for TruthfulQA MC1) -- same doc_to_text convention as MMLU_TEMPLATE
    (question, then lettered choices, then a bare 'Answer:' cue) so the same extract_letter/A-D
    protocol carries over unchanged."""
    lines = [question.strip()]
    for i, c in enumerate(choices):
        lines.append(f"{_LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)

_GSM8K_GT_RE = re.compile(r"####\s*(-?\d+[\d,]*)")


def build_chat_prompt(tokenizer, user_content):
    """Wrap raw doc_to_text content in the Instruct chat template, mirrors Prophet/README.md's
    'Basic Generation with Prophet Early Exit' usage pattern exactly."""
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def load_gsm8k(n, seed=0, skip=0, split="test"):
    ds = load_dataset("gsm8k", "main", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        m = _GSM8K_GT_RE.search(item["answer"])
        gt = m.group(1).replace(",", "") if m else None
        # Few-shot exemplar rationale: GSM8K's own `answer` field already ships a full worked
        # solution ending in "#### <number>" -- reused verbatim (calculator annotations `<<...>>`
        # stripped, and the "####" marker replaced with a plain sentence so the shown exemplar reads
        # like natural continuation text, not a dataset artifact) rather than writing new rationales,
        # since this is the exact correct derivation for that exemplar's own question.
        rationale = _CALC_ANNOTATION_RE.sub("", item["answer"])
        rationale = re.sub(r"####\s*(-?[\d,]+)", r"The answer is \1.", rationale)
        out.append({
            "prompt_text": GSM8K_TEMPLATE.format(question=item["question"]),
            "shot_answer_text": " " + rationale.strip(),
            "gt": gt,
            "task": "gsm8k",
        })
    return out


def load_gsm8k_strict(n, seed=0, skip=0, split="test"):
    """Same GSM8K questions/split/shuffle as load_gsm8k (SAME seed=0 shuffle -> same skip range is
    the same underlying questions), only the prompt template differs (GSM8K_TEMPLATE_STRICT)."""
    ds = load_dataset("gsm8k", "main", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        m = _GSM8K_GT_RE.search(item["answer"])
        gt = m.group(1).replace(",", "") if m else None
        rationale = _CALC_ANNOTATION_RE.sub("", item["answer"])
        rationale = re.sub(r"####\s*(-?[\d,]+)", r"The answer is \1.", rationale)
        out.append({
            "prompt_text": GSM8K_TEMPLATE_STRICT.format(question=item["question"]),
            "shot_answer_text": " " + rationale.strip(),
            "gt": gt,
            "task": "gsm8k_strict",
        })
    return out


def load_mmlu(n, seed=0, skip=0, split="test"):
    ds = load_dataset("cais/mmlu", "all", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        choices = item["choices"]
        gt = ["A", "B", "C", "D"][item["answer"]]
        out.append({
            "prompt_text": MMLU_TEMPLATE.format(
                question=item["question"].strip(), a=choices[0], b=choices[1],
                c=choices[2], d=choices[3]),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "mmlu",
        })
    return out


def load_math(n, seed=0, skip=0, split="test"):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        # MATH-500 ships a full worked `solution` field (already ending in `\boxed{...}`) -- reused
        # verbatim as the few-shot exemplar rationale, same reasoning as GSM8K above.
        out.append({
            "prompt_text": MATH_TEMPLATE.format(problem=item["problem"]),
            "shot_answer_text": " " + item["solution"].strip(),
            "gt": item["answer"],
            "task": "math",
        })
    return out


def load_arc_c(n, seed=0, skip=0, split="test"):
    """ARC-Challenge. Loaded via the legacy `ai2_arc` script name -- `allenai/ai2_arc` (canonical
    repo) fails to load under this environment's datasets==2.17.1 (a library-side dataclass-parsing
    bug unrelated to this project, confirmed by reproducing the identical failure with
    trust_remote_code=True); `ai2_arc` (no org prefix) is the same underlying data via the older
    script path and loads cleanly. Filtered to the 1144/1172 four-choice A-D items (a small number
    of ARC-C items use 3, 5, or numeric 1-4 labels; excluded for template consistency, matches the
    common lm-eval convention of dropping non-4-way items for this task)."""
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    ds = ds.filter(lambda x: tuple(x["choices"]["label"]) == ("A", "B", "C", "D"))
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        gt = item["answerKey"]
        out.append({
            "prompt_text": _mc_template(item["question"], item["choices"]["text"]),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "arc_c",
        })
    return out


def load_hellaswag(n, seed=0, skip=0, split="validation"):
    ds = load_dataset("Rowan/hellaswag", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        ctx = (item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip() if item["ctx_b"] else item["ctx"]
        gt = _LETTERS[int(item["label"])]
        out.append({
            "prompt_text": _mc_template(f"{item['activity_label']}: {ctx}", item["endings"]),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "hellaswag",
        })
    return out


def load_winogrande(n, seed=0, skip=0, split="validation"):
    ds = load_dataset("winogrande", "winogrande_xl", split=split, trust_remote_code=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        question = f"{item['sentence']}\nWhich option correctly fills in the blank (_)?"
        gt = _LETTERS[int(item["answer"]) - 1]
        out.append({
            "prompt_text": _mc_template(question, [item["option1"], item["option2"]]),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "winogrande",
        })
    return out


def load_piqa(n, seed=0, skip=0, split="validation"):
    # ybisk/piqa is a script-based HF dataset, no longer loadable under current `datasets` (script
    # datasets deprecated) -- baber/piqa is a script-free Parquet mirror with an identical schema
    # (goal/sol1/sol2/label), same underlying PIQA validation split.
    ds = load_dataset("baber/piqa", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        gt = _LETTERS[int(item["label"])]
        out.append({
            "prompt_text": _mc_template(item["goal"], [item["sol1"], item["sol2"]]),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "piqa",
        })
    return out


def load_truthfulqa(n, seed=0, skip=0, split="validation"):
    """MC1 (single-true-answer) variant, ungated HF mirror -- substituted for GPQA (gated, requires
    HF auth not available in this environment) to keep the 8-task suite fully reproducible without
    credentials; both are "general reasoning" MC tasks in Prophet's own taxonomy.

    The raw `mc1_targets` field always lists the correct answer first (`labels[0] == 1` on all
    817/817 validation items, confirmed empirically) -- an unshuffled loader makes "A" the answer
    key on every item, trivially exploitable zero-shot and directly demonstrated by every few-shot
    exemplar (root cause of an anomalous 99%/83% few-shot score, Appendix~\ref{sec:fewshot}). Each
    item's own (choices, labels) pair is shuffled with a per-item deterministic seed derived from
    `seed`/`skip`/position, so the correct answer's letter is randomized but reproducible."""
    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for pos, item in enumerate(ds):
        choices = list(item["mc1_targets"]["choices"])
        labels = list(item["mc1_targets"]["labels"])
        order = list(range(len(choices)))
        random.Random((seed or 0) * 100003 + skip + pos).shuffle(order)
        choices = [choices[i] for i in order]
        labels = [labels[i] for i in order]
        gt_idx = labels.index(1)
        gt = _LETTERS[gt_idx]
        out.append({
            "prompt_text": _mc_template(item["question"], choices),
            "shot_answer_text": f" {gt}",
            "gt": gt,
            "task": "truthfulqa",
        })
    return out


def load_svamp(n, seed=0, skip=0, split="test"):
    """SVAMP: short (1-2 step) arithmetic word problems, same free-form CoT + last-number grading
    as GSM8K -- `question_concat` (HF field) already joins the problem's `Body` and `Question`
    halves, `Answer` is already a clean unit-free numeric string in this split (no decimals,
    verified empirically)."""
    ds = load_dataset("ChilleD/SVAMP", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        gt = item["Answer"].strip()
        # SVAMP ships no natural-language rationale, only a symbolic `Equation` -- the few-shot
        # exemplar rationale is synthesized directly from it (e.g. "( 62.0 - 35.0 ) = 27."), a
        # minimal but faithful derivation rather than an invented explanation, documented here as a
        # deliberate choice (mirrors this codebase's existing MATH-template deviation note).
        out.append({
            "prompt_text": GSM8K_TEMPLATE.format(question=item["question_concat"]),
            "shot_answer_text": f" {item['Equation']} = {gt}. The answer is {gt}.",
            "gt": gt,
            "task": "svamp",
        })
    return out


_ASDIV_ANSWER_RE = re.compile(r"^\s*(-?[\d.,]+)")


def load_asdiv(n, seed=0, skip=0, split="validation"):
    """ASDiv: diverse math word problems, same free-form CoT + last-number grading as GSM8K.
    `answer` ships with a trailing unit annotation, e.g. "9 (apples)" -- strip everything after the
    leading number so grading matches `extract_number`'s bare-number convention."""
    ds = load_dataset("EleutherAI/asdiv", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        question = f"{item['body'].strip()} {item['question'].strip()}"
        m = _ASDIV_ANSWER_RE.match(item["answer"])
        gt = m.group(1).replace(",", "") if m else item["answer"]
        # Same synthesized-rationale approach as SVAMP above, using ASDiv's own `formula` field
        # (e.g. "7+2=9") -- a minimal, faithful derivation rather than an invented explanation.
        out.append({
            "prompt_text": GSM8K_TEMPLATE.format(question=question),
            "shot_answer_text": f" {item['formula']}. The answer is {gt}.",
            "gt": gt,
            "task": "asdiv",
        })
    return out


HUMANEVAL_TEMPLATE = ("Complete the following Python function. Respond with ONLY the full function "
                      "implementation (signature and body), no explanation, no markdown code "
                      "fences.\n\n{prompt}")
MBPP_TEMPLATE = ("Write a Python function for the following task. Respond with ONLY the full "
                 "function implementation, no explanation, no markdown code fences. Your function "
                 "must satisfy this example usage:\n{example_test}\n\nTask: {text}")


def load_humaneval(n, seed=0, skip=0, split="test"):
    """HumanEval (Chen et al., 2021): 164 hand-written Python programming problems, function
    signature + docstring as prompt, execution-based grading against held-out unit tests
    (`common/codegen_grade.py`). CVEE's answer span here is the completed function definition itself,
    delimited by Python's own indentation structure (`common/codegen_grade.py::code_candidate`)
    rather than by a numeric or single-letter pattern; the whole 164-problem set is evaluated, since
    no code task contributes to any calibration pool."""
    ds = load_dataset("openai/openai_humaneval", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        out.append({
            "prompt_text": HUMANEVAL_TEMPLATE.format(prompt=item["prompt"]),
            "gt": {"test": item["test"], "entry_point": item["entry_point"]},
            "task": "humaneval",
        })
    return out


def load_mbpp(n, seed=0, skip=0, split="test"):
    """MBPP (Austin et al., 2021), sanitized config: crowd-sourced Python programming problems, NL
    description + one example assertion as prompt (standard MBPP prompting practice -- gives the
    model the expected function name/signature without handing it the full test suite), graded by
    executing the generated function against `test_list` (`common/codegen_grade.py`). Same
    function-definition answer span as HumanEval above."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        example_test = item["test_list"][0] if item["test_list"] else ""
        out.append({
            "prompt_text": MBPP_TEMPLATE.format(text=item["prompt"], example_test=example_test),
            "gt": {"test_list": item["test_list"], "test_setup_code": item.get("test_setup_code", "")},
            "task": "mbpp",
        })
    return out


def load_gsmhard(n, seed=0, skip=0, split="train"):
    """GSM-Hard: GSM8K questions with the small numbers substituted for large/awkward ones (same
    reasoning structure, much larger arithmetic -- a harder variant of the exact same task family,
    not a different skill). `input` is already the full self-contained question; `target` is a
    float64, normalized to a bare int string when it's a whole number (the common case, since the
    model's CoT answer is not expected to spell out a trailing ".0"), else rounded to 2 decimal
    places -- roughly a quarter of targets in this dataset are non-integer (the substituted numbers
    don't divide evenly) and ship with raw floating-point noise (e.g. `2047414.7999999996`); rounding
    is a necessary, minimal correction, not an accuracy nudge -- an unrounded 16-significant-digit
    target could never match any free-form generated answer regardless of whether the model's
    reasoning was correct."""
    ds = load_dataset("reasoning-machines/gsm-hard", split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(skip, min(skip + n, len(ds))))
    out = []
    for item in ds:
        t = item["target"]
        gt = str(int(t)) if float(t).is_integer() else str(round(float(t), 2))
        # GSM-Hard ships only a Python `code` field, no natural-language rationale -- unlike SVAMP/
        # ASDiv there is no simple symbolic formula to quote either (the substituted numbers make the
        # arithmetic chain long), so the few-shot exemplar rationale here is deliberately minimal
        # (states the answer directly, no derivation shown). This is weaker as a reasoning
        # demonstration than the other tasks' exemplars, a documented limitation of this task's
        # source data rather than an oversight.
        out.append({
            "prompt_text": GSM8K_TEMPLATE.format(question=item["input"]),
            "shot_answer_text": f" The answer is {gt}.",
            "gt": gt,
            "task": "gsmhard",
        })
    return out


LOADERS = {
    "gsm8k": load_gsm8k, "gsm8k_strict": load_gsm8k_strict, "mmlu": load_mmlu, "math": load_math,
    "arc_c": load_arc_c, "hellaswag": load_hellaswag, "winogrande": load_winogrande,
    "piqa": load_piqa, "truthfulqa": load_truthfulqa,
    "svamp": load_svamp, "asdiv": load_asdiv, "gsmhard": load_gsmhard,
    "countdown": load_countdown, "sudoku": load_sudoku,
    "humaneval": load_humaneval, "mbpp": load_mbpp,
}
GEN_SHAPE = {  # gen_length, steps, block_length -- frozen protocol, SPEC.md §5
    "gsm8k": (256, 256, 32),
    "gsm8k_strict": (256, 256, 32),
    "mmlu": (64, 64, 16),
    "math": (256, 256, 32),
    # Same shape as MMLU: all five are short, single-letter-answer MC tasks, matching our own
    # established MMLU protocol convention rather than Prophet's own L=128 choice (kept internally
    # consistent within this paper's own task-shape convention instead).
    "arc_c": (64, 64, 16),
    "hellaswag": (64, 64, 16),
    "winogrande": (64, 64, 16),
    "piqa": (64, 64, 16),
    "truthfulqa": (64, 64, 16),
    # Same shape as GSM8K/MATH: all three are multi-step CoT arithmetic/word-problem tasks in the
    # same "long-reasoning" family, not short-answer MC.
    "svamp": (256, 256, 32),
    "asdiv": (256, 256, 32),
    "gsmhard": (256, 256, 32),
    # Planning tasks (Prophet Table 6): single-block (block_length == gen_length), matching their
    # own L=T=B choice exactly.
    "countdown": (32, 32, 32),
    "sudoku": (24, 24, 24),
    # Code generation (Table~\ref{tab:main}'s code family): longer budget than GSM8K/MATH since a full
    # function implementation commonly runs longer than a CoT arithmetic derivation; same
    # block_length=32 convention as the rest of the long-form family.
    "humaneval": (384, 384, 32),
    "mbpp": (384, 384, 32),
}


def build_fewshot_prefix(task, k, seed, exemplar_skip):
    """K-shot prefix: K fully-worked (question+answer) exemplars from a range of the SAME seed=0
    shuffle that is guaranteed disjoint from whatever range the eval run itself uses
    (`exemplar_skip` is chosen by the caller for exactly this reason -- see live_eval.py's
    `--shots` handling), joined by blank lines, ending in one trailing blank line so the real
    (unanswered) test question can simply be appended after it. Reuses each task's own loader
    (and therefore its own `shot_answer_text` derivation) rather than a separate exemplar path, so
    a fix to a loader's prompt format can never silently drift out of sync with its few-shot
    version."""
    shots = LOADERS[task](k, seed=seed, skip=exemplar_skip)
    return "\n\n".join(d["prompt_text"] + d["shot_answer_text"] for d in shots) + "\n\n"
