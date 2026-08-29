"""Answer extractors, one per task family. Ported+extended from
`Prophet/ours_components.py::taec_extract_number/letter` (same regex philosophy: grade what the model
actually said, mirrors lm-eval's own flexible-extract/lenient filters already established in this
codebase) plus a new MATH extractor. Copied rather than imported so this repo stays self-contained
and reproducible independent of Prophet's own future state (see SPEC.md).
"""
import re

# Mirrors lm-eval's flexible-extract regex for GSM8K: "(-?[$0-9.,]{2,})|(-?[0-9]+)", take the LAST match.
_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
# MMLU/ARC-C/HellaSwag/WinoGrande/PIQA/TruthfulQA: prompt ends "...Answer:", first standalone
# letter anywhere in the response is graded. A-D covers every task except TruthfulQA MC1, whose
# choice count varies up to 13 (A-M) -- extended here rather than kept at A-D so that task doesn't
# silently under-extract on its own long tail of many-choice questions.
_LETTER_RE = re.compile(r"\b([A-M])\b")
# MATH: prefer the LAST \boxed{...} (handles nested braces one level deep, which covers the vast
# majority of MATH-500 answers e.g. \boxed{\frac{1}{2}}).
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_number(text):
    """Last number-looking substring in `text`, normalized (strip $ and , and trailing .)."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    last = next(g for g in matches[-1] if g)
    return last.strip().rstrip(".").replace("$", "").replace(",", "")


def extract_letter(text):
    """First standalone A/B/C/D in `text`."""
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None


def extract_boxed(text):
    """Last \\boxed{...} content; fallback to the last number if no \\boxed{} is present (mirrors
    the GSM8K/MMLU lenient-grading philosophy: don't zero out an otherwise-correct answer over
    formatting)."""
    matches = _BOXED_RE.findall(text)
    if matches:
        return _normalize_math(matches[-1])
    num = extract_number(text)
    return _normalize_math(num) if num is not None else None


_BARE_FRAC_RE = re.compile(r"\\frac(\d)(\d)(?!\d)")


def _normalize_math(s):
    """Loose normalization for MATH-style answers: strip whitespace/$, drop \\left \\right, collapse
    spaces. Not a full CAS-equivalence checker (out of scope) -- matches the "lenient" grading
    philosophy already used for GSM8K/MMLU in this codebase, not a claim of mathematical rigor.

    Also inserts braces around MATH-500's own occasional bare-digit \\frac shorthand (e.g. the
    ground-truth string "\\frac65" meaning 6/5, confirmed present verbatim in 3/500 MATH-500
    answers) so it compares equal to a model's equivalently-valued but properly-braced
    "\\frac{6}{5}" -- both denote the identical fraction, so treating them as a mismatch was
    exactly the kind of formatting-not-correctness gap this function already exists to avoid."""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    s = s.rstrip(".")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = _BARE_FRAC_RE.sub(r"\\frac{\1}{\2}", s)
    return s


EXTRACTORS = {
    "number": extract_number,
    "letter": extract_letter,
    "boxed": extract_boxed,
}


_MC_TASKS = {"mmlu", "arc_c", "hellaswag", "winogrande", "piqa", "truthfulqa"}
# Free-form CoT, last-number grading -- same family as GSM8K (short arithmetic word problems or a
# harder-number variant of it), not a different extraction convention.
_NUMERIC_TASKS = {"gsm8k", "gsm8k_strict", "svamp", "asdiv", "gsmhard"}


def grade(task, pred_text, gt):
    """task in _NUMERIC_TASKS union {'math'} union _MC_TASKS union {'countdown','sudoku'} ->
    bool correct."""
    if task in _NUMERIC_TASKS:
        pred = extract_number(pred_text)
        return pred is not None and pred == str(gt).strip().replace(",", "")
    if task in _MC_TASKS:
        pred = extract_letter(pred_text)
        return pred is not None and pred == gt
    if task == "math":
        pred = extract_boxed(pred_text)
        return pred is not None and pred == _normalize_math(str(gt))
    if task == "countdown":
        from .planning import grade_countdown
        return grade_countdown(pred_text, gt)
    if task == "sudoku":
        from .planning import grade_sudoku
        return grade_sudoku(pred_text, gt)
    if task == "humaneval":
        from .codegen_grade import grade_humaneval
        return grade_humaneval(pred_text, gt)
    if task == "mbpp":
        from .codegen_grade import grade_mbpp
        return grade_mbpp(pred_text, gt)
    raise ValueError(task)


def extract_answer(task, pred_text):
    """Same extractor dispatch as grade(), but returns the extracted string (or None) instead of a
    bool -- used to log a per-example prediction for paired significance tests / output-identical-
    rate comparisons across variants, without duplicating grade()'s task-routing logic."""
    if task in _NUMERIC_TASKS:
        return extract_number(pred_text)
    if task in _MC_TASKS:
        return extract_letter(pred_text)
    if task == "math":
        return extract_boxed(pred_text)
    if task in ("countdown", "sudoku"):
        return None  # planning tasks grade the full trace, no single extracted answer string
    if task in ("humaneval", "mbpp"):
        from .codegen_grade import extract_code
        return extract_code(pred_text)  # full function body, not a single-token answer
    raise ValueError(task)
