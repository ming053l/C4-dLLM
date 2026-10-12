"""Answer extraction, one parser per output format.

The same parsers serve twice: CVEE calls them every step on the provisional text to obtain the
candidate a_t, and grading calls them once on the final text. They are fixed by each task's
output format and never tuned.
"""
import re

# lm-eval's GSM8K "flexible-extract" pattern; the last match is the answer.
_NUM_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
# First standalone capital letter. The range A-M is kept from the original release (it once
# served a 13-choice task); narrowing it would change which letter is extracted.
_LETTER_RE = re.compile(r"\b([A-M])\b")
# Last \boxed{...}, one level of nested braces (covers e.g. \boxed{\frac{1}{2}}).
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_BARE_FRAC_RE = re.compile(r"\\frac(\d)(\d)(?!\d)")
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_number(text):
    """Last number-like substring, with $ and , removed and a trailing period stripped."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    last = next(g for g in matches[-1] if g)
    return last.strip().rstrip(".").replace("$", "").replace(",", "")


def extract_letter(text):
    """First standalone choice letter."""
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None


def normalize_math(s):
    """Lenient MATH normalization: drop spaces, $, \\left/\\right, thin spaces and a trailing
    period, and brace MATH-500's bare-digit shorthand ("\\frac65" -> "\\frac{6}{5}"). Not a CAS
    equivalence check."""
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    s = s.rstrip(".")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return _BARE_FRAC_RE.sub(r"\\frac{\1}{\2}", s)


def extract_boxed(text):
    """Last \\boxed{...} content, falling back to the last number, normalized."""
    matches = _BOXED_RE.findall(text)
    if matches:
        return normalize_math(matches[-1])
    num = extract_number(text)
    return normalize_math(num) if num is not None else None


def extract_code(text):
    """The program to execute: the first fenced block if the model added fences despite being
    asked not to, otherwise the raw text."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def code_candidate(text):
    """CVEE's candidate for code tasks: everything from the start of the program through the end
    of the first function definition, or None before a `def` appears.

    The function ends at the first non-blank column-0 line after its body (Python's own
    end-of-block rule), so trailing usage examples and the provisional text of undecoded
    positions are excluded and a settled function reads as settled. Imports and comments before
    the `def` are kept as part of the program."""
    lines = extract_code(text).split("\n")
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("def ") or line.startswith("async def ")), None)
    if start is None:
        return None
    end = start
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue  # a blank line inside a body is not a dedent
        if line[0] in " \t":
            end = i
            continue
        break
    candidate = "\n".join(lines[:end + 1]).rstrip()
    return candidate or None
