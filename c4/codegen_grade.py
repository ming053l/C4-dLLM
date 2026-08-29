"""Execution-based grading for HumanEval/MBPP, plus CVEE's own candidate parser for the same two
tasks. Grading mirrors the standard community pattern (OpenAI's own human-eval `execution.py`,
bigcode-evaluation-harness): run the generated code plus its test suite in a separate subprocess
with a wall-clock timeout, so a hanging or crashing candidate cannot affect the parent evaluation
process.

`code_candidate` is the code family's entry in the same per-task-format extractor table every other
task already has (`live_eval.py::_EXTRACTORS`): it returns the completed function definition, i.e.
the answer span CVEE tracks the identity and confidence of, with the trailing region (still-churning
provisional guesses at not-yet-decoded positions, any test/example chatter the model appends after
the function) excluded. The span is delimited by Python's own indentation structure -- the function
ends at the first non-blank column-0 line after its body -- so it is fixed by the task's output
format exactly like the boxed-expression and single-letter parsers are, never tuned.
"""
import re
import subprocess
import sys
import tempfile

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)

TIMEOUT_SECONDS = 10


def extract_code(text):
    """Model was asked for a bare function implementation, no fences -- but strip markdown fences
    if the model added them anyway (observed occasionally despite the instruction), taking the
    first fenced block if present, else the raw text as-is."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def code_candidate(text):
    """CVEE's extracted candidate $a_t$ for humaneval/mbpp: the completed function definition, as a
    normalized string, or None when no function has been started yet (which leaves run/changes
    paused and fails the confidence gate automatically, exactly like a missing number on GSM8K).

    Everything from the buffer's start through the end of the function's body is kept -- imports
    and comments the model emits before the `def` are part of the program and must not be dropped.
    The body ends at the first non-blank line at column 0 after the definition, Python's own
    end-of-block rule; whatever follows (usage examples, asserts, prose, or the not-yet-decoded
    tail's provisional garbage) is not part of the answer and is excluded so that a settled
    function reads as settled."""
    code = extract_code(text)
    lines = code.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if l.startswith("def ") or l.startswith("async def ")), None)
    if start is None:
        return None
    end = start
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue  # blank lines inside a body are not a dedent
        if line[0] in " \t":
            end = i
            continue
        break
    cand = "\n".join(lines[:end + 1]).rstrip()
    return cand or None


def _run_in_subprocess(program):
    """Executes `program` in a fresh Python subprocess, wall-clock-limited to TIMEOUT_SECONDS.
    Returns True iff the subprocess exits 0 (no assertion/exception raised)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run([sys.executable, path], capture_output=True,
                                 timeout=TIMEOUT_SECONDS)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        os.unlink(path)


def grade_humaneval(pred_text, gt):
    """gt = {"test": <check(candidate) function source>, "entry_point": <function name>}."""
    code = extract_code(pred_text)
    program = f"{code}\n\n{gt['test']}\n\ncheck({gt['entry_point']})\n"
    return _run_in_subprocess(program)


def grade_mbpp(pred_text, gt):
    """gt = {"test_list": [assert stmt, ...], "test_setup_code": <optional setup source>}."""
    code = extract_code(pred_text)
    setup = gt.get("test_setup_code") or ""
    tests = "\n".join(gt["test_list"])
    program = f"{code}\n\n{setup}\n\n{tests}\n"
    return _run_in_subprocess(program)
