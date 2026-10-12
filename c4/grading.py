"""Grading: answer match for the 10 short-answer and reasoning tasks, execution for code.

Code grading runs model-written programs. Each program runs in a fresh Python subprocess, in its
own temporary working directory, with no stdin, a wall-clock timeout, and its own process group so
that a timeout also kills anything it spawned. This isolates the evaluation process from a hanging
or crashing candidate; it is not a security sandbox, so run untrusted generations in a container.
"""
import os
import signal
import subprocess
import sys
import tempfile

from .answers import extract_boxed, extract_code, extract_letter, extract_number, normalize_math
from .tasks import CHOICE_TASKS, CODE_TASKS, NUMERIC_TASKS

TIMEOUT_SECONDS = 10


def run_program(program, timeout=TIMEOUT_SECONDS):
    """True iff `program` exits with status 0 within `timeout` seconds."""
    with tempfile.TemporaryDirectory(prefix="c4_grade_") as workdir:
        path = os.path.join(workdir, "program.py")
        with open(path, "w") as f:
            f.write(program)
        proc = subprocess.Popen([sys.executable, path], cwd=workdir, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return False


def grade_humaneval(pred_text, gt):
    """gt = {"test": source defining check(candidate), "entry_point": function name}."""
    program = f"{extract_code(pred_text)}\n\n{gt['test']}\n\ncheck({gt['entry_point']})\n"
    return run_program(program)


def grade_mbpp(pred_text, gt):
    """gt = {"test_list": [assert statements], "test_setup_code": optional setup source}."""
    setup = gt.get("test_setup_code") or ""
    tests = "\n".join(gt["test_list"])
    return run_program(f"{extract_code(pred_text)}\n\n{setup}\n\n{tests}\n")


def extract_answer(task, text):
    """The prediction logged per example: the extracted answer, or the program for code."""
    if task in NUMERIC_TASKS:
        return extract_number(text)
    if task in CHOICE_TASKS:
        return extract_letter(text)
    if task == "math":
        return extract_boxed(text)
    if task in CODE_TASKS:
        return extract_code(text)
    raise ValueError(f"unknown task {task!r}")


def grade(task, text, gt):
    """True iff the generated `text` answers the example correctly."""
    if task == "humaneval":
        return grade_humaneval(text, gt)
    if task == "mbpp":
        return grade_mbpp(text, gt)
    pred = extract_answer(task, text)
    if pred is None:
        return False
    if task in NUMERIC_TASKS:
        return pred == str(gt).strip().replace(",", "")
    if task == "math":
        return pred == normalize_math(str(gt))
    return pred == gt
