"""Planning tasks (Countdown, Sudoku), matching Prophet's own Table 1 "Planning Tasks" bucket and
Table 6 configuration (Sudoku L=T=B=24, Countdown L=T=B=32 -- both single-block, i.e. no
block-wise structure at all, unlike every other task in this codebase). Both evaluated 8-shot per
Prophet's own footnote ("Sudoku and Countdown are evaluated using 8-shot setting; all other
benchmarks use zero-shot evaluation").

Countdown: `Jiayi-Pan/Countdown-Tasks-3to4` (nums, target) -- standard "use each number exactly
once with +-*/ to reach target" puzzle, same dataset used throughout the RL-reasoning literature
(e.g. TinyZero). We brute-force a valid solution for each few-shot exemplar (the dataset itself
only ships nums+target, not a worked solution).

Sudoku: no ready small-scale (4x4) HF dataset was available locally, so puzzles are generated here
directly -- start from a complete valid 4x4 Latin-square-plus-subgrid solution, permute rows/cols/
digits for variety, then remove a fixed number of cells to form the puzzle. Grading logic (row/
column/2x2-subgrid validity + must match given clues) ported from
`Streaming-dLLM/Dream/sudoku_metric.py::is_valid_sudoku`.
"""
import itertools
import random
import re

from datasets import load_dataset

COUNTDOWN_TEMPLATE = ("Using the numbers {nums}, create an equation that equals {target}. "
                      "You can use the basic arithmetic operations (+, -, *, /) and each number "
                      "can only be used once. Show your work, then write your final equation "
                      "on the last line after \"Answer:\".")
SUDOKU_TEMPLATE = ("Solve this 4x4 Sudoku puzzle. Each row, column, and 2x2 box must contain the "
                   "digits 1-4 exactly once. 0 marks an empty cell.\n{grid}\nWrite the completed "
                   "grid (4 lines of 4 digits, no spaces) on the lines after \"Answer:\".")

_EQN_RE = re.compile(r"Answer:\s*([-+*/()\d\s.]+?)\s*=\s*-?\d+(?:\.\d+)?", re.IGNORECASE)
_EQN_RE_NOEQ = re.compile(r"Answer:\s*([-+*/()\d\s.]+)")


def _solve_countdown(nums, target):
    """Brute-force: try every permutation of nums and every combination of +-*/ between them,
    evaluated with STANDARD operator precedence (via `eval`, matching `grade_countdown`'s own
    evaluation -- these two must agree, or a self-consistent shot example can silently fail its
    own grading check)."""
    ops = "+-*/"
    for perm in set(itertools.permutations(nums)):
        for op_combo in itertools.product(ops, repeat=len(nums) - 1):
            expr = str(perm[0])
            for op, n in zip(op_combo, perm[1:]):
                expr += f" {op} {n}"
            try:
                val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 -- digit/operator-only
            except ZeroDivisionError:
                continue
            if abs(val - target) < 1e-6:
                return expr, val
    return None, None


def load_countdown(n, seed=0, skip=0):
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=seed if seed is not None else 0)
    out = []
    i = 0
    idx = skip
    while len(out) < n and idx < len(ds):
        item = ds[idx]
        idx += 1
        nums, target = item["nums"], item["target"]
        expr, val = _solve_countdown(nums, target)
        if expr is None:
            continue  # skip unsolvable-by-our-brute-forcer items (rare, keeps loader simple)
        out.append({
            "prompt_text": COUNTDOWN_TEMPLATE.format(nums=nums, target=target),
            "shot_answer_text": f" Let me work through this.\n{expr} = {target}\nAnswer: {expr} = {target}",
            "gt": {"nums": list(nums), "target": target},
            "task": "countdown",
        })
    return out


def grade_countdown(pred_text, gt):
    nums, target = gt["nums"], gt["target"]
    m = _EQN_RE.search(pred_text) or _EQN_RE_NOEQ.search(pred_text)
    if not m:
        return False
    expr = m.group(1).strip().rstrip("=").strip()
    used = sorted(int(x) for x in re.findall(r"\d+", expr))
    if used != sorted(nums):
        return False
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 -- expr is digit/operator-only, checked above
    except Exception:
        return False
    return isinstance(val, (int, float)) and abs(val - target) < 1e-6


_BASE_SOLUTIONS_4X4 = [
    [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]],
]


def _random_solution(rng):
    base = [row[:] for row in _BASE_SOLUTIONS_4X4[0]]
    digit_perm = rng.sample([1, 2, 3, 4], 4)
    remap = {i + 1: digit_perm[i] for i in range(4)}
    grid = [[remap[v] for v in row] for row in base]
    row_band_order = rng.sample([0, 1], 2)
    rows_within = [rng.sample([0, 1], 2), rng.sample([0, 1], 2)]
    new_rows = []
    for band in row_band_order:
        for r in rows_within[band]:
            new_rows.append(grid[band * 2 + r])
    grid = new_rows
    col_band_order = rng.sample([0, 1], 2)
    cols_within = [rng.sample([0, 1], 2), rng.sample([0, 1], 2)]
    new_cols = []
    for band in col_band_order:
        for c in cols_within[band]:
            new_cols.append(band * 2 + c)
    grid = [[row[c] for c in new_cols] for row in grid]
    return grid


def load_sudoku(n, seed=0, skip=0, n_blank=5):
    rng = random.Random(seed if seed is not None else 0)
    # burn `skip` draws so different skip windows give disjoint puzzles from the same seed
    for _ in range(skip):
        _random_solution(rng)
        rng.sample(range(16), n_blank)
    out = []
    for _ in range(n):
        solution = _random_solution(rng)
        blanks = set(rng.sample(range(16), n_blank))
        puzzle = [[0 if (r * 4 + c) in blanks else solution[r][c] for c in range(4)] for r in range(4)]
        grid_str = "\n".join("".join(str(v) for v in row) for row in puzzle)
        sol_str = "\n".join("".join(str(v) for v in row) for row in solution)
        out.append({
            "prompt_text": SUDOKU_TEMPLATE.format(grid=grid_str),
            "shot_answer_text": f" Answer:\n{sol_str}",
            "gt": {"puzzle": puzzle, "solution": solution},
            "task": "sudoku",
        })
    return out


def grade_sudoku(pred_text, gt):
    puzzle = gt["puzzle"]
    m = re.search(r"Answer:\s*\n?((?:[0-4\s]*\n){3}[0-4\s]*)", pred_text, re.IGNORECASE)
    text = m.group(1) if m else pred_text
    rows = [re.sub(r"[^0-4]", "", line) for line in text.strip().split("\n")]
    rows = [r for r in rows if r]
    if len(rows) < 4:
        return False
    try:
        grid = [[int(ch) for ch in row[:4]] for row in rows[-4:]]
    except ValueError:
        return False
    if any(len(row) != 4 for row in grid):
        return False
    for r in range(4):
        for c in range(4):
            if puzzle[r][c] != 0 and grid[r][c] != puzzle[r][c]:
                return False
    expected = {1, 2, 3, 4}
    for row in grid:
        if set(row) != expected:
            return False
    for c in range(4):
        if {grid[r][c] for r in range(4)} != expected:
            return False
    for br in (0, 2):
        for bc in (0, 2):
            if {grid[r][c] for r in range(br, br + 2) for c in range(bc, bc + 2)} != expected:
                return False
    return True
