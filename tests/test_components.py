"""CPU unit tests for the pieces of C4 that do not need a model. Run: python -m unittest discover tests

Decoding equivalence against the release the paper used needs a GPU and lives in
tools/check_equivalence.py.
"""
import unittest

import numpy as np
import torch

from c4.answers import code_candidate, extract_boxed, extract_letter, extract_number
from c4.cctc import CCTCConfig, CoreWindow
from c4.cvee import CVEETracker, conf_floor, confidence_only, required_run, stability_only
from c4.grading import grade, run_program


class CharTokenizer:
    """One token per character, enough to exercise candidate location."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in np.asarray(ids).tolist())

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def bools(s):
    return torch.tensor([c == "1" for c in s])


class GateTests(unittest.TestCase):
    def test_required_run(self):
        self.assertEqual(required_run(0, 2.0, 3), 3)
        self.assertEqual(required_run(2, 2.0, 3), 4)
        self.assertEqual(required_run(1, 2.5, 1), 3)

    def test_joint_gate_needs_both(self):
        self.assertTrue(conf_floor(run=3, changes=0, conf=0.7))
        self.assertFalse(conf_floor(run=3, changes=0, conf=0.69))
        self.assertFalse(conf_floor(run=3, changes=2, conf=0.99))   # two changes demand run 4
        self.assertTrue(conf_floor(run=4, changes=2, conf=0.99))
        self.assertFalse(conf_floor(run=10, changes=0, conf=None))

    def test_ablations(self):
        self.assertTrue(confidence_only(conf=0.8))
        self.assertFalse(confidence_only(conf=None))
        self.assertTrue(stability_only(run=3, changes=0, conf=None))


class CoreWindowTests(unittest.TestCase):
    def window(self, **kw):
        cfg = CCTCConfig(**{"holes": 1, "confirm": True, "stall_escalate": 3, **kw})
        return CoreWindow(cfg, 7, "cpu")

    def test_core_stops_after_hole_budget(self):
        w = self.window(confirm=False)
        commit = w.select(bools("1101101"), bools("1111111"), torch.arange(7), last_step=False)
        self.assertEqual(commit.tolist(), bools("1101100").tolist())

    def test_committed_positions_are_transparent(self):
        w = self.window(holes=0, confirm=False)
        # positions 0-1 already committed (not alive), so the core starts at 2
        commit = w.select(bools("0011000"), bools("0011111"), torch.arange(7), last_step=False)
        self.assertEqual(commit.tolist(), bools("0011000").tolist())

    def test_deferred_position_confirmed_if_prediction_holds(self):
        w = self.window(holes=0)
        x0 = torch.tensor([5, 6, 7, 8, 9, 10, 11])
        first = w.select(bools("1011000"), bools("1111111"), x0, last_step=False)
        self.assertEqual(first.tolist(), bools("1000000").tolist())       # 2,3 deferred
        x0_next = x0.clone()
        x0_next[3] = 99                                                     # 3 changes its mind
        second = w.select(bools("0011000"), bools("0111111"), x0_next, last_step=False)
        self.assertEqual(second.tolist(), bools("0010000").tolist())      # only 2 confirmed

    def test_stall_escalation_and_last_step(self):
        w = self.window(holes=0, confirm=False, stall_escalate=2)
        alive, elig = bools("1111111"), bools("0001000")
        self.assertFalse(w.select(elig, alive, torch.arange(7), last_step=False).any())
        self.assertEqual(w.select(elig, alive, torch.arange(7), last_step=False).tolist(),
                         elig.tolist())                                     # escalated
        w = self.window(holes=0, confirm=False, stall_escalate=0)
        self.assertEqual(w.select(elig, alive, torch.arange(7), last_step=True).tolist(),
                         elig.tolist())


class CVEETrackerTests(unittest.TestCase):
    def test_stops_after_stable_confident_candidate(self):
        tok = CharTokenizer()
        prompt_len, gen = 2, "So the answer is 42"
        tracker = CVEETracker(tok, extract_number, prompt_len, len(gen), search_mode="last",
                              tail_frac=0.3, gamma=2.0, p_min=3, tau=0.7)
        x0 = torch.tensor([[0, 0] + [ord(c) for c in gen]])
        conf = torch.full(x0.shape, 0.9)
        fired = [tracker.step(x0, conf) for _ in range(3)]
        self.assertEqual(fired, [False, False, True])
        self.assertEqual((tracker.run, tracker.changes), (3, 0))


class AnswerAndGradingTests(unittest.TestCase):
    def test_extractors(self):
        self.assertEqual(extract_number("it costs $1,234."), "1234")
        self.assertEqual(extract_letter("The answer is B because"), "B")
        self.assertEqual(extract_boxed(r"so \boxed{\frac65}"), r"\frac{6}{5}")
        prog = "import math\ndef f(x):\n    return x\n\nprint(f(1))\n"
        self.assertEqual(code_candidate(prog), "import math\ndef f(x):\n    return x")
        self.assertIsNone(code_candidate("no function yet"))

    def test_grading(self):
        self.assertTrue(grade("gsm8k", "... so 1,200", "1200"))
        self.assertFalse(grade("mmlu", "no letter here", "A"))
        gt = {"test": "def check(c):\n    assert c(1) == 2\n", "entry_point": "f"}
        self.assertTrue(grade("humaneval", "def f(x):\n    return x + 1\n", gt))
        self.assertFalse(grade("humaneval", "def f(x):\n    return x\n", gt))

    def test_program_timeout(self):
        self.assertFalse(run_program("while True:\n    pass\n", timeout=1))


if __name__ == "__main__":
    unittest.main()
