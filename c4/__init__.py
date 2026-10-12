"""C4: commit locally, exit globally.

`generate` decodes one sequence with CCTC (which positions a step may commit, `cctc.py`) and
CVEE (when the sequence may stop, `cvee.py`) layered on the reference block-wise sampler. Both are
off unless configured, so the same call also runs the unaccelerated baseline. `generate_live` is
the flat keyword interface of earlier releases; `CVEETracker` is the exit gate alone, for a
decoding loop you do not own.
"""
from .cctc import CCTCConfig
from .cvee import EXIT_GATES, POLICIES, CVEEConfig, CVEETracker
from .decode import DecodeResult, generate, generate_live
from .peer_exits import ProphetExit, SchEDExit
from .trajectory import generate_full

__all__ = ["generate", "generate_live", "generate_full", "DecodeResult", "CCTCConfig",
           "CVEEConfig", "CVEETracker", "EXIT_GATES", "POLICIES", "ProphetExit", "SchEDExit"]
