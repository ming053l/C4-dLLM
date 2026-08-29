"""C4: commit locally, exit globally.

`decode.generate_live` is the accelerated decoding loop. It takes a masked diffusion language model
and returns the generated text along with the step count it actually spent, applying two gates.
CCTC decides which token positions a step may commit; CVEE decides when the sequence may stop.
Both are off unless their arguments are supplied, so the same function reproduces the unaccelerated
schedule for a baseline.
"""
from .decode import generate_live, generate_full          # noqa: F401
from .gate import CVEETracker                              # noqa: F401
from .policies import POLICIES                             # noqa: F401
