"""Make `rlm_train` and the per-env eval-suite packages importable in tests.

These packages live in sibling source trees rather than being installed, so a
plain `pytest` from the repo root would fail to import them. The validators
under `tools/` do the same sys.path insertion; this keeps the test suite
runnable with a bare `pytest` invocation. Each eval environment is its own
package directory under `training/environments/` (mirroring `oolong/`), so we
add every one to the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
ENVS = TRAINING / "environments"
_PATHS = [TRAINING / "src"] + [
    ENVS / name
    for name in (
        "oolong",
        "oolong_pairs",
        "browsecomp_plus",
        "longbench_codeqa",
        "longcot_mini",
    )
]
for p in _PATHS:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
