"""Monte-Carlo simulation of voting methods.

Describe an electorate and a set of voting methods with a config, run some
number of trials, and get one Arrow row per trial back::

    import mcelect

    config = mcelect.load_config("configs/default.toml")
    table = mcelect.simulate(config, 10_000)

The simulation runs on all available cores and releases the GIL while it works.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ._mcelect import __version__
from ._mcelect import simulate as _simulate

if TYPE_CHECKING:
    import pyarrow

__all__ = ["__version__", "load_config", "simulate"]


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a config file into a plain dict, ready to pass to `simulate`.

    `.toml` files are parsed as TOML, everything else as JSON.
    """
    path = os.fspath(path)
    if path.endswith(".toml"):
        with open(path, "rb") as f:
            return tomllib.load(f)
    with open(path, "rb") as f:
        return json.load(f)


def simulate(config: Mapping[str, Any] | str, trials: int) -> pyarrow.Table:
    """Run `trials` elections and return the results as a `pyarrow.Table`.

    `config` is either a mapping (as `load_config` returns) or a JSON string.
    The resulting columns depend on the config's `mode`; see the crate docs for
    `ExperimentResult` (single-winner) and `CommitteeResult` (multi-winner).
    """
    if not isinstance(config, str):
        config = json.dumps(config)
    return _simulate(config, trials)
