"""Shared paths and helpers.  Nothing here imports anything but NumPy."""

import json
import os

import numpy as np
import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CHECKPOINT = os.path.join(FIXTURES, "checkpoint")


def rel_err(got, ref) -> float:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.abs(got - ref).max() / max(float(np.abs(ref).max()), 1e-9))


@pytest.fixture(scope="session")
def fixture_config():
    with open(os.path.join(FIXTURES, "config.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def reference_io():
    with np.load(os.path.join(FIXTURES, "reference_io.npz")) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="session")
def reference_values():
    with open(os.path.join(FIXTURES, "..", "reference_values.json"), encoding="utf-8") as fh:
        return json.load(fh)
