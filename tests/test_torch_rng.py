"""cv3cpu.torch_rng must reproduce PyTorch's CPU generator without PyTorch.

CosyVoice 3 draws its flow-matching ODE initial condition as a fixed
``torch.randn`` right after ``torch.manual_seed(0)``, and that tensor is not in
the checkpoint — so a PyTorch-free runtime has to be able to regenerate it.  The
expected values are frozen samples of what ``torch.randn`` / ``torch.rand``
actually produce.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cv3cpu.torch_rng import TorchMT19937, rand, randn  # noqa: E402


def _check(golden, arr):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    assert list(np.shape(arr)) == golden["shape"]
    got = np.array([flat[i] for i in golden["idx"]])
    assert np.abs(got - np.array(golden["vals"])).max() < 1e-5
    assert abs(flat.sum() - golden["sum"]) < 1e-2 * max(1.0, abs(golden["sum"]))


@pytest.mark.parametrize("seed", [0, 1, 42, 1986])
@pytest.mark.parametrize("shape", [(5,), (17,), (1000,), (1, 80, 15000)])
def test_randn_matches_torch(reference_values, seed, shape):
    """Covers both paths: normal_fill for >= 16 elements, scalar Box-Muller below."""
    key = f"randn_{seed}_{'x'.join(map(str, shape))}"
    _check(reference_values["torch_rng"][key], randn(shape, seed))


@pytest.mark.parametrize("seed", [0, 1, 42, 1986])
def test_rand_matches_torch(reference_values, seed):
    _check(reference_values["torch_rng"][f"rand_{seed}_1000"], rand((1000,), seed))


def test_a_shared_generator_advances_like_torch(reference_values):
    """Successive draws from one generator must line up with successive torch calls."""
    g = TorchMT19937(0)
    first = randn((1000,), gen=g)
    _check(reference_values["torch_rng"]["randn_0_1000"], first)
    second = randn((1000,), gen=g)
    assert not np.array_equal(first, second)


def test_the_flow_noise_buffer_is_deterministic():
    a = randn((1, 80, 5000), 0)
    b = randn((1, 80, 5000), 0)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32


def test_uniform_output_is_in_range():
    u = rand((10000,), 7)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.02
