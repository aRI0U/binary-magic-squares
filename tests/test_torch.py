import itertools
from math import log2
import pytest

import torch

from binary_magic_squares.pytorch import generate_bms, is_bms


_valid_bms = torch.tensor([
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1]
], dtype=torch.bool)


_invalid_bms = torch.tensor([
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1]
], dtype=torch.bool)


@pytest.fixture
def valid_bms() -> torch.Tensor:
    # Fixture to generate a valid binary magic square
    return _valid_bms.clone()


@pytest.fixture
def invalid_bms() -> torch.Tensor:
    # Fixture to generate an invalid binary magic square
    return _invalid_bms.clone()


@pytest.mark.parametrize('bms', itertools.permutations(_valid_bms))
def test_is_bms_true(bms):
    # Test that a valid binary magic square is recognized as such (all permutations of a BMS are BMS as well)
    assert is_bms(torch.stack(bms, dim=0))
    assert is_bms(torch.stack(bms, dim=1))


@pytest.mark.parametrize('bms', itertools.permutations(_invalid_bms))
def test_is_bms_false(bms):
    # Test that an invalid binary magic square is recognized as such (none of the permutations of a non-BMS are BMS)
    assert not is_bms(torch.stack(bms, dim=0))
    assert not is_bms(torch.stack(bms, dim=1))


def test_is_bms_true_multiple(valid_bms):
    assert is_bms(torch.stack([
        valid_bms,
        valid_bms.roll(2, 0),
        valid_bms.roll(3, 1),
        valid_bms.roll((-1, 1), (0, 1))
    ], dim=0))


def test_is_bms_false_multiple(invalid_bms):
    assert not is_bms(torch.stack([
        invalid_bms,
        invalid_bms.roll(2, 0),
        invalid_bms.roll(3, 1),
        invalid_bms.roll((-1, 1), (0, 1))
    ], dim=0))


def test_is_bms_both(valid_bms, invalid_bms):
    assert not is_bms(torch.stack([valid_bms, invalid_bms]))


def _test_generate_bms_it(size_max, num_masks_max):
    # Generate all combinations of size, line_sum, and num_masks
    for size in range(1, size_max + 1):
        for line_sum in range(size):
            for num_masks in [2**i for i in range(int(log2(num_masks_max))+1)]:
                yield size, line_sum, num_masks

@pytest.mark.parametrize("size, line_sum, num_masks", _test_generate_bms_it(30, 64))
def test_generate_bms(size, line_sum, num_masks):
    # Test that generate_bms produces a valid binary magic square with the specified size, line_sum, and num_masks
    bms = generate_bms(line_sum, size, num_masks=num_masks)
    assert is_bms(bms)


@pytest.mark.parametrize("k1, k2, size", ((k1, k2, s) for s in range(1, 21) for k1, k2 in itertools.product(range(1, s), range(1, s))))
def test_generate_bms_different_sums(k1, k2, size):
    assert is_bms(torch.stack([generate_bms(k1, size), generate_bms(k2, size)]))

@pytest.mark.parametrize("k, m, n", ((k, m, i*m) for m in range(1, 10) for k in range(1, m) for i in range(1, 5)))
def test_shape_no_batch_dim(k, m, n):
    assert generate_bms(k, m, n).shape == (m, n)


@pytest.mark.parametrize("k, m, n, masks", ((k, m, i*m, masks) for m in range(1, 10) for k in range(1, m) for i in range(1, 5) for masks in range(1, 10)))
def test_shape_batch_dim(k, m, n, masks):
    assert generate_bms(k, m, n, num_masks=masks).shape == (masks, m, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_device(device):
    assert generate_bms(3, 5, device=device).device == device


def test_invalid_shape():
    with pytest.raises(AssertionError):
        generate_bms(3, 5, 8)