import itertools

import numpy as np
import pytest


from binary_magic_squares import generate_bms, is_bms


_valid_bms = np.array([
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1]
], dtype=bool)


_invalid_bms = np.array([
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1]
], dtype=bool)


@pytest.mark.parametrize('bms', itertools.permutations(_valid_bms))
def test_is_bms_true(bms):
    # Test that a valid binary magic square is recognized as such (all permutations of a BMS are BMS as well)
    assert is_bms(np.stack(bms, axis=0))
    assert is_bms(np.stack(bms, axis=1))


@pytest.mark.parametrize('bms', itertools.permutations(_invalid_bms))
def test_is_bms_false(bms):
    # Test that an invalid binary magic square is recognized as such (none of the permutations of a non-BMS are BMS)
    assert not is_bms(np.stack(bms, axis=0))
    assert not is_bms(np.stack(bms, axis=1))


@pytest.mark.parametrize("k, n", ((k, n) for n in range(1, 31) for k in range(1, n+1)))
def test_generate_bms(k, n):
    # Test that generate_bms produces a valid binary magic square with the specified size, line_sum, and num_masks
    bms = generate_bms(k, n)
    assert is_bms(bms)


@pytest.mark.parametrize("k, m, n", ((k, m, i*m) for m in range(1, 10) for k in range(1, m) for i in range(1, 5)))
def test_shape(k, m, n):
    assert generate_bms(k, m, n).shape == (m, n)


def test_invalid_shape():
    with pytest.raises(AssertionError):
        generate_bms(3, 5, 8)
