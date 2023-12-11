from typing import Optional

import numpy as np


def random_subset(mask, n_elems):
    if not isinstance(mask, np.ndarray) or not mask.dtype == bool:
        raise ValueError("The 'mask' parameter must be a boolean numpy array.")

    if n_elems < 0 or n_elems > len(mask):
        raise ValueError("Invalid value for 'n_elems' parameter.")

    indices = np.where(mask)[0]
    np.random.shuffle(indices)

    subset = np.zeros_like(mask)
    subset[indices[:n_elems]] = True

    return subset


def generate_bms(k: int, m: int, n: Optional[int] = None) -> np.ndarray:
    r""""""
    # By default we generate a square (i.e. m = n)
    n = n or m

    # handle trivial cases
    if k == 0:
        return np.zeros((m, n), dtype=bool)

    if k == n:
        return np.ones((m, n), dtype=bool)

    # the transpose of a BMS is also a BMS, so we eventually transpose for having m >= n
    # so that we always iterate on the smallest dimension
    transpose = m < n
    if transpose:
        m, n = n, m

    q, r = divmod(m, n)
    assert r == 0, "For non-trivial magic squares to exist, the number of rows and columns must divide each other."
    km, kn = q*k, k

    bms = np.zeros((m, n), dtype=bool)
    s = np.zeros(m, dtype=int)

    for t in range(n):
        a1 = s == kn + t - n
        a3 = s == kn
        a2 = ~(a1 | a3)

        to_check = a1 | random_subset(a2, km - a1.sum())

        s += to_check
        bms[:, t] = to_check

    return bms.T if transpose else bms


def is_bms(mask: np.ndarray) -> bool:
    """
    Args:
        mask (numpy.ndarray): boolean mask. Shape (m, n)

    Returns:
        bool: Whether the mask is a Binary Magic Square or not.
    """
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_, "Only bool ndarrays can be Binary Magic Squares"

    assert mask.ndim == 2, "Shape of the mask should be (m, n)"

    sum_rows = np.sum(mask, axis=0)
    sum_cols = np.sum(mask, axis=1)

    return np.all(np.equal(sum_rows, sum_rows[0])) and np.all(np.equal(sum_cols, sum_cols[0]))
