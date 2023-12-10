import numpy as np


def random_subset(mask, n_elems):  # TODO: see if can be optimized
    if not isinstance(mask, np.ndarray) or not mask.dtype == bool:
        raise ValueError("The 'mask' parameter must be a boolean numpy array.")

    if n_elems < 0 or n_elems > len(mask):
        raise ValueError("Invalid value for 'n_elems' parameter.")

    indices = np.where(mask)[0]
    np.random.shuffle(indices)

    subset = np.zeros_like(mask)
    subset[indices[:n_elems]] = True

    return subset


def generate_bms(k, m, n=None):
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


if __name__ == '__main__':
    import sys

    assert len(sys.argv) >= 3

    k, m, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None

    mat = generate_bms(k, m, n)
    print(mat.astype(int))
