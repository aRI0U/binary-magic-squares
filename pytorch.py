from typing import Tuple

import torch


def random_subset(mask: torch.BoolTensor, n_elems: int) -> torch.BoolTensor:
    indices = torch.where(mask)[0]
    indices = indices[torch.randperm(len(indices))[:n_elems]]

    subset = torch.zeros_like(mask)
    subset[indices] = True

    return subset


def batch_random_subset(mask: torch.BoolTensor, n_elems: torch.LongTensor) -> torch.BoolTensor:
    return torch.stack([random_subset(m, n) for m, n in zip(mask, n_elems)])


def generate_bms(m: int,
                 n: int,
                 masking_ratio: float,
                 num_masks: int = 1,
                 device: torch.device = torch.device("cpu")) -> Tuple[torch.BoolTensor, torch.BoolTensor]:

    r""""""
    # By default we generate a square (i.e. m = n)
    n = n or m
    k = round(masking_ratio * n)

    # the transpose of a BMS is also a BMS, so we eventually transpose for having m >= n
    # so that we always iterate on the smallest dimension
    transpose = m < n
    if transpose:
        m, n = n, m

    q, r = divmod(m, n)
    assert r == 0, "For non-trivial magic squares to exist, the number of rows and columns must divide each other."
    km, kn = q*k, k

    bms = torch.zeros(num_masks, m, n, dtype=torch.bool, device=device)
    s = torch.zeros(num_masks, m, dtype=torch.long, device=device)

    for t in range(n):
        a1 = torch.eq(s, kn + t - n)
        a3 = torch.eq(s, kn)
        a2 = ~torch.logical_or(a1, a3)

        to_check = a1 | batch_random_subset(a2, km - a1.sum(dim=1))

        s += to_check
        bms[:, :, t] = to_check

    if transpose:
        bms = bms.transpose(-2, -1)

    return bms, ~bms


if __name__ == '__main__':
    import sys

    assert len(sys.argv) >= 3

    a, b, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None

    mat, _ = generate_bms(b, c, a / c, 7)
    print(mat.long())
