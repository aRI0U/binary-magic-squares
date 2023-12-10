import torch


def _batch_random_subset(mask: torch.BoolTensor, n_elems: torch.LongTensor) -> torch.BoolTensor:
    r"""Randomly creates a submask of the original one so that each row i has exactly n_elems[i] True elements.

    Args:
        mask (torch.BoolTensor): boolean mask, shape (batch_size, m)
        n_elems (torch.LongTensor): number of elements to keep True per row

    Returns:

    """
    m = mask.size(1)
    sum_lines = mask.sum(dim=1, keepdim=True)
    permutations = torch.randn_like(mask, dtype=torch.float32).argsort(dim=1)

    valid_mask = permutations < sum_lines
    valid_indices = permutations[valid_mask]

    arange = torch.arange(m).unsqueeze(0)
    to_keep_mask = arange < n_elems.unsqueeze(1)
    to_keep_mask = to_keep_mask[arange < sum_lines]
    to_keep_indices = valid_indices[to_keep_mask]

    true_indices = torch.nonzero(mask, as_tuple=False)

    sum_lines.squeeze_(1)
    offset = torch.nn.functional.pad(sum_lines[:-1].cumsum(dim=0), (1, 0))
    offset = offset.gather(0, true_indices[:, 0])[to_keep_mask]
    true_indices = true_indices[to_keep_indices + offset]

    final_mask = torch.zeros_like(mask)
    # final_mask.scatter_(1, true_indices, 1)
    final_mask[true_indices[:, 0], true_indices[:, 1]] = True

    return final_mask


def generate_bms(k: int,
                 m: int,
                 n: int,
                 num_masks: int = 1,
                 device: torch.device = torch.device("cpu")) -> torch.BoolTensor:
    r""""""
    # By default we generate a square (i.e. m = n)
    n = n or m

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

        to_check = a1 | _batch_random_subset(a2, km - a1.sum(dim=1))

        s += to_check
        bms[:, :, t] = to_check

    if transpose:
        bms = bms.transpose(-2, -1)

    return bms


if __name__ == '__main__':
    import sys

    assert len(sys.argv) >= 3

    a, b, c = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None

    mat = generate_bms(a, b, c, 2)
    print(mat.long())
