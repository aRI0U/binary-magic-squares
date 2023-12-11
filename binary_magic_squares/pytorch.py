import torch


def _batch_random_subset(mask: torch.BoolTensor, n_elems: torch.LongTensor) -> torch.Tensor:
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
    final_mask[true_indices[:, 0], true_indices[:, 1]] = True

    return final_mask


def generate_bms(k: int,
                 m: int,
                 n: int | None = None,
                 num_masks: int | None = None,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
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

    not_batched = num_masks is None
    num_masks = num_masks or 1

    bms = torch.zeros(num_masks, m, n, dtype=torch.bool, device=device)
    s = torch.zeros(num_masks, m, dtype=torch.long, device=device)

    for t in range(n):
        a1 = torch.eq(s, kn + t - n)
        a3 = torch.eq(s, kn)
        a2 = ~torch.logical_or(a1, a3)

        to_check = a1 | _batch_random_subset(a2, km - a1.sum(dim=-1))

        s += to_check
        bms[:, :, t] = to_check

    if transpose:
        bms = bms.transpose(-2, -1)

    if not_batched:
        bms.squeeze_(0)

    return bms


def is_bms(masks: torch.Tensor) -> bool:
    r"""

    Args:
        masks (torch.Tensor): boolean mask (or batch of masks). Shape (*, m, n)

    Returns:
        bool: Whether the masks all are Binary Magic Squares or not.
    """
    assert torch.is_tensor(masks) and masks.dtype == torch.bool, "Only BoolTensors can be Binary Magic Squares"

    assert masks.ndim >= 2

    sum_rows = masks.sum(dim=-2)

    if not torch.all(torch.eq(sum_rows, sum_rows[..., 0].unsqueeze(-1))):
        return False

    sum_cols = masks.sum(dim=-1)
    return torch.all(torch.eq(sum_cols, sum_cols[..., 0].unsqueeze(-1))).item()
