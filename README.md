# Binary Magic Squares generation

**tl;dr**: Efficient algorithm for generating Binary Magic Squares (boolean matrices whose sum of all rows and columns are equal).

Two implementations are provided.
The first one has minimal requirements and enables to easily generate a Binary Magic Square.
The second one has a more advanced implementation and can take advantage of parallel GPU computing to generate many Binary Magic Squares efficiently.

## Installation

```shell
pip install binary-magic-squares
```

The default implementation relies on [NumPy](https://numpy.org/).
If not installed, it will be automatically installed while installing this package.

The batched implementation relies on [PyTorch](https://pytorch.org/).
Please install it beforehand if you want to use this implementation.
You do not need to install it if you want to use only the default implementation.


## Usage

### Default implementation

```python
import numpy as np

from binary_magic_squares import generate_bms, is_bms

# Generate a 5x5 array whose sum of each row and each column equals 2
bms1 = generate_bms(2, 5)

# Generate a 4x8 array whose sum of each row (resp. column) equals 3 (resp. 6)
bms2 = generate_bms(3, 4, 8)

not_bms = np.array(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool))

# Check whether input arrays are BMS or not
print(is_bms(bms1))         # True
print(is_bms(bms2))         # True
print(is_bms(not_bms))      # False
print(is_bms(np.eye(3)))    # True
```

### PyTorch implementation

```python
import torch

from binary_magic_squares.pytorch import generate_bms, is_bms

# Generate a 5x5 tensor whose sum of each row and each column equals 2
bms1 = generate_bms(2, 5)

# Generate a 5x5 tensor whose sum of each row and each column equals 4
bms2 = generate_bms(4, 5)

# Generate a 4x8 tensor on GPU 0 whose sum of each row (resp. column) equals 3 (resp. 6)
bms3 = generate_bms(3, 4, 8, device=torch.device("cuda:0"))

# Generate in parallel a batch of 128 15x15 boolean tensors on GPU 1 whose sum of each row and column equals 11
bms_batch = generate_bms(11, 15, num_masks=128, device=torch.device("cuda:1"))

not_bms = torch.tensor([
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1]
], dtype=torch.bool)

# Check whether input arrays are BMS or not
print(is_bms(bms1))         # True
print(is_bms(bms2))         # True
print(is_bms(bms3))         # True
print(is_bms(not_bms))      # False
print(is_bms(bms_batch))    # True, performs test in parallel
print(is_bms(torch.stack([bms1, bms2])))    # True even if the sum of the rows of bms1 and bms2 is different
print(is_bms(torch.stack([bms1, not_bms]))) # False since one of the tensors is not a BMS
```
