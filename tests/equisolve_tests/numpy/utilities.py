import functools

import metatensor
import numpy as np
from metatensor import Labels, TensorMap

from ..utilities import random_single_block_no_components_tensor_map


random_single_block_no_components_tensor_map = functools.partial(
    random_single_block_no_components_tensor_map,
    use_torch=False,
    use_metatensor_torch=False,
)


def tensor_to_tensormap(a: np.ndarray, key_name: str = "keys") -> TensorMap:
    """Create a :class:`metatensor.TensorMap` from 3d :class`numpy.ndarray`.

    First dimension of a defines the number of blocks.
    The values of each block are taken from the second and the third dimension.
    The name of the property labels in each block is `'property' and name of the sample
    labels is `'sample'`. The blocks have no components.

    :param a:
        3d numpy array for the block of the TensorMap values
    :param key_name:
        name of the TensorMaps' keys

    :returns:
        TensorMap with filled values


    Example:
    >>> a = np.zeros([2,2])
    >>> # make 2d array 3d tensor
    >>> tensor = tensor_to_tensormap(a[np.newaxis, :])
    >>> print(tensor)
    """
    if len(a.shape) != 3:
        raise ValueError(f"`a` has {len(a.shape)} but must have exactly 3")

    blocks = []
    for values in a:
        blocks.append(metatensor.block_from_array(values))

    keys = Labels.range(key_name, end=len(blocks))
    return TensorMap(keys, blocks)
