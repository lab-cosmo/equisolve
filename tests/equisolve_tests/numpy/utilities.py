import equistore
import numpy as np
from equistore import Labels, TensorBlock, TensorMap


def random_single_block_no_components_tensor_map():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.random.rand(4, 2),
        samples=Labels(
            ["sample", "structure"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
        ),
        components=[],
        properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
    )
    positions_gradient = TensorBlock(
        values=np.random.rand(7, 3, 2),
        samples=Labels(
            ["sample", "structure", "center"],
            np.array(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [2, 2, 0],
                    [3, 3, 0],
                ],
                dtype=np.int32,
            ),
        ),
        components=[Labels(["direction"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=block_1.properties,
    )
    block_1.add_gradient("positions", positions_gradient)

    cell_gradient = TensorBlock(
        values=np.random.rand(4, 6, 2),
        samples=Labels(
            ["sample", "structure"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["direction_xx_yy_zz_yz_xz_xy"],
                np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32),
            )
        ],
        properties=block_1.properties,
    )
    block_1.add_gradient("cell", cell_gradient)

    return TensorMap(Labels.single(), [block_1])


def tensor_to_tensormap(a: np.ndarray, key_name: str = "keys") -> TensorMap:
    """Create a :class:`equistore.TensorMap` from 3d :class`numpy.ndarray`.

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
        blocks.append(equistore.block_from_array(values))

    keys = Labels.range(key_name, end=len(blocks))
    return TensorMap(keys, blocks)
