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
