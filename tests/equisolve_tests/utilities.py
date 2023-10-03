import functools


def random_single_block_no_components_tensor_map(use_torch, use_metatensor_torch):
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    if not use_torch and use_metatensor_torch:
        raise ValueError(
            "torch.TensorMap cannot be created without torch.Tensor block values."
        )
    if use_metatensor_torch:
        import torch
        from metatensor.torch import Labels, TensorBlock, TensorMap

        create_int32_array = functools.partial(torch.tensor, dtype=torch.int32)
    else:
        import numpy as np
        from metatensor import Labels, TensorBlock, TensorMap

        create_int32_array = functools.partial(np.array, dtype=np.int32)

    if use_torch:
        import torch

        create_random_array = torch.rand
    else:
        import numpy as np

        create_random_array = np.random.rand

    block_1 = TensorBlock(
        values=create_random_array(4, 2),
        samples=Labels(
            ["sample", "structure"],
            create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ),
        components=[],
        properties=Labels(["properties"], create_int32_array([[0], [1]])),
    )
    positions_gradient = TensorBlock(
        values=create_random_array(7, 3, 2),
        samples=Labels(
            ["sample", "structure", "center"],
            create_int32_array(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [2, 2, 0],
                    [3, 3, 0],
                ],
            ),
        ),
        components=[Labels(["direction"], create_int32_array([[0], [1], [2]]))],
        properties=block_1.properties,
    )
    block_1.add_gradient("positions", positions_gradient)

    cell_gradient = TensorBlock(
        values=create_random_array(4, 6, 2),
        samples=Labels(
            ["sample", "structure"],
            create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ),
        components=[
            Labels(
                ["direction_xx_yy_zz_yz_xz_xy"],
                create_int32_array([[0], [1], [2], [3], [4], [5]]),
            )
        ],
        properties=block_1.properties,
    )
    block_1.add_gradient("cell", cell_gradient)

    return TensorMap(Labels.single(), [block_1])


def random_tensor_map_with_components(use_torch, use_metatensor_torch):
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    if not use_torch and use_metatensor_torch:
        raise ValueError(
            "torch.TensorMap cannot be created without torch.Tensor block values."
        )
    if use_metatensor_torch:
        import torch
        from metatensor.torch import Labels, TensorBlock, TensorMap

        create_int32_array = functools.partial(torch.tensor, dtype=torch.int32)
    else:
        import numpy as np
        from metatensor import Labels, TensorBlock, TensorMap

        create_int32_array = functools.partial(np.array, dtype=np.int32)

    if use_torch:
        import torch

        create_random_array = torch.rand
    else:
        import numpy as np

        create_random_array = np.random.rand

    blocks = []
    for i in range(3):
        block = TensorBlock(
            values=create_random_array(4, 2 * i + 1, 5),
            samples=Labels(
                ["sample", "structure"],
                create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            ),
            components=[
                Labels(names=["component"], values=np.arange(2 * i + 1).reshape(-1, 1)),
            ],
            properties=Labels(
                ["properties"], create_int32_array([[0], [1], [2], [5], [10]])
            ),
        )
        positions_gradient = TensorBlock(
            values=create_random_array(7, 3, 2 * i + 1, 5),
            samples=Labels(
                ["sample", "structure", "center"],
                create_int32_array(
                    [
                        [0, 0, 1],
                        [0, 0, 2],
                        [1, 1, 0],
                        [1, 1, 1],
                        [1, 1, 2],
                        [2, 2, 0],
                        [3, 3, 0],
                    ],
                ),
            ),
            components=[
                Labels(["direction"], create_int32_array([[0], [1], [2]])),
                Labels(names=["component"], values=np.arange(2 * i + 1).reshape(-1, 1)),
            ],
            properties=block.properties,
        )
        block.add_gradient("positions", positions_gradient)

        cell_gradient = TensorBlock(
            values=create_random_array(4, 6, 2 * i + 1, 5),
            samples=Labels(
                ["sample", "structure"],
                create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
            ),
            components=[
                Labels(
                    ["direction_xx_yy_zz_yz_xz_xy"],
                    create_int32_array([[0], [1], [2], [3], [4], [5]]),
                ),
                Labels(names=["component"], values=np.arange(2 * i + 1).reshape(-1, 1)),
            ],
            properties=block.properties,
        )
        block.add_gradient("cell", cell_gradient)
        blocks.append(block)

    return TensorMap(Labels(names=["key"], values=np.arange(3).reshape(-1, 1)), blocks)
