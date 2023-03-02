# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from equistore import io, Labels, TensorBlock, TensorMap
from equisolve.utils import split_data


class TestSplitData:
    def set_up(self):
        self.tensor_a = test_tensor_map_a()
        self.tensor_b = test_tensor_map_b()

    def test_split_data_no_shuffle(self):
        """
        Tests splitting a single TensorMap along "samples" axis, using sample
        name "samples", with no shuffling of the indices.
        """
        # Passing ``tensors`` and ``names`` not in a list shoudl not throw an
        # error. Passing ``n_groups=8`` should split into 8 even groups
        samples_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        target_grouped_idxs = [
            Labels(
                names=["samples"],
                values=np.array([[s]]).reshape(-1, 1),
            )
            for s in samples_idxs
        ]
        split_tensors, actual_grouped_idxs = split_data(
            tensors=self.tensor_a,
            axis="samples",
            names="samples",
            n_groups=-1,
        )
        # for i in range(len(target_grouped_idxs)):

        # TODO: Intersection between all groups of indices is zero


def test_tensor_map_a():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(
            ["samples"],
            np.array([[0], [3], [6], [8]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[
            Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32)),
        ],
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def test_tensor_map_b():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 101.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 111.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 102.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(
            ["properties"],
            np.array([[3], [4], [5]], dtype=np.int32),
        ),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 112.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 103.0),
        samples=Labels(
            ["samples"],
            np.array([[0], [3], [6], [8]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 113.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 104.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[
            Labels(
                ["components"],
                np.array([[0], [1], [2]], dtype=np.int32),
            )
        ],
        properties=Labels(
            ["properties"],
            np.array([[0]], dtype=np.int32),
        ),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 114.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])