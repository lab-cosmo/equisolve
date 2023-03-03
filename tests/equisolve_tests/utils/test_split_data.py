# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
from typing import List

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import equistore
from equistore import io, Labels, TensorBlock, TensorMap
from equisolve.utils import split_data


@pytest.fixture
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


@pytest.fixture
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


def check_no_overlap_indices(
    split_tensors: List[TensorMap], axis: str, names: List[str]
):
    """
    Checks that there is no overlap (i.e. intersection) between the indices in
    the split tensors, for the specified axis and names.
    """
    # no overlap in indices between groups
    for i in range(len(split_tensors[0])):
        for j in range(i + 1, len(split_tensors[0])):
            uniq_a = equistore.unique_metadata(
                split_tensors[0][i],
                axis,
                names,
            )
            uniq_b = equistore.unique_metadata(
                split_tensors[0][j],
                axis,
                names,
            )
            if np.any(np.isin(uniq_a, uniq_b)):
                pytest.fail(f"{uniq_a} not equal to {uniq_b}")


def check_metadata(split_tensors: List[TensorMap], axis: str):
    """
    Checks that the components metadata are the same for the corresponding
    blocks in the split tensors. If the split axis is "samples", check the
    properties are the same, otherwise check the samples are the same.
    """
    ref_tensor = split_tensors[0][0]
    for i in range(1, len(split_tensors[0])):
        for key in ref_tensor.keys:
            ref_block = ref_tensor[key]
            check_block = split_tensors[0][i][key]
            # check samples the same if splitting on properties
            if axis == "samples":
                assert np.all(ref_block.properties == check_block.properties)
            else:
                assert np.all(ref_block.samples == check_block.samples)
            # components are the same
            for c_i in range(len(ref_block.components)):
                if not np.all(ref_block.components[c_i] == check_block.components[c_i]):
                    pytest.fail(
                        f"components don't match: {ref_block.components[c_i]}"
                        + f" != {check_block.components[c_i]}"
                    )


def check_values(
    original_tensor: TensorMap,
    split_tensors: List[TensorMap],
    target_indices: List[Labels],
):
    """
    Checks that the block values have been sliced correctly.
    """
    for i, indices in enumerate(target_indices):
        ref_tensor = equistore.slice(original_tensor, samples=indices)
        for key, ref_block in ref_tensor:
            check_block = split_tensors[0][i][key]
            if not np.all(ref_block.values == check_block.values):
                pytest.fail(f"Expected {ref_block.values}, got {check_block.values}")


@pytest.mark.parametrize(
    "test_tensor_map",
    ["test_tensor_map_a", "test_tensor_map_b"],
)
class TestSplitData:
    def test_split_data_no_shuffle(self, test_tensor_map, request):
        """
        Tests splitting a single TensorMap along "samples" axis, using sample
        name "samples", with no shuffling of the indices.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        samples_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        target_grouped_idxs = [
            Labels(
                names=["samples"],
                values=np.array([[s]]).reshape(-1, 1),
            )
            for s in samples_idxs
        ]
        axis, names, n_groups = "samples", "samples", 8
        split_tensors, actual_grouped_idxs = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
        )
        # actual and target grouped indices are the same
        assert_equal(actual_grouped_idxs, target_grouped_idxs)
        # 1 tensor in, so 1 list of tensors out
        assert_equal(len(split_tensors), 1)
        # in that list there should be 8 tensors
        assert_equal(len(split_tensors[0]), n_groups)
        # check metadata along split axis
        check_no_overlap_indices(split_tensors, axis, names)
        # check other metadata
        check_metadata(split_tensors, axis)
        # check values
        check_values(test_tensor_map, split_tensors, target_grouped_idxs)

    def test_split_data_with_shuffle(self, test_tensor_map, request):
        """
        Tests splitting a single TensorMap along "samples" axis, using sample
        name "samples", with shuffling of the indices.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # Define the sample indices
        samples_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        # Copy and shuffle the indices using a seed
        shuffled_sample_idxs = np.array(samples_idxs, copy=True)
        random_seed = 1
        np.random.seed(random_seed)
        np.random.shuffle(shuffled_sample_idxs)
        # Set the random seed to a different random seed
        np.random.seed(31415)
        # Define the target grouped indices
        target_grouped_idxs = [
            Labels(
                names=["samples"],
                values=np.array([[s]]).reshape(-1, 1),
            )
            for s in shuffled_sample_idxs
        ]
        # Split the data
        axis, names, n_groups = "samples", "samples", 8
        split_tensors, actual_grouped_idxs = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            random_seed=random_seed,
        )
        # actual and target grouped indices are the same
        assert_equal(actual_grouped_idxs, target_grouped_idxs)
        # 1 tensor in, so 1 list of tensors out
        assert_equal(len(split_tensors), 1)
        # in that list there should be 8 tensors
        assert_equal(len(split_tensors[0]), n_groups)
        # check metadata along split axis
        check_no_overlap_indices(split_tensors, axis, names)
        # check other metadata
        check_metadata(split_tensors, axis)
        # check values
        check_values(test_tensor_map, split_tensors, target_grouped_idxs)

    def test_split_data_abs_rel_same_result(self, test_tensor_map, request):
        """
        Tests that splitting data using the `group_sizes_abs` and
        `group_sizes_rel` args gives equivalent results.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        samples_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        target_grouped_idxs = [
            Labels(
                names=["samples"],
                values=np.array([[0]]).reshape(-1, 1),
            ),
            Labels(
                names=["samples"],
                values=np.array([[1, 2]]).reshape(-1, 1),
            ),
            Labels(
                names=["samples"],
                values=np.array([[3, 4, 5, 6]]).reshape(-1, 1),
            ),
            Labels(
                names=["samples"],
                values=np.array([[8]]).reshape(-1, 1),
            ),
        ]
        # Split the data using abs and rel group sizes
        axis, names, n_groups = "samples", "samples", 4
        split_tensors_1, actual_grouped_idxs_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes_abs=[1, 2, 4, 1],
        )
        split_tensors_2, actual_grouped_idxs_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes_rel=[1/8, 2/8, 4/8, 1/8],
        )
        assert_equal(target_grouped_idxs, actual_grouped_idxs_1)
        assert_equal(actual_grouped_idxs_1, actual_grouped_idxs_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)

    def test_split_data_abs_rel_same_result(self, test_tensor_map, request):
        """
        Tests that splitting data using the `group_sizes_abs` and
        `group_sizes_rel` args gives equivalent results.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # There are 8 unique structure idxs in total
        samples_idxs = np.array([0, 1, 2, 3, 4, 5, 6, 8])
        # Split only the first 7 of these
        target_grouped_idxs = [
            Labels(
                names=["samples"],
                values=np.array([[0]]).reshape(-1, 1),
            ),
            Labels(
                names=["samples"],
                values=np.array([[1, 2]]).reshape(-1, 1),
            ),
            Labels(
                names=["samples"],
                values=np.array([[3, 4, 5, 6]]).reshape(-1, 1),
            ),
        ]
        # Split the data using abs and rel group sizes
        axis, names, n_groups = "samples", "samples", 3
        split_tensors_1, actual_grouped_idxs_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes_abs=[1, 2, 4],  # adds up to 7
        )
        split_tensors_2, actual_grouped_idxs_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes_rel=[1/8, 2/8, 4/8],  # adds up to 7/8
        )
        assert_equal(target_grouped_idxs, actual_grouped_idxs_1)
        assert_equal(actual_grouped_idxs_1, actual_grouped_idxs_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)


# class TestSplitDataErrors:
#     def test_split_data_error(self):
