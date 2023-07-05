# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import re
from typing import List

import equistore
import numpy as np
import pytest
from equistore import Labels, TensorBlock, TensorMap
from numpy.testing import assert_equal

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
    gradient_1 = TensorBlock(
        values=np.full((2, 1, 1), 11.0),
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=block_1.properties,
    )
    block_1.add_gradient("parameter", gradient_1)

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    gradient_2 = TensorBlock(
        values=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=block_2.properties,
    )
    block_2.add_gradient("parameter", gradient_2)

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
    gradient_3 = TensorBlock(
        values=np.full((1, 3, 1), 13.0),
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
        properties=block_3.properties,
    )
    block_3.add_gradient("parameter", gradient_3)

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
    gradient_4 = TensorBlock(
        values=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[
            Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32)),
        ],
        properties=block_4.properties,
    )
    block_4.add_gradient("parameter", gradient_4)

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
    gradient_1 = TensorBlock(
        values=np.full((2, 1, 1), 111.0),
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=block_1.properties,
    )
    block_1.add_gradient("parameter", gradient_1)

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 102.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(
            ["properties"],
            np.array([[3], [4], [5]], dtype=np.int32),
        ),
    )
    gradient_2 = TensorBlock(
        values=np.full((3, 1, 3), 112.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=block_2.properties,
    )
    block_2.add_gradient("parameter", gradient_2)

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
    gradient_3 = TensorBlock(
        values=np.full((1, 3, 1), 113.0),
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
        properties=block_3.properties,
    )
    block_3.add_gradient("parameter", gradient_3)

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
    gradient_4 = TensorBlock(
        values=np.full((2, 3, 1), 114.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=block_4.properties,
    )
    block_4.add_gradient("parameter", gradient_4)

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def check_no_intersection_indices_same_group(
    split_tensors: List[TensorMap], axis: str, names: List[str]
):
    """
    Checks that there is no overlap (i.e. intersection) between the indices in
    the split tensors, for the specified axis and names.
    """
    # no overlap in indices between groups
    for group in split_tensors:
        for i in range(len(group)):
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


def check_equivalent_indices_different_groups(
    split_tensors: List[TensorMap], axis: str, names: List[str]
):
    """
    Checks that the jth tensor in every list of split tensors has equivalent
    unique metadata for the specified axis and names.
    """
    if len(split_tensors) == 1:
        return
    ref_list = split_tensors[0]
    assert np.all([len(t) == len(ref_list) for t in split_tensors])
    for i in range(1, len(split_tensors)):
        check_list = split_tensors[i]
        for j in range(len(ref_list)):
            ref_uniq = equistore.unique_metadata(ref_list[j], axis, names)
            check_uniq = equistore.unique_metadata(check_list[j], axis, names)
            assert np.all(ref_uniq == check_uniq)


def check_other_metadata(split_tensors: List[TensorMap], axis: str):
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
                        f" != {check_block.components[c_i]}"
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
        ref_tensor = equistore.slice(original_tensor, axis="samples", labels=indices)
        for key, ref_block in ref_tensor.items():
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
        target_grouped_labels = [
            Labels(
                names=["samples"],
                values=np.array([[s]]).reshape(-1, 1),
            )
            for s in samples_idxs
        ]
        axis, names, n_groups = "samples", "samples", 8
        split_tensors, actual_grouped_labels = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
        )
        # actual and target grouped indices are the same
        assert_equal(actual_grouped_labels, target_grouped_labels)
        # 1 tensor in, so 1 list of tensors out
        assert_equal(len(split_tensors), 1)
        # in that list there should be 8 tensors
        assert_equal(len(split_tensors[0]), n_groups)
        # check metadata along split axis: no intersection between indices of
        # TensorMaps in the same group, exact equivalence between indices of
        # corresponding TensorMaps in different groups
        check_no_intersection_indices_same_group(split_tensors, axis, names)
        check_equivalent_indices_different_groups(split_tensors, axis, names)
        # check other metadata
        check_other_metadata(split_tensors, axis)
        # check values
        check_values(test_tensor_map, split_tensors, target_grouped_labels)

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
        seed = 1
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled_sample_idxs)
        # Set the random seed to a different random seed to check that the
        # function is using the correct seed
        rng = np.random.default_rng(31415)
        # Define the target grouped indices
        target_grouped_labels = [
            Labels(
                names=["samples"],
                values=np.array([[s]]).reshape(-1, 1),
            )
            for s in shuffled_sample_idxs
        ]
        # Split the data
        axis, names, n_groups = "samples", "samples", 8
        split_tensors, actual_grouped_labels = split_data(
            tensors=[test_tensor_map, test_tensor_map],
            axis=axis,
            names=names,
            n_groups=n_groups,
            seed=seed,
        )
        # actual and target grouped indices are the same
        assert_equal(actual_grouped_labels, target_grouped_labels)
        # 1 tensor in, so 1 list of tensors out
        assert_equal(len(split_tensors), 2)
        # in that list there should be 8 tensors
        assert_equal(len(split_tensors[0]), n_groups)
        assert_equal(len(split_tensors[1]), n_groups)
        # check metadata along split axis: no intersection between indices of
        # TensorMaps in the same group, exact equivalence between indices of
        # corresponding TensorMaps in different groups
        check_no_intersection_indices_same_group(split_tensors, axis, names)
        check_equivalent_indices_different_groups(split_tensors, axis, names)
        # check other metadata
        check_other_metadata(split_tensors, axis)
        # check values
        check_values(test_tensor_map, split_tensors, target_grouped_labels)

    def test_split_data_abs_rel_same_result_samples(self, test_tensor_map, request):
        """
        Tests that splitting data using the `group_sizes` arg in absolute and
        relative mode gives equivalent results.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # There are 8 unique samples indices - [0, 1, 2, 3, 4, 5, 6, 8]
        target_grouped_labels = [
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
        split_tensors_1, actual_grouped_labels_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1, 2, 4, 1],  # absolute
        )
        split_tensors_2, actual_grouped_labels_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1 / 8, 2 / 8, 4 / 8, 1 / 8],  # relative
        )
        assert_equal(target_grouped_labels, actual_grouped_labels_1)
        assert_equal(actual_grouped_labels_1, actual_grouped_labels_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)

    def test_split_data_abs_rel_same_result_properties(self, test_tensor_map, request):
        """
        Tests that splitting data using the `group_sizes` arg in absolute and
        relative mode gives equivalent results.
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # There are 4 unique properties indices - [0, 3, 4, 5]
        target_grouped_labels = [
            Labels(
                names=["properties"],
                values=np.array([[0]]).reshape(-1, 1),
            ),
            Labels(
                names=["properties"],
                values=np.array([[3, 4]]).reshape(-1, 1),
            ),
            Labels(
                names=["properties"],
                values=np.array([[5]]).reshape(-1, 1),
            ),
        ]
        # Split the data using abs and rel group sizes
        axis, names, n_groups = "properties", "properties", 3
        split_tensors_1, actual_grouped_labels_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1, 2, 1],  # absolute
        )
        split_tensors_2, actual_grouped_labels_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1 / 4, 2 / 4, 1 / 4],  # relative
        )
        assert_equal(target_grouped_labels, actual_grouped_labels_1)
        assert_equal(actual_grouped_labels_1, actual_grouped_labels_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)

    def test_split_data_abs_rel_same_result_less_than_n(self, test_tensor_map, request):
        """
        Tests that splitting data using the `group_sizes` arg in abs and rel
        modes gives equivalent results. Completes for passing these as summing
        to less than the number of unique indices (in the case of abs mode) or
        less than 1 (in the case of rel mode).
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # There are 8 unique structure idxs in total: [0, 1, 2, 3, 4, 5, 6, 8]
        # Split only the first 7 of these
        target_grouped_labels = [
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
        split_tensors_1, actual_grouped_labels_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1, 2, 4],  # adds up to 7
        )
        split_tensors_2, actual_grouped_labels_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1 / 8, 2 / 8, 4 / 8],  # adds up to 7/8
        )
        assert_equal(target_grouped_labels, actual_grouped_labels_1)
        assert_equal(actual_grouped_labels_1, actual_grouped_labels_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)

    def test_split_data_unequal_groups(self, test_tensor_map, request):
        """
        Tests that splitting data into equal groups using only the `n_groups`
        splits as expected when the number of unique indices is not divisible
        by `n_groups`. In this case, there are 8 unique structure idxs in the
        test tensor maps,
        """
        # Get the parameterized TensorMap
        test_tensor_map = request.getfixturevalue(test_tensor_map)
        # There are 8 unique structure idxs in total: [0, 1, 2, 3, 4, 5, 6, 8]
        # Split only the first 7 of these
        target_grouped_labels = [
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
        split_tensors_1, actual_grouped_labels_1 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1, 2, 4],  # adds up to 7
        )
        split_tensors_2, actual_grouped_labels_2 = split_data(
            tensors=test_tensor_map,
            axis=axis,
            names=names,
            n_groups=n_groups,
            group_sizes=[1 / 8, 2 / 8, 4 / 8],  # adds up to 7/8
        )
        assert_equal(target_grouped_labels, actual_grouped_labels_1)
        assert_equal(actual_grouped_labels_1, actual_grouped_labels_2)
        for i, j in zip(split_tensors_1[0], split_tensors_2[0]):
            assert equistore.equal(i, j)


class TestSplitDataErrors:
    def test_split_data_errors_arg_tensors(self, test_tensor_map_a):
        """
        Checks for exceptions on the `tensors` arg.
        """
        # Not passing a TensorMap
        tensors = 5
        msg = f"`tensors` must be a list of equistore `TensorMap`, got {type(tensors)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=tensors,
                axis="samples",
                names="samples",
                n_groups=3,
            )
        # Not passing a list of TensorMap
        tensors = [test_tensor_map_a, 3]
        msg = f"`tensors` must be a list of equistore `TensorMap`, got {type(tensors)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=tensors,
                axis="samples",
                names="samples",
                n_groups=3,
            )
        # Passing just a TensorMap, not in a list is ok
        split_data(
            tensors=test_tensor_map_a,
            axis="samples",
            names="samples",
            n_groups=3,
        )

    def test_split_data_errors_arg_axis(self, test_tensor_map_a):
        """
        Checks for exceptions on the `axis` arg.
        """
        # Passing axis not as a string
        axis = 3.14
        msg = f"`axis` must be passed as a str, got {type(axis)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis=axis,
                names="samples",
                n_groups=3,
            )
        # Passing axis as an incorrect str\
        axis = "not_samples"
        msg = f"`axis` must be passsed as either 'samples' or 'properties', got {axis}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis=axis,
                names="samples",
                n_groups=3,
            )

    def test_split_data_errors_arg_names(self, test_tensor_map_a):
        """
        Checks for exceptions on the `names` arg.
        """
        # Passing axis not as a list
        names = 3.14
        msg = f"`names` must be a list of str, got {type(names)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names=names,
                n_groups=3,
            )
        # Passing axis not as a list
        names = [3.14, 6.28]
        msg = f"`names` must be a list of str, got {type(names)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names=names,
                n_groups=3,
            )
        # Passing non-existent names
        axis = "samples"
        names = ["not_samples"]
        tmp_names = ["samples"]
        msg = (
            f"the passed `TensorMap` objects have {axis} names {tmp_names}"
            f" that do not match the one passed in `names` {names}"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis=axis,
                names=names,
                n_groups=3,
            )

    def test_split_data_errors_arg_n_groups(self, test_tensor_map_a):
        """
        Checks for exceptions on the `n_groups` arg.
        """
        # Passing n_groups not as an int
        n_groups = 3.14
        msg = f"`n_groups` must be passed as an int, got {type(n_groups)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names="samples",
                n_groups=n_groups,
            )
        # Passing n_groups as a negative int
        n_groups = -3
        msg = f"`n_groups` must be greater than 0, got {n_groups}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names="samples",
                n_groups=n_groups,
            )

    def test_split_data_errors_arg_group_sizes(self, test_tensor_map_a):
        """
        Checks for exceptions on the `group_sizes` arg.
        """
        # Passing group_sizes not as a list
        group_sizes = 3.14
        msg = (
            "`group_sizes` must be passed as a list of float or int,"
            f" got {type(group_sizes)}"
        )
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names="samples",
                n_groups=3,
                group_sizes=group_sizes,
            )
        # Passing group_sizes not as a list of int or float
        group_sizes = [3.14, "3.14", "6.28"]
        msg = (
            "`group_sizes` must be passed as a list of float or int,"
            f" got {type(group_sizes)}"
        )
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="properties",
                names="properties",
                n_groups=3,
                group_sizes=group_sizes,
            )
        # Passing group_sizes as a list of negative int or float
        group_sizes = [-3, 3]
        msg = f"all elements of `group_sizes` must be greater than 0, got {group_sizes}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names="samples",
                n_groups=2,
                group_sizes=group_sizes,
            )
        # Passing group_sizes as a list of float whose sum is > 1
        group_sizes = [0.7, 0.4]
        msg = (
            "if specifying `group_sizes` as a list of float, the sum of"
            " the list must be less than or equal to 1"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="properties",
                names="properties",
                n_groups=2,
                group_sizes=group_sizes,
            )
        # Passing group_sizes as a list of int whose sum is greater than the
        # number of unique properties
        group_sizes = [3, 3]
        unique_idxs = equistore.unique_metadata(
            test_tensor_map_a, "properties", "properties"
        )
        msg = (
            f"the sum of the absolute group sizes ({sum(group_sizes)}) is greater"
            f" than the number of unique metadata indices ({4}) for the chosen "
            f"axis {'properties'} and names {['properties']}: {unique_idxs}"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="properties",
                names="properties",
                n_groups=2,
                group_sizes=group_sizes,
            )
        # Passing group_sizes as a list of a single int which is greater than
        # the number of unique properties
        group_sizes = [1000]
        unique_idxs = equistore.unique_metadata(
            test_tensor_map_a, "properties", "properties"
        )
        msg = (
            f"the sum of the absolute group sizes ({sum(group_sizes)}) is greater"
            f" than the number of unique metadata indices ({4}) for the chosen "
            f"axis {'properties'} and names {['properties']}: {unique_idxs}"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="properties",
                names="properties",
                n_groups=1,
                group_sizes=group_sizes,
            )

    def test_split_data_errors_seed(self, test_tensor_map_a):
        """
        Checks for exceptions on the `seed` arg.
        """
        # Passing seed not as an int
        seed = 3.14
        msg = f"`seed` must be passed as an `int`, got {type(seed)}"
        with pytest.raises(TypeError, match=re.escape(msg)):
            split_data(
                tensors=test_tensor_map_a,
                axis="samples",
                names="samples",
                n_groups=3,
                seed=seed,
            )
